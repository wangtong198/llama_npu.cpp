"""
GGML_OP_SOFT_MAX 的 PyTorch 参考实现（对齐 ggml_compute_forward_soft_max_f32，ops.cpp）。

约定张量形状与 softmax.md / GGML ne 一致：
  inp:  [n_kv, n_tokens_per_stream, n_heads, n_stream]   FP32
  mask: [n_kv, mask_ne1, n_heads_or_1, n_stream]         FP16 或 FP32（可选）
  sinks: [n_heads]                                     FP32（可选）

softmax 沿 n_kv（ne0）逐行计算；scale、max_bias（alibi_bias）含义与 GGML op_params 一致。
"""
from __future__ import annotations

import math
import sys
from typing import Optional, Tuple

import torch


def alibi_slopes(n_heads: int, max_bias: float, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    L = 2^floor(log2(n_heads))
    m0 = 2^(-max_bias/L), m1 = 2^(-max_bias/(2*L))
    slope(h) = 1 (max_bias<=0); 否则 h<L 用 m0^(h+1)，否则 m1^(2*(h-L)+1)。
    与 ops.cpp 一致。
    """
    device = device or torch.device("cpu")
    h = torch.arange(n_heads, device=device, dtype=torch.int64)
    if max_bias <= 0.0:
        return torch.ones((n_heads,), device=device, dtype=dtype)

    n_head_log2 = 1 << int(math.floor(math.log2(float(n_heads))))
    m0 = math.pow(2.0, -(max_bias) / float(n_head_log2))
    m1 = math.pow(2.0, -(max_bias / 2.0) / float(n_head_log2))

    slope = torch.empty((n_heads,), device=device, dtype=dtype)
    mask_low = h < n_head_log2
    slope[mask_low] = torch.tensor(m0, dtype=dtype, device=device).pow(h[mask_low].to(dtype).add(1.0))
    hl = h[~mask_low].to(dtype) - float(n_head_log2)
    slope[~mask_low] = torch.tensor(m1, dtype=dtype, device=device).pow(2.0 * hl + 1.0)
    return slope


def _promote_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.float16:
        return mask.to(torch.float32)
    return mask


def ggml_soft_max(
    inp: torch.Tensor,
    scale: float,
    max_bias: float,
    attn_mask: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """
    inp: [n_kv, T, H, S]  contiguous 逻辑与 GGML 一致；沿 dim=0（n_kv）做 softmax。

    有 mask 时：z_i = scale * s_i + slope(h) * M_i（逐元素，沿 n_kv）。
    有 sinks：m = max(max(z), sinks[h])，分母 S = sum(exp(z-m)) + exp(sinks[h]-m)。
    """
    if inp.dtype != torch.float32:
        raise ValueError("inp must be FP32")
    if inp.dim() != 4:
        raise ValueError("inp must be 4-D [n_kv, T, H, S]")

    n_kv, t, n_heads, n_s = inp.shape
    slopes = alibi_slopes(n_heads, max_bias, device=inp.device, dtype=torch.float32)

    # (S, H, T, K) 与 ops 中沿 ne00 的向量一致：最后一维为 n_kv
    x = inp.permute(3, 2, 1, 0).contiguous()
    if attn_mask is not None:
        m = _promote_mask(attn_mask)
        if m.shape[0] != n_kv or m.shape[3] != n_s:
            raise ValueError("attn_mask ne[0]/ne[3] must match inp")
        if m.shape[1] < t or m.shape[2] not in (1, n_heads):
            raise ValueError("attn_mask broadcast shape invalid")
        m = m[:, :t, :, :].permute(3, 2, 1, 0).contiguous()
        # m: (S, Hm, T, K)，Hm 为 1 时对所有 head 广播
        wp = x * float(scale) + m * slopes.view(1, n_heads, 1, 1)
    else:
        if max_bias > 0.0:
            raise ValueError("alibi_bias > 0 requires attn_mask (per softmax.md)")
        wp = x * float(scale)

    # wp: (S, H, T, K) — 沿 K softmax
    wp_max = wp.max(dim=-1).values  # (S, H, T)
    if sinks is not None:
        if sinks.shape != (n_heads,):
            raise ValueError("sinks must be [n_heads]")
        sk = sinks.to(device=wp.device, dtype=torch.float32).view(1, n_heads, 1)
        wp_max = torch.maximum(wp_max, sk)

    ex = torch.exp(wp - wp_max.unsqueeze(-1))
    denom = ex.sum(dim=-1)  # (S, H, T)
    if sinks is not None:
        sk = sinks.to(device=wp.device, dtype=torch.float32).view(1, n_heads, 1)
        denom = denom + torch.exp(sk - wp_max)

    out = ex / denom.unsqueeze(-1)
    out_perm = out.permute(3, 2, 1, 0).contiguous()

    if inplace:
        inp.copy_(out_perm)
        return inp
    return out_perm


def ggml_soft_max_loop_reference(
    inp: torch.Tensor,
    scale: float,
    max_bias: float,
    attn_mask: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """三重循环参考实现，用于与向量化版对拍。"""
    n_kv, ne1, ne2, ne3 = inp.shape
    slopes = alibi_slopes(ne2, max_bias, device=inp.device, dtype=torch.float32)
    out = torch.empty_like(inp)
    mask_f = _promote_mask(attn_mask) if attn_mask is not None else None

    for i3 in range(ne3):
        for i2 in range(ne2):
            for i1 in range(ne1):
                sp = inp[:, i1, i2, i3]
                wp = sp * scale
                if mask_f is not None:
                    i12 = i2 % mask_f.shape[2]
                    i13 = i3 % mask_f.shape[3]
                    mp = mask_f[:, i1, i12, i13]
                    wp = wp + slopes[i2] * mp
                mx = wp.max()
                if sinks is not None:
                    mx = torch.maximum(mx, sinks[i2])
                t_exp = torch.exp(wp - mx)
                ssum = t_exp.sum()
                if sinks is not None:
                    ssum = ssum + torch.exp(sinks[i2] - mx)
                out[:, i1, i2, i3] = t_exp / ssum
    return out


def _assert_close(a: torch.Tensor, b: torch.Tensor, msg: str, atol: float = 1e-5) -> None:
    d = (a - b).abs().max().item()
    if d > atol:
        raise AssertionError(f"{msg}: max abs diff {d} > {atol}")


def run_checks() -> None:
    torch.manual_seed(0)
    n_kv, T, H, S = 32, 4, 8, 2
    scale = 1.0 / math.sqrt(64.0)

    # 1) 无 mask、无 alibi、无 sink：应对齐标准 softmax(scale * x) 沿 n_kv
    inp = torch.randn(n_kv, T, H, S, dtype=torch.float32)
    y = ggml_soft_max(inp, scale, 0.0, None, None)
    ref = torch.softmax(inp * scale, dim=0)
    _assert_close(y, ref, "no mask / no sink vs torch.softmax on dim 0")

    ref_loop = ggml_soft_max_loop_reference(inp, scale, 0.0, None, None)
    _assert_close(y, ref_loop, "vectorized vs loop no mask")

    # 2) 仅 mask（无 alibi 斜率，max_bias=0 时 slope=1，等价于加 mask 后 softmax… 但 max_bias=0 时公式是无 mask 分支）
    # 文档：无 mask 时 z=scale*s；有 mask 时 z=scale*s+slope*M。若 max_bias=0，GGML slope=1，仍需走 mask 分支才能加 M。
    # 校验：max_bias=0 + mask，应等价于 softmax(scale*inp + mask) on dim 0
    mask = torch.randn(n_kv, T, 1, S, dtype=torch.float32) * 0.1
    y_m = ggml_soft_max(inp, scale, 0.0, mask, None)
    ref_m = torch.softmax(inp * scale + mask, dim=0)
    _assert_close(y_m, ref_m, "mask only max_bias=0")
    _assert_close(y_m, ggml_soft_max_loop_reference(inp, scale, 0.0, mask, None), "mask loop")

    # 3) Alibi：max_bias>0 必须带 mask
    mb = 8.0
    y_alibi = ggml_soft_max(inp, scale, mb, mask, None)
    _assert_close(y_alibi, ggml_soft_max_loop_reference(inp, scale, mb, mask, None), "alibi loop")

    # 4) FP16 mask
    mask_h = mask.half()
    y_h = ggml_soft_max(inp, scale, 0.0, mask_h, None)
    _assert_close(y_h, y_m, "f16 mask vs f32", atol=2e-4)

    # 5) sinks
    sk = torch.randn(H, dtype=torch.float32) * 0.05
    y_sk = ggml_soft_max(inp, scale, 0.0, None, sk)
    _assert_close(y_sk, ggml_soft_max_loop_reference(inp, scale, 0.0, None, sk), "sinks loop")
    y_sk2 = ggml_soft_max(inp, scale, mb, mask, sk)
    _assert_close(y_sk2, ggml_soft_max_loop_reference(inp, scale, mb, mask, sk), "sinks+alibi loop")

    print("all softmax_torch checks: PASS", file=sys.stderr)


def main() -> None:
    run_checks()


if __name__ == "__main__":
    main()
