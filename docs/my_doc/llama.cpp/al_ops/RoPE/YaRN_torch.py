"""
用 PyTorch（float32）实现：corr_dims、rope_yarn、
rope cache、rotate_pairs（NORMAL）、rope_copy_tail、自测与演示输出。
"""
import math
import sys
from typing import Optional, Sequence, Tuple, Union

import torch

ROPE_TEST_EPS = 1e-4


def _f32(x: Union[float, int, torch.Tensor]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    return torch.tensor(float(x), dtype=torch.float32)


def ggml_rope_yarn_corr_dim(n_dims: int, n_ctx_orig: int, n_rot: float, base: float) -> torch.Tensor:
    v = (
        n_dims
        * math.log(n_ctx_orig / (n_rot * 2.0 * math.pi))
        / (2.0 * math.log(base))
    )
    return torch.tensor(v, dtype=torch.float32)


def ggml_rope_yarn_corr_dims(
    n_dims: int,
    n_ctx_orig: int,
    freq_base: float,
    beta_fast: float,
    beta_slow: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    start = torch.tensor(
        math.floor(float(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base))),
        dtype=torch.float32,
    )
    end = torch.tensor(
        math.ceil(float(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base))),
        dtype=torch.float32,
    )
    d0 = torch.maximum(_f32(0.0), start)
    d1 = torch.minimum(_f32(n_dims - 1), end)
    return d0, d1


def rope_yarn_ramp(low: float, high: float, i0: int) -> torch.Tensor:
    y = (_f32(i0 // 2) - _f32(low)) / torch.maximum(
        _f32(0.001),
        _f32(high) - _f32(low),
    )
    return _f32(1.0) - torch.minimum(
        _f32(1.0),
        torch.maximum(_f32(0.0), y),
    )


def rope_yarn(
    theta_extrap: Union[torch.Tensor, float],
    freq_scale: float,
    corr_dims: Tuple[torch.Tensor, torch.Tensor],
    i0: int,
    ext_factor: float,
    mscale: Union[torch.Tensor, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    theta_extrap = _f32(theta_extrap)
    mscale = _f32(mscale)
    theta_interp = _f32(freq_scale) * theta_extrap
    theta = theta_interp
    if ext_factor != 0.0:
        ramp_mix = rope_yarn_ramp(float(corr_dims[0]), float(corr_dims[1]), i0) * _f32(ext_factor)
        theta = theta_interp * (_f32(1.0) - ramp_mix) + theta_extrap * ramp_mix
        mscale = mscale * (
            _f32(1.0) + _f32(0.1) * torch.log(_f32(1.0) / _f32(freq_scale))
        )
    c = torch.cos(theta) * mscale
    s = torch.sin(theta) * mscale
    return c.to(torch.float32), s.to(torch.float32)


def ggml_rope_cache_init(
    m: float,
    freq_scale: float,
    freq_factors: Optional[Sequence[float]],
    corr_dims: Tuple[torch.Tensor, torch.Tensor],
    ne0: int,
    ext_factor: float,
    attn_factor: float,
    sin_sign: float,
    theta_scale: Union[torch.Tensor, float],
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    theta_scale = _f32(theta_scale)
    cache = torch.zeros((ne0,), dtype=torch.float32, device=device)
    theta = _f32(m)
    for i0 in range(0, ne0, 2):
        ff = _f32(freq_factors[i0 // 2]) if freq_factors is not None else _f32(1.0)
        mscale = _f32(attn_factor)
        cos_t, sin_t = rope_yarn(
            theta / ff,
            freq_scale,
            corr_dims,
            i0,
            ext_factor,
            mscale,
        )
        cache[i0] = cos_t
        cache[i0 + 1] = sin_t * _f32(sin_sign)
        theta = theta * theta_scale
    return cache


def rotate_pairs_f32_normal(
    n_dims: int,
    n_offset: int,
    cache: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    scale: int,
) -> None:
    """与 ops.cpp rotate_pairs<float>；NORMAL 时 n_offset=1, scale=1。原地写入 dst。"""
    for i0 in range(0, n_dims, 2):
        ic = i0 // scale
        cos_theta = cache[i0]
        sin_theta = cache[i0 + 1]
        x0 = src[ic].to(torch.float32)
        x1 = src[ic + n_offset].to(torch.float32)
        dst[ic] = (x0 * cos_theta - x1 * sin_theta).to(torch.float32)
        dst[ic + n_offset] = (x0 * sin_theta + x1 * cos_theta).to(torch.float32)


def rope_copy_tail(n_dims: int, ne0: int, src: torch.Tensor, dst: torch.Tensor) -> None:
    for i0 in range(n_dims, ne0, 2):
        dst[i0] = src[i0]
        dst[i0 + 1] = src[i0 + 1]


def test_rope_yarn_rotate_normal() -> None:
    p = 128
    n_dims = 128
    head_dim = 128
    freq_base = 10000.0
    freq_scale = 0.5
    ext_factor = 1.0
    attn_factor = 1.0
    beta_fast = 32.0
    beta_slow = 1.0
    n_ctx_orig = 4096
    sin_sign = 1.0

    theta_scale = torch.tensor(freq_base, dtype=torch.float32).pow(
        torch.tensor(-2.0 / n_dims, dtype=torch.float32)
    )
    corr_dims = ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow)

    cache = ggml_rope_cache_init(
        float(p),
        freq_scale,
        None,
        corr_dims,
        head_dim,
        ext_factor,
        attn_factor,
        sin_sign,
        theta_scale,
    )

    src = torch.tensor(
        [math.sin(i * 0.13) + 0.25 * (i % 7) for i in range(head_dim)],
        dtype=torch.float32,
    )
    dst = torch.zeros((head_dim,), dtype=torch.float32)
    rotate_pairs_f32_normal(n_dims, 1, cache, src, dst, 1)
    rope_copy_tail(n_dims, head_dim, src, dst)

    for i0 in range(0, n_dims, 2):
        c = cache[i0]
        s = cache[i0 + 1]
        x0 = src[i0]
        x1 = src[i0 + 1]
        e0 = (x0 * c - x1 * s).to(torch.float32)
        e1 = (x0 * s + x1 * c).to(torch.float32)
        if (dst[i0] - e0).abs().item() > ROPE_TEST_EPS or (dst[i0 + 1] - e1).abs().item() > ROPE_TEST_EPS:
            raise AssertionError(
                f"FAIL pair i0={i0}: dst=({dst[i0].item()}, {dst[i0+1].item()}) expect=({e0.item()}, {e1.item()})"
            )

    hd, nd = 160, 128
    cache2 = ggml_rope_cache_init(
        float(p),
        freq_scale,
        None,
        corr_dims,
        hd,
        ext_factor,
        attn_factor,
        sin_sign,
        theta_scale,
    )
    src2 = torch.tensor([i * 0.01 for i in range(hd)], dtype=torch.float32)
    dst2 = torch.zeros((hd,), dtype=torch.float32)
    rotate_pairs_f32_normal(nd, 1, cache2, src2, dst2, 1)
    rope_copy_tail(nd, hd, src2, dst2)
    for i in range(nd, hd):
        if (dst2[i] - src2[i]).abs().item() > ROPE_TEST_EPS:
            raise AssertionError(
                f"FAIL tail copy i={i}: dst={dst2[i].item()} src={src2[i].item()}"
            )

    print("test_rope_yarn_rotate_normal: PASS", file=sys.stderr)


def main() -> None:
    test_rope_yarn_rotate_normal()

    # p = 128
    # n_dims = 128
    # head_dim = 128
    # freq_base = 10000.0
    # freq_scale = 0.5
    # ext_factor = 1.0
    # attn_factor = 1.0
    # beta_fast = 32.0
    # beta_slow = 1.0
    # n_ctx_orig = 4096
    # sin_sign = 1.0

    # theta_scale = torch.tensor(freq_base, dtype=torch.float32).pow(
    #     torch.tensor(-2.0 / n_dims, dtype=torch.float32)
    # )
    # corr_dims = ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow)

    # print(
    #     f"corr_dims[0]={float(corr_dims[0]):.6f} corr_dims[1]={float(corr_dims[1]):.6f} "
    #     f"theta_scale={float(theta_scale):.9g}"
    # )

    # cache = ggml_rope_cache_init(
    #     float(p),
    #     freq_scale,
    #     None,
    #     corr_dims,
    #     head_dim,
    #     ext_factor,
    #     attn_factor,
    #     sin_sign,
    #     theta_scale,
    # )

    # print(f"cache (ne0={head_dim}, float32), one row for position m={p}:")
    # print(cache)

    # src_demo = torch.tensor([(i + 1) * 0.01 for i in range(head_dim)], dtype=torch.float32)
    # dst_demo = torch.zeros((head_dim,), dtype=torch.float32)
    # rotate_pairs_f32_normal(n_dims, 1, cache, src_demo, dst_demo, 1)
    # rope_copy_tail(n_dims, head_dim, src_demo, dst_demo)
    # print("dst[0..7] after rotate_pairs (demo src[i]=(i+1)*0.01):")
    # print(dst_demo[:8])


if __name__ == "__main__":
    main()
