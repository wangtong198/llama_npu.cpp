"""
与 print_rope_cache_ggml.c 一致：ggml_rope_yarn_corr_dims（ggml.c）、
rope_yarn / ggml_rope_cache_init、rotate_pairs（NORMAL）、rope_copy_tail、自测。
全部 cache / 旋转中间量使用 float32，以对齐 ggml_compute_forward_rope_flt。
"""
from __future__ import annotations

import math
import sys
from typing import Optional, Sequence, Tuple

import numpy as np

ROPE_TEST_EPS = np.float32(1e-4)


def ggml_rope_yarn_corr_dim(n_dims: int, n_ctx_orig: int, n_rot: float, base: float) -> np.float32:
    return np.float32(
        n_dims
        * math.log(n_ctx_orig / (n_rot * 2.0 * math.pi))
        / (2.0 * math.log(base))
    )


def ggml_rope_yarn_corr_dims(
    n_dims: int,
    n_ctx_orig: int,
    freq_base: float,
    beta_fast: float,
    beta_slow: float,
) -> Tuple[np.float32, np.float32]:
    start = np.float32(math.floor(float(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base))))
    end = np.float32(math.ceil(float(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base))))
    d0 = max(np.float32(0.0), start)
    d1 = min(np.float32(n_dims - 1), end)
    return d0, d1


def rope_yarn_ramp(low: float, high: float, i0: int) -> np.float32:
    y = (np.float32(i0 // 2) - np.float32(low)) / np.maximum(
        np.float32(0.001), np.float32(high) - np.float32(low)
    )
    return np.float32(1.0) - np.minimum(
        np.float32(1.0), np.maximum(np.float32(0.0), y)
    )


def rope_yarn(
    theta_extrap: np.float32,
    freq_scale: float,
    corr_dims: Tuple[np.float32, np.float32],
    i0: int,
    ext_factor: float,
    mscale: np.float32,
) -> Tuple[np.float32, np.float32]:
    theta_interp = np.float32(freq_scale) * np.float32(theta_extrap)
    theta = theta_interp
    if ext_factor != 0.0:
        ramp_mix = rope_yarn_ramp(float(corr_dims[0]), float(corr_dims[1]), i0) * np.float32(ext_factor)
        theta = theta_interp * (np.float32(1.0) - ramp_mix) + np.float32(theta_extrap) * ramp_mix
        mscale = mscale * (np.float32(1.0) + np.float32(0.1) * np.log(np.float32(1.0) / np.float32(freq_scale)))
    c = np.cos(theta) * mscale
    s = np.sin(theta) * mscale
    return np.float32(c), np.float32(s)


def ggml_rope_cache_init(
    m: float,
    freq_scale: float,
    freq_factors: Optional[Sequence[float]],
    corr_dims: Tuple[np.float32, np.float32],
    ne0: int,
    ext_factor: float,
    attn_factor: float,
    sin_sign: float,
    theta_scale: np.float32,
) -> np.ndarray:
    cache = np.zeros((ne0,), dtype=np.float32)
    theta = np.float32(m)
    for i0 in range(0, ne0, 2):
        ff = np.float32(freq_factors[i0 // 2]) if freq_factors is not None else np.float32(1.0)
        mscale = np.float32(attn_factor)
        cos_t, sin_t = rope_yarn(
            np.float32(theta / ff),
            freq_scale,
            corr_dims,
            i0,
            ext_factor,
            mscale,
        )
        cache[i0] = cos_t
        cache[i0 + 1] = sin_t * np.float32(sin_sign)
        theta = theta * theta_scale
    return cache


def rotate_pairs_f32_normal(
    n_dims: int,
    n_offset: int,
    cache: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    scale: int,
) -> None:
    """与 ops.cpp rotate_pairs<float>(n_dims, n_offset, cache, src, dst, scale)；NORMAL 时 n_offset=1, scale=1。"""
    for i0 in range(0, n_dims, 2):
        ic = i0 // scale
        cos_theta = cache[i0]
        sin_theta = cache[i0 + 1]
        x0 = np.float32(src[ic])
        x1 = np.float32(src[ic + n_offset])
        dst[ic] = np.float32(x0 * cos_theta - x1 * sin_theta)
        dst[ic + n_offset] = np.float32(x0 * sin_theta + x1 * cos_theta)


def rope_copy_tail(n_dims: int, ne0: int, src: np.ndarray, dst: np.ndarray) -> None:
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

    theta_scale = np.power(np.float32(freq_base), np.float32(-2.0 / n_dims))
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

    src = np.array(
        [math.sin(i * 0.13) + 0.25 * (i % 7) for i in range(head_dim)],
        dtype=np.float32,
    )
    dst = np.zeros((head_dim,), dtype=np.float32)
    rotate_pairs_f32_normal(n_dims, 1, cache, src, dst, 1)
    rope_copy_tail(n_dims, head_dim, src, dst)

    for i0 in range(0, n_dims, 2):
        c = cache[i0]
        s = cache[i0 + 1]
        x0 = src[i0]
        x1 = src[i0 + 1]
        e0 = np.float32(x0 * c - x1 * s)
        e1 = np.float32(x0 * s + x1 * c)
        if abs(float(dst[i0]) - float(e0)) > float(ROPE_TEST_EPS) or abs(
            float(dst[i0 + 1]) - float(e1)
        ) > float(ROPE_TEST_EPS):
            raise AssertionError(
                f"FAIL pair i0={i0}: dst=({dst[i0]}, {dst[i0+1]}) expect=({e0}, {e1})"
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
    src2 = np.array([i * 0.01 for i in range(hd)], dtype=np.float32)
    dst2 = np.zeros((hd,), dtype=np.float32)
    rotate_pairs_f32_normal(nd, 1, cache2, src2, dst2, 1)
    rope_copy_tail(nd, hd, src2, dst2)
    for i in range(nd, hd):
        if abs(float(dst2[i]) - float(src2[i])) > float(ROPE_TEST_EPS):
            raise AssertionError(f"FAIL tail copy i={i}: dst={dst2[i]} src={src2[i]}")

    print("test_rope_yarn_rotate_normal: PASS", file=sys.stderr)


def main() -> None:
    test_rope_yarn_rotate_normal()

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

    theta_scale = np.power(np.float32(freq_base), np.float32(-2.0 / n_dims))
    corr_dims = ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow)

    print(
        f"corr_dims[0]={float(corr_dims[0]):.6f} corr_dims[1]={float(corr_dims[1]):.6f} "
        f"theta_scale={float(theta_scale):.9g}"
    )

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

    print(f"cache (ne0={head_dim}, float32), one row for position m={p}:")
    print(cache)

    src_demo = np.array([(i + 1) * 0.01 for i in range(head_dim)], dtype=np.float32)
    dst_demo = np.zeros((head_dim,), dtype=np.float32)
    rotate_pairs_f32_normal(n_dims, 1, cache, src_demo, dst_demo, 1)
    rope_copy_tail(n_dims, head_dim, src_demo, dst_demo)
    print("dst[0..7] after rotate_pairs (demo src[i]=(i+1)*0.01):")
    print(dst_demo[:8])


if __name__ == "__main__":
    main()
