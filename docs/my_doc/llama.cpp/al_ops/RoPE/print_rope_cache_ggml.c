/*
 * 与 ggml_compute_forward_rope_flt 中 cache 与旋转一致：
 *   ggml_rope_cache_init / rope_yarn / rope_yarn_ramp（ops.cpp）
 *   ggml_rope_yarn_corr_dims（ggml.c）
 *   rotate_pairs<T>(n_dims, 1, cache, src, dst, 1) —— GGML_ROPE_TYPE_NORMAL（ops.cpp）
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GGML_MAX(a, b) ((a) > (b) ? (a) : (b))
#define GGML_MIN(a, b) ((a) < (b) ? (a) : (b))

#ifndef ROPE_TEST_EPS
#define ROPE_TEST_EPS 1e-4f
#endif

static float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2.0f * (float)M_PI)) / (2.0f * logf(base));
}

static void ggml_rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   =  ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = GGML_MAX(0.0f, start);
    dims[1] = GGML_MIN((float)(n_dims - 1), end);
}

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = ((float)(i0 / 2) - low) / GGML_MAX(0.001f, high - low);
    return 1.0f - GGML_MIN(1.0f, GGML_MAX(0.0f, y));
}

static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta
) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static void ggml_rope_cache_init(
    float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0,
    float ext_factor, float mscale, float * cache, float sin_sign, float theta_scale
) {
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn(
            theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;
        theta *= theta_scale;
    }
}

/* 与 ops.cpp rotate_pairs<float>(n_dims, 1, cache, src, dst, 1) 一致：NORMAL 布局，相邻两维一对 */
static void rotate_pairs_f32_normal(
    int64_t n_dims, int64_t n_offset, const float * cache, const float * src_data, float * dst_data, int scale
) {
    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
        const int64_t ic = i0 / scale;
        const float cos_theta = cache[i0 + 0];
        const float sin_theta = cache[i0 + 1];
        const float * src = src_data + ic;
        float * dst = dst_data + ic;
        const float x0 = src[0];
        const float x1 = src[n_offset];
        dst[0] = x0 * cos_theta - x1 * sin_theta;
        dst[n_offset] = x0 * sin_theta + x1 * cos_theta;
    }
}

/* 与 ggml_compute_forward_rope_flt 中 !is_vision 时对尾部维的拷贝一致 */
static void rope_copy_tail(int64_t n_dims, int64_t ne0, const float * src, float * dst) {
    for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
        dst[i0 + 0] = src[i0 + 0];
        dst[i0 + 1] = src[i0 + 1];
    }
}

/* ---------- 测试：与公式逐对校验 + n_dims < head_dim 时尾部直通 ---------- */
static int test_rope_yarn_rotate_normal(void) {
    const int64_t p = 128;
    const int n_dims = 128;
    const int head_dim = 128;
    const float freq_base = 10000.0f;
    const float freq_scale = 0.5f;
    const float ext_factor = 1.0f;
    const float attn_factor = 1.0f;
    const float beta_fast = 32.0f;
    const float beta_slow = 1.0f;
    const int n_ctx_orig = 4096;
    const float sin_sign = 1.0f;

    const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    float cache[256];
    memset(cache, 0, sizeof(cache));
    ggml_rope_cache_init(
        (float)p, freq_scale, NULL, corr_dims, (int64_t)head_dim,
        ext_factor, attn_factor, cache, sin_sign, theta_scale
    );

    float src[256];
    float dst[256];
    for (int i = 0; i < head_dim; i++) {
        src[i] = sinf((float)i * 0.13f) + 0.25f * (float)(i % 7);
    }
    memset(dst, 0, sizeof(dst));

    rotate_pairs_f32_normal((int64_t)n_dims, 1, cache, src, dst, 1);
    rope_copy_tail((int64_t)n_dims, (int64_t)head_dim, src, dst);

    /* 逐块与显式公式对照（NORMAL：块内为 src[i0], src[i0+1]） */
    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
        float c = cache[i0];
        float s = cache[i0 + 1];
        float x0 = src[i0];
        float x1 = src[i0 + 1];
        float e0 = x0 * c - x1 * s;
        float e1 = x0 * s + x1 * c;
        if (fabsf(dst[i0] - e0) > ROPE_TEST_EPS || fabsf(dst[i0 + 1] - e1) > ROPE_TEST_EPS) {
            fprintf(stderr, "FAIL pair i0=%lld: dst=(%.7g,%.7g) expect=(%.7g,%.7g)\n",
                    (long long)i0, dst[i0], dst[i0 + 1], e0, e1);
            return 1;
        }
    }

    /* head_dim > n_dims 时尾部应与 src 相同 */
    {
        const int hd = 160;
        const int nd = 128;
        float cache2[256];
        float src2[256];
        float dst2[256];
        memset(cache2, 0, sizeof(cache2));
        ggml_rope_cache_init(
            (float)p, freq_scale, NULL, corr_dims, (int64_t)hd,
            ext_factor, attn_factor, cache2, sin_sign, theta_scale
        );
        for (int i = 0; i < hd; i++) {
            src2[i] = (float)i * 0.01f;
        }
        memset(dst2, 0, sizeof(dst2));
        rotate_pairs_f32_normal((int64_t)nd, 1, cache2, src2, dst2, 1);
        rope_copy_tail((int64_t)nd, (int64_t)hd, src2, dst2);
        for (int i = nd; i < hd; i++) {
            if (fabsf(dst2[i] - src2[i]) > ROPE_TEST_EPS) {
                fprintf(stderr, "FAIL tail copy i=%d: dst=%.7g src=%.7g\n", i, dst2[i], src2[i]);
                return 2;
            }
        }
    }

    printf("test_rope_yarn_rotate_normal: PASS\n");
    return 0;
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;

    if (test_rope_yarn_rotate_normal() != 0) {
        return 1;
    }

    const int64_t p = 128;
    const int n_dims = 128;
    const int head_dim = 128;
    const float freq_base = 10000.0f;
    const float freq_scale = 0.5f;
    const float ext_factor = 1.0f;
    const float attn_factor = 1.0f;
    const float beta_fast = 32.0f;
    const float beta_slow = 1.0f;
    const int n_ctx_orig = 4096;
    const float sin_sign = 1.0f;

    const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    printf("corr_dims[0]=%.6f corr_dims[1]=%.6f theta_scale=%.9g\n", corr_dims[0], corr_dims[1], theta_scale);

    float cache[128];
    ggml_rope_cache_init(
        (float)p, freq_scale, NULL, corr_dims, (int64_t)head_dim,
        ext_factor, attn_factor, cache, sin_sign, theta_scale
    );

    printf("cache (ne0=%d, float32), one row for position m=%lld:\n", head_dim, (long long)p);
    for (int i = 0; i < head_dim; i++) {
        printf("% .9e%s", cache[i], (i + 1) % 8 == 0 ? "\n" : " ");
    }
    if (head_dim % 8 != 0) {
        printf("\n");
    }

    /* 演示：同一参数下旋转一行 src -> dst（前 8 个数） */
    float src_demo[128];
    for (int i = 0; i < head_dim; i++) {
        src_demo[i] = (float)(i + 1) * 0.01f;
    }
    float dst_demo[128];
    rotate_pairs_f32_normal((int64_t)n_dims, 1, cache, src_demo, dst_demo, 1);
    rope_copy_tail((int64_t)n_dims, (int64_t)head_dim, src_demo, dst_demo);
    printf("dst[0..7] after rotate_pairs (demo src[i]=(i+1)*0.01):\n");
    for (int i = 0; i < 8; i++) {
        printf("% .9e%s", dst_demo[i], (i + 1) % 8 == 0 ? "\n" : " ");
    }
    printf("\n");

    return 0;
}
