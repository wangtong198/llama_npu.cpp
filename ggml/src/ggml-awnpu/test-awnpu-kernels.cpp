// Correctness test: compare AWNPU native kernel outputs against CPU reference.
// Manual compile example (from ggml/src/ggml-awnpu):
//   g++ -std=c++17 -I../.. -I../../.. -I../../../include
//       test-awnpu-kernels.cpp ggml-awnpu-kernels-native.cpp
//       -L../../../build/awnpu-native/bin -lggml-base -lggml-cpu
//       -Wl,-rpath,../../../build/awnpu-native/bin -o test-awnpu-kernels
//
// Or build via CMake with -DNATIVE_KERNELS=ON (see CMakeLists.txt).

#include "ggml-awnpu-kernels-native.h"
#include "ggml-impl.h"   // ggml_get_op_params_i32

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float randf(float lo = -1.f, float hi = 1.f) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static void fill_rand(float * p, int n, float lo = -1.f, float hi = 1.f) {
    for (int i = 0; i < n; ++i) p[i] = randf(lo, hi);
}

static bool near(float a, float b, float tol = 1e-4f) {
    return fabsf(a - b) <= tol + 1e-6f * fmaxf(fabsf(a), fabsf(b));
}

// Allocate a simple contiguous F32 tensor (no ggml allocator needed).
static ggml_tensor make_tensor_f32(int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
    ggml_tensor t = {};
    t.type = GGML_TYPE_F32;
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = ne3;
    t.nb[0] = sizeof(float);
    t.nb[1] = ne0 * sizeof(float);
    t.nb[2] = ne0 * ne1 * sizeof(float);
    t.nb[3] = ne0 * ne1 * ne2 * sizeof(float);
    int64_t n = ne0 * ne1 * ne2 * ne3;
    t.data = malloc(n * sizeof(float));
    memset(t.data, 0, n * sizeof(float));
    return t;
}

static ggml_tensor make_tensor_i32(int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1) {
    ggml_tensor t = {};
    t.type = GGML_TYPE_I32;
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = 1;
    t.nb[0] = sizeof(int32_t);
    t.nb[1] = ne0 * sizeof(int32_t);
    t.nb[2] = ne0 * ne1 * sizeof(int32_t);
    t.nb[3] = ne0 * ne1 * ne2 * sizeof(int32_t);
    t.data = malloc(ne0 * ne1 * ne2 * sizeof(int32_t));
    memset(t.data, 0, ne0 * ne1 * ne2 * sizeof(int32_t));
    return t;
}

static float * fp(ggml_tensor & t) { return (float *)t.data; }
static int32_t * ip(ggml_tensor & t) { return (int32_t *)t.data; }
static int64_t nel(const ggml_tensor & t) { return t.ne[0]*t.ne[1]*t.ne[2]*t.ne[3]; }

static void free_tensor(ggml_tensor & t) { free(t.data); t.data = nullptr; }

static bool check_equal(const ggml_tensor & a, const ggml_tensor & b,
                        const char * name, float tol = 1e-4f) {
    assert(nel(a) == nel(b));
    const float * pa = (const float *)a.data;
    const float * pb = (const float *)b.data;
    int bad = 0;
    for (int64_t i = 0; i < nel(a); ++i) {
        if (!near(pa[i], pb[i], tol)) {
            if (bad < 5)
                printf("  [%s] mismatch at %ld: awnpu=%.6f  cpu=%.6f\n",
                       name, (long)i, pa[i], pb[i]);
            ++bad;
        }
    }
    if (bad == 0) {
        printf("PASS  %s\n", name);
        return true;
    }
    printf("FAIL  %s  (%d/%ld mismatches)\n", name, bad, (long)nel(a));
    return false;
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

static void cpu_binary_f32(float * dst, const float * a, const float * b, int n, char op) {
    for (int i = 0; i < n; ++i) {
        switch (op) {
            case '+': dst[i] = a[i] + b[i]; break;
            case '-': dst[i] = a[i] - b[i]; break;
            case '*': dst[i] = a[i] * b[i]; break;
            case '/': dst[i] = a[i] / b[i]; break;
        }
    }
}

static void cpu_rms_norm(float * dst, const float * src, int64_t ne0, int64_t rows, float eps) {
    for (int64_t r = 0; r < rows; ++r) {
        const float * x = src + r * ne0;
        float * y = dst + r * ne0;
        double ss = 0;
        for (int64_t i = 0; i < ne0; ++i) ss += (double)x[i]*x[i];
        float scale = 1.f / sqrtf((float)(ss/ne0) + eps);
        for (int64_t i = 0; i < ne0; ++i) y[i] = x[i] * scale;
    }
}

static void cpu_norm(float * dst, const float * src, int64_t ne0, int64_t rows, float eps) {
    for (int64_t r = 0; r < rows; ++r) {
        const float * x = src + r * ne0;
        float * y = dst + r * ne0;
        double mean = 0;
        for (int64_t i = 0; i < ne0; ++i) mean += x[i];
        mean /= ne0;
        double var = 0;
        for (int64_t i = 0; i < ne0; ++i) var += (double)(x[i]-mean)*(x[i]-mean);
        var /= ne0;
        float scale = 1.f / sqrtf((float)var + eps);
        for (int64_t i = 0; i < ne0; ++i) y[i] = (x[i] - (float)mean) * scale;
    }
}

static void cpu_softmax(float * dst, const float * src, int64_t ne0, int64_t rows, float scale) {
    for (int64_t r = 0; r < rows; ++r) {
        const float * x = src + r * ne0;
        float * y = dst + r * ne0;
        float mx = *std::max_element(x, x+ne0);
        double s = 0;
        for (int64_t i = 0; i < ne0; ++i) { y[i] = expf((x[i]*scale) - mx*scale); s += y[i]; }
        float inv = (float)(1.0/s);
        for (int64_t i = 0; i < ne0; ++i) y[i] *= inv;
    }
}

static void cpu_mul_mat(float * dst,
                        const float * src0, int64_t M, // [K, M]
                        const float * src1, int64_t N, // [K, N]
                        int64_t K) {
    for (int64_t n = 0; n < N; ++n)
    for (int64_t m = 0; m < M; ++m) {
        float s = 0;
        for (int64_t k = 0; k < K; ++k) s += src0[m*K + k] * src1[n*K + k];
        dst[n*M + m] = s;
    }
}

static float gelu_ref(float x) {
    static const float K = 0.044715f, S = 0.7978845608f;
    return 0.5f * x * (1.f + tanhf(S * x * (1.f + K*x*x)));
}
static float silu_ref(float x) { return x / (1.f + expf(-x)); }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;
static void record(bool ok) { if (ok) ++g_pass; else ++g_fail; }

// --- add ---
static void test_add() {
    constexpr int N = 256;
    auto a = make_tensor_f32(N); auto b = make_tensor_f32(N);
    auto dst_k = make_tensor_f32(N); auto dst_r = make_tensor_f32(N);
    fill_rand(fp(a), N); fill_rand(fp(b), N);
    // awnpu kernel
    dst_k.src[0] = &a; dst_k.src[1] = &b;
    ggml_backend_awnpu_kernel_add(0, &dst_k);
    // cpu ref
    cpu_binary_f32(fp(dst_r), fp(a), fp(b), N, '+');
    record(check_equal(dst_k, dst_r, "add"));
    free_tensor(a); free_tensor(b); free_tensor(dst_k); free_tensor(dst_r);
}

// --- sub ---
static void test_sub() {
    constexpr int N = 128;
    auto a = make_tensor_f32(N); auto b = make_tensor_f32(N);
    auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N); fill_rand(fp(b), N);
    dk.src[0] = &a; dk.src[1] = &b;
    ggml_backend_awnpu_kernel_sub(0, &dk);
    cpu_binary_f32(fp(dr), fp(a), fp(b), N, '-');
    record(check_equal(dk, dr, "sub"));
    free_tensor(a); free_tensor(b); free_tensor(dk); free_tensor(dr);
}

// --- mul ---
static void test_mul() {
    constexpr int N = 128;
    auto a = make_tensor_f32(N); auto b = make_tensor_f32(N);
    auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N); fill_rand(fp(b), N);
    dk.src[0] = &a; dk.src[1] = &b;
    ggml_backend_awnpu_kernel_mul(0, &dk);
    cpu_binary_f32(fp(dr), fp(a), fp(b), N, '*');
    record(check_equal(dk, dr, "mul"));
    free_tensor(a); free_tensor(b); free_tensor(dk); free_tensor(dr);
}

// --- div ---
static void test_div() {
    constexpr int N = 64;
    auto a = make_tensor_f32(N); auto b = make_tensor_f32(N);
    auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N, 1.f, 2.f); fill_rand(fp(b), N, 0.5f, 2.f);
    dk.src[0] = &a; dk.src[1] = &b;
    ggml_backend_awnpu_kernel_div(0, &dk);
    cpu_binary_f32(fp(dr), fp(a), fp(b), N, '/');
    record(check_equal(dk, dr, "div"));
    free_tensor(a); free_tensor(b); free_tensor(dk); free_tensor(dr);
}

// --- sqr ---
static void test_sqr() {
    constexpr int N = 64;
    auto a = make_tensor_f32(N); auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N);
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_sqr(0, &dk);
    for (int i = 0; i < N; ++i) fp(dr)[i] = fp(a)[i] * fp(a)[i];
    record(check_equal(dk, dr, "sqr"));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- sqrt ---
static void test_sqrt() {
    constexpr int N = 64;
    auto a = make_tensor_f32(N); auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N, 0.1f, 4.f);
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_sqrt(0, &dk);
    for (int i = 0; i < N; ++i) fp(dr)[i] = sqrtf(fp(a)[i]);
    record(check_equal(dk, dr, "sqrt"));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- log ---
static void test_log() {
    constexpr int N = 64;
    auto a = make_tensor_f32(N); auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N, 0.1f, 4.f);
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_log(0, &dk);
    for (int i = 0; i < N; ++i) fp(dr)[i] = logf(fp(a)[i]);
    record(check_equal(dk, dr, "log"));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- add1 ---
static void test_add1() {
    constexpr int N = 64;
    auto a = make_tensor_f32(N); auto sc = make_tensor_f32(1);
    auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N); fp(sc)[0] = 3.14f;
    dk.src[0] = &a; dk.src[1] = &sc;
    ggml_backend_awnpu_kernel_add1(0, &dk);
    for (int i = 0; i < N; ++i) fp(dr)[i] = fp(a)[i] + 3.14f;
    record(check_equal(dk, dr, "add1"));
    free_tensor(a); free_tensor(sc); free_tensor(dk); free_tensor(dr);
}

// --- scale ---
static void test_scale() {
    constexpr int N = 64;
    auto a = make_tensor_f32(N); auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N);
    float s = 2.5f, b = 0.0f;
    memcpy((float *)dk.op_params + 0, &s, sizeof(float));
    memcpy((float *)dk.op_params + 1, &b, sizeof(float));
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_scale(0, &dk);
    for (int i = 0; i < N; ++i) fp(dr)[i] = fp(a)[i] * s + b;
    record(check_equal(dk, dr, "scale"));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- clamp ---
static void test_clamp() {
    constexpr int N = 64;
    auto a = make_tensor_f32(N); auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N, -2.f, 2.f);
    float vmin = -0.5f, vmax = 0.5f;
    memcpy((float *)dk.op_params + 0, &vmin, sizeof(float));
    memcpy((float *)dk.op_params + 1, &vmax, sizeof(float));
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_clamp(0, &dk);
    for (int i = 0; i < N; ++i) fp(dr)[i] = std::max(vmin, std::min(vmax, fp(a)[i]));
    record(check_equal(dk, dr, "clamp"));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- rms_norm ---
static void test_rms_norm() {
    constexpr int D = 128, R = 8;
    auto a = make_tensor_f32(D, R); auto dk = make_tensor_f32(D, R); auto dr = make_tensor_f32(D, R);
    fill_rand(fp(a), D*R);
    float eps = 1e-5f;
    memcpy(dk.op_params, &eps, sizeof(float));
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_rms_norm(0, &dk);
    cpu_rms_norm(fp(dr), fp(a), D, R, eps);
    record(check_equal(dk, dr, "rms_norm", 5e-5f));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- norm ---
static void test_norm() {
    constexpr int D = 64, R = 4;
    auto a = make_tensor_f32(D, R); auto dk = make_tensor_f32(D, R); auto dr = make_tensor_f32(D, R);
    fill_rand(fp(a), D*R);
    float eps = 1e-5f;
    memcpy(dk.op_params, &eps, sizeof(float));
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_norm(0, &dk);
    cpu_norm(fp(dr), fp(a), D, R, eps);
    record(check_equal(dk, dr, "norm", 5e-5f));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- soft_max ---
static void test_soft_max() {
    constexpr int D = 32, R = 8;
    auto a = make_tensor_f32(D, R); auto dk = make_tensor_f32(D, R); auto dr = make_tensor_f32(D, R);
    fill_rand(fp(a), D*R);
    float scale = 1.0f, mb = 0.0f;
    memcpy((float *)dk.op_params + 0, &scale, sizeof(float));
    memcpy((float *)dk.op_params + 1, &mb,    sizeof(float));
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_soft_max(0, &dk);
    cpu_softmax(fp(dr), fp(a), D, R, scale);
    record(check_equal(dk, dr, "soft_max", 1e-4f));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- mul_mat ---
static void test_mul_mat() {
    constexpr int K = 16, M = 8, N = 4;
    // src0: [K, M]  src1: [K, N]  dst: [M, N]
    auto s0 = make_tensor_f32(K, M); auto s1 = make_tensor_f32(K, N);
    auto dk = make_tensor_f32(M, N); auto dr = make_tensor_f32(M, N);
    fill_rand(fp(s0), K*M); fill_rand(fp(s1), K*N);
    dk.src[0] = &s0; dk.src[1] = &s1;
    ggml_backend_awnpu_kernel_mul_mat(0, &dk);
    cpu_mul_mat(fp(dr), fp(s0), M, fp(s1), N, K);
    record(check_equal(dk, dr, "mul_mat", 1e-3f));
    free_tensor(s0); free_tensor(s1); free_tensor(dk); free_tensor(dr);
}

// --- get_rows ---
static void test_get_rows() {
    constexpr int D = 16, VOCAB = 32, SEQ = 8;
    auto emb = make_tensor_f32(D, VOCAB); // embedding table
    auto idx = make_tensor_i32(SEQ);      // token indices
    auto dk  = make_tensor_f32(D, SEQ);
    auto dr  = make_tensor_f32(D, SEQ);

    fill_rand(fp(emb), D*VOCAB);
    for (int i = 0; i < SEQ; ++i) ip(idx)[i] = rand() % VOCAB;

    dk.src[0] = &emb; dk.src[1] = &idx;
    ggml_backend_awnpu_kernel_get_rows(0, &dk);

    // cpu ref
    for (int i = 0; i < SEQ; ++i) {
        int row = ip(idx)[i];
        memcpy(fp(dr) + i*D, fp(emb) + row*D, D*sizeof(float));
    }
    record(check_equal(dk, dr, "get_rows"));
    free_tensor(emb); free_tensor(idx); free_tensor(dk); free_tensor(dr);
}

// --- unary: relu ---
static void test_unary_relu() {
    constexpr int N = 64;
    auto a = make_tensor_f32(N); auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N);
    // pack unary op into op_params[0]
    int32_t uop = (int32_t)GGML_UNARY_OP_RELU;
    memcpy(dk.op_params, &uop, sizeof(int32_t));
    dk.op = GGML_OP_UNARY;
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_unary(0, &dk);
    for (int i = 0; i < N; ++i) fp(dr)[i] = fp(a)[i] > 0.f ? fp(a)[i] : 0.f;
    record(check_equal(dk, dr, "unary_relu"));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- unary: silu ---
static void test_unary_silu() {
    constexpr int N = 64;
    auto a = make_tensor_f32(N); auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N);
    int32_t uop = (int32_t)GGML_UNARY_OP_SILU;
    memcpy(dk.op_params, &uop, sizeof(int32_t));
    dk.op = GGML_OP_UNARY;
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_unary(0, &dk);
    for (int i = 0; i < N; ++i) fp(dr)[i] = silu_ref(fp(a)[i]);
    record(check_equal(dk, dr, "unary_silu"));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- unary: gelu ---
static void test_unary_gelu() {
    constexpr int N = 64;
    auto a = make_tensor_f32(N); auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N);
    int32_t uop = (int32_t)GGML_UNARY_OP_GELU;
    memcpy(dk.op_params, &uop, sizeof(int32_t));
    dk.op = GGML_OP_UNARY;
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_unary(0, &dk);
    for (int i = 0; i < N; ++i) fp(dr)[i] = gelu_ref(fp(a)[i]);
    record(check_equal(dk, dr, "unary_gelu"));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- glu swiglu ---
static void test_glu_swiglu() {
    constexpr int D = 32, R = 4;
    // single-tensor mode: gate = first half, x = second half
    auto a  = make_tensor_f32(D*2, R); // gate || x
    auto dk = make_tensor_f32(D,   R);
    auto dr = make_tensor_f32(D,   R);
    fill_rand(fp(a), D*2*R);

    int32_t gop = (int32_t)GGML_GLU_OP_SWIGLU;
    int32_t swapped = 0;
    memcpy((int32_t *)dk.op_params + 0, &gop,     sizeof(int32_t));
    memcpy((int32_t *)dk.op_params + 1, &swapped,  sizeof(int32_t));
    dk.op = GGML_OP_GLU;
    dk.src[0] = &a; dk.src[1] = nullptr;
    ggml_backend_awnpu_kernel_glu(0, &dk);

    // cpu: gate = first D, x = second D
    for (int r = 0; r < R; ++r) {
        const float * gate = fp(a) + r*D*2;
        const float * x    = gate + D;
        float * y = fp(dr) + r*D;
        for (int i = 0; i < D; ++i) y[i] = silu_ref(gate[i]) * x[i];
    }
    record(check_equal(dk, dr, "glu_swiglu"));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- dup (same type, contiguous) ---
static void test_dup() {
    constexpr int N = 128;
    auto a = make_tensor_f32(N); auto dk = make_tensor_f32(N); auto dr = make_tensor_f32(N);
    fill_rand(fp(a), N);
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_dup(0, &dk);
    memcpy(fp(dr), fp(a), N*sizeof(float));
    record(check_equal(dk, dr, "dup"));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- pad (zero-pad axis-0) ---
static void test_pad() {
    constexpr int D = 8, R = 4;
    constexpr int lp0 = 2, rp0 = 2; // pad 2 on each side → D+4
    constexpr int D2 = D + lp0 + rp0;

    auto a  = make_tensor_f32(D,  R);
    auto dk = make_tensor_f32(D2, R);
    auto dr = make_tensor_f32(D2, R);
    fill_rand(fp(a), D*R);

    memset(fp(dk), 0, D2*R*sizeof(float));
    int32_t pp[9] = { lp0, rp0, 0, 0, 0, 0, 0, 0, 0 };
    memcpy(dk.op_params, pp, sizeof(pp));
    dk.src[0] = &a;
    ggml_backend_awnpu_kernel_pad(0, &dk);

    // cpu ref
    memset(fp(dr), 0, D2*R*sizeof(float));
    for (int r = 0; r < R; ++r)
    for (int i = 0; i < D; ++i)
        fp(dr)[r*D2 + lp0 + i] = fp(a)[r*D + i];

    record(check_equal(dk, dr, "pad"));
    free_tensor(a); free_tensor(dk); free_tensor(dr);
}

// --- rope (NORMAL mode, F32) ---
static void test_rope() {
    constexpr int D = 64; // head_dim
    constexpr int N = 8;  // seq_len
    constexpr int H = 2;  // heads
    constexpr int n_dims = D; // all dims rotated

    auto src  = make_tensor_f32(D, N, H);
    auto pos  = make_tensor_i32(N);
    auto dstk = make_tensor_f32(D, N, H);
    auto dstr = make_tensor_f32(D, N, H);

    fill_rand(fp(src), D*N*H);
    for (int i = 0; i < N; ++i) ip(pos)[i] = i;

    // op_params layout (matches rope_impl)
    int32_t * op = (int32_t *)dstk.op_params;
    op[1] = n_dims;
    op[2] = GGML_ROPE_TYPE_NEOX;
    op[4] = 2048; // n_ctx_orig
    float freq_base = 10000.f, freq_scale = 1.f, ext = 0.f, attn = 1.f, bf = 32.f, bs = 1.f;
    memcpy(op+5, &freq_base,  sizeof(float));
    memcpy(op+6, &freq_scale, sizeof(float));
    memcpy(op+7, &ext,        sizeof(float));
    memcpy(op+8, &attn,       sizeof(float));
    memcpy(op+9, &bf,         sizeof(float));
    memcpy(op+10,&bs,         sizeof(float));

    dstk.src[0] = &src; dstk.src[1] = &pos; dstk.src[2] = nullptr;
    ggml_backend_awnpu_kernel_rope(0, &dstk);

    // cpu ref: NEOX adjacent-pair rotation
    const float theta_scale = powf(freq_base, -2.f / (float)n_dims);
    for (int h = 0; h < H; ++h)
    for (int t = 0; t < N; ++t) {
        int p = ip(pos)[t];
        float * d   = fp(dstr) + (h*N + t)*D;
        const float * s = fp(src) + (h*N + t)*D;
        float theta = (float)p;
        for (int i = 0; i < n_dims; i += 2) {
            float cs = cosf(theta), ss = sinf(theta);
            d[i]   = s[i]*cs   - s[i+1]*ss;
            d[i+1] = s[i]*ss   + s[i+1]*cs;
            theta *= theta_scale;
        }
    }
    record(check_equal(dstk, dstr, "rope_neox", 5e-5f));
    free_tensor(src); free_tensor(pos); free_tensor(dstk); free_tensor(dstr);
}

// --- flash_attn_ext (F32, no mask, no ALiBi) ---
static void test_flash_attn_ext() {
    constexpr int DK = 16, DV = 16;
    constexpr int N = 4;   // query tokens
    constexpr int M = 8;   // kv tokens
    constexpr int H = 2;   // heads

    // Shapes: Q=[DK,N,H,1]  K=[DK,M,H,1]  V=[DV,M,H,1]
    // dst=[DV,H,N,1]  (permuted)
    auto q   = make_tensor_f32(DK, N, H);
    auto k   = make_tensor_f32(DK, M, H);
    auto v   = make_tensor_f32(DV, M, H);
    auto dstk = make_tensor_f32(DV, H, N); // layout: [DV, H, N]
    auto dstr = make_tensor_f32(DV, H, N);

    fill_rand(fp(q), DK*N*H, -1.f, 1.f);
    fill_rand(fp(k), DK*M*H, -1.f, 1.f);
    fill_rand(fp(v), DV*M*H, -1.f, 1.f);

    float scale = 1.f / sqrtf((float)DK), mb = 0.f, lsc = 0.f;
    memcpy((float *)dstk.op_params + 0, &scale, sizeof(float));
    memcpy((float *)dstk.op_params + 1, &mb,    sizeof(float));
    memcpy((float *)dstk.op_params + 2, &lsc,   sizeof(float));
    dstk.src[0] = &q; dstk.src[1] = &k; dstk.src[2] = &v; dstk.src[3] = nullptr;

    ggml_backend_awnpu_kernel_flash_attn_ext(0, &dstk);

    // cpu ref (naive)
    std::vector<float> scores(M);
    for (int h = 0; h < H; ++h)
    for (int iq = 0; iq < N; ++iq) {
        const float * qrow = fp(q) + (h*N + iq)*DK;
        float mval = -INFINITY;
        for (int ic = 0; ic < M; ++ic) {
            const float * krow = fp(k) + (h*M + ic)*DK;
            float s = 0;
            for (int d = 0; d < DK; ++d) s += qrow[d] * krow[d];
            scores[ic] = s * scale;
            if (scores[ic] > mval) mval = scores[ic];
        }
        double S = 0;
        for (int ic = 0; ic < M; ++ic) { scores[ic] = expf(scores[ic]-mval); S += scores[ic]; }
        float inv = (float)(1.0/S);
        // output: dst[iq, h, :] = weighted sum of v rows
        // dst layout [DV, H, N] → out_ptr = dstr + (h + iq*H)*DV
        float * out = fp(dstr) + (h + iq*H)*DV;
        memset(out, 0, DV*sizeof(float));
        for (int ic = 0; ic < M; ++ic) {
            const float * vrow = fp(v) + (h*M + ic)*DV;
            float w = scores[ic] * inv;
            for (int d = 0; d < DV; ++d) out[d] += w * vrow[d];
        }
    }
    record(check_equal(dstk, dstr, "flash_attn_ext", 2e-4f));
    free_tensor(q); free_tensor(k); free_tensor(v); free_tensor(dstk); free_tensor(dstr);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    srand(42);
    printf("=== AWNPU kernel correctness tests ===\n\n");

    test_add();
    test_sub();
    test_mul();
    test_div();
    test_sqr();
    test_sqrt();
    test_log();
    test_add1();
    test_scale();
    test_clamp();
    test_rms_norm();
    test_norm();
    test_soft_max();
    test_mul_mat();
    test_get_rows();
    test_unary_relu();
    test_unary_silu();
    test_unary_gelu();
    test_glu_swiglu();
    test_dup();
    test_pad();
    test_rope();
    test_flash_attn_ext();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
