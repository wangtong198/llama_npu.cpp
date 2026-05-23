// Single-threaded reference implementations for AWNPU kernels.
// Only F32 (and F16/BF16 where trivial) types are handled;
// all other type combinations return GGML_STATUS_FAILED so the
// caller can fall back to the CPU backend.

#include "ggml-awnpu-kernels-native.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline float load_f32(const ggml_tensor * t, size_t byte_off) {
    const char * p = (const char *)t->data + byte_off;
    switch (t->type) {
        case GGML_TYPE_F32:  return *(const float *)p;
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *)p);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *)p);
        default:             return 0.0f;
    }
}

static inline void store_f32(ggml_tensor * t, size_t byte_off, float v) {
    char * p = (char *)t->data + byte_off;
    switch (t->type) {
        case GGML_TYPE_F32:  *(float *)p = v;                         break;
        case GGML_TYPE_F16:  *(ggml_fp16_t *)p = ggml_fp32_to_fp16(v); break;
        case GGML_TYPE_BF16: *(ggml_bf16_t *)p = ggml_fp32_to_bf16(v); break;
        default:                                                        break;
    }
}

static inline bool is_float_type(ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
}

static enum ggml_status copy_same_type(struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1],
                  ne2 = dst->ne[2], ne3 = dst->ne[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1],
                 nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb0  = dst->nb[0],  nb1  = dst->nb[1],
                 nb2  = dst->nb[2],  nb3  = dst->nb[3];

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        memcpy(dst->data, src0->data, ggml_nbytes(dst));
        return GGML_STATUS_SUCCESS;
    }

    if (!ggml_are_same_shape(src0, dst)) {
        return GGML_STATUS_FAILED;
    }

    const size_t type_size = ggml_type_size(src0->type);
    if (nb00 != type_size || nb0 != type_size) {
        if (!ggml_is_quantized(src0->type)) {
            for (int64_t i3 = 0; i3 < ne3; ++i3)
            for (int64_t i2 = 0; i2 < ne2; ++i2)
            for (int64_t i1 = 0; i1 < ne1; ++i1)
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const size_t src_off = i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03;
                const size_t dst_off = i0*nb0  + i1*nb1  + i2*nb2  + i3*nb3;
                memcpy((char *) dst->data + dst_off, (const char *) src0->data + src_off, type_size);
            }
            return GGML_STATUS_SUCCESS;
        }
        return GGML_STATUS_FAILED;
    }

    const size_t row_size = ggml_row_size(src0->type, ne0);

    for (int64_t i3 = 0; i3 < ne3; ++i3)
    for (int64_t i2 = 0; i2 < ne2; ++i2)
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
        memcpy(
            (char *) dst->data + i1*nb1 + i2*nb2 + i3*nb3,
            (const char *) src0->data + i1*nb01 + i2*nb02 + i3*nb03,
            row_size);
    }

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status copy_i32_and_float(struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1],
                  ne2 = dst->ne[2], ne3 = dst->ne[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1],
                 nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb0  = dst->nb[0],  nb1  = dst->nb[1],
                 nb2  = dst->nb[2],  nb3  = dst->nb[3];

    for (int64_t i3 = 0; i3 < ne3; ++i3)
    for (int64_t i2 = 0; i2 < ne2; ++i2)
    for (int64_t i1 = 0; i1 < ne1; ++i1)
    for (int64_t i0 = 0; i0 < ne0; ++i0) {
        const size_t src_off = i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03;
        const size_t dst_off = i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3;
        float v;
        if (src0->type == GGML_TYPE_I32) {
            v = (float) *(const int32_t *)((const char *) src0->data + src_off);
        } else {
            v = load_f32(src0, src_off);
        }

        if (dst->type == GGML_TYPE_I32) {
            *(int32_t *)((char *) dst->data + dst_off) = (int32_t) v;
        } else {
            store_f32(dst, dst_off, v);
        }
    }

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status copy_via_traits(struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_type src_type = src0->type;
    const ggml_type dst_type = dst->type;
    const ggml_type_traits * src_tt = ggml_get_type_traits(src_type);
    const ggml_type_traits * dst_tt = ggml_get_type_traits(dst_type);

    if (!src_tt || !dst_tt) {
        return GGML_STATUS_FAILED;
    }

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1],
                  ne2 = dst->ne[2], ne3 = dst->ne[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1],
                 nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb0  = dst->nb[0],  nb1  = dst->nb[1],
                 nb2  = dst->nb[2],  nb3  = dst->nb[3];
    const size_t src_row_bytes = ggml_row_size(src_type, ne0);
    const size_t dst_row_bytes = ggml_row_size(dst_type, ne0);

    if (nb00 != ggml_type_size(src_type) || nb0 != ggml_type_size(dst_type)) {
        return GGML_STATUS_FAILED;
    }

    if (src_type == GGML_TYPE_F32 && dst_tt->from_float_ref) {
        for (int64_t i3 = 0; i3 < ne3; ++i3)
        for (int64_t i2 = 0; i2 < ne2; ++i2)
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const float * src_row = (const float *) ((const char *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
            void * dst_row = (char *) dst->data + i1*nb1 + i2*nb2 + i3*nb3;
            (void) src_row_bytes;
            (void) dst_row_bytes;
            dst_tt->from_float_ref(src_row, dst_row, ne0);
        }
        return GGML_STATUS_SUCCESS;
    }

    if (dst_type == GGML_TYPE_F32 && src_tt->to_float) {
        for (int64_t i3 = 0; i3 < ne3; ++i3)
        for (int64_t i2 = 0; i2 < ne2; ++i2)
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const void * src_row = (const char *) src0->data + i1*nb01 + i2*nb02 + i3*nb03;
            float * dst_row = (float *) ((char *) dst->data + i1*nb1 + i2*nb2 + i3*nb3);
            src_tt->to_float(src_row, dst_row, ne0);
        }
        return GGML_STATUS_SUCCESS;
    }

    // Generic fallback for other non-quantized integer/float combinations.
    if (!src_tt->is_quantized && !dst_tt->is_quantized) {
        if (!(is_float_type(src_type) || src_type == GGML_TYPE_I32) ||
            !(is_float_type(dst_type) || dst_type == GGML_TYPE_I32)) {
            return GGML_STATUS_FAILED;
        }
        for (int64_t i3 = 0; i3 < ne3; ++i3)
        for (int64_t i2 = 0; i2 < ne2; ++i2)
        for (int64_t i1 = 0; i1 < ne1; ++i1)
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            const size_t src_off = i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03;
            const size_t dst_off = i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3;

            float v;
            if (src_type == GGML_TYPE_I32) {
                v = (float) *(const int32_t *)((const char *) src0->data + src_off);
            } else {
                v = load_f32(src0, src_off);
            }

            if (dst_type == GGML_TYPE_I32) {
                *(int32_t *)((char *) dst->data + dst_off) = (int32_t) v;
            } else {
                store_f32(dst, dst_off, v);
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    return GGML_STATUS_FAILED;
}

// Generic 4-D binary op with full broadcasting (F32 / F16 / BF16).
template<float (*op)(float, float)>
static enum ggml_status binary_op_f32(struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!is_float_type(src0->type) || !is_float_type(src1->type) ||
        !is_float_type(dst->type)) {
        return GGML_STATUS_FAILED;
    }

    const int64_t ne0  = dst->ne[0], ne1  = dst->ne[1],
                  ne2  = dst->ne[2], ne3  = dst->ne[3];
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1],
                  ne02 = src0->ne[2], ne03 = src0->ne[3];
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1],
                  ne12 = src1->ne[2], ne13 = src1->ne[3];

    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1],
                 nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb10 = src1->nb[0], nb11 = src1->nb[1],
                 nb12 = src1->nb[2], nb13 = src1->nb[3];
    const size_t nb0  = dst->nb[0],  nb1  = dst->nb[1],
                 nb2  = dst->nb[2],  nb3  = dst->nb[3];

    for (int64_t i3 = 0; i3 < ne3; ++i3)
    for (int64_t i2 = 0; i2 < ne2; ++i2)
    for (int64_t i1 = 0; i1 < ne1; ++i1)
    for (int64_t i0 = 0; i0 < ne0; ++i0) {
        const int64_t s0_i0 = i0 % ne00, s0_i1 = i1 % ne01,
                      s0_i2 = i2 % ne02, s0_i3 = i3 % ne03;
        const int64_t s1_i0 = i0 % ne10, s1_i1 = i1 % ne11,
                      s1_i2 = i2 % ne12, s1_i3 = i3 % ne13;
        float a = load_f32(src0, s0_i0*nb00 + s0_i1*nb01 + s0_i2*nb02 + s0_i3*nb03);
        float b = load_f32(src1, s1_i0*nb10 + s1_i1*nb11 + s1_i2*nb12 + s1_i3*nb13);
        store_f32(dst, i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3, op(a, b));
    }
    return GGML_STATUS_SUCCESS;
}

// Generic 4-D unary op (F32 / F16 / BF16).
template<float (*op)(float)>
static enum ggml_status unary_op_f32(struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    if (!is_float_type(src0->type) || !is_float_type(dst->type))
        return GGML_STATUS_FAILED;

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1],
                  ne2 = dst->ne[2], ne3 = dst->ne[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1],
                 nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb0  = dst->nb[0],  nb1  = dst->nb[1],
                 nb2  = dst->nb[2],  nb3  = dst->nb[3];

    for (int64_t i3 = 0; i3 < ne3; ++i3)
    for (int64_t i2 = 0; i2 < ne2; ++i2)
    for (int64_t i1 = 0; i1 < ne1; ++i1)
    for (int64_t i0 = 0; i0 < ne0; ++i0) {
        float v = load_f32(src0, i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
        store_f32(dst, i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3, op(v));
    }
    return GGML_STATUS_SUCCESS;
}

static inline float op_add(float a, float b) { return a + b; }
static inline float op_sub(float a, float b) { return a - b; }
static inline float op_mul(float a, float b) { return a * b; }
static inline float op_div(float a, float b) { return a / b; }

static inline float op_sqr (float x) { return x * x; }
static inline float op_sqrt(float x) { return sqrtf(x); }
static inline float op_log (float x) { return logf(x); }

// ---------------------------------------------------------------------------
// GET_ROWS
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_get_rows(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0]; // embedding table
    const ggml_tensor * src1 = node->src[1]; // indices (I32)
    ggml_tensor       * dst  = node;

    if (dst->type != GGML_TYPE_F32) return GGML_STATUS_FAILED;
    if (src1->type != GGML_TYPE_I32) return GGML_STATUS_FAILED;

    const int64_t ne00 = src0->ne[0]; // embedding dim
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1];
    const size_t  nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2];
    const size_t  nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];
    (void)ne10;

    const int64_t nr = ggml_nelements(src1);

    // choose per-element converter
    ggml_to_float_t to_float = nullptr;
    if (src0->type == GGML_TYPE_F32) {
        to_float = nullptr; // will use memcpy path
    } else if (is_float_type(src0->type)) {
        // handled inline below
    } else {
        const ggml_type_traits * tt = ggml_get_type_traits(src0->type);
        to_float = tt->to_float;
        if (!to_float) return GGML_STATUS_FAILED;
    }

    for (int64_t i = 0; i < nr; ++i) {
        const int64_t i12 = i / (ne11 * ne10);
        const int64_t i11 = (i - i12*ne11*ne10) / ne10;
        const int64_t i10 = i - i12*ne11*ne10 - i11*ne10;
        const int64_t i01 = *(const int32_t *)((const char *)src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < src0->ne[1]);

        float * dst_row = (float *)((char *)dst->data + i10*nb1 + i11*nb2 + i12*nb3);
        const char * src_row = (const char *)src0->data + i01*nb01 + i11*nb02 + i12*nb03;

        if (src0->type == GGML_TYPE_F32) {
            memcpy(dst_row, src_row, ne00 * sizeof(float));
        } else if (src0->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)src_row, dst_row, ne00);
        } else if (src0->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *)src_row, dst_row, ne00);
        } else {
            to_float(src_row, dst_row, ne00);
        }
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// MUL_MAT  — mirrors ggml_compute_forward_mul_mat (single-thread reference)
//
//   dst[m,p,qq,rr] = src0[n,m,q1,r1]^T @ src1[n,p,qq,rr]
//   ne0==ne01, ne1==ne11, ne2==ne12, ne3==ne13, ne00==ne10
//   dst is always F32; src1 is F32; src0 may be F32/F16/BF16 (quantized → fallback)
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_mul_mat(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    ggml_tensor       * dst  = node;

    GGML_TENSOR_BINARY_OP_LOCALS

    if (dst->type != GGML_TYPE_F32 || nb0 != sizeof(float)) {
        return GGML_STATUS_FAILED;
    }
    if (src1->type != GGML_TYPE_F32 || nb10 != ggml_type_size(GGML_TYPE_F32)) {
        return GGML_STATUS_FAILED;
    }
    if (!is_float_type(src0->type) || nb00 != ggml_type_size(src0->type)) {
        return GGML_STATUS_FAILED;
    }
    if (ne0 != ne01 || ne1 != ne11 || ne2 != ne12 || ne3 != ne13 || ne00 != ne10) {
        return GGML_STATUS_FAILED;
    }
    if (ne12 % ne02 != 0 || ne13 % ne03 != 0) {
        return GGML_STATUS_FAILED;
    }

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            for (int64_t i11 = 0; i11 < ne11; ++i11) {
                const char * src1_col = (const char *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13;

                for (int64_t i01 = 0; i01 < ne01; ++i01) {
                    const char * src0_row = (const char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;

                    float sum = 0.0f;
                    for (int64_t i00 = 0; i00 < ne00; ++i00) {
                        const float v0 = load_f32(src0, (size_t) (src0_row - (const char *) src0->data) + i00 * nb00);
                        const float v1 = load_f32(src1, (size_t) (src1_col - (const char *) src1->data) + i00 * nb10);
                        sum += v0 * v1;
                    }

                    float * out = (float *)((char *) dst->data + i01 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);
                    *out = sum;
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// MUL_MAT_ID  — mixture-of-experts gather-matmul (F32 only, expert-per-row)
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_mul_mat_id(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0]; // experts [K, M, n_expert]
    const ggml_tensor * src1 = node->src[1]; // tokens  [K, N]
    const ggml_tensor * ids  = node->src[2]; // expert ids [n_tokens_per_expert, N]
    ggml_tensor       * dst  = node;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 ||
        dst->type  != GGML_TYPE_F32 || ids->type  != GGML_TYPE_I32)
        return GGML_STATUS_FAILED;

    const int64_t K         = src0->ne[0];
    const int64_t M         = src0->ne[1]; // output cols per expert
    const int64_t n_expert  = src0->ne[2];
    const int64_t n_ids     = ids->ne[0]; // tokens per expert slot
    const int64_t N         = ids->ne[1]; // number of expert slots

    GGML_UNUSED(n_expert);

    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2];
    const size_t nb11 = src1->nb[1];
    const size_t nb0  = dst->nb[0],  nb1  = dst->nb[1];
    const size_t ids_nb0 = ids->nb[0], ids_nb1 = ids->nb[1];

    // zero output
    memset(dst->data, 0, ggml_nbytes(dst));

    for (int64_t ie = 0; ie < N; ++ie)
    for (int64_t it = 0; it < n_ids; ++it) {
        const int32_t expert_id = *(const int32_t *)((const char *)ids->data + it*ids_nb0 + ie*ids_nb1);
        if (expert_id < 0) continue;

        const float * w   = (const float *)((const char *)src0->data + expert_id*nb02);
        const float * in  = (const float *)((const char *)src1->data + it*nb11);
        float       * out = (float       *)((char *)dst->data         + it*nb1);

        for (int64_t im = 0; im < M; ++im) {
            const float * row = w + im*(nb01/sizeof(float));
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) sum += row[k] * in[k];
            out[im*(nb0/sizeof(float))] += sum;
        }
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// ADD / ADD1 / ADD_ID
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_add(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    return binary_op_f32<op_add>(node);
}

enum ggml_status ggml_backend_awnpu_kernel_add_id(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const ggml_tensor * src2 = node->src[2]; // I32 row indices
    ggml_tensor       * dst  = node;

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 ||
        dst->type  != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32)
        return GGML_STATUS_FAILED;

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1],
                  ne2 = dst->ne[2], ne3 = dst->ne[3];
    const int64_t ne11 = src1->ne[1];
    const size_t  nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb11 = src1->nb[1];
    const size_t  nb20 = src2->nb[0], nb21 = src2->nb[1];
    const size_t  nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];

    for (int64_t i3 = 0; i3 < ne3; ++i3)
    for (int64_t i2 = 0; i2 < ne2; ++i2)
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
        const int32_t i11 = *(const int32_t *)((const char *)src2->data + i1*nb20 + i2*nb21);
        GGML_ASSERT(i11 >= 0 && i11 < ne11);
        const float * s0  = (const float *)((const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03);
        const float * s1  = (const float *)((const char *)src1->data + i11*nb11);
        float       * out = (float       *)((char *)dst->data         + i1*nb1  + i2*nb2  + i3*nb3);
        for (int64_t i0 = 0; i0 < ne0; ++i0) out[i0] = s0[i0] + s1[i0];
    }
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_awnpu_kernel_add1(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1]; // scalar tensor
    ggml_tensor       * dst  = node;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return GGML_STATUS_FAILED;

    const float scalar = *(const float *)src1->data;

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1],
                  ne2 = dst->ne[2], ne3 = dst->ne[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1],
                 nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb0  = dst->nb[0],  nb1  = dst->nb[1],
                 nb2  = dst->nb[2],  nb3  = dst->nb[3];

    for (int64_t i3 = 0; i3 < ne3; ++i3)
    for (int64_t i2 = 0; i2 < ne2; ++i2)
    for (int64_t i1 = 0; i1 < ne1; ++i1)
    for (int64_t i0 = 0; i0 < ne0; ++i0) {
        float v = load_f32(src0, i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
        store_f32(dst, i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3, v + scalar);
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// MUL / DIV / SUB
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_mul(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    return binary_op_f32<op_mul>(node);
}

enum ggml_status ggml_backend_awnpu_kernel_div(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    return binary_op_f32<op_div>(node);
}

enum ggml_status ggml_backend_awnpu_kernel_sub(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    return binary_op_f32<op_sub>(node);
}

// ---------------------------------------------------------------------------
// SQR / SQRT / LOG
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_sqr(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    return unary_op_f32<op_sqr>(node);
}

enum ggml_status ggml_backend_awnpu_kernel_sqrt(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    return unary_op_f32<op_sqrt>(node);
}

enum ggml_status ggml_backend_awnpu_kernel_log(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    return unary_op_f32<op_log>(node);
}

// ---------------------------------------------------------------------------
// UNARY  (dispatch based on op_params[0])
// ---------------------------------------------------------------------------
static inline float gelu_f32(float x) {
    // GELU approximation used by llama.cpp
    static const float K = 0.044715f;
    static const float S = 0.7978845608f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(S * x * (1.0f + K * x * x)));
}
static inline float gelu_quick_f32(float x) {
    return x * (1.0f / (1.0f + expf(-1.702f * x)));
}
static inline float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}
static inline float gelu_erf_f32(float x) {
    static const float SQRT_2_INV = 0.70710678118f;
    return 0.5f * x * (1.0f + erff(x * SQRT_2_INV));
}

enum ggml_status ggml_backend_awnpu_kernel_unary(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0];
    ggml_tensor       * dst  = node;

    if (!is_float_type(src0->type) || !is_float_type(dst->type))
        return GGML_STATUS_FAILED;

    const enum ggml_unary_op uop = ggml_get_unary_op(node);

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1],
                  ne2 = dst->ne[2], ne3 = dst->ne[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1],
                 nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb0  = dst->nb[0],  nb1  = dst->nb[1],
                 nb2  = dst->nb[2],  nb3  = dst->nb[3];

    for (int64_t i3 = 0; i3 < ne3; ++i3)
    for (int64_t i2 = 0; i2 < ne2; ++i2)
    for (int64_t i1 = 0; i1 < ne1; ++i1)
    for (int64_t i0 = 0; i0 < ne0; ++i0) {
        float x = load_f32(src0, i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
        float y;
        switch (uop) {
            case GGML_UNARY_OP_ABS:         y = fabsf(x);               break;
            case GGML_UNARY_OP_SGN:         y = (x > 0.f) ? 1.f : (x < 0.f) ? -1.f : 0.f; break;
            case GGML_UNARY_OP_NEG:         y = -x;                     break;
            case GGML_UNARY_OP_STEP:        y = (x > 0.f) ? 1.f : 0.f; break;
            case GGML_UNARY_OP_TANH:        y = tanhf(x);               break;
            case GGML_UNARY_OP_ELU:         y = (x >= 0.f) ? x : expm1f(x); break;
            case GGML_UNARY_OP_RELU:        y = (x > 0.f) ? x : 0.f;   break;
            case GGML_UNARY_OP_SIGMOID:     y = 1.f / (1.f + expf(-x)); break;
            case GGML_UNARY_OP_GELU:        y = gelu_f32(x);            break;
            case GGML_UNARY_OP_GELU_QUICK:  y = gelu_quick_f32(x);      break;
            case GGML_UNARY_OP_SILU:        y = silu_f32(x);            break;
            case GGML_UNARY_OP_HARDSWISH:   y = x * fminf(1.f, fmaxf(0.f, (x+3.f)/6.f)); break;
            case GGML_UNARY_OP_HARDSIGMOID: y = fminf(1.f, fmaxf(0.f, (x+3.f)/6.f));     break;
            case GGML_UNARY_OP_EXP:         y = expf(x);                break;
            case GGML_UNARY_OP_EXPM1:       y = expf(x) - 1.f;         break;
            case GGML_UNARY_OP_SOFTPLUS:    y = (x > 20.f) ? x : logf(1.f + expf(x)); break;
            case GGML_UNARY_OP_GELU_ERF:    y = gelu_erf_f32(x);        break;
            case GGML_UNARY_OP_FLOOR:       y = floorf(x);              break;
            case GGML_UNARY_OP_CEIL:        y = ceilf(x);               break;
            case GGML_UNARY_OP_ROUND:       y = roundf(x);              break;
            case GGML_UNARY_OP_TRUNC:       y = truncf(x);              break;
            case GGML_UNARY_OP_XIELU: {
                float alpha_n = ggml_get_op_params_f32(node, 1);
                float alpha_p = ggml_get_op_params_f32(node, 2);
                float beta    = ggml_get_op_params_f32(node, 3);
                float eps     = ggml_get_op_params_f32(node, 4);
                if (x > 0.f) {
                    y = alpha_p * x * x + beta * x;
                } else {
                    float mxe = fminf(x, eps);
                    y = (expm1f(mxe) - x) * alpha_n + beta * x;
                }
                break;
            }
            default: return GGML_STATUS_FAILED;
        }
        store_f32(dst, i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3, y);
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// NORM  (layer norm)
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_norm(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0];
    ggml_tensor       * dst  = node;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return GGML_STATUS_FAILED;

    float eps;
    memcpy(&eps, node->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1],
                  ne02 = src0->ne[2], ne03 = src0->ne[3];
    const size_t  nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];

    for (int64_t i03 = 0; i03 < ne03; ++i03)
    for (int64_t i02 = 0; i02 < ne02; ++i02)
    for (int64_t i01 = 0; i01 < ne01; ++i01) {
        const float * x = (const float *)((const char *)src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        float       * y = (float       *)((char *)dst->data         + i01*nb1  + i02*nb2  + i03*nb3);

        double sum = 0.0;
        for (int64_t i = 0; i < ne00; ++i) sum += x[i];
        float mean = (float)(sum / ne00);

        double var = 0.0;
        for (int64_t i = 0; i < ne00; ++i) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= ne00;

        float scale = 1.0f / sqrtf((float)var + eps);
        for (int64_t i = 0; i < ne00; ++i) y[i] = (x[i] - mean) * scale;
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// RMS_NORM
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_rms_norm(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0];
    ggml_tensor       * dst  = node;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return GGML_STATUS_FAILED;

    float eps;
    memcpy(&eps, node->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1],
                  ne02 = src0->ne[2], ne03 = src0->ne[3];
    const size_t  nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];

    for (int64_t i03 = 0; i03 < ne03; ++i03)
    for (int64_t i02 = 0; i02 < ne02; ++i02)
    for (int64_t i01 = 0; i01 < ne01; ++i01) {
        const float * x = (const float *)((const char *)src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        float       * y = (float       *)((char *)dst->data         + i01*nb1  + i02*nb2  + i03*nb3);

        double ss = 0.0;
        for (int64_t i = 0; i < ne00; ++i) ss += (double)x[i] * x[i];
        float scale = 1.0f / sqrtf((float)(ss / ne00) + eps);
        for (int64_t i = 0; i < ne00; ++i) y[i] = x[i] * scale;
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// SOFT_MAX  (F32 only, with optional F16/F32 mask and ALiBi)
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_soft_max(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1]; // optional mask
    const ggml_tensor * src2 = node->src[2]; // optional sink
    ggml_tensor       * dst  = node;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return GGML_STATUS_FAILED;
    if (src1 && src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16)
        return GGML_STATUS_FAILED;
    if (src2 && src2->type != GGML_TYPE_F32 && src2->type != GGML_TYPE_F16)
        return GGML_STATUS_FAILED;

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (const float *)node->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *)node->op_params + 1, sizeof(float));

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1],
                  ne02 = src0->ne[2], ne03 = src0->ne[3];
    const size_t  nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];

    const int64_t nb11 = src1 ? (int64_t)src1->nb[1] : 1;
    const int64_t nb12 = src1 ? (int64_t)src1->nb[2] : 1;
    const int64_t nb13 = src1 ? (int64_t)src1->nb[3] : 1;
    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;
    const int64_t nb21 = src2 ? (int64_t)src2->nb[1] : 1;

    const uint32_t n_head      = (uint32_t)ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    const bool use_f16_mask = (src1 && src1->type == GGML_TYPE_F16);
    const bool use_f16_sink = (src2 && src2->type == GGML_TYPE_F16);

    for (int64_t i03 = 0; i03 < ne03; ++i03)
    for (int64_t i02 = 0; i02 < ne02; ++i02)
    for (int64_t i01 = 0; i01 < ne01; ++i01) {
        const uint32_t h  = (uint32_t)i02;
        const float slope = (max_bias > 0.0f) ?
            (h < n_head_log2 ? powf(m0, h+1) : powf(m1, 2*(h - n_head_log2)+1)) : 1.0f;

        const float * row = (const float *)((const char *)src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        float       * dp = (float       *)((char *)dst->data         + i01*nb1  + i02*nb2  + i03*nb3);

        const int64_t i12 = i02 % ne12, i13 = i03 % ne13;
        const char * mp = src1 ? (const char *)src1->data + i01*nb11 + i12*nb12 + i13*nb13 : nullptr;
        const char * sink_ptr = src2 ? (const char *)src2->data + i02*nb21 : nullptr;
        float sink = 0.0f;
        if (sink_ptr != nullptr) {
            sink = use_f16_sink ? ggml_fp16_to_fp32(*(const ggml_fp16_t *)sink_ptr) : *(const float *)sink_ptr;
        }

        // accumulate into dst (as work buffer), apply scale + mask
        float vmax = -INFINITY;
        for (int64_t i = 0; i < ne00; ++i) {
            float v = row[i] * scale;
            if (mp) {
                v += slope * (use_f16_mask ?
                    ggml_fp16_to_fp32(((const ggml_fp16_t *)mp)[i]) :
                    ((const float *)mp)[i]);
            }
            dp[i] = v;
            if (v > vmax) vmax = v;
        }

        if (sink_ptr != nullptr) {
            vmax = std::max(vmax, sink);
        }

        double sum = 0.0;
        for (int64_t i = 0; i < ne00; ++i) {
            dp[i] = expf(dp[i] - vmax);
            sum += dp[i];
        }
        if (sink_ptr != nullptr) {
            sum += expf(sink - vmax);
        }
        float inv = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
        for (int64_t i = 0; i < ne00; ++i) dp[i] *= inv;
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// ROPE helpers (YaRN)
// ---------------------------------------------------------------------------
static inline float rope_yarn_ramp_k(float low, float high, int i0) {
    float y = ((float)(i0/2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static void rope_compute_cos_sin(float theta, float freq_scale,
                                  float corr_dims[2], int64_t i0,
                                  float ext_factor, float mscale,
                                  float * cos_out, float * sin_out) {
    float theta_interp = freq_scale * theta;
    float th = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp = rope_yarn_ramp_k(corr_dims[0], corr_dims[1], (int)i0) * ext_factor;
        th = theta_interp * (1.0f - ramp) + theta * ramp;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_out = cosf(th) * mscale;
    *sin_out = sinf(th) * mscale;
}

template<typename T>
static inline float rope_load_value(const T * row, int64_t i0);

template<>
inline float rope_load_value<float>(const float * row, int64_t i0) {
    return row[i0];
}

template<>
inline float rope_load_value<ggml_fp16_t>(const ggml_fp16_t * row, int64_t i0) {
    return ggml_fp16_to_fp32(row[i0]);
}

template<typename T>
static inline void rope_store_value(T * row, int64_t i0, float v);

template<>
inline void rope_store_value<float>(float * row, int64_t i0, float v) {
    row[i0] = v;
}

template<>
inline void rope_store_value<ggml_fp16_t>(ggml_fp16_t * row, int64_t i0, float v) {
    row[i0] = ggml_fp32_to_fp16(v);
}

template<typename T>
static enum ggml_status rope_impl_t(struct ggml_tensor * node, bool forward) {
    const ggml_tensor * src0 = node->src[0]; // Q or K  [D, N, H, B]
    const ggml_tensor * src1 = node->src[1]; // positions I32
    const ggml_tensor * src2 = node->src[2]; // freq factors (optional)
    ggml_tensor       * dst  = node;

    if (src0->type != dst->type || (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16))
        return GGML_STATUS_FAILED;
    if (src1->type != GGML_TYPE_I32)
        return GGML_STATUS_FAILED;

    const int n_dims     = ggml_get_op_params_i32(node, 1);
    const int mode       = ggml_get_op_params_i32(node, 2);
    const int n_ctx_orig = ggml_get_op_params_i32(node, 4);

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (const int32_t *)node->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *)node->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *)node->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *)node->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *)node->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *)node->op_params + 10, sizeof(float));

    // mrope: only support NORMAL / NEOX for now
    if (mode & GGML_ROPE_TYPE_MROPE) return GGML_STATUS_FAILED;
    if (mode == GGML_ROPE_TYPE_VISION) return GGML_STATUS_FAILED;
    if (n_dims <= 0 || (n_dims & 1) != 0) return GGML_STATUS_FAILED;

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const float sin_sign   = forward ? 1.0f : -1.0f;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);

    const int64_t ne0 = src0->ne[0], ne1 = src0->ne[1],
                  ne2 = src0->ne[2], ne3 = src0->ne[3];
    const size_t  nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t  nb0  = dst->nb[0],  nb1  = dst->nb[1],  nb2  = dst->nb[2],  nb3  = dst->nb[3];

    if (nb0 != nb00 || nb0 != sizeof(T)) {
        return GGML_STATUS_FAILED;
    }
    if (n_dims > ne0) {
        return GGML_STATUS_FAILED;
    }

    const float * freq_factors = nullptr;
    if (src2) {
        if (src2->type != GGML_TYPE_F32 || src2->ne[0] < n_dims / 2) {
            return GGML_STATUS_FAILED;
        }
        freq_factors = (const float *)src2->data;
    }

    const int32_t * pos = (const int32_t *)src1->data;

    for (int64_t i3 = 0; i3 < ne3; ++i3)
    for (int64_t i2 = 0; i2 < ne2; ++i2) // token
    for (int64_t i1 = 0; i1 < ne1; ++i1) { // head
        const int64_t p = pos[i2];

        const T * src_row = (const T *)((const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03);
        T       * dst_row = (T       *)((char *)dst->data         + i1*nb1  + i2*nb2  + i3*nb3);

        float theta = (float)p;
        for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
            const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
            float cs, ss;
            rope_compute_cos_sin(theta/ff, freq_scale, corr_dims, i0,
                                 ext_factor, attn_factor, &cs, &ss);
            ss *= sin_sign;

            int64_t idx0, idx1;
            if (mode == GGML_ROPE_TYPE_NORMAL) {
                // CPU semantics: normal RoPE rotates adjacent pairs.
                idx0 = i0;
                idx1 = i0 + 1;
            } else if (mode == GGML_ROPE_TYPE_NEOX) {
                // NEOX pairs the first half with the second half.
                idx0 = i0/2;
                idx1 = i0/2 + n_dims/2;
            } else {
                return GGML_STATUS_FAILED;
            }

            float x0 = rope_load_value(src_row, idx0);
            float x1 = rope_load_value(src_row, idx1);
            rope_store_value(dst_row, idx0, x0 * cs - x1 * ss);
            rope_store_value(dst_row, idx1, x0 * ss + x1 * cs);

            theta *= theta_scale;
        }
        // copy remaining dimensions verbatim
        for (int64_t i0 = n_dims; i0 < ne0; ++i0) {
            rope_store_value(dst_row, i0, rope_load_value(src_row, i0));
        }
    }
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_awnpu_kernel_rope(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    switch (node->src[0]->type) {
        case GGML_TYPE_F32: return rope_impl_t<float>(node, true);
        case GGML_TYPE_F16: return rope_impl_t<ggml_fp16_t>(node, true);
        default:            return GGML_STATUS_FAILED;
    }
}

enum ggml_status ggml_backend_awnpu_kernel_rope_back(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    switch (node->src[0]->type) {
        case GGML_TYPE_F32: return rope_impl_t<float>(node, false);
        case GGML_TYPE_F16: return rope_impl_t<ggml_fp16_t>(node, false);
        default:            return GGML_STATUS_FAILED;
    }
}

// ---------------------------------------------------------------------------
// SCALE  (dst = src0 * s + b)
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_scale(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0];
    ggml_tensor       * dst  = node;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return GGML_STATUS_FAILED;

    float s, b;
    memcpy(&s, (const float *)node->op_params + 0, sizeof(float));
    memcpy(&b, (const float *)node->op_params + 1, sizeof(float));

    const int64_t nr = ggml_nrows(src0);
    const int64_t nc = src0->ne[0];
    const size_t  nb01 = src0->nb[1];
    const size_t  nb1  = dst->nb[1];

    for (int64_t ir = 0; ir < nr; ++ir) {
        const float * s0 = (const float *)((const char *)src0->data + ir * nb01);
        float       * d  = (float       *)((char *)dst->data         + ir * nb1);
        for (int64_t i = 0; i < nc; ++i) d[i] = s0[i] * s + b;
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// CLAMP
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_clamp(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0];
    ggml_tensor       * dst  = node;

    if (!is_float_type(src0->type) || !is_float_type(dst->type))
        return GGML_STATUS_FAILED;

    float vmin, vmax;
    memcpy(&vmin, (const float *)node->op_params + 0, sizeof(float));
    memcpy(&vmax, (const float *)node->op_params + 1, sizeof(float));

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1],
                  ne2 = dst->ne[2], ne3 = dst->ne[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1],
                 nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb0  = dst->nb[0],  nb1  = dst->nb[1],
                 nb2  = dst->nb[2],  nb3  = dst->nb[3];

    for (int64_t i3 = 0; i3 < ne3; ++i3)
    for (int64_t i2 = 0; i2 < ne2; ++i2)
    for (int64_t i1 = 0; i1 < ne1; ++i1)
    for (int64_t i0 = 0; i0 < ne0; ++i0) {
        float v = load_f32(src0, i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
        store_f32(dst, i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3, fmaxf(fminf(v, vmax), vmin));
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// SET_ROWS  — mirrors ggml_compute_forward_set_rows (F32 src0 → float/quant dst)
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_set_rows(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0]; // F32 source rows
    const ggml_tensor * src1 = node->src[1]; // I32 / I64 destination row indices
    ggml_tensor       * dst  = node;

    GGML_TENSOR_BINARY_OP_LOCALS

    if (src0->type != GGML_TYPE_F32) {
        return GGML_STATUS_FAILED;
    }
    if (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        return GGML_STATUS_FAILED;
    }
    if (!is_float_type(dst->type)) {
        return GGML_STATUS_FAILED;
    }

    const int64_t nc = ne00;
    const int64_t nr = ne01;

    if (ne0 != nc || ne2 != ne02 || ne3 != ne03) {
        return GGML_STATUS_FAILED;
    }
    if (ne02 % ne11 != 0 || ne03 % ne12 != 0) {
        return GGML_STATUS_FAILED;
    }

    const bool use_i64 = (src1->type == GGML_TYPE_I64);

    for (int64_t i3 = 0; i3 < ne03; ++i3) {
        for (int64_t i2 = 0; i2 < ne02; ++i2) {
            for (int64_t ir = 0; ir < nr; ++ir) {
                const int64_t i12 = i3 % ne12;
                const int64_t i11 = i2 % ne11;
                const char * idx_ptr = (const char *) src1->data + ir * nb10 + i11 * nb11 + i12 * nb12;
                const int64_t i1 = use_i64 ? *(const int64_t *) idx_ptr : *(const int32_t *) idx_ptr;
                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                const float * src_row = (const float *)((const char *) src0->data + ir * nb01 + i2 * nb02 + i3 * nb03);
                char        * dst_row = (char       *) dst->data         + i1 * nb1  + i2 * nb2  + i3 * nb3;

                switch (dst->type) {
                    case GGML_TYPE_F32:
                        memcpy(dst_row, src_row, (size_t) nc * sizeof(float));
                        break;
                    case GGML_TYPE_F16:
                        ggml_fp32_to_fp16_row(src_row, (ggml_fp16_t *) dst_row, nc);
                        break;
                    case GGML_TYPE_BF16:
                        ggml_fp32_to_bf16_row(src_row, (ggml_bf16_t *) dst_row, nc);
                        break;
                    default:
                        return GGML_STATUS_FAILED;
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// GLU variants
// ---------------------------------------------------------------------------
static enum ggml_status glu_f32(struct ggml_tensor * node, enum ggml_glu_op op) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1]; // may be null → split src0
    ggml_tensor       * dst  = node;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return GGML_STATUS_FAILED;
    if (src1 && src1->type != GGML_TYPE_F32)
        return GGML_STATUS_FAILED;

    const int nc  = (int)(src1 ? src0->ne[0] : src0->ne[0] / 2);
    const int nr  = (int)ggml_nrows(src0);
    const int32_t swapped = ggml_get_op_params_i32(node, 1);

    const size_t src0_o = src0->nb[1];
    const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    const size_t dst_o  = dst->nb[1];

    for (int ir = 0; ir < nr; ++ir) {
        const float * gate = (const float *)((const char *)src0->data + ir * src0_o);
        const float * x;
        if (src1) {
            x = (const float *)((const char *)src1->data + ir * src1_o);
        } else {
            gate = (const float *)((const char *)src0->data + ir*src0_o) + (swapped ? nc : 0);
            x    = (const float *)((const char *)src0->data + ir*src0_o) + (swapped ? 0  : nc);
        }
        float * y = (float *)((char *)dst->data + ir * dst_o);

        switch (op) {
            case GGML_GLU_OP_REGLU:
                for (int i = 0; i < nc; ++i)
                    y[i] = (gate[i] > 0.f) ? gate[i] * x[i] : 0.f;
                break;
            case GGML_GLU_OP_GEGLU:
                for (int i = 0; i < nc; ++i)
                    y[i] = gelu_f32(gate[i]) * x[i];
                break;
            case GGML_GLU_OP_SWIGLU:
            case GGML_GLU_OP_SWIGLU_OAI:
                for (int i = 0; i < nc; ++i)
                    y[i] = silu_f32(gate[i]) * x[i];
                break;
            case GGML_GLU_OP_GEGLU_ERF:
                for (int i = 0; i < nc; ++i)
                    y[i] = gelu_erf_f32(gate[i]) * x[i];
                break;
            case GGML_GLU_OP_GEGLU_QUICK:
                for (int i = 0; i < nc; ++i)
                    y[i] = gelu_quick_f32(gate[i]) * x[i];
                break;
            default:
                return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_awnpu_kernel_glu(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    return glu_f32(node, ggml_get_glu_op(node));
}

// ---------------------------------------------------------------------------
// FLASH_ATTN_EXT  (F32 Q/K/V, standard scaled dot-product attention)
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_flash_attn_ext(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * q    = node->src[0]; // [DK, N, H, B]
    const ggml_tensor * k    = node->src[1]; // [DK, M, H, B]
    const ggml_tensor * v    = node->src[2]; // [DV, M, H, B]
    const ggml_tensor * mask = node->src[3]; // optional [M, N, ...]  F16
    ggml_tensor       * dst  = node;         // [DV, H, N, B]

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 ||
        v->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return GGML_STATUS_FAILED;
    if (mask && mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32)
        return GGML_STATUS_FAILED;

    float scale    = 1.0f;
    float max_bias = 0.0f;
    float logit_sc = 0.0f;
    memcpy(&scale,    (const float *)node->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *)node->op_params + 1, sizeof(float));
    memcpy(&logit_sc, (const float *)node->op_params + 2, sizeof(float));
    if (logit_sc != 0.0f) scale /= logit_sc;

    const int64_t DK  = k->ne[0],  DV  = v->ne[0];
    const int64_t N   = q->ne[1];  // query tokens
    const int64_t H   = q->ne[2];  // heads
    const int64_t B   = q->ne[3];  // batch
    const int64_t M   = k->ne[1];  // kv tokens
    const int64_t Hk  = k->ne[2],  Bk  = k->ne[3];
    const int64_t Hv  = v->ne[2],  Bv  = v->ne[3];

    const size_t nbq1 = q->nb[1], nbq2 = q->nb[2], nbq3 = q->nb[3];
    const size_t nbk1 = k->nb[1], nbk2 = k->nb[2], nbk3 = k->nb[3];
    const size_t nbv1 = v->nb[1], nbv2 = v->nb[2], nbv3 = v->nb[3];
    const size_t nb1  = dst->nb[1];
    const int64_t ne1 = dst->ne[1]; // H in dst layout

    const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)H));
    const float m0 = powf(2.0f, -(max_bias       ) / (float)n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float)n_head_log2);

    std::vector<float> VKQ(DV), scores(M);

    for (int64_t i3 = 0; i3 < B;  ++i3)
    for (int64_t i2 = 0; i2 < H;  ++i2) // head
    for (int64_t i1 = 0; i1 < N;  ++i1) { // query token
        const int64_t ik2 = i2 % Hk, ik3 = i3 % Bk;
        const int64_t iv2 = i2 % Hv, iv3 = i3 % Bv;

        const uint32_t h = (uint32_t)i2;
        const float slope = (max_bias > 0.0f) ?
            (h < n_head_log2 ? powf(m0, h+1) : powf(m1, 2*(h - n_head_log2)+1)) : 1.0f;

        const float * qrow = (const float *)((const char *)q->data + i1*nbq1 + i2*nbq2 + i3*nbq3);

        // optional mask row
        const char * mrow = nullptr;
        bool use_f16_mask = false;
        if (mask) {
            const int64_t mi1 = i1 % mask->ne[1];
            const int64_t mi2 = i2 % mask->ne[2];
            const int64_t mi3 = i3 % mask->ne[3];
            mrow = (const char *)mask->data + mi1*mask->nb[1] + mi2*mask->nb[2] + mi3*mask->nb[3];
            use_f16_mask = (mask->type == GGML_TYPE_F16);
        }

        // compute scores
        float M_val = -INFINITY;
        for (int64_t ic = 0; ic < M; ++ic) {
            const float * krow = (const float *)((const char *)k->data + ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            float s = 0.0f;
            for (int64_t d = 0; d < DK; ++d) s += qrow[d] * krow[d];
            s *= scale;
            if (logit_sc != 0.0f) s = logit_sc * tanhf(s);
            if (mrow) {
                float mv = use_f16_mask ?
                    slope * ggml_fp16_to_fp32(((const ggml_fp16_t *)mrow)[ic]) :
                    slope * ((const float *)mrow)[ic];
                s += mv;
            }
            scores[ic] = s;
            if (s > M_val) M_val = s;
        }

        // softmax + weighted sum (online, F32 accumulation)
        std::fill(VKQ.begin(), VKQ.end(), 0.0f);
        float S = 0.0f;
        for (int64_t ic = 0; ic < M; ++ic) {
            float w = expf(scores[ic] - M_val);
            S += w;
            const float * vrow = (const float *)((const char *)v->data + ic*nbv1 + iv2*nbv2 + iv3*nbv3);
            for (int64_t d = 0; d < DV; ++d) VKQ[d] += w * vrow[d];
        }
        float inv_S = (S > 0.0f) ? 1.0f / S : 0.0f;

        // write output: permute(0,2,1,3) → dst[i3*ne1*N*nb1 + i2 + i1*ne1]
        float * out = (float *)((char *)dst->data + (i3*ne1*N + i2 + i1*ne1) * nb1);
        for (int64_t d = 0; d < DV; ++d) out[d] = VKQ[d] * inv_S;
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// DUP / CPY / CONT  (copy with optional type conversion, F32↔F16/BF16)
// ---------------------------------------------------------------------------
static enum ggml_status dup_impl(struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    // same-type copy path: keep byte layout intact, even for quantized tensors.
    if (src0->type == dst->type) {
        return copy_same_type(dst);
    }

    // compatible float/int conversion path.
    if ((src0->type == GGML_TYPE_I32 && is_float_type(dst->type)) ||
        (dst->type == GGML_TYPE_I32 && is_float_type(src0->type))) {
        return copy_i32_and_float(dst);
    }

    // quantized <-> float path when the public type traits provide converters.
    if (copy_via_traits(dst) == GGML_STATUS_SUCCESS) {
        return GGML_STATUS_SUCCESS;
    }

    if (!is_float_type(src0->type) || !is_float_type(dst->type)) {
        return GGML_STATUS_FAILED;
    }

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1],
                  ne2 = dst->ne[2], ne3 = dst->ne[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1],
                 nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb0  = dst->nb[0],  nb1  = dst->nb[1],
                 nb2  = dst->nb[2],  nb3  = dst->nb[3];

    for (int64_t i3 = 0; i3 < ne3; ++i3)
    for (int64_t i2 = 0; i2 < ne2; ++i2)
    for (int64_t i1 = 0; i1 < ne1; ++i1)
    for (int64_t i0 = 0; i0 < ne0; ++i0) {
        float v = load_f32(src0, i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
        store_f32(dst, i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3, v);
    }
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_awnpu_kernel_cpy(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    return dup_impl(node);
}

enum ggml_status ggml_backend_awnpu_kernel_cont(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    return dup_impl(node);
}

enum ggml_status ggml_backend_awnpu_kernel_dup(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    return dup_impl(node);
}

// ---------------------------------------------------------------------------
// PAD  (zero-pad, F32 only)
// ---------------------------------------------------------------------------
enum ggml_status ggml_backend_awnpu_kernel_pad(int device_id, struct ggml_tensor * node) {
    GGML_UNUSED(device_id);

    const ggml_tensor * src0 = node->src[0];
    ggml_tensor       * dst  = node;

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
        return GGML_STATUS_FAILED;

    const int64_t ne0 = dst->ne[0], ne1 = dst->ne[1],
                  ne2 = dst->ne[2], ne3 = dst->ne[3];
    const size_t  nb00 = src0->nb[0], nb01 = src0->nb[1],
                  nb02 = src0->nb[2], nb03 = src0->nb[3];

    const int32_t lp0 = ggml_get_op_params_i32(node, 0);
    const int32_t rp0 = ggml_get_op_params_i32(node, 1);
    const int32_t lp1 = ggml_get_op_params_i32(node, 2);
    const int32_t rp1 = ggml_get_op_params_i32(node, 3);
    const int32_t lp2 = ggml_get_op_params_i32(node, 4);
    const int32_t rp2 = ggml_get_op_params_i32(node, 5);
    const int32_t lp3 = ggml_get_op_params_i32(node, 6);
    const int32_t rp3 = ggml_get_op_params_i32(node, 7);

    float * out = (float *)dst->data;

    for (int64_t i3 = 0; i3 < ne3; ++i3)
    for (int64_t i2 = 0; i2 < ne2; ++i2)
    for (int64_t i1 = 0; i1 < ne1; ++i1)
    for (int64_t i0 = 0; i0 < ne0; ++i0) {
        const int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
        if (i0 >= lp0 && i0 < ne0 - rp0 &&
            i1 >= lp1 && i1 < ne1 - rp1 &&
            i2 >= lp2 && i2 < ne2 - rp2 &&
            i3 >= lp3 && i3 < ne3 - rp3) {
            const int64_t src_idx = (i3-lp3)*nb03 + (i2-lp2)*nb02 + (i1-lp1)*nb01 + (i0-lp0)*nb00;
            out[dst_idx] = *(const float *)((const char *)src0->data + src_idx);
        } else {
            out[dst_idx] = 0.0f;
        }
    }
    return GGML_STATUS_SUCCESS;
}
