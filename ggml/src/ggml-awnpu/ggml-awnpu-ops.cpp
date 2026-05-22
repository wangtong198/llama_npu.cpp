#include "ggml-awnpu-ops.h"

#include "ggml-awnpu-layer-map.h"
#include "ggml-impl.h"
#include "ggml.h"

static bool ggml_backend_awnpu_op_is_layout_only(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static bool ggml_backend_awnpu_op_is_compute_supported(enum ggml_op op) {
    switch (op) {
        case GGML_OP_GET_ROWS:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_UNARY: // ggml_silu / ggml_gelu / ...
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_SET_ROWS:
        case GGML_OP_GLU:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
        case GGML_OP_PAD:
            return true;
        default:
            return false;
    }
}

bool ggml_backend_awnpu_op_supported(enum ggml_op op) {
    return ggml_backend_awnpu_op_is_layout_only(op) || ggml_backend_awnpu_op_is_compute_supported(op);
}

static bool ggml_backend_awnpu_npu_abort_requested(
        ggml_abort_callback abort_callback,
        void * abort_callback_data) {
    return abort_callback != nullptr && abort_callback(abort_callback_data);
}

static enum ggml_status ggml_backend_awnpu_npu_compute_op(
        int device_id,
        struct ggml_tensor * node) {
    GGML_UNUSED(device_id);
    GGML_ASSERT(node != nullptr);

    if (ggml_backend_awnpu_op_is_layout_only(node->op)) {
        return GGML_STATUS_SUCCESS;
    }

    switch (node->op) {
        case GGML_OP_GET_ROWS:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_SET_ROWS:
        case GGML_OP_GLU:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
        case GGML_OP_PAD:
            // Per-op NPU kernels are implemented here.
            return GGML_STATUS_FAILED;
        default:
            return GGML_STATUS_FAILED;
    }
}

enum ggml_status ggml_backend_awnpu_npu_compute_node(
        int device_id,
        struct ggml_tensor * node,
        ggml_abort_callback abort_callback,
        void * abort_callback_data) {
    GGML_ASSERT(node != nullptr);

    if (ggml_backend_awnpu_npu_abort_requested(abort_callback, abort_callback_data)) {
        return GGML_STATUS_ABORTED;
    }

    if (ggml_backend_awnpu_node_is_layout_only(node)) {
        return GGML_STATUS_SUCCESS;
    }

    return ggml_backend_awnpu_npu_compute_op(device_id, node);
}
