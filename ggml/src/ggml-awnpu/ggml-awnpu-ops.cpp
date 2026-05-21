#include "ggml-awnpu-ops.h"

#include "ggml-impl.h"

static enum ggml_status ggml_backend_awnpu_sim_forward_to_node(
        void * ctx,
        const struct ggml_backend_awnpu_op_dispatch * dispatch,
        struct ggml_cgraph * cgraph,
        int node_idx) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(dispatch != nullptr);
    GGML_ASSERT(dispatch->get_last_node_idx != nullptr);
    GGML_ASSERT(dispatch->set_last_node_idx != nullptr);
    GGML_ASSERT(dispatch->forward_range != nullptr);
    GGML_ASSERT(cgraph != nullptr);

    const int i1 = node_idx + 1;
    const int last_node_idx = dispatch->get_last_node_idx(ctx);
    const int i0 = last_node_idx < 0 ? 0 : last_node_idx;

    if (i1 <= i0) {
        return GGML_STATUS_SUCCESS;
    }

    const enum ggml_status status = dispatch->forward_range(ctx, cgraph, i0, i1);
    if (status == GGML_STATUS_SUCCESS) {
        dispatch->set_last_node_idx(ctx, i1);
    }

    return status;
}

static bool ggml_backend_awnpu_op_is_supported(enum ggml_op op) {
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
    return ggml_backend_awnpu_op_is_supported(op);
}

enum ggml_status ggml_backend_awnpu_compute_node_sim_op(
        void * ctx,
        const struct ggml_backend_awnpu_op_dispatch * dispatch,
        struct ggml_cgraph * cgraph,
        int node_idx,
        struct ggml_tensor * node) {
    GGML_ASSERT(node != nullptr);

    if (ggml_backend_awnpu_op_is_supported(node->op)) {
        return ggml_backend_awnpu_sim_forward_to_node(ctx, dispatch, cgraph, node_idx);
    }

    return GGML_STATUS_FAILED;
}

enum ggml_status ggml_backend_awnpu_compute_node_op(
        void * ctx,
        struct ggml_tensor * node) {
    GGML_ASSERT(node != nullptr);
    GGML_UNUSED(ctx);

    if (ggml_backend_awnpu_op_is_supported(node->op)) {
        return GGML_STATUS_SUCCESS;
    }

    return GGML_STATUS_FAILED;
}
