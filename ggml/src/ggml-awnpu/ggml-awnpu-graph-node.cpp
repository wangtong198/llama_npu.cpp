#include "ggml-awnpu-graph-node.h"

#if defined(GGML_AWNPU_KERNEL_TYPE_NATIVE)
#include "ggml-awnpu-kernels-native.h"
#elif defined(GGML_AWNPU_KERNEL_TYPE_AWNPU)
#include "ggml-awnpu-kernels-npu.h"
#else
#error "KERNEL_TYPE must be native or awnpu when prebuilding the library"
#endif

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
#define GGML_AWNPU_KERNEL(ggml_op, name) case GGML_OP_##ggml_op:
#include "ggml-awnpu-kernels.def"
#undef GGML_AWNPU_KERNEL
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
    GGML_ASSERT(node != nullptr);

    if (ggml_backend_awnpu_op_is_layout_only(node->op)) {
        return GGML_STATUS_SUCCESS;
    }

    switch (node->op) {
#define GGML_AWNPU_KERNEL(ggml_op, name) \
        case GGML_OP_##ggml_op: return ggml_backend_awnpu_kernel_##name(device_id, node);
#include "ggml-awnpu-kernels.def"
#undef GGML_AWNPU_KERNEL
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
