// AWNPU kernel entry points for the device-backed implementation.
// The actual op implementations will be filled in later; for now each
// kernel returns GGML_STATUS_FAILED so the caller can fall back cleanly.

#include "ggml-awnpu-kernels-npu.h"

#define GGML_AWNPU_KERNEL(op, name) \
    enum ggml_status ggml_backend_awnpu_kernel_##name(int device_id, struct ggml_tensor * node) { \
        (void) device_id; \
        (void) node; \
        return GGML_STATUS_FAILED; \
    }

#include "ggml-awnpu-kernels.def"

#undef GGML_AWNPU_KERNEL
