// Placeholder kernel implementations when native reference kernels are disabled.
// Each op returns GGML_STATUS_FAILED so the AWNPU backend can fall back to CPU.

#include "ggml-awnpu-kernels-native.h"

#define GGML_AWNPU_KERNEL(op, name) \
    enum ggml_status ggml_backend_awnpu_kernel_##name(int device_id, struct ggml_tensor * node) { \
        (void) device_id; \
        (void) node; \
        return GGML_STATUS_FAILED; \
    }

#include "ggml-awnpu-kernels.def"

#undef GGML_AWNPU_KERNEL
