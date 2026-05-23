#pragma once

#include "ggml.h"

// Per-op AWNPU kernel entry points (native CPU reference implementations).
// Return GGML_STATUS_SUCCESS when implemented.
#define GGML_AWNPU_KERNEL(op, name) \
    enum ggml_status ggml_backend_awnpu_kernel_##name(int device_id, struct ggml_tensor * node);

#include "ggml-awnpu-kernels.def"

#undef GGML_AWNPU_KERNEL
