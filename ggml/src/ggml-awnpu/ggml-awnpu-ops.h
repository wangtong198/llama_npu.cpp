#pragma once

#include "ggml-backend.h"
#include "ggml.h"

bool ggml_backend_awnpu_op_supported(enum ggml_op op);

enum ggml_status ggml_backend_awnpu_npu_compute_node(
        int device_id,
        struct ggml_tensor * node,
        ggml_abort_callback abort_callback,
        void * abort_callback_data);
