#pragma once

#include "ggml-awnpu.h"
#include "ggml.h"

#include <unordered_set>

bool ggml_backend_awnpu_buffer_name_is_npu(const char * buft_name);

bool ggml_backend_awnpu_tensor_on_npu(const struct ggml_tensor * tensor);

bool ggml_backend_awnpu_weight_on_npu(const struct ggml_tensor * tensor);

bool ggml_backend_awnpu_node_is_layout_only(const struct ggml_tensor * node);

bool ggml_backend_awnpu_node_on_npu(
        const struct ggml_tensor * node,
        int depth,
        std::unordered_set<const struct ggml_tensor *> * visited,
        int max_depth = 1024);
