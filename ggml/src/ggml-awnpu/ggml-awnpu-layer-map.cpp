#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-awnpu-layer-map.h"

#include <cstring>

static bool name_has_prefix(const char * name, const char * prefix, size_t prefix_len) {
    return name != nullptr && std::strncmp(name, prefix, prefix_len) == 0;
}

bool ggml_backend_awnpu_buffer_name_is_npu(const char * buft_name) {
    if (buft_name == nullptr || buft_name[0] == '\0') {
        return false;
    }

    return name_has_prefix(buft_name, GGML_AWNPU_NAME, sizeof(GGML_AWNPU_NAME) - 1) &&
           buft_name[sizeof(GGML_AWNPU_NAME) - 1] != '\0';
}

static ggml_backend_buffer_t ggml_backend_awnpu_tensor_buffer(const struct ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return nullptr;
    }

    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer == nullptr && tensor->view_src != nullptr) {
        buffer = tensor->view_src->buffer;
    }

    return buffer;
}

bool ggml_backend_awnpu_tensor_on_npu(const struct ggml_tensor * tensor) {
    ggml_backend_buffer_t buffer = ggml_backend_awnpu_tensor_buffer(tensor);
    if (buffer == nullptr) {
        return false;
    }

    return ggml_backend_awnpu_buffer_name_is_npu(ggml_backend_buffer_name(buffer));
}

bool ggml_backend_awnpu_weight_on_npu(const struct ggml_tensor * tensor) {
    ggml_backend_buffer_t buffer = ggml_backend_awnpu_tensor_buffer(tensor);
    if (buffer == nullptr || !ggml_backend_awnpu_tensor_on_npu(tensor)) {
        return false;
    }

    return buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
           (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY && tensor->op == GGML_OP_NONE);
}

bool ggml_backend_awnpu_node_is_layout_only(const struct ggml_tensor * node) {
    if (node == nullptr) {
        return false;
    }

    switch (node->op) {
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

static bool ggml_backend_awnpu_node_has_npu_weight_operand(const struct ggml_tensor * node) {
    if (node == nullptr) {
        return false;
    }

    switch (node->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (ggml_backend_awnpu_weight_on_npu(node->src[i])) {
                    return true;
                }
            }
            return false;
        case GGML_OP_MUL:
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                const struct ggml_tensor * src = node->src[i];
                if (src != nullptr && src->buffer != nullptr &&
                    src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_awnpu_weight_on_npu(src)) {
                    return true;
                }
            }
            return false;
        default:
            return false;
    }
}

bool ggml_backend_awnpu_node_on_npu(
        const struct ggml_tensor * node,
        int depth,
        std::unordered_set<const struct ggml_tensor *> * visited,
        int max_depth) {
    if (node == nullptr || depth > max_depth) {
        return false;
    }

    if (visited != nullptr) {
        if (!visited->insert(node).second) {
            return false;
        }
    }

    if (ggml_backend_awnpu_node_is_layout_only(node)) {
        if (node->view_src != nullptr) {
            return ggml_backend_awnpu_node_on_npu(node->view_src, depth + 1, visited, max_depth);
        }

        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (ggml_backend_awnpu_node_on_npu(node->src[i], depth + 1, visited, max_depth)) {
                return true;
            }
        }

        return ggml_backend_awnpu_tensor_on_npu(node);
    }

    if (node->op == GGML_OP_NONE) {
        return ggml_backend_awnpu_tensor_on_npu(node);
    }

    if (ggml_backend_awnpu_node_has_npu_weight_operand(node)) {
        return true;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (ggml_backend_awnpu_node_on_npu(node->src[i], depth + 1, visited, max_depth)) {
            return true;
        }
    }

    return false;
}
