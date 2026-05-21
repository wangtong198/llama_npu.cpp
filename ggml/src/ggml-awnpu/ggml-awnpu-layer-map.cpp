#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-awnpu-layer-map.h"
#include "ggml-awnpu-names.h"

#include <cctype>
#include <cstring>
#include <unordered_map>

static std::unordered_map<int, bool> g_layer_on_npu;
static bool                          g_output_on_npu = false;
static int                           g_model_n_layer = 0;
static bool                          g_reset_pending = false;

static int ggml_backend_awnpu_max_recorded_layer_index(void) {
    int max_il = -1;
    for (const auto & entry : g_layer_on_npu) {
        if (entry.first > max_il) {
            max_il = entry.first;
        }
    }
    return max_il;
}

// Upper bound for node->src[] recursion in node_runs_on_npu().
// Primary source: hparams.n_layer passed from llama.cpp after model load.
static int ggml_backend_awnpu_node_layer_infer_max_depth(void) {
    if (g_model_n_layer > 0) {
        return g_model_n_layer;
    }

    // Before llama notifies us, fall back to the highest layer index seen in weights.
    const int max_il = ggml_backend_awnpu_max_recorded_layer_index();
    return max_il >= 0 ? max_il + 1 : 0;
}

static bool name_has_prefix(const char * name, const char * prefix, size_t prefix_len) {
    return name != nullptr && std::strncmp(name, prefix, prefix_len) == 0;
}

bool ggml_backend_awnpu_buffer_name_is_npu(const char * buft_name) {
    if (buft_name == nullptr || buft_name[0] == '\0') {
        return false;
    }

    // AWNPU buft: "AWNPU0", "AWNPU1", ... (GGML_AWNPU_NAME + device index).
    return name_has_prefix(buft_name, GGML_AWNPU_NAME, GGML_AWNPU_BACKEND_NAME_LEN) &&
           buft_name[GGML_AWNPU_BACKEND_NAME_LEN] != '\0';
}

int ggml_backend_awnpu_parse_layer_index(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return -1;
    }

    // llama-arch: "blk.{il}.attn_q.weight" etc.
    if (name_has_prefix(name, GGML_AWNPU_TN_BLK, GGML_AWNPU_TN_BLK_LEN)) {
        const char * p = name + GGML_AWNPU_TN_BLK_LEN;
        char * end = nullptr;
        const long il = std::strtol(p, &end, 10);
        if (end != p && *end == '.') {
            return (int) il;
        }
    }

    static const struct {
        const char * prefix;
        size_t       len;
    } cache_prefixes[] = {
        { GGML_AWNPU_TN_CACHE_K, GGML_AWNPU_TN_CACHE_K_LEN },
        { GGML_AWNPU_TN_CACHE_V, GGML_AWNPU_TN_CACHE_V_LEN },
    };

    for (const auto & cp : cache_prefixes) {
        if (name_has_prefix(name, cp.prefix, cp.len)) {
            const char * p = name + cp.len;
            char * end = nullptr;
            const long il = std::strtol(p, &end, 10);
            if (end != p) {
                return (int) il;
            }
        }
    }

    // Intermediate activations: "ffn_inp-12", "l_out-23", ... (cb suffix in llama-graph).
    const char * dash = std::strrchr(name, '-');
    if (dash == nullptr || *(dash + 1) == '\0') {
        return -1;
    }

    for (const char * p = dash + 1; *p != '\0'; ++p) {
        if (!std::isdigit((unsigned char) *p)) {
            return -1;
        }
    }

    return (int) std::atoi(dash + 1);
}

void ggml_backend_awnpu_set_output_on_npu(bool on_npu) {
    if (on_npu) {
        g_output_on_npu = true;
    }
}

void ggml_backend_awnpu_set_model_n_layer(int n_layer) {
    // Weights may already be loaded; do not clear layer placement recorded in buffer init.
    if (n_layer > 0) {
        g_model_n_layer = n_layer;
        g_reset_pending = true;
    }
}

void ggml_backend_awnpu_record_layer_placement(const struct ggml_tensor * tensor, bool on_npu) {
    if (tensor == nullptr) {
        return;
    }

    if (g_reset_pending) {
        g_layer_on_npu.clear();
        g_output_on_npu = false;
        g_model_n_layer = 0;
        g_reset_pending = false;
    }

    if (ggml_backend_awnpu_is_output_weight_name(tensor->name)) {
        if (on_npu) {
            g_output_on_npu = true;
        }
    }

    const int il = ggml_backend_awnpu_parse_layer_index(tensor->name);
    if (il < 0) {
        return;
    }

    if (on_npu) {
        g_layer_on_npu[il] = true;
    } else if (g_layer_on_npu.find(il) == g_layer_on_npu.end()) {
        g_layer_on_npu[il] = false;
    }
}

bool ggml_backend_awnpu_layer_on_npu(int il) {
    if (il < 0) {
        return false;
    }

    const auto it = g_layer_on_npu.find(il);
    return it != g_layer_on_npu.end() && it->second;
}

bool ggml_backend_awnpu_is_output_weight_name(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }

    return std::strcmp(name, GGML_AWNPU_TN_OUTPUT_NORM_W) == 0 ||
           std::strcmp(name, GGML_AWNPU_TN_OUTPUT_W) == 0 ||
           std::strcmp(name, GGML_AWNPU_TN_OUTPUT_B) == 0 ||
           name_has_prefix(name, GGML_AWNPU_TN_OUTPUT_B_PREFIX, GGML_AWNPU_TN_OUTPUT_B_PREFIX_LEN);
}

// LLM_TENSOR_LAYER_INPUT (llama dev_input); never probed on AWNPU buft at load.
bool ggml_backend_awnpu_is_input_embedding_weight(const struct ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->name[0] == '\0') {
        return false;
    }

    const char * name = tensor->name;
    if (std::strcmp(name, "token_embd.weight") == 0 ||
        std::strcmp(name, "position_embd.weight") == 0 ||
        std::strcmp(name, "token_types.weight") == 0) {
        return true;
    }

    return name_has_prefix(name, "per_layer_token_embd", sizeof("per_layer_token_embd") - 1);
}

static bool ggml_backend_awnpu_tensor_is_static_weight(const struct ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->name[0] == '\0') {
        return false;
    }

    if (ggml_backend_awnpu_is_input_embedding_weight(tensor)) {
        return false;
    }

    const char * name = tensor->name;
    if (ggml_backend_awnpu_is_output_weight_name(name)) {
        return true;
    }

    return std::strstr(name, ".weight") != nullptr || std::strstr(name, ".bias") != nullptr;
}

int ggml_backend_awnpu_max_npu_layer(void) {
    int max_il = -1;
    for (const auto & entry : g_layer_on_npu) {
        if (entry.second && entry.first > max_il) {
            max_il = entry.first;
        }
    }
    return max_il;
}

bool ggml_backend_awnpu_lm_head_on_npu(void) {
    if (g_output_on_npu) {
        return true;
    }

    const int max_npu = ggml_backend_awnpu_max_npu_layer();
    return max_npu >= 0 && ggml_backend_awnpu_layer_on_npu(max_npu);
}

bool ggml_backend_awnpu_is_output_head_node(const struct ggml_tensor * op) {
    if (op == nullptr || op->name[0] == '\0') {
        return false;
    }

    if (std::strcmp(op->name, GGML_AWNPU_TN_RESULT_NORM) == 0 ||
        std::strcmp(op->name, GGML_AWNPU_TN_RESULT_OUTPUT) == 0) {
        return true;
    }

    // qwen2: final RMS before LM head; name is "norm" without layer suffix.
    if (std::strcmp(op->name, GGML_AWNPU_TN_NORM) == 0 && op->op == GGML_OP_RMS_NORM) {
        return true;
    }

    return false;
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

int ggml_backend_awnpu_infer_layer_index(const struct ggml_tensor * op) {
    if (op == nullptr) {
        return -1;
    }

    int il = ggml_backend_awnpu_parse_layer_index(op->name);
    if (il >= 0) {
        return il;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const struct ggml_tensor * src = op->src[i];
        if (src == nullptr || src->buffer == nullptr ||
            src->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            continue;
        }

        il = ggml_backend_awnpu_parse_layer_index(src->name);
        if (il >= 0) {
            return il;
        }
    }

    return -1;
}

/*
TODO：和模型有绑定，最好写成通用的，不依赖模型架构架构
*/
bool ggml_backend_awnpu_weight_on_npu(const struct ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }

    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer == nullptr && tensor->view_src != nullptr) {
        buffer = tensor->view_src->buffer;
    }

    if (buffer == nullptr) {
        return false;
    }

    if (ggml_backend_awnpu_buffer_name_is_npu(ggml_backend_buffer_name(buffer))) {
        if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return true;
        }
        // weight_buft_supported() load probe: temp buffer on NPU buft, usage still ANY.
        if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY &&
            ggml_backend_awnpu_tensor_is_static_weight(tensor)) {
            return true;
        }
        return false;
    }

    if (buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }

    const int il = ggml_backend_awnpu_parse_layer_index(tensor->name);
    if (il >= 0 && ggml_backend_awnpu_layer_on_npu(il)) {
        return true;
    }

    if (ggml_backend_awnpu_is_output_weight_name(tensor->name) && ggml_backend_awnpu_lm_head_on_npu()) {
        return true;
    }

    return false;
}

/*
在模型加载之后，通过调用回调函数ggml_backend_set_model_n_layer()设置模型层数
Q1,： 能不能不依赖模型层数？
*/
bool ggml_backend_awnpu_node_runs_on_npu(const struct ggml_tensor * node, int depth) {
    const int max_depth = ggml_backend_awnpu_node_layer_infer_max_depth();
    if (node == nullptr || max_depth <= 0 || depth > max_depth) {
        return false;
    }

    const int il = ggml_backend_awnpu_infer_layer_index(node);
    if (il >= 0) {
        return ggml_backend_awnpu_layer_on_npu(il);
    }

    if (ggml_backend_awnpu_is_output_head_node(node) && ggml_backend_awnpu_lm_head_on_npu()) {
        return true;
    }

    if (node->op == GGML_OP_NONE) {
        // KV cache views: physical buft reflects dev_layer(il) placement.
        if (name_has_prefix(node->name, GGML_AWNPU_TN_CACHE, GGML_AWNPU_TN_CACHE_LEN)) {
            ggml_backend_buffer_t buffer = node->buffer;
            if (buffer == nullptr && node->view_src != nullptr) {
                buffer = node->view_src->buffer;
            }
            return buffer != nullptr &&
                ggml_backend_awnpu_buffer_name_is_npu(ggml_backend_buffer_name(buffer));
        }
        return ggml_backend_awnpu_weight_on_npu(node);
    }

    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            const struct ggml_tensor * src = node->src[i];
            if (src != nullptr && ggml_backend_awnpu_weight_on_npu(src)) {
                return true;
            }
        }
        return false;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const struct ggml_tensor * src = node->src[i];
        if (src == nullptr) {
            continue;
        }

        if (ggml_backend_awnpu_node_runs_on_npu(src, depth + 1)) {
            return true;
        }
    }

    return false;
}

void ggml_backend_awnpu_build_layer_map(const struct ggml_cgraph * cgraph) {
    if (cgraph == nullptr) {
        return;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            continue;
        }

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const struct ggml_tensor * src = node->src[j];
            if (src == nullptr || src->buffer == nullptr) {
                continue;
            }
            if (src->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                continue;
            }

            ggml_backend_awnpu_record_layer_placement(src, ggml_backend_awnpu_weight_on_npu(src));
        }
    }
}

const char * ggml_backend_awnpu_buffer_logical_backend_name(
        ggml_backend_t awnpu_backend,
        ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return "unallocated";
    }

    return ggml_backend_awnpu_buffer_name_is_npu(ggml_backend_buffer_name(buffer)) ?
        ggml_backend_name(awnpu_backend) : "CPU";
}

const char * ggml_backend_awnpu_tensor_logical_backend_name(
        ggml_backend_t awnpu_backend,
        const struct ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return "none";
    }

    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer == nullptr && tensor->view_src != nullptr) {
        buffer = tensor->view_src->buffer;
    }

    if (buffer == nullptr) {
        return "unallocated";
    }

    if (name_has_prefix(tensor->name, GGML_AWNPU_TN_CACHE, GGML_AWNPU_TN_CACHE_LEN)) {
        return ggml_backend_awnpu_buffer_logical_backend_name(awnpu_backend, buffer);
    }

    if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return ggml_backend_awnpu_weight_on_npu(tensor) ?
            ggml_backend_name(awnpu_backend) : "CPU";
    }

    return ggml_backend_awnpu_node_runs_on_npu(tensor, 0) ?
        ggml_backend_name(awnpu_backend) : "CPU";
}
