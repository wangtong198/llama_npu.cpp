#include "ggml-impl.h"

#include "ggml-awnpu-exec-log-resolver.h"

#include "ggml-awnpu-layer-map.h"
#include "ggml-awnpu-names.h"
#include "ggml-awnpu.h"
#include "llama-graph-exec-log.h"

#include <algorithm>
#include <cstring>

namespace {

static bool name_has_prefix(const char * name, const char * prefix, size_t prefix_len) {
    return name != nullptr && std::strncmp(name, prefix, prefix_len) == 0;
}

static bool tensor_is_sampler_node(const struct ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->name[0] == '\0') {
        return false;
    }

    static const char * prefixes[] = {
        "greedy_",
        "dist_",
        "top_k",
        "top_p_",
        "min_p_",
        "temp_ext_",
        "temp_",
        "logit_bias",
        "logits_seq_",
    };

    return std::any_of(std::begin(prefixes), std::end(prefixes), [tensor](const char * prefix) {
        return name_has_prefix(tensor->name, prefix, std::strlen(prefix));
    });
}

static bool tensor_is_prefix_node(const struct ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->name[0] == '\0') {
        return false;
    }

    const char * name = tensor->name;
    if (std::strcmp(name, GGML_AWNPU_TN_EMBD) == 0) {
        return true;
    }

    return name_has_prefix(name, GGML_AWNPU_TN_INP, GGML_AWNPU_TN_INP_LEN);
}

static bool is_awnpu_backend(ggml_backend_t backend) {
    const char * name = ggml_backend_name(backend);
    return name != nullptr &&
        name_has_prefix(name, GGML_AWNPU_NAME, GGML_AWNPU_BACKEND_NAME_LEN);
}

static bool is_layout_only_cb(void * user_data, const struct ggml_tensor * node) {
    GGML_UNUSED(user_data);
    return ggml_backend_awnpu_node_is_layout_only(node);
}

static const char * resolve_node_exec(
        void * user_data,
        ggml_backend_t dispatch,
        const struct ggml_tensor * node) {
    GGML_UNUSED(user_data);

    if (is_awnpu_backend(dispatch)) {
        return ggml_backend_awnpu_node_runs_on_npu(node, 0) ?
            ggml_backend_name(dispatch) : "CPU";
    }

    return ggml_backend_name(dispatch);
}

static const char * resolve_tensor(
        void * user_data,
        ggml_backend_t dispatch,
        const struct ggml_tensor * tensor) {
    GGML_UNUSED(user_data);
    return ggml_backend_awnpu_tensor_logical_backend_name(dispatch, tensor);
}

static void on_graph_begin(
        void * user_data,
        ggml_backend_t dispatch,
        const struct ggml_cgraph * graph) {
    GGML_UNUSED(user_data);
    GGML_UNUSED(dispatch);
    ggml_backend_awnpu_build_layer_map(graph);
}

} // namespace

ggml_backend_awnpu_graph_split ggml_backend_awnpu_detect_graph_split(const struct ggml_cgraph * cgraph) {
    ggml_backend_awnpu_graph_split split {};
    split.prefix_end   = 0;
    split.suffix_start = cgraph != nullptr ? cgraph->n_nodes : 0;

    if (cgraph == nullptr || cgraph->n_nodes <= 0) {
        return split;
    }

    // AWNPU graph_compute runtime only: prefix (inp/embd) -> middle -> suffix (sampler).
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            continue;
        }

        if (tensor_is_sampler_node(node)) {
            split.suffix_start = i;
            break;
        }

        if (split.prefix_end == i && tensor_is_prefix_node(node)) {
            split.prefix_end = i + 1;
            continue;
        }

        break;
    }

    if (split.suffix_start < split.prefix_end) {
        split.suffix_start = split.prefix_end;
    }

    return split;
}

const llama_graph_exec_log_callbacks * ggml_backend_awnpu_graph_exec_log_get_callbacks(void) {
    static const llama_graph_exec_log_callbacks callbacks = {
        /* .user_data         = */ nullptr,
        /* .resolve_node_exec = */ resolve_node_exec,
        /* .resolve_tensor    = */ resolve_tensor,
        /* .on_graph_begin    = */ on_graph_begin,
        /* .is_layout_only    = */ is_layout_only_cb,
    };

    return &callbacks;
}
