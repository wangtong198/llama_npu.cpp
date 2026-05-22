#include "ggml-awnpu-exec-log-resolver.h"

#include "ggml-awnpu-layer-map.h"

namespace {

static void on_graph_begin(
        void * user_data,
        ggml_backend_t dispatch,
        const struct ggml_cgraph * graph) {
    GGML_UNUSED(user_data);
    GGML_UNUSED(dispatch);
    ggml_backend_awnpu_build_layer_map(graph);
}

} // namespace

const llama_graph_exec_log_callbacks * ggml_backend_awnpu_graph_exec_log_get_callbacks(void) {
    static const llama_graph_exec_log_callbacks callbacks = {
        /* .user_data      = */ nullptr,
        /* .on_graph_begin = */ on_graph_begin,
    };

    return &callbacks;
}
