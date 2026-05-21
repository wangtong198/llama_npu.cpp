#pragma once

typedef struct ggml_backend_awnpu_graph_split {
    int prefix_end;
    int suffix_start;
} ggml_backend_awnpu_graph_split;

struct llama_graph_exec_log_callbacks;

const struct llama_graph_exec_log_callbacks * ggml_backend_awnpu_graph_exec_log_get_callbacks(void);

ggml_backend_awnpu_graph_split ggml_backend_awnpu_detect_graph_split(const struct ggml_cgraph * cgraph);
