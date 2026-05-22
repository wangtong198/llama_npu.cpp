#pragma once

#include "llama-graph-exec-log.h"

// AWNPU-specific llama_graph_exec_log resolver.
// Exported to llama-graph-exec-log via ggml_backend_reg_get_proc_address("llama_graph_exec_log_get_callbacks").
const llama_graph_exec_log_callbacks * ggml_backend_awnpu_graph_exec_log_get_callbacks(void);
