#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// LLAMA_GRAPH_EXEC_LOG=1: after sched_reserve, register hook (no file I/O yet).
GGML_API void llama_graph_exec_log_prepare(void);

// Suppress node-done logging for nested fallback compute regions.
GGML_API void llama_graph_exec_log_suspend_node_done(void);
GGML_API void llama_graph_exec_log_resume_node_done(void);

// Mark the current node as running through a CPU fallback path.
GGML_API void llama_graph_exec_log_set_current_node_fallback(bool fallback);

#ifdef __cplusplus
}
#endif
