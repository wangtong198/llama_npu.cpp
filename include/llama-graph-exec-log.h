#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llama_graph_exec_log_callbacks {
    void * user_data;

    void (*on_graph_begin)(
            void * user_data,
            ggml_backend_t dispatch,
            const struct ggml_cgraph * graph);
} llama_graph_exec_log_callbacks;

typedef const llama_graph_exec_log_callbacks * (*llama_graph_exec_log_get_callbacks_t)(void);

// LLAMA_GRAPH_EXEC_LOG=1: after sched_reserve, register hook (no file I/O yet).
GGML_API void llama_graph_exec_log_prepare(ggml_backend_reg_t * regs, size_t n_regs);

// Suppress node-done logging for nested fallback compute regions.
GGML_API void llama_graph_exec_log_suspend_node_done(void);
GGML_API void llama_graph_exec_log_resume_node_done(void);

// Mark the current node as running through a CPU fallback path.
GGML_API void llama_graph_exec_log_set_current_node_fallback(bool fallback);

#ifdef __cplusplus
}
#endif
