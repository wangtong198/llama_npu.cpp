#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llama_graph_exec_log_callbacks {
    void * user_data;

    const char * (*resolve_node_exec)(
            void * user_data,
            ggml_backend_t dispatch,
            const struct ggml_tensor * node);

    const char * (*resolve_tensor)(
            void * user_data,
            ggml_backend_t dispatch,
            const struct ggml_tensor * tensor);

    void (*on_graph_begin)(
            void * user_data,
            ggml_backend_t dispatch,
            const struct ggml_cgraph * graph);

    bool (*is_layout_only)(void * user_data, const struct ggml_tensor * node);
} llama_graph_exec_log_callbacks;

typedef const llama_graph_exec_log_callbacks * (*llama_graph_exec_log_get_callbacks_t)(void);

// LLAMA_GRAPH_EXEC_LOG=1: after sched_reserve, register hook (no file I/O yet).
GGML_API void llama_graph_exec_log_prepare(ggml_backend_reg_t * regs, size_t n_regs);

#ifdef __cplusplus
}
#endif
