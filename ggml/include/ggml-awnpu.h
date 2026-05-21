#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_AWNPU_NAME "AWNPU"

GGML_API ggml_backend_reg_t ggml_backend_awnpu_reg(void);

// Set from llama_context via ggml_backend_reg_get_proc_address("ggml_backend_set_model_n_layer").
// n_layer is hparams.n_layer (LLM_KV_BLOCK_COUNT in GGUF).
GGML_API void ggml_backend_awnpu_set_model_n_layer(int n_layer);

#ifdef __cplusplus
}
#endif
