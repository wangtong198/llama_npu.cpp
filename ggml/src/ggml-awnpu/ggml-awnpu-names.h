#pragma once

#include "ggml-awnpu.h"

#include <stddef.h>

// Tensor / graph node name conventions from llama.cpp (not AWNPU-specific).
// We match these strings when inferring transformer layer index or graph segment.
//
// Weight / block tensors: src/llama-arch.cpp LLM_TENSOR_NAMES ("blk.%d.*")
// KV cache views:         src/llama-kv-cache.cpp ggml_format_name(..., "cache_k_l%d", il)
// Graph inputs:           src/llama-graph.cpp cb(..., "inp_*", ...)
// LM head (qwen2 etc.):   src/models/qwen2.cpp "norm", "result_norm", "result_output"
// Output bias:            src/llama-model.cpp tn(LLM_TENSOR_OUTPUT, "bias") -> "output_b"

#define GGML_AWNPU_TN_BLK              "blk."
#define GGML_AWNPU_TN_BLK_LEN          (sizeof(GGML_AWNPU_TN_BLK) - 1)

#define GGML_AWNPU_TN_CACHE_K          "cache_k_l"
#define GGML_AWNPU_TN_CACHE_K_LEN      (sizeof(GGML_AWNPU_TN_CACHE_K) - 1)

#define GGML_AWNPU_TN_CACHE_V          "cache_v_l"
#define GGML_AWNPU_TN_CACHE_V_LEN      (sizeof(GGML_AWNPU_TN_CACHE_V) - 1)

// Prefix for all KV / recurrent cache tensors; see llama-model.cpp pattern_kv_cache.
#define GGML_AWNPU_TN_CACHE            "cache_"
#define GGML_AWNPU_TN_CACHE_LEN        (sizeof(GGML_AWNPU_TN_CACHE) - 1)

// Graph input nodes (token ids, masks, pos, ...).
#define GGML_AWNPU_TN_INP              "inp_"
#define GGML_AWNPU_TN_INP_LEN          (sizeof(GGML_AWNPU_TN_INP) - 1)

#define GGML_AWNPU_TN_EMBD             "embd"

#define GGML_AWNPU_TN_OUTPUT_NORM_W    "output_norm.weight"
#define GGML_AWNPU_TN_OUTPUT_W         "output.weight"
#define GGML_AWNPU_TN_OUTPUT_B         "output_b"
#define GGML_AWNPU_TN_OUTPUT_B_PREFIX  "output_b."
#define GGML_AWNPU_TN_OUTPUT_B_PREFIX_LEN (sizeof(GGML_AWNPU_TN_OUTPUT_B_PREFIX) - 1)

#define GGML_AWNPU_TN_RESULT_NORM      "result_norm"
#define GGML_AWNPU_TN_RESULT_OUTPUT    "result_output"
#define GGML_AWNPU_TN_NORM             "norm"

// AWNPU backend device name prefix; buft names are "AWNPU0", "AWNPU1", ...
#define GGML_AWNPU_BACKEND_NAME_LEN    (sizeof(GGML_AWNPU_NAME) - 1)
