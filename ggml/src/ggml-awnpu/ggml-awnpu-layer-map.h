#pragma once

#include "ggml-awnpu.h"
#include "ggml.h"

bool ggml_backend_awnpu_buffer_name_is_npu(const char * buft_name);

int  ggml_backend_awnpu_parse_layer_index(const char * name);

void ggml_backend_awnpu_record_layer_placement(const struct ggml_tensor * tensor, bool on_npu);

void ggml_backend_awnpu_set_output_on_npu(bool on_npu);

void ggml_backend_awnpu_set_model_n_layer(int n_layer);

bool ggml_backend_awnpu_layer_on_npu(int il);

bool ggml_backend_awnpu_weight_on_npu(const struct ggml_tensor * tensor);

int  ggml_backend_awnpu_infer_layer_index(const struct ggml_tensor * op);

bool ggml_backend_awnpu_is_output_weight_name(const char * name);

bool ggml_backend_awnpu_is_input_embedding_weight(const struct ggml_tensor * tensor);

bool ggml_backend_awnpu_is_output_head_node(const struct ggml_tensor * op);

bool ggml_backend_awnpu_lm_head_on_npu(void);

int  ggml_backend_awnpu_max_npu_layer(void);

bool ggml_backend_awnpu_node_is_layout_only(const struct ggml_tensor * node);

bool ggml_backend_awnpu_node_runs_on_npu(const struct ggml_tensor * node, int depth);

void ggml_backend_awnpu_build_layer_map(const struct ggml_cgraph * cgraph);
