#pragma once

#include "ggml-backend.h"
#include "ggml.h"

struct ggml_backend_awnpu_op_dispatch {
    int (*get_last_node_idx)(const void * ctx);
    void (*set_last_node_idx)(void * ctx, int node_idx);
    enum ggml_status (*forward_range)(void * ctx, struct ggml_cgraph * cgraph, int i0, int i1);
};

bool ggml_backend_awnpu_op_supported(enum ggml_op op);

enum ggml_status ggml_backend_awnpu_compute_node_sim_op(
        void * ctx,
        const struct ggml_backend_awnpu_op_dispatch * dispatch,
        struct ggml_cgraph * cgraph,
        int node_idx,
        struct ggml_tensor * node);

enum ggml_status ggml_backend_awnpu_compute_node_real_op(
        void * ctx,
        struct ggml_tensor * node);
