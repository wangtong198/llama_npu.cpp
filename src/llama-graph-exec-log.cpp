#include "llama-graph-exec-log.h"

#include "ggml-impl.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <inttypes.h>
#include <mutex>

namespace {

struct llama_graph_exec_log_session {
    std::mutex  mutex;
    std::FILE * file         = nullptr;
    bool        enabled      = false;
    bool        want_log     = false;
    uint64_t    node_counter = 0;
};

llama_graph_exec_log_session g_session;
thread_local int  g_node_done_suppression = 0;
thread_local bool g_current_node_fallback = false;

static bool env_enabled(void) {
    static std::atomic<int> cached {-1};

    int value = cached.load(std::memory_order_acquire);
    if (value == -1) {
        const char * env = std::getenv("LLAMA_GRAPH_EXEC_LOG");
        if (env != nullptr && *env != '\0' &&
            (std::strcmp(env, "1") == 0 ||
             std::strcmp(env, "true") == 0 ||
             std::strcmp(env, "TRUE") == 0)) {
            value = 1;
        } else {
            value = 0;
        }
        cached.store(value, std::memory_order_release);
    }

    return value == 1;
}

static void write_node_log(const struct ggml_tensor * node, double elapsed_ms, bool fallback) {
    if (node == nullptr) {
        return;
    }

    const uint64_t seq = ++g_session.node_counter;

    std::fprintf(
            g_session.file,
            "seq=%" PRIu64 " op=%s name=%s type=%s shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] contiguous: %s fallback: %s time-elapsed: %.3fms\n",
            seq,
            ggml_op_name(node->op),
            node->name[0] != '\0' ? node->name : "<unnamed>",
            ggml_type_name(node->type),
            (int64_t) node->ne[0],
            (int64_t) node->ne[1],
            (int64_t) node->ne[2],
            (int64_t) node->ne[3],
            ggml_is_contiguous(node) ? "true" : "false",
            fallback ? "true" : "false",
            elapsed_ms);

    for (int j = 0; j < GGML_MAX_SRC; j++) {
        const struct ggml_tensor * src_node = node->src[j];
        if (src_node == nullptr) {
            continue;
        }

        std::fprintf(
                g_session.file,
                "    src[%d] op=%s name=%s type=%s shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] contiguous: %s\n",
                j,
                ggml_op_name(src_node->op),
                src_node->name[0] != '\0' ? src_node->name : "<unnamed>",
                ggml_type_name(src_node->type),
                (int64_t) src_node->ne[0],
                (int64_t) src_node->ne[1],
                (int64_t) src_node->ne[2],
                (int64_t) src_node->ne[3],
                ggml_is_contiguous(src_node) ? "true" : "false");
    }

    std::fflush(g_session.file);
}

static void write_split_header(
        ggml_backend_t dispatch,
        const struct ggml_cgraph * graph) {
    if (!g_session.enabled || g_session.file == nullptr || dispatch == nullptr || graph == nullptr) {
        return;
    }

    ggml_backend_graph_exec_info exec_info {};
    ggml_backend_get_graph_exec_info(&exec_info);

    std::lock_guard<std::mutex> lock(g_session.mutex);
    std::fprintf(
            g_session.file,
            "==================== sched %" PRIu64 " split %d/%d dispatch=%s n_nodes=%d ====================\n",
            exec_info.sched_round,
            exec_info.split_index,
            exec_info.n_splits,
            ggml_backend_name(dispatch),
            graph->n_nodes);
    std::fflush(g_session.file);
}

static bool ensure_enabled(void) {
    static std::once_flag enable_once;

    std::call_once(enable_once, []() {
        if (!g_session.want_log) {
            return;
        }

        g_session.file = std::fopen("llama_graph_exec.log", "w+");
        if (g_session.file == nullptr) {
            GGML_LOG_WARN("%s: failed to open exec log file llama_graph_exec.log\n", __func__);
            return;
        }

        g_session.enabled = true;
        GGML_LOG_INFO("LLAMA_GRAPH_EXEC_LOG enabled, writing to llama_graph_exec.log\n");
        std::fprintf(g_session.file, "=== llama graph exec log start ===\n");
        std::fprintf(g_session.file, "=== header: sched split dispatch n_nodes | node: exec= ===\n");
        std::fflush(g_session.file);
    });

    return g_session.enabled && g_session.file != nullptr;
}

static void exec_log_before_split_compute(
        ggml_backend_t backend,
        struct ggml_cgraph * cgraph,
        void * user_data) {
    GGML_UNUSED(user_data);

    if (!ensure_enabled() || backend == nullptr || cgraph == nullptr) {
        return;
    }

    write_split_header(backend, cgraph);
}

static void exec_log_node_done(
        ggml_backend_t backend,
        const struct ggml_cgraph * cgraph,
        int node_idx,
        int64_t elapsed_us,
        void * user_data) {
    GGML_UNUSED(user_data);
    GGML_UNUSED(backend);

    if (!g_session.enabled || g_session.file == nullptr || cgraph == nullptr) {
        return;
    }

    if (g_node_done_suppression > 0) {
        return;
    }

    if (node_idx < 0 || node_idx >= cgraph->n_nodes) {
        return;
    }

    const struct ggml_tensor * node = cgraph->nodes[node_idx];
    if (node == nullptr) {
        return;
    }

    const double elapsed_ms = elapsed_us / 1000.0;
    const bool fallback = g_current_node_fallback;

    std::lock_guard<std::mutex> lock(g_session.mutex);
    write_node_log(node, elapsed_ms, fallback);
}

} // namespace

void llama_graph_exec_log_prepare(void) {
    if (!env_enabled()) {
        return;
    }

    g_session.want_log = true;

    static std::once_flag hook_once;
    std::call_once(hook_once, []() {
        ggml_backend_set_graph_split_begin_callback(exec_log_before_split_compute, nullptr);
        ggml_backend_set_graph_node_done_callback(exec_log_node_done, nullptr);
    });
}

void llama_graph_exec_log_suspend_node_done(void) {
    ++g_node_done_suppression;
}

void llama_graph_exec_log_resume_node_done(void) {
    if (g_node_done_suppression > 0) {
        --g_node_done_suppression;
    }
}

void llama_graph_exec_log_set_current_node_fallback(bool fallback) {
    g_current_node_fallback = fallback;
}
