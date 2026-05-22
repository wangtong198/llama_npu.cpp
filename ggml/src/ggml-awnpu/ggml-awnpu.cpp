#include "ggml-awnpu.h"

#include "ggml-awnpu-layer-map.h"
#include "ggml-awnpu-exec-log-resolver.h"
#include "ggml-awnpu-ops.h"
#include "llama-graph-exec-log.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

struct ggml_backend_awnpu_buffer_context {
    void * data = nullptr;
    size_t size = 0;
    bool owned = false;
};

struct ggml_backend_awnpu_buffer_type_context {
    int device = 0;
    std::string name;
    size_t alignment = GGML_MEM_ALIGN;
    size_t max_size = 0;
    bool is_host = true;
};

struct ggml_backend_awnpu_device_context {
    int device = 0;
    std::string name;
    std::string description;
    std::string device_id;
    size_t memory_total = 0;
    size_t memory_free = 0;
    ggml_backend_buffer_type_t buft = nullptr;
};

// 这些字段不是“AWNPU 硬件必须有的状态”，而是 backend 执行图时的宿主机上下文。
struct ggml_backend_awnpu_context {
    // CPU fallback 计算时，控制 CPU 算子并行线程数
    int n_threads = 1;  
    // CPU fallback 计算时，传给 cplan.use_ref，是否用 reference 实现（ref=1 参数）
    bool use_ref = false;
    // CPU fallback 计算时，CPU graph compute 的工作缓冲区，按 cplan.work_size 按需分配/复用
    void * work_data = nullptr;
    size_t work_size = 0;
    
    int device_id = 0;
    // 图执行中的中断回调
    // 用于外部请求停止推理时，执行过程能及时退出
    // 对 NPU 来说，如果未来接真实 runtime，也可以映射成设备执行过程中的取消/中止检查。
    ggml_abort_callback abort_callback = nullptr;
    void * abort_callback_data = nullptr; // 传给 abort_callback 的用户数据指针。让回调能访问上下文状态。
};

struct ggml_backend_awnpu_reg_context {
    std::vector<std::unique_ptr<ggml_backend_buffer_type>> bufts;
    std::vector<std::unique_ptr<ggml_backend_awnpu_buffer_type_context>> buft_contexts;
    std::vector<std::unique_ptr<ggml_backend_device>> devices;
    std::vector<std::unique_ptr<ggml_backend_awnpu_device_context>> device_contexts;
};

struct ggml_backend_awnpu_device_probe_info {
    size_t memory_total = 0;
    size_t memory_free = 0;
    size_t buffer_alignment = GGML_MEM_ALIGN;
    size_t buffer_max_size = 0;
    bool shared_host_buffer = true;
};

struct ggml_backend_awnpu_probe_info {
    int n_devices = 0;
    std::vector<ggml_backend_awnpu_device_probe_info> devices;
};

// Runtime-facing probe ABI.
// The runtime can omit shared_host_buffer by setting has_shared_host_buffer = false,
// in which case the backend defaults it to true.
struct awnpu_probe_device_info {
    size_t memory_total = 0;
    size_t memory_free = 0;
    size_t buffer_alignment = GGML_MEM_ALIGN;
    size_t buffer_max_size = 0;
    bool has_shared_host_buffer = false;
    bool shared_host_buffer = true;
};

struct awnpu_probe_result {
    int n_devices = 0;
    const awnpu_probe_device_info * devices = nullptr;
};

#if defined(__GNUC__) || defined(__clang__)
#define GGML_AWNPU_WEAK __attribute__((weak))
#else
#define GGML_AWNPU_WEAK
#endif

extern "C" bool awnpu_probe(awnpu_probe_result * result) GGML_AWNPU_WEAK;

static bool ggml_backend_awnpu_parse_bool_env(const char * name, bool def = false) {
    const char * env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return def;
    }

    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0 ||
        std::strcmp(env, "on") == 0 || std::strcmp(env, "ON") == 0 || std::strcmp(env, "yes") == 0 ||
        std::strcmp(env, "YES") == 0) {
        return true;
    }

    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0 ||
        std::strcmp(env, "off") == 0 || std::strcmp(env, "OFF") == 0 || std::strcmp(env, "no") == 0 ||
        std::strcmp(env, "NO") == 0) {
        return false;
    }

    return def;
}

static bool ggml_backend_awnpu_sim_probe_enabled(void) {
    static std::atomic<int> cached {-1};

    int value = cached.load(std::memory_order_acquire);
    if (value == -1) {
        value = ggml_backend_awnpu_parse_bool_env("SIM_PROBE", false) ? 1 : 0;
        cached.store(value, std::memory_order_release);
    }

    return value == 1;
}

static int ggml_backend_awnpu_parse_int_env(const char * name, int def) {
    const char * env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return def;
    }

    char * end = nullptr;
    long value = std::strtol(env, &end, 10);
    if (end == env) {
        return def;
    }

    return (int) value;
}

static std::string ggml_backend_awnpu_fallback_warning_key(const struct ggml_tensor * node) {
    GGML_ASSERT(node != nullptr);

    std::string key = ggml_op_name(node->op);
    key += '|';
    key += ggml_type_name(node->type);
    key += '|';
    key += std::to_string((int64_t) node->ne[0]);
    key += 'x';
    key += std::to_string((int64_t) node->ne[1]);
    key += 'x';
    key += std::to_string((int64_t) node->ne[2]);
    key += 'x';
    key += std::to_string((int64_t) node->ne[3]);
    return key;
}

static bool ggml_backend_awnpu_warn_fallback_once(
        const char * func_name,
        const struct ggml_tensor * node,
        bool op_supported) {
    GGML_ASSERT(node != nullptr);
    GGML_ASSERT(func_name != nullptr);

    static std::mutex mutex;
    static std::unordered_set<std::string> seen;

    const std::string key = ggml_backend_awnpu_fallback_warning_key(node);

    std::lock_guard<std::mutex> lock(mutex);
    const bool inserted = seen.insert(key).second;
    if (inserted) {
        if (!op_supported) {
            GGML_LOG_WARN("%s: AWNPU op %s type=%s shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] is not supported, fallback to CPU\n",
                    func_name, ggml_op_name(node->op), ggml_type_name(node->type),
                    (int64_t) node->ne[0], (int64_t) node->ne[1], (int64_t) node->ne[2], (int64_t) node->ne[3]);
        } else {
            GGML_LOG_WARN("%s: NPU execution is not implemented for op %s type=%s shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "], fallback to CPU\n",
                    func_name, ggml_op_name(node->op), ggml_type_name(node->type),
                    (int64_t) node->ne[0], (int64_t) node->ne[1], (int64_t) node->ne[2], (int64_t) node->ne[3]);
        }
    }

    return inserted;
}

extern const struct ggml_backend_buffer_i ggml_backend_awnpu_buffer_i;
extern const struct ggml_backend_i ggml_backend_awnpu_i;
static enum ggml_status ggml_backend_awnpu_cpu_forward_range(
        ggml_backend_awnpu_context * ctx,
        struct ggml_cgraph * cgraph,
        int i0,
        int i1);
static bool ggml_backend_awnpu_op_has_npu_weights(const struct ggml_tensor * op);

static void ggml_backend_awnpu_get_host_memory(size_t * memory_free, size_t * memory_total) {
    size_t total = 8ull * 1024 * 1024 * 1024;
    size_t free = 6ull * 1024 * 1024 * 1024;

#if !defined(_WIN32) && defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    const long page_size = sysconf(_SC_PAGE_SIZE);
    const long total_pages = sysconf(_SC_PHYS_PAGES);
    if (page_size > 0 && total_pages > 0) {
        total = (size_t) page_size * (size_t) total_pages;
    }
#if defined(_SC_AVPHYS_PAGES)
    const long free_pages = sysconf(_SC_AVPHYS_PAGES);
    if (page_size > 0 && free_pages > 0) {
        free = (size_t) page_size * (size_t) free_pages;
    } else {
        free = total;
    }
#else
    free = total;
#endif
#else
    free = total;
#endif

    if (memory_total != nullptr) {
        *memory_total = total;
    }
    if (memory_free != nullptr) {
        *memory_free = std::min(free, total);
    }
}

static bool ggml_backend_awnpu_probe_simulated(ggml_backend_awnpu_probe_info * info) {
    GGML_ASSERT(info != nullptr);

    size_t memory_total = 0;
    size_t memory_free = 0;
    ggml_backend_awnpu_get_host_memory(&memory_free, &memory_total);

    info->n_devices = 1;
    info->devices.clear();
    info->devices.push_back({
        /* .memory_total       = */ memory_total,
        /* .memory_free        = */ memory_free,
        /* .buffer_alignment   = */ GGML_MEM_ALIGN,
        /* .buffer_max_size    = */ memory_free > 0 ? memory_free : memory_total,
        /* .shared_host_buffer = */ true,
    });

    return true;
}

static bool ggml_backend_awnpu_probe(ggml_backend_awnpu_probe_info * info) {
    GGML_ASSERT(info != nullptr);

    if (ggml_backend_awnpu_sim_probe_enabled()) {
        return ggml_backend_awnpu_probe_simulated(info);
    }

    if (awnpu_probe == nullptr) {
        GGML_LOG_ERROR("%s: awnpu_probe runtime API is unavailable\n", __func__);
        return false;
    }

    awnpu_probe_result runtime_probe {};
    if (!awnpu_probe(&runtime_probe)) {
        GGML_LOG_ERROR("%s: awnpu_probe runtime API failed\n", __func__);
        return false;
    }

    if (runtime_probe.n_devices <= 0 || runtime_probe.devices == nullptr) {
        return false;
    }

    info->n_devices = runtime_probe.n_devices;
    info->devices.clear();
    info->devices.reserve((size_t) runtime_probe.n_devices);

    for (int i = 0; i < runtime_probe.n_devices; ++i) {
        const awnpu_probe_device_info & dev = runtime_probe.devices[i];

        ggml_backend_awnpu_device_probe_info dev_info {};
        dev_info.memory_total = dev.memory_total;
        dev_info.memory_free = dev.memory_free;
        dev_info.buffer_alignment = std::max<size_t>(GGML_MEM_ALIGN, dev.buffer_alignment);
        dev_info.buffer_max_size = dev.buffer_max_size > 0 ? dev.buffer_max_size : dev.memory_total;
        dev_info.shared_host_buffer = dev.has_shared_host_buffer ? dev.shared_host_buffer : true;

        info->devices.push_back(dev_info);
    }

    return true;
}

static size_t ggml_backend_awnpu_normalize_alignment(size_t alignment) {
    const size_t min_alignment = sizeof(void *);
    if (alignment < min_alignment) {
        alignment = min_alignment;
    }

    // posix_memalign/_aligned_malloc require a power-of-two alignment.
    if ((alignment & (alignment - 1)) != 0) {
        size_t normalized = min_alignment;
        while (normalized < alignment) {
            normalized <<= 1;
        }
        alignment = normalized;
    }

    return alignment;
}

static void * ggml_backend_awnpu_host_aligned_malloc(size_t size, size_t alignment) {
    alignment = ggml_backend_awnpu_normalize_alignment(alignment);

#if defined(_MSC_VER) || defined(__MINGW32__)
    return _aligned_malloc(size, alignment);
#else
    if (size == 0) {
        GGML_LOG_WARN("%s: allocating 0 bytes\n", __func__);
        return nullptr;
    }

    void * aligned_memory = nullptr;
    const int result = posix_memalign(&aligned_memory, alignment, size);
    if (result != 0) {
        const char * error_desc = "unknown allocation error";
        switch (result) {
            case EINVAL:
                error_desc = "invalid alignment value";
                break;
            case ENOMEM:
                error_desc = "insufficient memory";
                break;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB, alignment = %zu)\n",
                __func__, error_desc, size/(1024.0*1024.0), alignment);
        return nullptr;
    }

    return aligned_memory;
#endif
}

static void ggml_backend_awnpu_host_aligned_free(void * ptr) {
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static ggml_backend_awnpu_buffer_context * ggml_backend_awnpu_buffer_context_init(void * data, size_t size, bool owned) {
    return new ggml_backend_awnpu_buffer_context {
        /* .data  = */ data,
        /* .size  = */ size,
        /* .owned = */ owned,
    };
}

static void ggml_backend_awnpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_awnpu_buffer_context *) buffer->context;
    if (ctx->owned) {
        ggml_backend_awnpu_host_aligned_free(ctx->data);
    }

    delete ctx;
}

static void * ggml_backend_awnpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_awnpu_buffer_context *) buffer->context;
    return ctx ? ctx->data : nullptr;
}


// 指针和布局主要由 ggml_backend_tensor_alloc / ggml_backend_view_init 及上游分配器在 CPU 上完成，
// ggml_backend_buffer_init_tensor 只是随后调用可选的 iface.init_tensor
static enum ggml_status ggml_backend_awnpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    if (buffer != nullptr && tensor != nullptr &&
        ggml_backend_awnpu_buffer_name_is_npu(ggml_backend_buffer_name(buffer))) {
        ggml_backend_awnpu_record_layer_placement(tensor, true);
        if (ggml_backend_awnpu_is_output_weight_name(tensor->name)) {
            ggml_backend_awnpu_set_output_on_npu(true);
        }
    }
    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_awnpu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                    struct ggml_tensor * tensor,
                                                    uint8_t value,
                                                    size_t offset,
                                                    size_t size) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->data != nullptr);
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    std::memset((uint8_t *) tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_awnpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                 struct ggml_tensor * tensor,
                                                 const void * data,
                                                 size_t offset,
                                                 size_t size) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->data != nullptr);
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    std::memcpy((uint8_t *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_awnpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                 const struct ggml_tensor * tensor,
                                                 void * data,
                                                 size_t offset,
                                                 size_t size) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->data != nullptr);
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
    std::memcpy(data, (const uint8_t *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static void ggml_backend_awnpu_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                     struct ggml_tensor * tensor,
                                                     const void * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    const uint8_t * src = (const uint8_t *) data;
    for (size_t i = 0; i < n_copies; ++i) {
        ggml_backend_awnpu_buffer_set_tensor(buffer, tensor, src + i * stride_data, offset + i * stride_tensor, size);
    }
}

static void ggml_backend_awnpu_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                     const struct ggml_tensor * tensor,
                                                     void * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    uint8_t * dst = (uint8_t *) data;
    for (size_t i = 0; i < n_copies; ++i) {
        ggml_backend_awnpu_buffer_get_tensor(buffer, tensor, dst + i * stride_data, offset + i * stride_tensor, size);
    }
}

static bool ggml_backend_awnpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                 const struct ggml_tensor * src,
                                                 struct ggml_tensor * dst) {
    GGML_ASSERT(src != nullptr && dst != nullptr);
    GGML_ASSERT(src->buffer != nullptr && dst->buffer != nullptr);
    const size_t size = ggml_nbytes(src);
    if (ggml_nbytes(dst) != size) {
        return false;
    }

    std::memcpy(dst->data, src->data, size);
    GGML_UNUSED(buffer);
    return true;
}

static void ggml_backend_awnpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_awnpu_buffer_context *) buffer->context;
    GGML_ASSERT(ctx != nullptr && ctx->data != nullptr);
    std::memset(ctx->data, value, ctx->size);
}

static void ggml_backend_awnpu_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
}

const struct ggml_backend_buffer_i ggml_backend_awnpu_buffer_i = {
    /* .free_buffer   = */ ggml_backend_awnpu_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_awnpu_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_awnpu_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_awnpu_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_awnpu_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_awnpu_buffer_get_tensor,
    /* .set_tensor_2d = */ ggml_backend_awnpu_buffer_set_tensor_2d,
    /* .get_tensor_2d = */ ggml_backend_awnpu_buffer_get_tensor_2d,
    /* .cpy_tensor    = */ ggml_backend_awnpu_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_awnpu_buffer_clear,
    /* .reset         = */ ggml_backend_awnpu_buffer_reset,
};

// 返回这个 buffer_type 要求的内存对齐。
// 这是为了让 ggml 在布局 tensor 时知道：
//  - tensor 起始地址要对齐到多少字节
// - tensor 之间的偏移要满足什么约束

// 对 NPU backend 来说很关键，因为：
//  - DMA / cache line / NPU kernel 常常有对齐要求
//  - 某些算子要求 16 / 32 / 64 / 4096 字节对齐
// TODO： 需要按照NPU真正的对齐大小
static size_t ggml_backend_awnpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_awnpu_buffer_type_context *) buft->context;
    return ctx ? ggml_backend_awnpu_normalize_alignment(ctx->alignment) : GGML_MEM_ALIGN;
}

// 返回这个 buffer_type 单次能支持的最大 buffer 大小
// 如果这个 backend 不需要限制，该函数被设置为 nullptr，表示没有明确上限，或者上层不走这个约束检查。
// 有些 backend 需要这个值，比如：
//  某些显存池有单块最大分配限制
//  某些硬件或固件限制 buffer 不能超过某个大小
// TODO: 修改为NPU真正的最大分配大小
static size_t ggml_backend_awnpu_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_awnpu_buffer_type_context *) buft->context;
    return ctx ? ctx->max_size : 0;
}

// 返回“实际应该申请多大”的大小。
// 它不是简单把请求 size 原样返回，而是可以做：
//  - 向上取整
//  - 加上头部/元数据开销
//  - 按页对齐
//  - 按硬件要求补齐

// 这个函数的意义是：
//  ggml 先告诉你“我需要多大”，buffer_type 再告诉你“实际去申请多少才够”。
// TODO： GGML 默认按照16字节对齐空间。 需要考虑NPU对齐大小，按照实际的NPU对齐大小来计算。
static size_t ggml_backend_awnpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    const size_t alignment = ggml_backend_awnpu_buffer_type_get_alignment(buft);
    return GGML_PAD(ggml_nbytes(tensor), alignment);
}

// 返回这个 buffer 是否属于 host memory
// 这个标志很重要，因为 ggml 会据此决定：
// - tensor 数据是否可直接 CPU 访问
// - 是否需要 memcpy
// - 是否需要额外同步
// - 是否可以直接从 CPU 侧读写
// 常见语义是：
//  - true：普通主存，CPU 可直接访问
//  - false：设备内存、显存、NPU 私有内存等
//  - NPU 后端通常会返回 false，除非它实际上就是共享内存并且希望被当成 host buffer 处理。
static bool ggml_backend_awnpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_awnpu_buffer_type_context *) buft->context;
    // The current AWNPU backend uses CPU-allocated shared DDR as its backing store,
    // so tensors in this buft remain directly accessible from the host.
    return ctx ? ctx->is_host : true;
}

// 返回这个 buffer_type 的名字，也就是一类 buffer 的标识字符串
// 用于：
//  - 日志打印
//  - 调试
//  - 诊断当前 tensor 是被放进了哪种 buffer
//  - 有些地方会用它区分不同 backend / 不同内存类型
static const char * ggml_backend_awnpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_awnpu_buffer_type_context *) buft->context;
    return ctx ? ctx->name.c_str() : GGML_AWNPU_NAME;
}

// 根据这个 buffer_type 申请一个真正可用的 buffer 实例
// 它通常需要做的事情包括：

//  - 根据 size 申请内存
//  - 绑定 backend 自己的内存句柄
//  - 设置 buffer 的元信息
//  - 返回一个 ggml_backend_buffer_t

// 在 NPU 后端里，这里可能不是简单 malloc，而是：
//  - 申请 NPU memory
//  - 申请可 DMA 的共享内存
//  - 返回一个可供 tensor 挂载的 buffer 对象
static ggml_backend_buffer_t ggml_backend_awnpu_buffer_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_ctx = (ggml_backend_awnpu_buffer_type_context *) buft->context;
    GGML_ASSERT(buft_ctx != nullptr);

    const size_t alignment = ggml_backend_awnpu_buffer_type_get_alignment(buft);
    const size_t max_size = ggml_backend_awnpu_buffer_type_get_max_size(buft);

    if (max_size > 0 && size > max_size) {
        GGML_LOG_ERROR("%s: requested %zu bytes exceeds AWNPU max host allocation size %zu bytes\n",
                __func__, size, max_size);
        return nullptr;
    }

    // 这里由 CPU/host 为 AWNPU 申请一块共享 DDR 空间，并按 NPU 要求的对齐方式分配。
    auto * buffer_ctx = ggml_backend_awnpu_buffer_context_init(
            ggml_backend_awnpu_host_aligned_malloc(size, alignment), size, true);
    if (buffer_ctx->data == nullptr) {
        delete buffer_ctx;
        return nullptr;
    }
    std::memset(buffer_ctx->data, 0, size);

    return ggml_backend_buffer_init(buft, ggml_backend_awnpu_buffer_i, buffer_ctx, size);
}

static ggml_backend_buffer_type_t ggml_backend_awnpu_buffer_type_from_device(ggml_backend_dev_t dev) {
    auto * dev_ctx = (ggml_backend_awnpu_device_context *) dev->context;
    GGML_ASSERT(dev_ctx != nullptr);
    return dev_ctx->buft;
}

// 根据一个已有的 host 指针包装成 buffer。
// 用途：
// - memory-mapped model
// - 从别的库导入已有内存
// - 避免重复拷贝
// - 让 ggml 直接把外部内存纳入管理
static ggml_backend_buffer_t ggml_backend_awnpu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(max_tensor_size);
    auto * dev_ctx = (ggml_backend_awnpu_device_context *) dev->context;
    GGML_ASSERT(dev_ctx != nullptr);
    auto * buffer_ctx = ggml_backend_awnpu_buffer_context_init(ptr, size, false);
    return ggml_backend_buffer_init(dev_ctx->buft, ggml_backend_awnpu_buffer_i, buffer_ctx, size);
}

static ggml_backend_t ggml_backend_awnpu_device_init_backend(ggml_backend_dev_t dev, const char * params);
static const char * ggml_backend_awnpu_backend_get_name(ggml_backend_t backend);
static void ggml_backend_awnpu_backend_free(ggml_backend_t backend);

// 返回设备的短名字，比如 "CPU"、"CUDA0"、"AWNPU0"
// 用途：
// - 枚举设备时显示
// - 日志和调试
// - 作为 device 的唯一可读标识之一
static const char * ggml_backend_awnpu_device_get_name(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_awnpu_device_context *) dev->context;
    return ctx ? ctx->name.c_str() : GGML_AWNPU_NAME;
}

// 返回设备的描述信息，通常比名字更具体。
// 比如可能是：
//  - 型号名
//  - 芯片代号
//  - 驱动版本摘要
//  - 平台说明
// 用途主要是给用户看，便于识别具体硬件。
static const char * ggml_backend_awnpu_device_get_description(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_awnpu_device_context *) dev->context;
    return ctx ? ctx->description.c_str() : GGML_AWNPU_NAME;
}

// 返回设备内存信息，通常是：
// - free：当前剩余可用内存
// - total：总内存
// 如果设备不方便精确报告，也可以返回 0。

// 调度器和上层工具会用它做：
// - 设备选择
// - 内存容量判断
// - 可用性展示
static void ggml_backend_awnpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = (ggml_backend_awnpu_device_context *) dev->context;
    if (ctx == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    *free = ctx->memory_free;
    *total = ctx->memory_total;
}

// 返回设备类型枚举，比如：CPU、GPU、AWNPU等
static enum ggml_backend_dev_type ggml_backend_awnpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_AWNPU;
}

// 一次性填充完整设备属性结构体 ggml_backend_dev_props。
// 里面包括：

// - name
// - description
// - memory_free
// - memory_total
// - type
// - device_id
// - caps
// 它本质上是把前面几个信息接口打包起来，方便一次取全。
// 如果 get_name/get_description/get_memory/get_type 已经实现了，这里通常就是聚合赋值。

// caps 里会描述能力，比如：
// - 是否支持异步
// - 是否支持 host buffer
// - 是否支持从 host ptr 创建 buffer
// - 是否支持 events
static void ggml_backend_awnpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    auto * ctx = (ggml_backend_awnpu_device_context *) dev->context;
    GGML_ASSERT(ctx != nullptr);
    props->name = ctx->name.c_str();
    props->description = ctx->description.c_str();
    props->memory_free = ctx->memory_free;
    props->memory_total = ctx->memory_total;
    props->type = GGML_BACKEND_DEVICE_TYPE_AWNPU;
    props->device_id = ctx->device_id.c_str();
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ true,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
    };
}

// 返回这个 device 的“首选 buffer type”。
// 典型含义：
// - 这个 device 上张量默认应该放在哪种内存里
// - 哪种 buffer type 最适合这个硬件执行
static ggml_backend_buffer_type_t ggml_backend_awnpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_awnpu_buffer_type_from_device(dev);
}

// 返回这个 device 相关的 host buffer type。
// 它通常表示：
// - 系统内存中的缓冲区
// - 但可能是 pinned / page-locked / DMA-friendly 内存
// 用途是做主机和设备之间更高效的数据传输。

// 对于 AWNPU 这种“CPU 统一申请并托管共享 DDR”的场景，这里应该返回 CPU buffer type：
// - 主机侧临时 buffer 由 CPU 侧分配和回收
// - AWNPU 只把这些内存视作可访问的 host/shared DDR
// - 设备自己的 buffer type 仍然保持 AWNPU 语义，用于权重和 graph tensor 绑定
static ggml_backend_buffer_type_t ggml_backend_awnpu_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

// AWNPU 接管 NPU 层（权重在 AWNPU buft）上的 op；其余由 scheduler 按 -ngl 切分与传播决定。
static bool ggml_backend_awnpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (op == nullptr) {
        return false;
    }

    if (ggml_backend_awnpu_node_is_layout_only(op)) {
        return true;
    }

    if (op->op == GGML_OP_GET_ROWS) {
        const struct ggml_tensor * rows = op->src[0];
        if (rows == nullptr) {
            return false;
        }
        // Input embedding lookup: always CPU (merge with CPU prefix layers).
        if (ggml_backend_awnpu_is_input_embedding_weight(rows)) {
            return false;
        }
        if (rows->buffer != nullptr &&
            rows->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            return false;
        }
        // Last-layer inp_out_ids / MoE row select: follow activation src.
        return ggml_backend_awnpu_node_runs_on_npu(rows, 0);
    }

    if (ggml_backend_awnpu_op_has_npu_weights(op)) {
        return true;
    }

    if (ggml_backend_awnpu_is_output_head_node(op) && ggml_backend_awnpu_lm_head_on_npu()) {
        return true;
    }

    const int il = ggml_backend_awnpu_infer_layer_index(op);
    if (il >= 0) {
        return ggml_backend_awnpu_layer_on_npu(il);
    }

    if (!ggml_backend_awnpu_op_supported(op->op)) {
        return ggml_backend_awnpu_node_runs_on_npu(op, 0);
    }

    return false;
}

// 判断这个 device 能否使用某种 buffer_type。 这个设备能不能处理放在这种内存里的 tensor
// 它用于调度和张量放置决策。
// 比如一个 NPU 可能只接受它自己的专用 buffer type，不接受普通 host buffer 作为计算输入。
// 对于在 CPU + NPU 上 部署的tensor，所有 buffer type 都必须能被 CPU 或者 NPU 支持
static bool ggml_backend_awnpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto * dev_ctx = (ggml_backend_awnpu_device_context *) dev->context;
    if (dev_ctx == nullptr || buft == nullptr) {
        return false;
    }

    if (buft == dev_ctx->buft) {
        return true;
    }

    // 共享 DDR 场景下，AWNPU 需要接受主机侧 buffer。
    if (ggml_backend_buft_is_host(buft)) {
        return true;
    }

    return false;
}

static bool ggml_backend_awnpu_op_has_npu_weights(const struct ggml_tensor * op) {
    GGML_ASSERT(op != nullptr);

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (ggml_backend_awnpu_weight_on_npu(op->src[i])) {
            return true;
        }
    }

    return false;
}

// 根据这个 device 创建一个 ggml_backend_t，也就是一个可执行的 backend 实例。

// 通常要做的事：
// - 创建 backend 上下文
// - 绑定设备句柄
// - 准备执行队列/流
// - 初始化分配器或运行时状态

// 你可以把它理解成：
// - device 是硬件实体
// - backend 是这个硬件上的一次运行实例
static ggml_backend_t ggml_backend_awnpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    auto * dev_ctx = dev != nullptr ? (ggml_backend_awnpu_device_context *) dev->context : nullptr;

    auto * backend_ctx = new ggml_backend_awnpu_context;
    backend_ctx->n_threads = std::max(1, ggml_backend_awnpu_parse_int_env("GGML_AWNPU_THREADS", (int) std::max(1u, std::thread::hardware_concurrency())));
    backend_ctx->use_ref = false;
    backend_ctx->device_id = dev_ctx != nullptr ? dev_ctx->device : 0;

    if (params != nullptr) {
        const std::string p = params;
        auto pos = p.find("threads=");
        if (pos != std::string::npos) {
            backend_ctx->n_threads = std::max(1, std::atoi(p.c_str() + pos + 8));
        }
        if (p.find("ref=1") != std::string::npos) {
            backend_ctx->use_ref = true;
        }
    }

    // 同一种 backend 的多个实例（例如 AWNPU0、AWNPU1）可以共用同一段静态 guid，因为它们属于同一类实现。
    static ggml_guid guid = { 0x3a, 0x52, 0x5d, 0x3f, 0x6e, 0x42, 0x49, 0x0c, 0x9c, 0x8a, 0x25, 0x52, 0x0d, 0x4a, 0x2b, 0x11 };

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ &guid,
        /* .iface   = */ ggml_backend_awnpu_i,
        /* .device  = */ dev,
        /* .context = */ backend_ctx,
    };

    return backend;
}

/*
作用：返回这个 backend 实例的名字。
用途：
- 日志输出
- 调试
- 区分不同 backend 实例

比如可能返回：
- "CPU"
- "CUDA"
- "AWNPU0"
它是 backend 名字，不是 device 名字，也不是 buffer type 名字。
*/
static const char * ggml_backend_awnpu_backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_AWNPU_NAME;
}

/*
作用：释放这个 backend 实例。
它需要做的事情通常包括：
- 释放 backend 私有上下文
- 释放 work buffer
- 停止后台线程
- 清理运行时资源
- 关闭 device session / stream / queue
ggml_backend_dev_init() 创建出来的 backend，最后都要通过这个函数销毁。
*/
static void ggml_backend_awnpu_backend_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_awnpu_context *) backend->context;
    if (ctx != nullptr) {
        ggml_aligned_free(ctx->work_data, ctx->work_size);
        delete ctx;
    }
    delete backend;
}

static void ggml_backend_awnpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    auto * ctx = (ggml_backend_awnpu_context *) backend->context;
    if (ctx != nullptr) {
        ctx->n_threads = std::max(1, n_threads);
    }
}

static void ggml_backend_awnpu_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * abort_callback_data) {
    auto * ctx = (ggml_backend_awnpu_context *) backend->context;
    if (ctx != nullptr) {
        ctx->abort_callback = abort_callback;
        ctx->abort_callback_data = abort_callback_data;
    }
}


// 执行单 node 子图 [node_idx, node_idx + 1)。
static enum ggml_status ggml_backend_awnpu_cpu_forward_range(
        ggml_backend_awnpu_context * ctx,
        struct ggml_cgraph * cgraph,
        int i0,
        int i1) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(cgraph != nullptr);

    if (i0 < 0) {
        i0 = 0;
    }
    if (i1 <= i0 || i1 > cgraph->n_nodes) {
        return GGML_STATUS_FAILED;
    }

    struct ggml_cgraph gv = ggml_graph_view(cgraph, i0, i1);
    struct ggml_cplan cplan = ggml_graph_plan(&gv, ctx->n_threads, nullptr);
    if (ctx->work_size < cplan.work_size) {
        ggml_aligned_free(ctx->work_data, ctx->work_size);
        ctx->work_data = ggml_aligned_malloc(cplan.work_size);
        if (ctx->work_data == nullptr && cplan.work_size > 0) {
            ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        ctx->work_size = cplan.work_size;
    }

    cplan.work_data           = (uint8_t *) ctx->work_data;
    cplan.abort_callback      = ctx->abort_callback;
    cplan.abort_callback_data = ctx->abort_callback_data;
    cplan.use_ref             = ctx->use_ref;
    return ggml_graph_compute(&gv, &cplan);
}

static enum ggml_status ggml_backend_awnpu_forward_cpu_node(
        ggml_backend_awnpu_context * ctx,
        struct ggml_cgraph * cgraph,
        int node_idx) {
    return ggml_backend_awnpu_cpu_forward_range(ctx, cgraph, node_idx, node_idx + 1);
}

static enum ggml_status ggml_backend_awnpu_forward_npu_node(
        ggml_backend_awnpu_context * ctx,
        struct ggml_tensor * node) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(node != nullptr);

    return ggml_backend_awnpu_npu_compute_node(
            ctx->device_id,
            node,
            ctx->abort_callback,
            ctx->abort_callback_data);
}

static enum ggml_status ggml_backend_awnpu_fallback_cpu_node(
        ggml_backend_awnpu_context * ctx,
        struct ggml_cgraph * cgraph,
        int node_idx,
        const struct ggml_tensor * node,
        bool op_supported) {
    ggml_backend_awnpu_warn_fallback_once(__func__, node, op_supported);
    llama_graph_exec_log_set_current_node_fallback(true);

    llama_graph_exec_log_suspend_node_done();
    const enum ggml_status status = ggml_backend_awnpu_forward_cpu_node(ctx, cgraph, node_idx);
    llama_graph_exec_log_resume_node_done();
    return status;
}

static enum ggml_status ggml_backend_awnpu_compute_node(
        ggml_backend_t backend,
        struct ggml_cgraph * cgraph,
        int node_idx,
        struct ggml_tensor * node) {
    auto * ctx = (ggml_backend_awnpu_context *) backend->context;
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(node != nullptr);

    if (ggml_backend_awnpu_node_is_layout_only(node)) {
        return GGML_STATUS_SUCCESS; //  ggml_backend_awnpu_forward_npu_node(ctx, node);
    }

    if (!ggml_backend_awnpu_op_supported(node->op)) {
        return ggml_backend_awnpu_fallback_cpu_node(
                ctx, cgraph, node_idx, node, false);
    }

    const enum ggml_status status = ggml_backend_awnpu_forward_npu_node(ctx, node);
    if (status == GGML_STATUS_SUCCESS || status == GGML_STATUS_ABORTED) {
        return status;
    }

    return ggml_backend_awnpu_fallback_cpu_node(
            ctx, cgraph, node_idx, node, true);
}

static enum ggml_status ggml_backend_awnpu_graph_compute_npu(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend != nullptr);
    GGML_ASSERT(cgraph != nullptr);

    auto * ctx = (ggml_backend_awnpu_context *) backend->context;
    GGML_ASSERT(ctx != nullptr);

    const bool log_nodes = ggml_backend_graph_node_done_callback_is_set();

    enum ggml_status status = GGML_STATUS_SUCCESS;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        llama_graph_exec_log_set_current_node_fallback(false);

        struct ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            continue;
        }

        int64_t t0 = 0;
        if (log_nodes) {
            t0 = ggml_time_us();
        }

        status = ggml_backend_awnpu_compute_node(backend, cgraph, i, node);

        if (log_nodes) {
            ggml_backend_invoke_graph_node_done_callback(backend, cgraph, i, ggml_time_us() - t0);
        }

        llama_graph_exec_log_set_current_node_fallback(false);

        if (status != GGML_STATUS_SUCCESS) {
            break;
        }
    }

    return status;
}

/*
作用：执行整个计算图。
它要做的事情包括：
- 遍历 graph 节点
- 判断每个 op 怎么执行
- 调用对应 kernel / runtime
- 处理中间 buffer
- 返回执行状态

如果 backend 支持异步，这个接口通常也会是异步提交式；
如果不支持异步，也可能是同步执行后返回。
*/
static enum ggml_status ggml_backend_awnpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_awnpu_context *) backend->context;
    GGML_ASSERT(ctx != nullptr);

    if (cgraph == nullptr || cgraph->n_nodes <= 0) {
        return GGML_STATUS_SUCCESS;
    }

    // Scheduler splits by layer/weight placement; per-node NPU/CPU fallback is handled in graph_compute_npu.
    return ggml_backend_awnpu_graph_compute_npu(backend, cgraph);
}

/*
synchronize = ggml_backend_awnpu_synchronize
作用：等待这个 backend 上所有尚未完成的异步操作结束。
它要做的事情一般是：
- flush 队列
- 等待设备执行完
- 保证后续 CPU 访问看到最新数据
如果 backend 只有同步执行，那这个函数可能几乎是空操作。
但只要支持 async，通常就必须实现它。
 */
static void ggml_backend_awnpu_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
}

const struct ggml_backend_i ggml_backend_awnpu_i = {
    /* .get_name                = */ ggml_backend_awnpu_backend_get_name,
    /* .free                    = */ ggml_backend_awnpu_backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_awnpu_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_awnpu_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

// 返回这个 backend registry 的名字，也就是整个后端模块的标识名。
// 这不是 device 名字，也不是 buffer type 名字，而是 registry / backend 模块名字。
static const char * ggml_backend_awnpu_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_AWNPU_NAME;
}

// 返回这个 registry 下有多少个 device。
// 一般要实现什么?
// - 扫描系统里的 NPU 设备
// - 返回可用 device 数量
// - 如果驱动没起来、硬件不可用，返回 0

// 这是后续 get_device(index) 的上界依据。
static size_t ggml_backend_awnpu_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * ctx = (ggml_backend_awnpu_reg_context *) reg->context;
    return ctx ? ctx->devices.size() : 0;
}

// 按索引返回一个具体的 device 对象。
// 这个函数是 registry 和 device 之间的桥梁。
static ggml_backend_dev_t ggml_backend_awnpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = (ggml_backend_awnpu_reg_context *) reg->context;
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index].get();
}

// 返回这个 backend 自定义扩展函数的函数指针。
// 除了 ggml 标准接口之外，某个 backend 可能还额外提供一些专用 API，比如：
// - 设备诊断
// - 特定参数设置
// - dump 调试信息
// - 查询私有能力
// - 专有内存管理函数
// 这些不属于通用 ggml 接口，所以通过 get_proc_address 暴露出去。

// 一般要实现什么 ?
// - 根据字符串 name 查找函数
// - 如果匹配某个自定义符号，就返回对应函数指针
// - 不支持的名字返回 nullptr

// 如果 backend 没有额外 API，也可以返回 nullptr。
static void * ggml_backend_awnpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_awnpu_set_n_threads;
    }
    if (std::strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *) ggml_backend_awnpu_set_abort_callback;
    }
    if (std::strcmp(name, "ggml_backend_set_model_n_layer") == 0) {
        return (void *) ggml_backend_awnpu_set_model_n_layer;
    }
    if (std::strcmp(name, "llama_graph_exec_log_get_callbacks") == 0) {
        return (void *) ggml_backend_awnpu_graph_exec_log_get_callbacks;
    }
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_awnpu_reg_i = {
    /* .get_name         = */ ggml_backend_awnpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_awnpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_awnpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_awnpu_get_proc_address,
};

static ggml_backend_awnpu_reg_context * ggml_backend_awnpu_reg_init(void) {
    auto * ctx = new ggml_backend_awnpu_reg_context;
    ggml_backend_awnpu_probe_info probe{};
    if (!ggml_backend_awnpu_probe(&probe)) {
        delete ctx;
        return nullptr;
    }

    GGML_LOG_INFO("ggml_backend_awnpu_reg_init: probe.n_devices = %d\n", probe.n_devices);
    for (int i = 0; i < probe.n_devices; ++i) {
        const ggml_backend_awnpu_device_probe_info & probe_dev = probe.devices[(size_t) i];
        GGML_LOG_INFO("Device %d: \n\
            memory_total = %zu MB, \n\
            memory_free = %zu MB, \n\
            buffer_alignment = %zu, \n\
            buffer_max_size = %zu, \n\
            shared_host_buffer = %d\n\n", 
            i, 
            probe_dev.memory_total / 1024 / 1024, 
            probe_dev.memory_free / 1024 / 1024, 
            probe_dev.buffer_alignment, 
            probe_dev.buffer_max_size, 
            probe_dev.shared_host_buffer);
    }

    for (int i = 0; i < probe.n_devices; ++i) {
        const ggml_backend_awnpu_device_probe_info & probe_dev = probe.devices[(size_t) i];

        auto buft_ctx = std::make_unique<ggml_backend_awnpu_buffer_type_context>();
        buft_ctx->device = i;
        buft_ctx->name = std::string(GGML_AWNPU_NAME) + std::to_string(i);
        buft_ctx->alignment = probe_dev.buffer_alignment;
        buft_ctx->max_size = probe_dev.buffer_max_size;
        buft_ctx->is_host = true; //probe_dev.shared_host_buffer;

        auto buft = std::make_unique<ggml_backend_buffer_type>();
        *buft = {
            /* .iface   = */ {
                /* .get_name       = */ ggml_backend_awnpu_buffer_type_get_name,
                /* .alloc_buffer   = */ ggml_backend_awnpu_buffer_alloc_buffer,
                /* .get_alignment  = */ ggml_backend_awnpu_buffer_type_get_alignment,
                /* .get_max_size   = */ ggml_backend_awnpu_buffer_type_get_max_size,    // 若设置为：nullptr，则表示没有明确上限，或者上层不走这个约束检查。
                /* .get_alloc_size = */ ggml_backend_awnpu_buffer_type_get_alloc_size,
                /* .is_host       = */ ggml_backend_awnpu_buffer_type_is_host,
            },
            /* .device  = */ nullptr,
            /* .context = */ buft_ctx.get(),
        };

        auto dev_ctx = std::make_unique<ggml_backend_awnpu_device_context>();
        dev_ctx->device = i;
        dev_ctx->name = buft_ctx->name;
        dev_ctx->description = "AWNPU device " + std::to_string(i);
        dev_ctx->device_id = "awnpu:" + std::to_string(i);
        dev_ctx->memory_total = probe_dev.memory_total;
        dev_ctx->memory_free = probe_dev.memory_free;

        // 这个 npu 的内存分配器规格/内存池类型描述
        // 描述怎么分配，包括名字、对齐、实际分配大小、是否 host memory、如何创建 buffer。
        dev_ctx->buft = buft.get();

        auto dev = std::make_unique<ggml_backend_device>();
        *dev = {
            /* .iface   = */ {
                /* .get_name             = */ ggml_backend_awnpu_device_get_name,
                /* .get_description      = */ ggml_backend_awnpu_device_get_description,
                /* .get_memory           = */ ggml_backend_awnpu_device_get_memory,
                /* .get_type             = */ ggml_backend_awnpu_device_get_type,
                /* .get_props            = */ ggml_backend_awnpu_device_get_props,
                /* .init_backend         = */ ggml_backend_awnpu_device_init_backend,
                /* .get_buffer_type      = */ ggml_backend_awnpu_device_get_buffer_type,
                /* .get_host_buffer_type = */ ggml_backend_awnpu_device_get_host_buffer_type,
                /* .buffer_from_host_ptr = */ ggml_backend_awnpu_device_buffer_from_host_ptr,
                /* .supports_op          = */ ggml_backend_awnpu_device_supports_op,
                /* .supports_buft        = */ ggml_backend_awnpu_device_supports_buft,
                /* .offload_op           = */ nullptr,
                /* .event_new            = */ nullptr,
                /* .event_free           = */ nullptr,
                /* .event_synchronize    = */ nullptr,
            },
            /* .reg     = */ nullptr,
            /* .context = */ dev_ctx.get(),
        };

        buft->device = dev.get();
        dev->reg = nullptr;

        ctx->buft_contexts.emplace_back(std::move(buft_ctx));
        ctx->bufts.emplace_back(std::move(buft));
        ctx->device_contexts.emplace_back(std::move(dev_ctx));
        ctx->devices.emplace_back(std::move(dev));
    }

    return ctx;
}

ggml_backend_reg_t ggml_backend_awnpu_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;
    static std::mutex mutex;
    static std::unique_ptr<ggml_backend_awnpu_reg_context> reg_ctx;

    std::lock_guard<std::mutex> lock(mutex);

    if (!initialized) {
        reg_ctx.reset(ggml_backend_awnpu_reg_init());
        if (reg_ctx == nullptr || reg_ctx->devices.empty()) {
            reg_ctx.reset();
            return nullptr;
        }

        reg = {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ ggml_backend_awnpu_reg_i,
            /* .context     = */ reg_ctx.get(),
        };
        for (auto & dev : reg_ctx->devices) {
            dev->reg = &reg;
        }
        initialized = true;
    }

    return &reg;
}

static int ggml_backend_awnpu_score_impl(void) {
    auto * reg = ggml_backend_awnpu_reg();
    return reg && ggml_backend_awnpu_reg_get_device_count(reg) > 0 ? 100 : 0;
}

// GGML_BACKEND_DL 的意思是：
// 把某个 backend 按“动态库插件”方式构建和加载，而不是直接编进主程序里
// 如果不使用动态库，那么 backend 的代码直接参与主工程编译，
// 启动时由 ggml_backend_registry() 直接注册。

// #ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_awnpu_reg)

// 这个 backend 值不值得被优先加载的打分函数。
// 动态后端加载器在扫描多个插件时，不是随便全装上，每个 backend 会返回一个分数,
// 分数高：更值得加载
// 分数低或 0：不加载，或者优先级很低
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_awnpu_score_impl)
// #endif
