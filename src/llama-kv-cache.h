#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    // 记录跨 stream 的待执行拷贝。
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]

    // 这是 ubatch 在 KV cache 里的落点描述。
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;    // 当前 ubatch 覆盖的 stream 范围
        uint32_t s1;    // 当前 ubatch 覆盖的 stream 范围

        std::vector<llama_seq_id> strm; // [ns]  第 s 路对应的 stream id。
        std::vector<idx_vec_t>    idxs; // [ns]  该 stream 上每个 token 最终落到哪些 cell 下标。

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    llama_kv_cache(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    // llama_kv_cache 是长期存在的 cache 对象，
    // 而 llama_kv_cache_context 是一次 batch/update 过程中的临时上下文，负责：
    //   1 顺序遍历当前要处理的 ubatch。
    //   2 对当前 ubatch 调 apply()，把它的槽位元数据提交到 KV cache。
    //   3 持有当前 ubatch 对应的 slot_info。
    //   4 提供 get_k/get_v/cpy_k/cpy_v 这组“当前 ubatch 视角”的转发接口。

    // 把一个 batch 切成若干 ubatch，并为每个 ubatch 准备 memory 状态
    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    // 构造“满 cache”的上下文，用来预留最坏情况 graph / buffer
    llama_memory_context_ptr init_full() override;

    // 对 pending 的 shift / copy 等做真正更新前的准备
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    // 作用：
    // 1 重置所有 v_cells。
    // 2 把所有 v_heads 置 0。
    // 3 若 data == true，还会把实际 K/V buffer 也清零。
    void clear(bool data) override;

    // 对 sequence 级别的 memory 做操作

    // 按位置区间删除某条 sequence 的 cache 条目。
    // 两种模式：
    //   1 seq_id >= 0：只删该 sequence 在 [p0, p1) 中的出现。
    //   2 seq_id == -1：匹配所有 sequence，等价于按位置区间整片清掉。
    // 删除过程中如果释放出了更早的槽位，会把对应 head 往前回退，以便后续 find_slot() 更快复用。
    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    // 复制 sequence。
    // 同 stream
    //  只改 metadata，不拷 K/V 数据：
    //      1 找到所有属于 seq_id_src 的 cell。
    //      2 给这些 cell 追加 seq_id_dst。
    // 因为它们本来就在同一套物理 K/V stream 上，不需要真复制张量内容。

    // 跨 stream
    //  这是较复杂的分支：
    //   1 要求是“整条 KV buffer 复制”，不支持任意子区间。
    //   2 先把 (src_stream, dst_stream) 加到 sc_info。
    //   3 然后立即重建 v_cells[dst] 的元数据。
    //   4 真正的 K/V tensor copy 在下一次 update() 中做。
    // 这是个典型的“控制面先行、数据面延迟”的设计。
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    // 只保留某个 sequence，删掉该 stream 上不属于它的部分。
    // 内部通过 cells.seq_keep(i, seq_id) 实现：
    //   1 若 cell 本来包含该 seq，则把该 cell 的 seq 集合压缩成只剩它。
    //   2 若 cell 不包含该 seq，但原先有别的 seq，则整格清空。
    void seq_keep(llama_seq_id seq_id)                                                          override;
    // 对某个序列在 [p0, p1) 范围内的 position 做整体平移。
    // 底层调用 cells.pos_add(i, shift)：
    //   1 修改 pos[i]。
    //   2 累计 shift[i]。
    //   3 同时更新 seq_pos 索引。
    //   4 若位置变成负数，则 cell 直接失效并释放。
    // 这个接口主要用于上下文滑窗、位置平移一类场景。
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    // 对某个序列的 position 做整除缩放。
    // 底层调用 cells.pos_div(i, d)，同样会累计到 shift 中。
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    // 查询某个序列在 cache 中当前出现的最小/最大位置。
    // 实现逻辑：
    //   1 先用 seq_to_stream[seq_id] 找到它归属的 stream。
    //   2 再对该 stream 的 v_cells 调 cells.seq_pos_min(seq_id) / cells.seq_pos_max(seq_id)。
    // 这也是为什么 llama_kv_cells 里要维护 seq_pos[seq_id] 这张有序计数表。
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    // 序列化与反序列化
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    // 整个 KV cache 依附的模型对象。
    // 它提供：
    //   1 每层放在哪个 device 上：model.dev_layer(il)。
    //   2 rope 参数。
    //   3 层数、架构、是否 MLA 等模型属性。
    const llama_model & model;
    // 模型超参数引用，频繁使用：
    //   1 n_layer / n_layer_kv()。
    //   2 n_embd_k_gqa(il) / n_embd_v_gqa(il)。
    //   3 n_head_kv(il)。
    //   4 n_embd_head_k(il) / n_embd_head_v(il)。
    //   5 n_pos_per_embd()。
    //   6 rope_type / SWA / ALiBi 等。
    const llama_hparams & hparams;

    // kv_layer 描述 「KV cache 里的一层」（注意：这是 cache 里的一层，不一定和模型 il 一一对应，因为过滤/reuse）。
    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;    // 模型里的层号：对应 hparams / llama_model 的 第 il 层 transformer block。

        // 这一层在 整块 device/CPU buffer 上的 K cache 父张量。ne[0] = n_embd_k_gqa， ne[1] = kv_size， ne[2] = n_stream
        ggml_tensor * k;
        ggml_tensor * v;    // 这一层在 整块 device/CPU buffer 上的 V cache 父张量。MLA 可能没有V

        // 多序列不 unified 时，每个序列/stream 单独的 K, [n_embd_k_gqa, kv_size]
        std::vector<ggml_tensor *> k_stream;
        // 多序列不 unified 时，每个序列/stream 单独的 V, [n_embd_v_gqa, kv_size]
        std::vector<ggml_tensor *> v_stream;
    };

    // 表示 V cache 是否采用转置布局。
    // false：通常是 Flash Attention 场景，V 的布局较直观。
    // true：非 FA 路径里，V 会采用转置后的写法，因此：
    //   1 get_v() 的视图逻辑不同。
    //   2 cpy_v() 也有专门的 transposed 分支。
    //   3 state_write_data() / state_read_data() 也要单独处理。
    bool v_trans = true;  // the value tensor is transposed

    // 支持的最大 sequence 数
    const uint32_t n_seq_max = 1;
    // 物理 stream 数
    // unified == true 时，n_stream = 1。 所有 sequence 共用一套物理 KV ring
    // unified == false 时，n_stream = n_seq_max。 每个 sequence 各有自己的一套 stream。
    const uint32_t n_stream  = 1;

    // required padding
    // KV 长度在图上会按这个值对齐；get_n_kv() 里还会进一步和 256 取大值，以便：
    //   1 保持 graph 形状稳定，利于 graph 复用。
    //   2 某些 backend 上性能更好。
    const uint32_t n_pad = 1;

    // SWA  SWA window 大小相关参数。
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    // 是否为 K/V 启用 attention rotation。
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    // 如果所有参与 cache 的层 head 维度一致，则记录该公共值；否则设为 -1。
    // 用途：
    //   1 用于判断是否能构造统一大小的 Hadamard rotation 输入。
    //   2 避免对“每层 head 维度不一致”的模型做错误的统一处理。
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    // 预计算的 Hadamard 矩阵，
    // 后续通过 set_input_k_rot() / set_input_v_rot() 直接灌到 graph input tensor。
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    // 每个 buffer type 对应一个 ggml context + backend buffer。
    // 构造时会按 buft 聚合层：
    //   1 同一种 buffer type 的层共享同一个 ggml context。
    //   2 最后一次性分配 buffer。
    // 这样减少了 context 和 buffer 的碎片化。
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

   // 每个 stream 一根“找空槽起始位置”的 head。
    // 它不是 KV 逻辑状态的一部分，而是搜索优化辅助状态。
    // 用途：
    //   1 find_slot() 从 v_heads[strm] 开始向后扫。
    //   2 apply_ubatch() 提交后把 head 移到当前写入范围之后。
    //   3 seq_rm / seq_keep / seq_add 释放槽位时可能把 head 回退到更早的空位。
    std::vector<uint32_t> v_heads;

    // 它不存 K/V 向量本身，而是存每个 cell 的元数据
    // v_cells[*] 是“控制面 / 目录索引” , layers[*].k / v 是“真正的数据面”

    // 每个 stream 一套 cell 元数据。
    // 其中每个 llama_kv_cells 里保存：
    //   1 哪些 cell 为空/非空。
    //   2 每个 cell 的逻辑位置。
    //   3 每个 cell 属于哪些 seq。
    //   4 各个 seq 当前的 min/max pos。
    //   5 是否存在 shift。
    std::vector<llama_kv_cells> v_cells;

    // sequence id 到 stream id 的映射。
    // 两种典型情况：
    //   1 unified：所有 seq_id -> 0
    //   2 non-unified：seq_id -> seq_id
    // 这层映射让上层逻辑统一基于 seq_id 工作，而内部可以决定物理落在哪个 stream。
    std::vector<uint32_t> seq_to_stream;

    // 待执行的跨 stream buffer 拷贝队列。由 seq_cp() 入队，由 update() 真正执行。
    stream_copy_info sc_info;

    // layers[*].k / v 是“真正的数据面”, v_cells[*] 是“控制面 / 目录索引” 
    // layers 是“cache 层数组”而不是“模型层数组”。因为有 filter 和 reuse：
    //   1 某些模型层可以不参与 cache。
    //   2 某些层可以复用别的层的 cache 存储。

    // 这是经过 has_kv / filter / reuse 处理后的紧凑层数组。
    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    // 模型层号 il -> layers 下标 ikv 的映射。
    // get_k/get_v/cpy_k/cpy_v 都先通过它找到对应的 cache 层。
    // 这是因为：
    //   1 模型层不一定全部有 KV。
    //   2 某些层可能被过滤掉。
    //   3 某些层可能复用别的层的 cache。
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    // 用于 save/load 阶段，表示某个 stream 上被选中的若干连续 cell range。
    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;   // 已经填充的 token 数量
};
