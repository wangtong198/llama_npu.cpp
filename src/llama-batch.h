#pragma once

#include "llama.h"

#include "llama-cparams.h"

#include <array>
#include <vector>
#include <set>
#include <bitset>
#include <memory>
#include <unordered_map>

// keep this struct lightweight
struct llama_ubatch {
    bool equal_seqs() const {
        return b_equal_seqs != 0;
    }

    // typical for M-RoPE cases:
    //   0 - sequential position of the tokens/embeddings in the sequence
    //   1 - y position in the image
    //   2 - x position in the image
    //   3 - other
    bool is_pos_2d() const {
        // TODO @ngxson : we may need to check for model arch when more models use >1 positions
        return n_pos >= 3;
    }

    // true表示：本次 ubatch 是 split_equal 一类「各条序列（更准确地说是各 sequence set）长度一致、按块对齐」 的划分；
    // false表示： 多来自 split_simple（连续截取一段行，各「set」长度可以不一）
    uint32_t b_equal_seqs; // note: this is a boolean, but we use an int32_t for alignment
                           //       otherwise address sanitizer complains
    // TODO: whole_seqs for embeddings?

    // 本 ubatch 里的 token 行数（也是 token / n_seq_id / output 等按行数组的长度）
    uint32_t n_tokens;     // total tokens (n_seq_tokens * n_seqs)
    // 每个 sequence set 在 本 ubatch 内占多少行；
    uint32_t n_seq_tokens; // tokens per sequence set
    // 在 ubatch_add 里满足 n_tokens % n_seqs == 0，且 n_seq_tokens = n_tokens / n_seqs。
    // 含义是：
    // 1 在 equal 布局下，把 n_tokens 行看成 n_seqs 个并列的「序列条带」，
    //   每条带 恰好 n_seq_tokens 行（多流 KV / 多序列同步一步时常用这种块结构）。
    // 2 split_simple 里调用的是 ubatch_add(idxs, idxs.size(), false)，
    //   此时 n_seqs == n_tokens、n_seq_tokens == 1，即「每一行自成一个 set」，不再有跨行对齐的多序列块语义。
    uint32_t n_seqs;       // sequence sets in the ubatch
    // 本 ubatch 里出现过的 llama_seq_id 去重后的个数（按 id 从小到大扫 0…n_seq_max-1 填进 seq_id_unq）。
    // 和 n_seqs 不同：n_seqs 是 切块几何（几条并列「带」）；n_seqs_unq 是 逻辑会话 id 有几个。
    // 例如两行共享同一个 seq_id 时，n_seqs 仍可能为 2（两条带），但 n_seqs_unq 可以为 1。
    // 读侧：非 kv_unified 时，attention mask 的 stream 维用 ubatch.n_seqs_unq（build_attn_inp_kq_mask）；
    //        embedding 的 mean/cls 等用 seq_id_unq + seq_idx 按序列聚合。
    uint32_t n_seqs_unq;   // unique sequence ids in the ubatch
    // 每个 token 占 多少个 llama_pos 标量（来自构造 batch 时的 n_pos_per_embd）。
    //    普通 1D RoPE 为 1；
    //    M-RoPE / 多段位置 时为 3、4 等，is_pos_2d() 用 n_pos >= 3 判断「是否多维位置」。
    // pos 在缓冲里的布局是 按维分块、块内按 token 下标：
    //     第 j 维、第 i 个 token 在 pos[j * n_tokens + i]（与 ubatch_add 里写入方式一致）。
    //     纯文本且 M-RoPE 维数=4 时，set_input 还会把 1D pos[i] 扩成 4 路再写入图张量。
    uint32_t n_pos;        // number of position inputs for each token/embedding

    // seq_id_unq: unique sequence ids in the ubatch
    // seq_idx:    indices of the unique sequence ids in the ubatch in [0, n_seqs_unq)
    //             used for extracting sequence pooled embeddings

    //                          // size               | idx | val
    llama_token  *  token;      // [n_tokens]         | i   | id, token
    float        *  embd;       // [n_embd, n_tokens] | i   | embd
    llama_pos    *  pos;        // [n_tokens*n_pos]   | i   | pos
    int32_t      *  n_seq_id;   // [n_tokens]         | i   | -
    llama_seq_id ** seq_id;     // [n_tokens]         | s   | s0, s1, seq_id
    // 长度 n_seqs_unq，列出本 ubatch 出现过的 llama_seq_id（有序枚举）。
    llama_seq_id *  seq_id_unq; // [n_seqs_unq]       | s   | seq_id
    // 长度 LLAMA_MAX_SEQ，对每个可能的 seq_id 标量，若在 ubatch 里出现过，则为 [0, n_seqs_unq) 里的下标；否则为 -1。
    // 作用：Pooling（如 mean）里用 seq_idx[seq_id] 把行归到「第几个唯一序列」，填 mean 张量里对应的列；
    // 避免每行都扫一遍 seq_id_unq。
    int32_t      *  seq_idx;    // [LLAMA_MAX_SEQ]    | -   | seq_idx
    int8_t       *  output;     // [n_tokens]         | i   | -

    struct data_t {
        std::vector<llama_token>    token;
        std::vector<float>          embd;
        std::vector<llama_pos>      pos;
        std::vector<int32_t>        n_seq_id;
        std::vector<llama_seq_id *> seq_id;      // these point into the seq_id_data below
        std::vector<llama_seq_id>   seq_id_unq;
        std::vector<int32_t>        seq_idx;
        std::vector<int8_t>         output;

        // 所有行的 seq_id 拷在一起
        std::vector<llama_seq_id> seq_id_data;
    };

    // the llama_ubatch pointers above point to this data if set. otherwise - point to external non-owning data
    // ubatch 顶层的裸指针在 data 非空时指向 data 里 .data()；这样 ubatch 可以 被移动/复制时仍保持指针有效。
    // 若将来有路径指向 调用方已有数组，则 data 可以为空，指针仅为 非拥有引用（头文件注释所写）。
    std::shared_ptr<data_t> data;
};

// a helper for sanitizing, fulfilling and splitting a batch
class llama_batch_allocr {
public:
    llama_batch_allocr(uint32_t n_pos_per_embd);

    // sanitize and auto-gen missing data in the input batch
    // memory is optional. if provided will be used to check for sequence continuity and to determine the positions
    bool init(
            const llama_batch & batch_inp,
            const llama_vocab & vocab,
            const llama_memory_i * memory,
            uint32_t n_embd,
            uint32_t n_seq_max,
            bool output_all);

    const llama_batch & get_batch() const;

    uint32_t get_n_tokens()  const;
    uint32_t get_n_outputs() const;
    uint32_t get_n_used()    const;

    // the array of output indices in the order they were encountered during the ubatch splitting
    std::vector<int32_t> & get_out_ids();

    // min/max positions of each sequence in the current ubatch
    llama_pos seq_pos_min(llama_seq_id seq_id) const;
    llama_pos seq_pos_max(llama_seq_id seq_id) const;

    // call once before splitting the batch to reset the internal state
    void split_reset();

    // simple split, unknown number of sequence sets of unequal lengths
    llama_ubatch split_simple(uint32_t n_ubatch);

    // make ubatches of equal-length sequences sets
    // if sequential == true, the tokens in the ubatch will have increasing sequential sequence ids
    llama_ubatch split_equal(uint32_t n_ubatch, bool sequential);

    // sequence-set-wise split - each ubatch contains a single sequence-set
    llama_ubatch split_seq(uint32_t n_ubatch);

    // a helper method for creating a well-defined ubatch of tokens
    // TODO: support embeddings if needed in the future
    llama_ubatch ubatch_reserve(uint32_t n_seq_tokens, uint32_t n_seqs);

private:
    void clear();

    // create the next ubatch based on the provided batch indices (idxs) and the number of sequence sets (n_seqs)
    // return llama_ubatch.n_tokens == 0 if the entire batch was consumed
    llama_ubatch ubatch_add(const std::vector<int32_t> & idxs, uint32_t n_seqs, bool equal_seqs);

    // for debugging, start with LLAMA_BATCH_DEBUG=2
    void ubatch_print(const llama_ubatch & ubatch, int debug);

    // 本次 init() 校验、补全字段之后使用的 llama_batch（会在缺少 pos / seq_id / logits 时填上内部缓冲区指针）。
    llama_batch batch;

    // only for debugging purposes
    // 仅用于调试输出（例如 LLAMA_BATCH_DEBUG 下打印 token 文本）。
    const llama_vocab * vocab;

    // TODO: this is more of a temporary solution until we have a better way to handle multiple positions per token/embd
    //       ref: https://github.com/ggml-org/llama.cpp/issues/13694#issuecomment-2983871762
    // 每个 token / 每条 embedding 占多少个 position 标量。
    // 普通因果 LM 多为 1；
    // M-RoPE、带 2D 坐标的视觉等多维位置时 ≥3（与 llama_ubatch::n_pos 一致）。
    // 来自模型 hparams.n_pos_per_embd()。
    const uint32_t n_pos_per_embd;

    // 连续向量输入路径下每条 embd 的维度
    uint32_t n_embd;
    // 当前上下文允许的序列 id 上界（与 cparams.n_seq_max 等一致），用于校验 seq_id 是否越界。
    uint32_t n_seq_max;
    // 本 batch 里 logits[i] != 0 的 token 个数（需要输出 logits / embedding 行数），在 init 末尾统计
    uint32_t n_outputs;

    // 默认单序列：当用户没提供 seq_id 时，每个 token 都指向 {0}。
    std::array<llama_seq_id, 1> seq_id_0 = {{ 0 }}; // default sequence id

    // 内部生成的 llama_pos 数组，
    // 在用户未提供 batch.pos 时，按 memory 里各 seq 的 seq_pos_max （或从 0） 递推出每条 token 的位置。
    std::vector<llama_pos>      pos;

    // 与 llama_ubatch 里同名字段同角色：
    //   每条 token 绑定几个序列、指针数组、去重后的 seq_id 列表，
    //   以及 seq_id → 在 seq_id_unq 里的下标（池化 embedding 等会用到）
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id *> seq_id;
    std::vector<llama_seq_id>   seq_id_unq;
    std::vector<int32_t>        seq_idx;

    // 当用户未提供 batch.logits 时，内部生成的 int8 布尔表：要么全 true（output_all），
    // 要么仅最后一个 token 为 true（常见自回归「只对最后一个算 logits」）。
    std::vector<int8_t>         output;

    using pos_set_t = std::set<llama_pos>;
    using seq_cpl_t = std::vector<bool>;

    // helper flag to quickly determine if there are any coupled sequences in the batch
    // 本 batch 里是否存在 「耦合序列」：
    //    同一 token 行挂多个 seq_id（一条前向被多条序列共享），
    //    split_equal(..., sequential) 时与 -kvu 等策略有关
    bool has_cpl = false;

    // 序列 s 在本 batch 里出现过的 所有 position，用于检查连续性、与 memory 对齐、多维 RoPE 约束等。
    std::vector<pos_set_t> seq_pos; // seq_pos[s]: the set of positions in sequence s
    // 耦合矩阵：若某条 token 同时属于 s0 和 s1，则标记两条序列在该 batch 中 通过共享 token 耦合；
    // 后面会用来检查耦合序列在 memory 里 position 范围是否一致等。
    std::vector<seq_cpl_t> seq_cpl; // seq_cpl[s0][s1]: if sequence s0 is coupled to sequence s1

    using idx_vec_t = std::vector<int32_t>;
    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

    // 第 i 个 token 所属的 序列 id 集合 的紧凑表示（多 seq 广播时一位可能被置多位）。
    std::vector<seq_set_t> seq_set; // seq_set[i]: the sequence set of token i

    // seq_set 的 bit pattern → 具有相同序列集合的 batch 下标列表。切 ubatch 时按「序列集合」成组取用，避免把不该放在一起的 token 混进同一微批。
    std::unordered_map<seq_set_t, idx_vec_t> seq_set_map; // the indices at which the sequence set appears

    // batch indices of the output
    // 在当前切分顺序下，各微批里「带 logits/output」的 token 在 原始 batch 中的下标，
    // 按 切分过程中遇到的顺序 追加（和 llama_context 里 output_ids、取 logits 行对齐有关）
    std::vector<int32_t> out_ids;

    // 已通过 split_simple / split_equal / split_seq 等 消耗掉的 token 个数（used[i]=true 的计数），
    // 用于判断是否整批已切完。
    uint32_t n_used;

    // used[i] indicates if token i has already been used in a previous ubatch
    // 第 i 个 token 是否已经划入某个已生成的 llama_ubatch；split_reset() 会清空并重置为全 false。
    std::vector<bool> used;

    int debug;
};
