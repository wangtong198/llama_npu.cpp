# llama.cpp 中 `llama_kv_cache` 的设计与实现梳理

本文围绕 `src/llama-kv-cache.h` / `src/llama-kv-cache.cpp` 中的 `llama_kv_cache` 展开，目标是系统回答三个问题：

1. `llama.cpp` 里的 KV cache 到底由哪些对象管理。
2. `llama_kv_cache` 的成员变量和成员函数分别负责什么。
3. 整个设计为什么是现在这个分层，以及主链路是如何串起来的。

---

## 1. 总体定位

`llama_kv_cache` 是 `llama_memory_i` 的一种具体实现：

- `llama_memory_i` 表示“模型前向过程中可被增量维护的 memory 抽象”。
- 对普通 decoder / attention 类模型，这个 memory 主要就是 KV cache。
- 对 recurrent / hybrid 模型，则会有别的 memory 实现。
- `llama_context` / `decode()` 并不直接操作某个具体 KV 结构，而是通过 `llama_memory_i` / `llama_memory_context_i` 这层抽象来做。

也就是说，`llama_kv_cache` 在工程里的角色不是“某个小工具类”，而是：

- 一方面，它管理真正存放 K/V 向量的 ggml 张量和后端 buffer。
- 另一方面，它还管理槽位元数据，比如哪些 cell 被谁占用、位置是多少、head 从哪里开始找空槽、是否有 shift 待应用等。
- 再一方面，它提供图构建所需的视图和输入 tensor，让 attention 在前向图里读历史 KV、写当前 KV。
- 最后，它还负责状态落盘 / 恢复。

因此，`llama_kv_cache` 实际上同时承担了四层职责：

1. 物理存储层：K/V tensor 和 buffer。
2. 元数据层：`v_cells` / `v_heads` / `seq_to_stream`。
3. 批处理准备层：为 `ubatch` 找槽位、预演、提交。
4. 图构建接口层：为 attention 提供 `get_k/get_v/cpy_k/cpy_v` 和各类输入。

---

## 2. 相关对象之间的层次关系

先看几个核心对象。

### 2.1 `llama_memory_i`

定义在 `src/llama-memory.h`。

它规定了 memory 子系统需要提供的统一接口：

- `init_batch()`：把一个 batch 切成若干 `ubatch`，并为每个 `ubatch` 准备 memory 状态。
- `init_full()`：构造“满 cache”的上下文，用来预留最坏情况 graph / buffer。
- `init_update()`：对 pending 的 shift / copy 等做真正更新前的准备。
- `seq_rm/seq_cp/seq_keep/seq_add/seq_div`：对 sequence 级别的 memory 做操作。
- `seq_pos_min/seq_pos_max`：查询某个 sequence 当前在 memory 里的位置边界。
- `state_write/state_read`：序列化与反序列化。

`llama_kv_cache` 就是这套接口的 KV 版本实现。

### 2.2 `llama_kv_cache`

这是本文主角。它实现了：

- KV tensor 的分配与持有。
- KV cell 元数据。
- `ubatch -> slot` 映射。
- attention 图构建所需的 KV 视图。
- KV 状态 save/load。

### 2.3 `llama_kv_cells`

定义在 `src/llama-kv-cells.h`。

它不存 K/V 向量本身，而是存每个 cell 的元数据：

- `pos[i]`：第 `i` 个 cell 的逻辑位置。
- `seq[i]`：第 `i` 个 cell 当前属于哪些 `seq_id`。
- `seq_pos[s]`：序列 `s` 当前在 cache 中出现了哪些 `pos`，且用计数而不是 set。
- `shift[i]`：位置平移累计量。
- `used`：哪些 cell 当前非空。

可以把它理解成：

- `layers[*].k / v` 是“真正的数据面”。
- `v_cells[*]` 是“控制面 / 目录索引”。

### 2.4 `llama_kv_cache_context`

它是 `llama_memory_context_i` 的 KV 实现。

`llama_kv_cache` 是长期存在的 cache 对象，而 `llama_kv_cache_context` 是一次 batch/update 过程中的临时上下文，负责：

- 顺序遍历当前要处理的 `ubatch`。
- 对当前 `ubatch` 调 `apply()`，把它的槽位元数据提交到 KV cache。
- 持有当前 `ubatch` 对应的 `slot_info`。
- 提供 `get_k/get_v/cpy_k/cpy_v` 这组“当前 `ubatch` 视角”的转发接口。

---

## 3. `llama_kv_cache` 的核心嵌套结构

### 3.1 `stream_copy_info`

```cpp
struct stream_copy_info {
    std::vector<uint32_t> ssrc;
    std::vector<uint32_t> sdst;
};
```

作用：记录跨 stream 的待执行拷贝。

典型场景是 `seq_cp(src, dst, ...)`：

- 如果 `src` 和 `dst` 在同一个 stream，只要改 metadata，不需要真拷贝 K/V buffer。
- 如果在不同 stream，则需要把整条 stream 的 K/V buffer 从 `ssrc` 拷到 `sdst`。
- 这个拷贝不会立刻做，而是先记录到 `sc_info`，等下一次 `init_update()` / `update()` 时统一执行。

这是一个典型的“先改控制面，后批量改数据面”设计。

### 3.2 `slot_info`

这是 `ubatch` 在 KV cache 里的落点描述。

关键字段：

- `s0/s1`：当前 `ubatch` 覆盖的 stream 范围。
- `strm[s]`：第 `s` 路对应的 stream id。
- `idxs[s]`：该 stream 上每个 token 最终落到哪些 cell 下标。

可以把它理解成：

- `ubatch` 描述“这批 token 是谁”。
- `slot_info` 描述“这批 token 应该写到 cache 的哪里”。

后面的：

- `get_k/get_v()` 用它决定读取哪个 stream、从哪里开始看。
- `cpy_k/cpy_v()` 用它决定写到哪些全局 cell 下标。
- `apply_ubatch()` 用它更新元数据。

### 3.3 `kv_layer`

这是“KV cache 里的一层”的描述，注意它不一定和模型层号一一对应。

字段：

- `il`：模型中的层号。
- `k` / `v`：该层的 K/V 父张量，形状都是 3D：
  - K：`[n_embd_k_gqa, kv_size, n_stream]`
  - V：`[n_embd_v_gqa, kv_size, n_stream]`
- `k_stream` / `v_stream`：从父张量按 stream 维切出来的 2D view：
  - `[n_embd_*, kv_size]`

这里有两个关键点：

1. `layers` 是“cache 层数组”而不是“模型层数组”。因为有 `filter` 和 `reuse`：
   - 某些模型层可以不参与 cache。
   - 某些层可以复用别的层的 cache 存储。
2. 真正的 K/V 向量都存在这里，不在 `v_cells` 里。`v_cells` 只知道 cell `i` 里是谁、位置是什么，但不存向量。

### 3.4 `cell_ranges_t`

用于 save/load 阶段，表示某个 stream 上被选中的若干连续 cell range。

为什么需要 range，而不是直接存一大串 idx？

- 因为写文件时，连续 range 可以直接做连续 tensor I/O。
- 这样能走更大的块状读写，比一个 cell 一个 cell 地 scatter 高效。

---

## 4. `llama_kv_cache` 的成员变量详解

下面按功能分层说明。

### 4.1 与模型和配置绑定的只读成员

#### `const llama_model & model`

整个 KV cache 依附的模型对象。

它提供：

- 每层放在哪个 device 上：`model.dev_layer(il)`。
- rope 参数。
- 层数、架构、是否 MLA 等模型属性。

#### `const llama_hparams & hparams`

模型超参数引用，频繁使用：

- `n_layer` / `n_layer_kv()`。
- `n_embd_k_gqa(il)` / `n_embd_v_gqa(il)`。
- `n_head_kv(il)`。
- `n_embd_head_k(il)` / `n_embd_head_v(il)`。
- `n_pos_per_embd()`。
- `rope_type` / SWA / ALiBi 等。

### 4.2 cache 布局和模式参数

#### `bool v_trans`

表示 V cache 是否采用转置布局。

- `false`：通常是 Flash Attention 场景，V 的布局较直观。
- `true`：非 FA 路径里，V 会采用转置后的写法，因此：
  - `get_v()` 的视图逻辑不同。
  - `cpy_v()` 也有专门的 transposed 分支。
  - `state_write_data()` / `state_read_data()` 也要单独处理。

它是整个类里最重要的布局分支之一。

#### `const uint32_t n_seq_max`

支持的最大 sequence 数。

#### `const uint32_t n_stream`

物理 stream 数。

核心关系：

- `unified == true` 时，`n_stream = 1`。
- `unified == false` 时，`n_stream = n_seq_max`。

这意味着：

- unified：所有 sequence 共用一套物理 KV ring。
- non-unified：每个 sequence 各有自己的一套 stream。

#### `const uint32_t n_pad`

KV 长度在图上会按这个值对齐；`get_n_kv()` 里还会进一步和 `256` 取大值，以便：

- 保持 graph 形状稳定，利于 graph 复用。
- 某些 backend 上性能更好。

#### `const uint32_t n_swa`

SWA window 大小相关参数。

#### `const llama_swa_type swa_type`

cache 这一侧采用的 SWA 语义。

注意它不等于“模型本身的 SWA 类型”，而是 cache 在 masking / slot 选择时使用的 SWA 规则。

### 4.3 与 K-shift / 旋转优化相关的成员

#### `bool attn_rot_k`
#### `bool attn_rot_v`

是否为 K/V 启用 attention rotation。

#### `int32_t n_embd_head_k_all`
#### `int32_t n_embd_head_v_all`

如果所有参与 cache 的层 head 维度一致，则记录该公共值；否则设为 `-1`。

用途：

- 用于判断是否能构造统一大小的 Hadamard rotation 输入。
- 避免对“每层 head 维度不一致”的模型做错误的统一处理。

#### `std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard`

预计算的 Hadamard 矩阵，后续通过 `set_input_k_rot()` / `set_input_v_rot()` 直接灌到 graph input tensor。

### 4.4 调试和内存统计相关

#### `int debug`

来自 `LLAMA_KV_CACHE_DEBUG` 环境变量，控制 `find_slot()` 等处的调试输出。

#### `std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs`

每个 buffer type 对应一个 ggml context + backend buffer。

构造时会按 `buft` 聚合层：

- 同一种 buffer type 的层共享同一个 ggml context。
- 最后一次性分配 buffer。

这样减少了 context 和 buffer 的碎片化。

### 4.5 核心运行时状态

#### `std::vector<uint32_t> v_heads`

每个 stream 一根“找空槽起始位置”的 head。

它不是 KV 逻辑状态的一部分，而是搜索优化辅助状态。

用途：

- `find_slot()` 从 `v_heads[strm]` 开始向后扫。
- `apply_ubatch()` 提交后把 head 移到当前写入范围之后。
- `seq_rm` / `seq_keep` / `seq_add` 释放槽位时可能把 head 回退到更早的空位。

#### `std::vector<llama_kv_cells> v_cells`

每个 stream 一套 cell 元数据。

这是 `llama_kv_cache` 的控制核心。

其中每个 `llama_kv_cells` 里保存：

- 哪些 cell 为空/非空。
- 每个 cell 的逻辑位置。
- 每个 cell 属于哪些 seq。
- 各个 seq 当前的 min/max pos。
- 是否存在 shift。

#### `std::vector<uint32_t> seq_to_stream`

sequence id 到 stream id 的映射。

两种典型情况：

- unified：所有 `seq_id -> 0`
- non-unified：`seq_id -> seq_id`

这层映射让上层逻辑统一基于 `seq_id` 工作，而内部可以决定物理落在哪个 stream。

#### `stream_copy_info sc_info`

待执行的跨 stream buffer 拷贝队列。由 `seq_cp()` 入队，由 `update()` 真正执行。

### 4.6 与层张量和层映射有关的成员

#### `std::vector<kv_layer> layers`

cache 真正参与存储的层集合。

这是经过 `has_kv` / `filter` / `reuse` 处理后的紧凑层数组。

#### `std::unordered_map<int32_t, int32_t> map_layer_ids`

模型层号 `il` -> `layers` 下标 `ikv` 的映射。

`get_k/get_v/cpy_k/cpy_v` 都先通过它找到对应的 cache 层。

这是因为：

- 模型层不一定全部有 KV。
- 某些层可能被过滤掉。
- 某些层可能复用别的层的 cache。

---

## 5. 构造函数做了什么

构造函数是整个类最密集的一段初始化逻辑。主线如下。

### 5.1 决定 stream 模式

```cpp
n_stream(unified ? 1 : n_seq_max)
```

即：

- unified：所有序列共享一个物理 stream。
- non-unified：每个序列一个 stream。

然后初始化：

- `v_heads.resize(n_stream)`，全部置 0。
- `v_cells.resize(n_stream)`，每个 `v_cells[s]` 再 `resize(kv_size)`。

### 5.2 初始化 `seq_to_stream`

- 默认先把 `LLAMA_MAX_SEQ` 个序列都映射到 0。
- 若 `n_stream > 1`，则改成 `seq_to_stream[s] = s`。

### 5.3 为不同 buffer type 建 ggml context

构造中有一个 `ctx_map<buft, ggml_context_ptr>`：

- 同一 `buft` 的层复用同一个 context。
- 最终再为该 context 上挂的 tensor 一次性分配 backend buffer。

这样做的好处是：

- 避免每层一个独立 context/buffer。
- 更适合多设备、多层混合 offload。

### 5.4 为每个参与 cache 的模型层分配 K/V tensor

对每个模型层 `il`：

1. 若 `!hparams.has_kv(il)`，跳过。
2. 若 `filter` 存在且 `filter(il)==false`，跳过。
3. 确定这层放在 CPU 还是 `model.dev_layer(il)` 对应设备。
4. 创建：
   - `k = [n_embd_k_gqa, kv_size, n_stream]`
   - `v = [n_embd_v_gqa, kv_size, n_stream]`
5. 为每个 stream 建 2D view：
   - `k_stream[s]`
   - `v_stream[s]`
6. 填入 `layers` 和 `map_layer_ids`。

若模型是 MLA，`has_v = !is_mla`，因此 MLA 可能没有 V tensor。

### 5.5 处理 layer reuse

如果提供了 `reuse(il)`：

- 某个层可以声明“复用另一个层的 KV cache”。
- 这时不会重新分配 tensor，而是让 `map_layer_ids[il]` 指向已有 `ikv`。

因此：

- `layers.size()` 不一定等于 `hparams.n_layer`。
- 也不一定等于 `hparams.n_layer_kv()`。

### 5.6 分配 backend buffer 并清零

最后把各 `ctx` 上挂的 tensor 真正分配到 buffer 中：

- `ggml_backend_alloc_ctx_tensors_from_buft(...)`
- `ggml_backend_buffer_clear(buf, 0)`

到这里，KV cache 的“物理空间”就准备好了。

---

## 6. 对外 memory 操作接口的逻辑

这部分对应 `llama_memory_i` 的一组 sequence 操作。

### 6.1 `clear(bool data)`

作用：

- 重置所有 `v_cells`。
- 把所有 `v_heads` 置 0。
- 若 `data == true`，还会把实际 K/V buffer 也清零。

因此它同时支持：

- 只清 metadata。
- metadata + tensor 数据一起清。

### 6.2 `seq_rm(seq_id, p0, p1)`

按位置区间删除某条 sequence 的 cache 条目。

两种模式：

- `seq_id >= 0`：只删该 sequence 在 `[p0, p1)` 中的出现。
- `seq_id == -1`：匹配所有 sequence，等价于按位置区间整片清掉。

删除过程中如果释放出了更早的槽位，会把对应 `head` 往前回退，以便后续 `find_slot()` 更快复用。

### 6.3 `seq_cp(seq_id_src, seq_id_dst, p0, p1)`

复制 sequence。

#### 同 stream

只改 metadata，不拷 K/V 数据：

- 找到所有属于 `seq_id_src` 的 cell。
- 给这些 cell 追加 `seq_id_dst`。

因为它们本来就在同一套物理 K/V stream 上，不需要真复制张量内容。

#### 跨 stream

这是较复杂的分支：

- 要求是“整条 KV buffer 复制”，不支持任意子区间。
- 先把 `(src_stream, dst_stream)` 加到 `sc_info`。
- 然后立即重建 `v_cells[dst]` 的元数据。
- 真正的 K/V tensor copy 在下一次 `update()` 中做。

这是个典型的“控制面先行、数据面延迟”的设计。

### 6.4 `seq_keep(seq_id)`

只保留某个 sequence，删掉该 stream 上不属于它的部分。

内部通过 `cells.seq_keep(i, seq_id)` 实现：

- 若 cell 本来包含该 seq，则把该 cell 的 seq 集合压缩成只剩它。
- 若 cell 不包含该 seq，但原先有别的 seq，则整格清空。

### 6.5 `seq_add(seq_id, p0, p1, shift)`

对某个序列在 `[p0, p1)` 范围内的 position 做整体平移。

底层调用 `cells.pos_add(i, shift)`：

- 修改 `pos[i]`。
- 累计 `shift[i]`。
- 同时更新 `seq_pos` 索引。
- 若位置变成负数，则 cell 直接失效并释放。

这个接口主要用于上下文滑窗、位置平移一类场景。

### 6.6 `seq_div(seq_id, p0, p1, d)`

对某个序列的 position 做整除缩放。

底层调用 `cells.pos_div(i, d)`，同样会累计到 `shift` 中。

### 6.7 `seq_pos_min/seq_pos_max`

查询某个序列在 cache 中当前出现的最小/最大位置。

实现逻辑：

1. 先用 `seq_to_stream[seq_id]` 找到它归属的 stream。
2. 再对该 stream 的 `v_cells` 调 `cells.seq_pos_min(seq_id)` / `cells.seq_pos_max(seq_id)`。

这也是为什么 `llama_kv_cells` 里要维护 `seq_pos[seq_id]` 这张有序计数表。

---

## 7. batch 处理主链

这部分是 `llama_kv_cache` 最核心的运行时主链。

### 7.1 `init_batch()`

输入：

- `llama_batch_allocr & balloc`
- `n_ubatch`

它做两件事：

1. 把 batch 切成若干 `llama_ubatch`。
2. 为每个 `ubatch` 预先找到可写入 cache 的 `slot_info`。

切分策略：

- `n_stream == 1`：`balloc.split_simple(n_ubatch)`
- `n_stream > 1`：`balloc.split_equal(n_ubatch, true)`

之所以 multi-stream 要 `split_equal(..., true)`，是因为：

- 每个 stream 实际对应一个 sequence。
- 需要保证 `ubatch` 中各 stream 对齐，便于后续 `slot_info` 和图输入构建。

若：

- batch 不能被完全切掉。
- 或者 `prepare(ubatches)` 失败。

就返回一个 `FAILED_PREPARE` 的 `llama_kv_cache_context`。

### 7.2 `prepare(ubatches)`

这是一个很重要的“试排布”阶段。

它并不真正提交，只是：

1. 对每个 `ubatch` 调 `find_slot(ubatch, false)` 找可用槽。
2. 把当前 `v_heads` 与相关 `v_cells` 片段备份起来。
3. 临时 `apply_ubatch()`，模拟“如果真的写进去会怎样”。
4. 全部成功后，再按逆序把 metadata 恢复回原状。

为什么要这么做？

- 后续 `ubatch` 的找槽位，依赖前面 `ubatch` 已经占掉了哪些槽。
- 但真正提交前又不能破坏当前 cache。

因此需要：

- 先模拟。
- 若全成功，再在运行时逐个 `apply()`。

### 7.3 `find_slot(ubatch, cont)`

这是“为一个 `ubatch` 找落点”的核心算法。

主要逻辑：

1. 先确定这个 `ubatch` 有几路 stream、每路多少 token。
2. 对每个参与的 seq/stream：
   - 从 `v_heads[strm]` 起开始扫。
   - 优先找空 cell。
   - 若 cell 非空但满足 SWA 覆盖条件，也可复用。
3. 得到每路 stream 的 `idxs[s]`。

`cont` 决定是否必须连续。

当前 `prepare()` 用的是 `false`，表示：

- 不要求一整段连续，只要能给每个 token 找到可用 cell 即可。

### 7.4 `apply_ubatch(sinfo, ubatch)`

这是“真正提交当前 `ubatch` 元数据”的函数。

它会：

1. 遍历 `sinfo` 指定的目标 cell。
2. 如果目标 cell 原先非空，记录被覆盖掉的 sequence 的最大位置。
3. 清掉旧 cell。
4. 写入新的：
   - `pos`
   - 2D `ext`（若是 M-RoPE）
   - `seq_id`
5. 最后为所有被覆盖的 sequence 做一次 purge，保证：
   - 对每个 sequence，cache 中 `[pos_min, pos_max]` 之间的位置仍然连续存在。
   - 不会留下“中间缺失、后面还有更大位置”的破坏性状态。
6. 把 `v_heads` 移到本次写入区间的尾后。

这一步只更新 metadata；真正的 K/V tensor 写入在 graph 执行时通过 `cpy_k/cpy_v` 完成。

---

## 8. update 主链：shift 和 stream copy

### 8.1 `init_update()`

它会生成一个 `llama_kv_cache_context`，携带：

- `do_shift = get_has_shift()`
- `sc_info = std::move(sc_info)`

如果：

- 没有 shift。
- 也没有 pending stream copy。

那么返回的 context 状态会是 `LLAMA_MEMORY_STATUS_NO_UPDATE`。

### 8.2 `update(lctx, do_shift, sc_info)`

分两块。

#### 先处理 `sc_info`

如果有跨 stream copy：

- 先 `llama_synchronize(lctx)`。
- 然后逐层 `ggml_backend_tensor_copy(k_stream[src], k_stream[dst])`。
- V 也同理。

#### 再处理 shift

如果 `do_shift == true`：

- 先检查 `get_can_shift()`。
- 然后构建一张专门的 K-shift graph。
- 运行 `graph_compute`。
- 最后对每个 stream `cells.reset_shift()`。

这里的“shift 应用”不是简单改 metadata，而是要把 cache 中已有的 K 向量按新的 position 做 RoPE shift，因此必须真正跑一张 compute graph。

---

## 9. 图构建相关 API

这组函数把 KV cache 接进 attention graph。

### 9.1 `get_n_kv(sinfo)`

返回当前 `ubatch` 实际要参与 attention 的 KV 长度。

它不是简单返回 `cells.used_max_p1()`，而是：

- 至少按 `n_pad_cur = max(n_pad, 256)` 对齐。
- 并且不超过 cache 总大小。

目的：

- graph 形状尽量稳定，便于复用。
- 某些 backend 性能更好。

### 9.2 `get_k()` / `get_v()`

返回 cache 中“当前可见状态”的 tensor view。

不是复制，而是 view。

#### `get_k()`

视图逻辑：

- 从父张量 `layer.k` 里切出 `[n_embd_head_k, n_head_kv, n_kv, ns]`。

#### `get_v()`

分 `v_trans` 两条路径：

- 非转置：`[n_embd_head_v, n_head_kv, n_kv, ns]`
- 转置：`[n_kv, n_head_kv, n_embd_head_v, ns]`

这正是为什么 `v_trans` 会影响后续所有读写逻辑。

### 9.3 `cpy_k()` / `cpy_v()`

作用：把当前 step 刚算出的 `k_cur` / `v_cur` scatter 写回 cache。

共同点：

- 最终都用 `ggml_set_rows`。
- `*_idxs` 是“全局行索引”。

不同点：

#### `cpy_k()`

比较简单：

- 把 `k_cur` reshape 成 2D。
- multi-stream 时把 `k` 也 reshape 成 `[n_embd_gqa, kv_size*n_stream]`。
- 直接 `ggml_set_rows(ctx, k, k_cur, k_idxs)`。

#### `cpy_v()`

分支更多：

- 非 `v_trans`：类似 `cpy_k()`。
- `v_trans`：要把 V 视作“按元素行展开”的大平面，再用专门构造的 `v_idxs` scatter。

### 9.4 `build_input_k_idxs()` / `build_input_v_idxs()`

构造 graph input tensor，占位用。

其中：

- `k_idxs` 长度就是 `n_tokens`
- `v_idxs` 在 `v_trans` 下长度会变成 `n_tokens * n_embd_v_gqa_max()`

### 9.5 `set_input_k_idxs()` / `set_input_v_idxs()`

在真正执行 graph 前，把 `slot_info` 变成写回索引。

统一思路：

- 全局索引 = `stream_offset + cell_idx`

其中：

- K 的 offset 是 `strm * kv_size`
- V 转置路径下，offset 还要再乘 embedding 维度，并展开成 `(token, embd)` 级别的散点索引。

### 9.6 `set_input_kq_mask()`

这是 attention mask 的填充逻辑。

它综合考虑：

- 当前 `ubatch`
- `v_cells`
- `seq_to_stream`
- causal mask
- SWA mask
- M-RoPE 的 2D 位置关系
- ALiBi

它本质上回答的是：

当前 query token 能看见 cache 里的哪些历史 cell。

### 9.7 `set_input_pos_bucket()`

为相对位置 bucket 模型填输入。

目前只支持 `n_stream == 1`。

### 9.8 `build_input_k_rot()` / `build_input_v_rot()`
### 9.9 `set_input_k_rot()` / `set_input_v_rot()`

这几组函数负责把预先算好的 Hadamard rotation matrix 作为 graph 输入送进去。

### 9.10 `build_graph_shift()` / `build_rope_shift()`

这组函数专门服务于 K-shift：

- 遍历所有 cache 层。
- 把已有 K cache 视作 tensor。
- 构建按 shift 重算 rope 的 graph。
- 在 `update()` 中执行。

---

## 10. 状态落盘与恢复

这是 `llama_kv_cache` 的另一个大职责。

### 10.1 整体协议

`state_write()` / `state_read()` 处理的是：

1. 每个 stream 上哪些 cell 被用到了。
2. 这些 cell 的 metadata：
   - pos
   - n_seq_id
   - ext（可选）
   - seq_id 列表
3. 各层 K/V 张量上对应 cell 的实际字节。

### 10.2 `state_write()`

对每个 stream：

1. 统计 `cell_count`
2. 收集被使用的连续 range
3. 写：
   - `cell_count`
   - `state_write_meta(...)`
   - `state_write_data(...)`

支持：

- `seq_id == -1`：全 cache
- `seq_id >= 0`：只写某个 sequence 相关的 cell

### 10.3 `state_write_meta()`

写每个 cell 的元数据：

- `pos`
- `n_seq_id`
- `ext`（如果 `n_pos_per_embd() > 1`）
- 该 cell 里的所有 `seq_id`

### 10.4 `state_write_data()`

写 K/V 张量字节。

K 总是“按 cell 行写”。

V 分两种：

- `!v_trans`：按 cell 行写。
- `v_trans`：要按 embedding 维度展开写。

所以 `state_write_data()` 里专门先写：

- `v_trans`
- `n_layer`
- 每层的 type / row size / element size / embd size

以便恢复时做兼容性检查。

### 10.5 `state_read()`

按 stream 读回：

1. `cell_count`
2. `state_read_meta()`
3. `state_read_data()`

任一环节失败就：

- 全恢复模式下 `clear(true)`。
- 单序列恢复模式下 `seq_rm(seq_id, -1, -1)`。
- 然后抛异常。

### 10.6 `state_read_meta()`

有两种恢复方式。

#### 全 cache 恢复

直接：

- `clear(true)`
- 从文件中逐个 `pos_set` / `ext_set` / `seq_add`
- 同时构造一份“连续下标”的 `slot_info`

#### 单序列恢复

不能简单按原始下标原样塞回去，因为目标 stream 上未必还空着。

它会：

- 构造一个 fake `ubatch`
- 把读到的 `pos` / `ext` / `seq_id` 转成该 `ubatch`
- 再调用 `find_slot()` 为它重新找落点
- 然后 `apply_ubatch()`

这说明单序列恢复本质上是“逻辑恢复”，不是“物理原位恢复”。

### 10.7 `state_read_data()`

校验：

- `n_layer`
- `v_trans`
- K/V type
- row size / element size

然后把字节 scatter/set 回各层 tensor。

若 `slot_info.is_contiguous()`：

- 可走连续 memcpy 快路径。

否则：

- 逐 cell scatter。

---

## 11. `llama_kv_cache_context` 做了什么

虽然用户问题主角是 `llama_kv_cache`，但真正让它接到 `llama_context` 前向流程里的，是 `llama_kv_cache_context`。

### 11.1 它的几种构造方式

#### 错误 context

只携带一个 `status`。

#### full-cache context

用于 `init_full()`，构造一个“最坏情况视角”的上下文，主要给 graph reserve 用。

#### update context

用于 `init_update()`，里面不持有 `ubatches`，而是持有：

- `lctx`
- `do_shift`
- `sc_info`

#### batch context

用于 `init_batch()`，里面持有：

- `sinfos`
- `ubatches`

### 11.2 `next()`

推进到下一个 `ubatch`。

### 11.3 `apply()`

分两种：

- 如果 `ubatches.empty()`：说明这是 update context，执行 `kv->update(...)`
- 否则：执行当前 `ubatch` 的 `kv->apply_ubatch(...)`

并在 batch 模式下更新 `n_kv = kv->get_n_kv(sinfos[i_cur])`。

### 11.4 `get_ubatch()` / `get_n_kv()`

- `get_ubatch()`：返回当前 micro-batch
- `get_n_kv()`：返回当前 micro-batch 对应的 attention 可见 KV 长度

### 11.5 其余接口

它基本上是一个“当前 `ubatch` 视角的代理层”：

- `get_k/get_v`
- `cpy_k/cpy_v`
- `build_input_*`
- `set_input_*`

都只是把当前 `sinfo` / `n_kv` 封装进去后转发给 `llama_kv_cache`。

因此可以把 `llama_kv_cache_context` 理解成：

- `llama_kv_cache`：全局 KV 管理器
- `llama_kv_cache_context`：当前一步 / 当前 ubatch 的 KV 访问句柄

---

## 12. 设计思路与原理总结

最后从更高层把设计意图归纳一下。

### 12.1 数据面与控制面分离

这是整个设计的第一原则。

数据面：

- `layers[*].k / v`
- 各类 backend buffer

负责真正存放大块 K/V 向量。

控制面：

- `v_cells`
- `v_heads`
- `seq_to_stream`
- `slot_info`

负责回答：

- 哪个 token 该落到哪个 cell？
- 这个 cell 现在属于谁？
- 当前某个 sequence 的最大位置是多少？
- 当前注意力 mask 应该怎么看 cache？

好处：

- 不需要频繁搬动大块 K/V 张量。
- 很多操作只动 metadata 即可。
- save/load、slot 分配、purge、sequence copy 都更容易做。

### 12.2 “先预演，再提交”

`prepare()` 的“试排布 + 回滚”是第二原则。

原因：

- 一个大 batch 会切成多个 `ubatch`。
- 后面的 `ubatch` 能否找到槽位，依赖前面的 `ubatch` 占位结果。
- 但真正提交前又不能破坏当前 cache。

因此需要：

- 先模拟。
- 若全成功，再在运行时逐个 `apply()`。

### 12.3 统一抽象 sequence，内部再映射到 stream

对外大家都用 `seq_id` 说话，但内部用 `seq_to_stream` 做物理映射。

好处：

- 上层逻辑不必区分 unified / non-unified。
- KV cache 自己决定：
  - 所有序列共用一套物理 ring。
  - 还是一条序列一路 stream。

### 12.4 把 graph 构建也纳入 KV cache 的职责

`llama_kv_cache` 不只是“内存容器”，它还直接参与 graph 构建：

- 提供 `get_k/get_v` 视图。
- 提供 `cpy_k/cpy_v` 写回节点。
- 提供 `k_idxs/v_idxs/kq_mask/pos_bucket/rot` 等输入填充。

这意味着：

- KV layout 的选择，例如 `v_trans`，只需要在这一层统一处理。
- attention 构图代码无需了解底层 buffer 是如何组织的。

### 12.5 save/load 设计成“元数据 + 数据字节”双通道

若只存字节，不知道 cell 对应的 `pos/seq_id`；
若只存 metadata，又恢复不了真正的 K/V 向量。

所以协议拆成：

- `state_write_meta` / `state_read_meta`
- `state_write_data` / `state_read_data`

前者恢复逻辑形态，后者恢复数值内容。

这也是它能支持：

- 全 cache 原位恢复
- 单序列逻辑恢复

的根本原因。

---

## 13. 一句话总结

`llama_kv_cache` 的本质不是“某个 K/V 张量数组”，而是：

一个把物理 K/V 存储、cell 元数据、sequence/stream 映射、ubatch 槽位分配、attention 图接入、以及状态持久化统一封装起来的 KV memory 子系统。

如果继续往下追主链，最值得结合阅读的几个点是：

- `llama_context::decode()` 如何通过 `memory->init_batch()` 拿到 `llama_kv_cache_context`
- `process_ubatch()` / graph build 如何调用 `get_k/get_v/cpy_k/cpy_v`
- `llama_kv_cells` 如何维护 `seq_pos`、`shift` 与 `seq` 的一致性
