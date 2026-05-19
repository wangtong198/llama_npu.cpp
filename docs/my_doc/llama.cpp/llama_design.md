# Step 1: `ggml_tensor` 结构说明

`ggml_tensor` 是 ggml 里最核心的数据结构。它不只是“张量数据”的描述符，同时也是**计算图里的一个节点对象**：既保存张量的形状、步长、数据地址，也保存这个节点对应的算子类型、输入依赖、view 关系等信息。后面分析 Qwen2.5 的计算图时，几乎所有 node 都是围绕这个结构展开的。

下面按字段说明其含义。

## 1. 数据类型与存储归属

### `enum ggml_type type`
- 表示张量的数据类型。
- 例如 `F32`、`F16`、`BF16`、`I32`，以及各种量化类型如 `Q4_0`。
- 它决定：
  - 一个底层存储块占多少字节；
  - 一个底层存储块对应多少个逻辑元素；
  - 这个张量在计算时该按什么数据格式解释。

### `struct ggml_backend_buffer * buffer`
- 表示这块张量数据当前挂在哪个 backend buffer 上。
- 可以理解为“这块数据实际归谁管理、放在哪块后端内存里”。
- 例如 CPU buffer、CUDA buffer、Metal buffer 等。
- 如果张量只是图中的一个逻辑节点、还没真正分配后端存储，这个字段可以为空。

## 2. 形状与步长

### `int64_t ne[GGML_MAX_DIMS]`
- `ne` 是每一维的元素个数，`ne` = number of elements。
- ggml 统一按最多 4 维张量处理，常记为：
  - `ne[0]`：第 0 维长度
  - `ne[1]`：第 1 维长度
  - `ne[2]`：第 2 维长度
  - `ne[3]`：第 3 维长度
- 在 llama.cpp 里，大部分张量都用这 4 个维度来表达，不足 4 维的高维通常为 1。

### `size_t nb[GGML_MAX_DIMS]`
- `nb` 是每一维的字节步长，`nb` = number of bytes。
- 它表示：某一维下标加 1 时，底层地址要前进多少字节。
- 这是理解 `view`、`permute`、`transpose`、`reshape` 是否真的搬数据的关键。

需要特别注意：

- `nb[0]` **不是总等于一个逻辑元素的字节数**。
- 对普通类型如 `F32`、`I32`，一个存储块只对应 1 个逻辑元素，所以 `nb[0]` 可以直接看成单元素字节数。
- 对量化类型，底层是“按 block 存”的，一个 block 对应多个逻辑元素，因此：
  - `ggml_type_size(type)`：一个存储块占多少字节；
  - `ggml_blck_size(type)`：一个存储块对应多少个逻辑元素。
- 所以量化张量里，`nb[0]` 更准确地说是**底层最小存储单元的字节大小**，不等于单个逻辑元素大小。

可以把 `nb` 理解为“地址计算规则”：

- `nb[0]`：沿第 0 维移动的最小字节跨度
- `nb[1]`：跨过一整段 `ne[0]` 后前进多少字节
- `nb[2]`：跨过一整段 `ne[1]` 后前进多少字节
- `nb[3]`：跨过一整段 `ne[2]` 后前进多少字节

如果张量是标准连续布局，那么高维步长通常满足逐层乘上前一维大小；如果是 `view`、`transpose`、`permute` 之类的结果，`nb` 可能会被重新排列或改写，此时 shape 看起来没问题，但内存不一定连续。

## 3. 计算图相关字段

### `enum ggml_op op`
- 表示这个张量节点对应的操作类型。
- 例如：
  - 常量/输入张量可视为没有实际算子计算；
  - 中间节点可能是 `GGML_OP_MUL_MAT`、`GGML_OP_RMS_NORM`、`GGML_OP_ROPE`、`GGML_OP_SOFT_MAX` 等。
- 后续执行计算图时，会根据这个字段决定如何计算当前节点。

### `int32_t op_params[...]`
- 保存算子的附加参数。
- 之所以用 `int32_t` 数组，是为了统一对齐和存储。
- 不同算子对它的解释不同，例如：
  - RoPE 会把若干旋转参数写进去；
  - Softmax 会把 scale、ALiBi 等相关参数写进去；
  - view / permute / reshape 类节点也可能借助它保存必要参数。
- 这个字段本身只是原始参数区，**语义取决于 `op`**。

### `int32_t flags`
- 保存张量/节点的一些标志位。
- 用来描述这个节点的额外属性，例如是否为参数、是否需要特殊处理等。
- 单看结构体定义时只能知道“这里是标志位”，具体每一位的语义要结合相关宏和使用位置看。

### `struct ggml_tensor * src[GGML_MAX_SRC]`
- 表示当前节点的输入张量。
- 可以理解为这个计算图节点的“前驱边”。
- 例如：
  - `MUL_MAT(dst = a @ b)` 时，`src` 里会挂 `a` 和 `b`；
  - `RMS_NORM` 时，`src[0]` 通常是待归一化输入；
  - `ROPE` 除输入张量外，还可能有位置相关输入。
- 计算图执行时，会先保证 `src` 对应的输入已经就绪，再计算当前节点。

## 4. view 相关字段

### `struct ggml_tensor * view_src`
- 如果当前张量是另一个张量的 view，这里指向源张量。
- `view` 的核心含义是：**共享底层存储，不复制数据**。
- 所以很多 shape 变换、切片、窗口映射，都是通过 view 实现的。

### `size_t view_offs`
- 表示当前 view 相对于 `view_src` 数据起点的字节偏移。
- 即：这个 view 从源张量底层存储的哪个位置开始看。
- 它和 `nb[]` 一起决定 view 后的逻辑坐标如何映射到底层地址。

## 5. 实际数据与辅助信息

### `void * data`
- 指向当前张量数据的地址。
- 如果是普通已分配张量，它就是实际数据起点。
- 如果是 view，通常可理解为逻辑上的当前数据入口；但真正的共享关系仍要结合 `view_src` / `view_offs` / `buffer` 一起理解。

### `char name[GGML_MAX_NAME]`
- 张量名字。
- 主要用于调试、打印计算图、排查问题。
- 在 llama.cpp 的图构建过程中，很多关键中间张量都会命名，便于定位。

### `void * extra`
- 预留给特定 backend 或扩展逻辑使用的附加信息。
- 例如某些后端实现可能会在这里挂接额外元数据。
- 从通用计算图语义上看，它不是核心字段，但在具体 backend 实现里可能很重要。

### `char padding[8]`
- 结构体对齐用的填充字段。
- 没有独立业务语义，主要是为了内存布局和对齐要求。

## 6. 怎样理解 `ggml_tensor`

后续分析时，可以把 `ggml_tensor` 同时看成三层含义：

1. **张量元信息**
   - `type`
   - `ne[]`
   - `nb[]`

2. **数据存储映射**
   - `buffer`
   - `data`
   - `view_src`
   - `view_offs`

3. **计算图节点信息**
   - `op`
   - `op_params`
   - `src[]`
   - `flags`

所以在 ggml / llama.cpp 里看到一个 `ggml_tensor *`，不能只把它理解成“一个数组”；更准确地说，它是：

- 一个张量的形状与存储描述；
- 一个可能共享底层内存的视图对象；
- 一个带输入依赖和算子类型的计算图节点。

后面讲 Qwen2.5 的构图流程时，所谓“创建一个 node”，本质上通常就是创建一个新的 `ggml_tensor`，填好它的 `op`、`src`、shape / stride / view 信息，然后把它串进整张前向计算图里。

## 7. 和 `torch.Tensor` 的对比

这一小结只对比最核心的几个方面：`shape`、`stride`、底层存储、数据遍历，以及 view / 转置这类“不搬数据的形状变化”。

## 7.1 shape：`torch.size()` vs `ne[]`

在 Torch 里，一个 tensor 的 shape 通常记为：

- `tensor.size()` 或 `tensor.shape`

在 ggml 里，对应的是：

- `ne[0]`、`ne[1]`、`ne[2]`、`ne[3]`

两者的共同点是：

- 都在描述每一维有多少个“逻辑元素”；
- 都不直接等于底层实际内存大小；
- 都只是“如何解释这块数据”的维度信息。

差异主要在表达习惯：

- Torch 的 shape 是更通用的 N 维抽象；
- ggml 固定最多按 4 维描述，大量模型张量都会被折叠或映射进这 4 维。

所以：

- 在 Torch 里常说一个 tensor 是 `[B, S, H, D]`；
- 在 ggml 里也能表达同样含义，但通常会根据具体算子习惯，把最内层计算维放在 `ne[0]`，其他维再依次往外放。

也就是说，`ne[]` 本质上就相当于 ggml 世界里的 shape 数组，只不过：

- 维度顺序约定更强；
- 最大维数固定为 4。

## 7.2 stride：`torch.stride()` vs `nb[]`

这是最需要重点区分的地方。

Torch 里：

- `tensor.stride()` 返回的是**按元素个数计**的 stride。
- 含义是：某一维下标加 1 时，底层地址要跨过多少个“该 tensor 的逻辑元素”。

例如一个连续 `float32` 张量，shape 为 `[3, 4]`，Torch 可能给出 stride：

- `(4, 1)`

意思是：

- 第 0 维前进一步，要跨过 4 个元素；
- 第 1 维前进一步，要跨过 1 个元素。

ggml 里：

- `nb[]` 返回的是**按字节计**的 stride。
- `nb[i]` 表示第 `i` 维下标加 1 时，底层地址前进多少字节。

所以两者的最核心差异是：

- Torch stride 的单位是“元素”
- ggml stride 的单位是“字节”

如果是普通非量化类型，这个换算还比较直接：

- `ggml nb[i] = torch stride[i] * 元素字节数`

但一旦到了 ggml 的量化张量，这个关系就不能再简单按“单元素字节数”去理解，因为：

- ggml 的底层最小存储单元可能是一个 quant block；
- 一个 block 对应多个逻辑元素；
- 所以 `nb[0]` 反映的是底层 block 存储步长，而不是一个逻辑标量的字节大小。

这也是为什么：

- Torch 的 stride 更偏“逻辑张量视角”；
- ggml 的 `nb[]` 更偏“底层物理存储视角”。

## 7.3 data / storage：Torch 的 storage 体系 vs ggml 的 `buffer` + `data`

Torch 里，一个 tensor 背后可以粗略理解为：

- 一块 storage；
- 一个 data pointer / storage offset；
- 一组 size；
- 一组 stride。

ggml 里的对应关系更接近：

- `buffer`：这块数据归属的后端存储区域；
- `data`：当前张量看到的数据地址；
- `view_src` + `view_offs`：如果当前张量是 view，它从哪个源张量、哪个字节偏移开始看。

共同点：

- 两边都把“底层存储”和“如何解释这块存储”分开了；
- 两边都支持多个 tensor/view 共享同一块底层数据；
- 两边都能通过 shape + stride + offset 把同一块内存解释成不同逻辑张量。

差异：

- Torch 更强调一个通用的 storage / storage_offset 抽象；
- ggml 更强调 backend buffer 和具体后端设备归属；
- ggml 的 `buffer` 直接把“数据在哪个 backend 上”作为张量元信息的一部分；
- Torch 虽然也有 device 概念，但它和 storage/stride 的公开使用体验和 ggml 不完全一样。

可以粗略对应成：

- Torch 的 `storage + storage_offset`，类似 ggml 的 `buffer + data/view_offs`
- Torch 的 `size`，类似 ggml 的 `ne[]`
- Torch 的 `stride`，类似 ggml 的 `nb[]`，但单位不同

## 7.4 连续性：Torch 的 contiguous vs ggml 的连续布局

两边都区分“逻辑 shape 正确”和“底层内存连续”这两件事。

Torch 里：

- `permute`、`transpose` 往往只改 size/stride，不搬数据；
- 如果后续算子要求连续，就需要 `contiguous()` 真正重排拷贝。

ggml 里也是同样的思想：

- `VIEW`、`RESHAPE`、`TRANSPOSE`、`PERMUTE` 这类 shape-node，大多只是改解释方式，不搬数据；
- 如果后续硬件执行单元或算子实现要求连续，就需要显式经过 `GGML_OP_CONT`，把数据拷成连续布局。

所以在概念上，两边非常接近：

- Torch 的 `contiguous()`，和 ggml 的 `CONT` 在“把逻辑视图落成真实连续数据”这个作用上是高度相似的。

但 ggml 的实现约束更强，因为它直接服务于图执行和后端内核调度，所以：

- 哪些节点只是 shape 变化；
- 哪些地方必须真的做 copy；
- 后端是否能接受非连续输入；

这些在 ggml / llama.cpp 里通常要比 Torch 用户态更明确地关心。

## 7.5 遍历方式：Torch 更偏逻辑索引，ggml 更强调地址公式

Torch 使用时，用户通常更少直接关心地址计算，更多是：

- 按逻辑索引访问；
- 由框架根据 size/stride 自动换算到底层地址。

ggml 里虽然本质也一样，但在读源码、看算子实现时，你会更频繁地看到“地址公式”思维：

- 某个元素或某一行的数据地址；
- 如何根据 `nb[]` 计算某维切片起点；
- view 的 offset 如何叠加；
- transpose / permute 后为什么 shape 没变错，但地址解释变了。

原因是 ggml 更接近底层执行引擎，很多算子实现都直接围绕：

- `ne[]` 决定循环边界；
- `nb[]` 决定地址步进；
- `data/view_offs` 决定起始地址；

来写。

也就是说：

- Torch 用户更常从“张量语义”思考；
- ggml 源码分析时更常从“循环 + 步长 + 地址”思考。

## 7.6 一个最实用的对应关系

如果你已经熟悉 Torch，可以把一个 `ggml_tensor` 先近似理解成下面这个组合：

- `shape` -> `ne[]`
- `stride` -> `nb[]`（但注意单位是字节，不是元素）
- `storage / storage_offset` -> `buffer + data/view_offs`
- `view tensor` -> `view_src` 指向源张量
- `contiguous()` -> `GGML_OP_CONT`

但还要再补上一层 ggml 特有语义：

- `ggml_tensor` 还是一个**计算图节点**
- 它自带 `op`、`src[]`、`op_params`

这一点和 Torch 的普通 `Tensor` 使用体验不同。Torch 当然也有 autograd graph，但在 ggml 的 C 结构里，这些“图节点属性”就直接和 tensor 放在了一起。

## 7.7 一句话总结

如果只看张量抽象：

- `ne[]` 很像 Torch 的 shape
- `nb[]` 很像 Torch 的 stride
- `buffer/data/view_offs` 很像 Torch 的 storage/data/offset

但 ggml 比 Torch 更底层，尤其体现在两点：

1. `nb[]` 是**字节步长**，而且必须兼容量化 block 存储；
2. `ggml_tensor` 不只是“数据视图”，还是“计算图节点”本身。

---

# Step 2: `llm_build_qwen2` 的计算图构建流程

这一节只围绕 `src/models/qwen2.cpp` 中的 `llm_build_qwen2::llm_build_qwen2` 来讲 Qwen2.5-0.5B 的前向图是如何被搭起来的。

本节采用以下口径：

- 只讲 **普通文本生成**
- 不考虑 **LoRA**
- 不考虑 **控制向量**
- RoPE 统一按 **标准 RoPE** 讲，不展开 YaRN
- 只在必要处提及底层辅助构图函数，不把讲解写成逐行代码解释

## 2.1 模型级固定信息

结合你给的 Qwen2.5-0.5B 配置，这条构图链里的核心维度是：

- `n_layer = 24`
- `hidden_size = 896`
- `num_attention_heads = 14`
- `num_key_value_heads = 2`
- `head_dim = 896 / 14 = 64`
- `intermediate_size = 4864`
- `vocab_size = 151936`
- `rms_norm_eps = 1e-6`
- `rope_theta = 1000000.0`

因此在后面的图里：

- 主隐藏状态通常记为 `[896, n_tokens]`
- Q 的 head 拆分后是 `[64, 14, n_tokens]`
- K / V 的 head 拆分后是 `[64, 2, n_tokens]`
- FFN 的中间通道是 `[4864, ...]`
- logits 最终是 `[151936, n_outputs]`

其中：

- `n_tokens` 表示当前这次构图输入的 token 数
- `n_outputs` 表示最后真正需要输出 logits 的 token 数

这两个量不一定相同。对 Qwen2 的这条图来说，一个非常关键的特点是：

- **前 23 层和最后一层的 attention，都是按全部 `n_tokens` 参与计算**
- **但在最后一层 attention 之后，会提前把只需要产出 logits 的 token 选出来**
- **因此最后一层的 FFN、最终 norm、LM head 只对 `n_outputs` 个 token 继续计算**

这是一处非常重要的图优化。

## 2.2 整体主流程

如果只看主干，这条构图链可以概括为：

1. 构造输入 embedding
2. 构造位置输入 `inp_pos`
3. 构造 attention 所需辅助输入 `inp_attn`
4. 构造输出 token 选择输入 `inp_out_ids`
5. 依次构造 24 层 Transformer
6. 做最终 RMSNorm
7. 做输出投影得到 logits

其中真正的主干数据流是：

`token ids -> hidden states -> 24 层 Transformer -> final hidden states -> vocab logits`

而 `inp_pos`、`inp_attn`、`inp_out_ids` 这三类输入，不是主隐藏状态的一部分，但它们分别为：

- RoPE
- self-attention / KV cache
- 最后输出 token 的筛选

提供必要条件。

## 2.3 模块 1：Word Embedding

### 1. 输入

在普通文本生成场景里，这个模块的输入是：

- token id 序列，长度为 `n_tokens`

对应的动态输入张量本质上是一个一维整数序列。

模型侧静态参数是：

- token embedding 表 `tok_embd`

对 Qwen2.5-0.5B，可以把它理解为一个词表大小为 `151936`、每个 token 映射到 `896` 维向量的查表矩阵。

### 2. 模块作用

这个模块完成一件事：

- 把离散 token id 映射成连续隐藏向量

得到的输出就是后续 Transformer 第 0 层的输入 `inpL`。

### 3. 输出

输出张量形状为：

- `[896, n_tokens]`

可以把它理解成：

- 每个 token 一列
- 每列是这个 token 的 896 维 embedding

### 4. 模块内的额外说明

`build_inp_embd` 其实同时兼容两种输入路径：

- token id 输入
- 外部直接提供 embedding 向量输入

但在本节的普通文本生成场景里，只走 **token id -> embedding 查表** 这一路。

因此本轮可以把这个模块理解为纯粹的 embedding lookup。

### 5. 这一模块的预计算/辅助内容

这一模块本身没有复杂的预计算逻辑，核心是：

- 预先在图里放好输入 token 节点
- 绑定模型自带的 embedding 参数表

然后让图的后续部分统一使用这个 embedding 输出。

## 2.4 模块 2 之前的三类辅助输入

在进入 Transformer 层循环之前，`llm_build_qwen2` 还会构造三类辅助输入，它们虽然不是主模块，但后续每层都会用到。

### 1. `inp_pos`

它保存每个 token 的位置编号。

在本轮口径下：

- 位置 id 连续递增
- 不考虑用户自定义 `position_ids`
- 不考虑 YaRN

因此这里可以把它理解为标准 RoPE 所需的位置序列输入。

它的作用是：

- 在 attention 内，对 Q 和 K 的前 `n_rot = 64` 维注入位置旋转信息。

### 2. `inp_attn`

它不是一张单独的“注意力张量”，而是一组围绕 self-attention 与 KV cache 组织出来的辅助输入。它的核心作用可以概括成三件事：

1. **告诉当前批次的新 K、V 应该写进 KV cache 的哪些槽位**
2. **告诉当前每个 query token 能看见 cache 里的哪些历史 token**
3. **在张量布局不是最直接形式时，提供额外的 K/V 布局适配信息**

这一组输入对后面 prefill / decode 的分析非常关键，因为它把“写入位置”和“可见范围”都提前组织好了，attention 子模块拿到这些输入后，就可以直接完成：

- K/V 写入 cache
- 从 cache 取历史 K/V
- 对合法可见位置做注意力计算

#### 先看第一件事：当前批次的 K / V 写入位置索引是怎么来的

当前批次进入前向图之前，KV cache 侧会先为这批 token 选择一组可写入的槽位。这里的基本思路不是“按 token 顺序永远线性追加”，而是：

- 先看当前这些 token 分别属于哪些 sequence
- 再看这些 sequence 当前被映射到哪个 stream
- 然后去对应 stream 的 KV cell 集合里寻找可用槽位

对每个 sequence 来说，KV cache 都维护了一组 cell 元数据。每个 cell 至少包含：

- 这个位置当前是否为空
- 这个位置存的是哪个 sequence 的 token
- 这个位置对应的逻辑 position 是多少

在给当前 ubatch 分配槽位时，原则上会优先寻找“可以安全复用”的 cell。一个 cell 可以被当前批次拿来写入，通常意味着以下几种情况之一：

- 这个 cell 现在是空的
- 这个 cell 虽然非空，但它对应的旧内容在当前 masking 规则下已经不再需要保留，可以被覆盖

在 Qwen2.5 这里按**标准 RoPE、非 SWA**来讲，所以可以把它理解为最普通的 decoder-only KV 管理：

- 每个 sequence 在各自 stream 的 cache 区域内维护自己的历史 token
- 当前批次的新 token 会被安放到为该 sequence 找到的那些空位/可复用位里

一旦这些槽位被选定，就会形成当前批次的 K 索引和 V 索引。

对 **K 索引** 来说，语义很直接：

- 第 `i` 个新 token 的 K，应该写到该 stream 下的第几个 KV 槽位

对 **V 索引** 来说，语义本质上也是一样的，只是 V cache 的底层排布有两种可能：

- 一种是和 K 类似，按 token 槽位直接索引
- 另一种是为了后续乘法更方便，V cache 在内存里做了转置式布局

如果 V cache 是转置存储，那么“写第 `i` 个 token 的 V”就不再只是一个单一槽位号，而是：

- 要把这个 token 对应的整条 V 向量，映射到转置后布局里的多段目标地址

所以：

- **K 索引的本质，是 token -> KV 槽位**
- **V 索引的本质，也是 token -> KV 存储位置，只是当 V 采用转置布局时，这个映射会更底层一些**

你可以把这一步理解成：在真正执行 attention 之前，系统先回答了一个问题：

- **“这一批新算出来的 K/V，应该落到缓存的哪里？”**

#### 第二件事：当前 token 能看到 cache 里的哪些历史位置

这件事由 attention mask 的生成逻辑决定。它不是简单地“所有历史都可见”，而是要逐个 query token、逐个 cache cell 判断是否合法。

在 Qwen2.5 当前这个普通文本生成场景里，可以把判断规则概括成下面几层。

##### 1. 先要求这个 cache cell 里真的有内容

空 cell 肯定不可见。

也就是说，一个 query token 首先只会考虑那些已经写入过 K 的有效 cache 槽位。

##### 2. 只能看属于同一 sequence 的历史

即使多个 sequence 共存在 KV cache 中，某个 query token 也只能看到和自己属于同一 sequence 的那些历史位置。

这是多请求并发/多 stream 下最基础的隔离条件。

所以 attention mask 的第二层语义是：

- **跨 sequence 的 KV 一律屏蔽**

##### 3. 因果约束：不能看未来位置

对 decoder-only 模型，query 位置 `p1` 只能看 key 位置 `p0 <= p1`。

因此即使某个 cache cell 属于同一 sequence，如果它的逻辑位置在当前 token 的“未来”，也必须屏蔽。

这就是最标准的 causal mask 语义。

##### 4. 如果启用了特殊窗口规则，还要额外收窄可见范围

代码里 attention mask 生成逻辑还兼容 SWA 等窗口化约束，即：

- 即使是同序列、且不是未来位置
- 如果已经落在窗口外，也要屏蔽

但对 Qwen2.5 分析，这一层可以先忽略，这里不考虑 SWA。

#### 第三件事：attention mask 是怎样编码这个“可见/不可见”关系的

mask 本质上是一张“query token x KV 槽位”的可见性表。

对当前批次的每个 query token，都会生成一整行 mask；这一行的长度等于当前可参与 attention 的 KV 槽位总数。

这张表的编码方式非常直接：

- **可见位置写 0**
- **不可见位置写 `-INF`**

这样在后续 attention logits 上加 mask 时：

- 合法位置保持原分数
- 非法位置在 softmax 前就被压成 0 概率

如果模型启用了 ALiBi 一类位置偏置，那么“可见位置”不一定写 0，而是写一个和相对距离有关的偏置值；但 暂时不考虑Alibi这条分支，因此可以直接按：

- **合法 = 0**
- **非法 = `-INF`**

来理解。

#### 这张 mask 的形状该怎么理解

它的本质维度是：

- 当前批次里的 query token
- 当前层可读的 KV cache 槽位

如果有多个 stream，那么会再多一层 stream 维度，把不同 stream 的 token 分开组织。

所以从语义上说，它描述的是：

- **“当前这个 stream 里的第几个 query token，对应能看见哪些 KV 槽位”**

#### 为什么这部分要在 attention 之前单独准备好

因为 attention 真正执行时，需要同时做两件事：

1. 从 cache 中读出历史 K/V
2. 对非法位置做严格屏蔽

如果不提前准备好这些索引与 mask，attention 内核就无法明确知道：

- 当前 batch 的新 K/V 写到哪里
- 当前 query 从哪里读取历史 K/V
- 哪些历史位置应该参与 softmax

因此 `inp_attn` 的本质不是“额外信息”，而是 self-attention 能够正确接上 KV cache 的必要接口层。

### 3. `inp_out_ids`

它表示：

- 最终哪些 token 需要继续走到 logits 输出

这不是 attention 的一部分，而是最后一层中间会用到的“输出 token 选择器”。

它的作用是：

- 避免最后一层 FFN 和 LM head 继续处理那些不需要产出 logits 的 token

因此它服务的是：

- **最后一层的计算裁剪**
- **而不是前 23 层的主干计算**

## 2.5 模块 2：Transformer 模块

这是整个 Qwen2.5-0.5B 图构建的主体部分。

它由 `24` 个结构相同的 block 串联而成。每一层都围绕一个输入隐藏状态 `inpL` 展开，层输出再作为下一层输入。

如果把单层抽象出来，它的主路径可以写成：

1. attention 前 RMSNorm
2. Q/K/V 投影
3. 对 Q/K 做标准 RoPE
4. self-attention
5. attention 残差相加
6. FFN 前 RMSNorm
7. SwiGLU 风格 FFN
8. FFN 残差相加

下面分开说明。

### 2.5.1 单层输入与输出

对第 `il` 层 Transformer：

- 输入：上一层输出 `inpL`
- 输出：当前层输出 `cur`

通常情况下：

- 输入 shape = `[896, 当前参与本层计算的 token 数]`
- 输出 shape = 同上

但最后一层有一个特殊点：

- attention 子模块仍按全部 `n_tokens` 计算
- 但在 attention 之后，会先裁剪成 `[896, n_outputs]`
- 之后 FFN 和层输出就只保留 `n_outputs` 个 token

因此：

- 第 0 到第 22 层：整层输入输出都是 `[896, n_tokens]`
- 第 23 层：attention 仍面向 `[896, n_tokens]`，但 FFN 输入以后变成 `[896, n_outputs]`

### 2.5.2 Attention 前 RMSNorm

每层一开始，先对输入隐藏状态做 RMSNorm：

- 输入：`inpL`
- 权重：该层的 `attn_norm`
- 输出：归一化后的隐藏状态 `cur`

Qwen2.5-0.5B 使用的是：

- RMSNorm
- eps = `1e-6`

这一步不改变 token 数，也不改变 hidden size，所以输出 shape 仍是：

- `[896, 当前 token 数]`

它的作用是：

- 为本层 attention 的 Q/K/V 投影准备稳定的输入特征

### 2.5.3 Q / K / V 投影

归一化后的隐藏状态会并行送入三条线性映射，得到 Q、K、V。

#### Q 投影

- 输入：`[896, 当前 token 数]`
- 输出：`[896, 当前 token 数]`

因为 Q 的总通道数是：

- `14 * 64 = 896`

随后它会被重新解释成：

- `[64, 14, 当前 token 数]`

#### K 投影

- 输入：`[896, 当前 token 数]`
- 输出：`[128, 当前 token 数]`

因为 K 的总通道数是：

- `2 * 64 = 128`

随后它会被重新解释成：

- `[64, 2, 当前 token 数]`

#### V 投影

- 输入：`[896, 当前 token 数]`
- 输出：`[128, 当前 token 数]`

随后它会被重新解释成：

- `[64, 2, 当前 token 数]`

#### 补充说明

Qwen2.5-0.5B 是典型的 GQA 风格头配置：

- Query 头数 `14`
- Key/Value 头数 `2`

所以：

- Q 的 head 数多于 K/V 的 head 数
- attention 计算时会基于较少的 KV 头去服务更多的 query 头

### 2.5.4 标准 RoPE

Q 和 K 在 reshape 成 head 结构后，会进入 RoPE。

这里本轮统一按 **标准 RoPE** 理解：

- 只对前 `n_rot = 64` 维做旋转
- 位置来自 `inp_pos`
- 基频来自 `rope_theta = 1000000.0`

对 Qwen2.5-0.5B 来说，因为：

- `head_dim = 64`
- `n_rot = 64`

所以可以直接理解为：

- **每个 head 的全部 64 维都参与 RoPE**

RoPE 之后：

- Q 的 shape 仍是 `[64, 14, 当前 token 数]`
- K 的 shape 仍是 `[64, 2, 当前 token 数]`

只是它们的数值已经带上了位置信息。

V 不做 RoPE。

### 2.5.5 Self-Attention 子模块

RoPE 之后，进入本层 self-attention。

这一部分从主语义上可以分成四件事：

1. 把本批次新得到的 K / V 写入 KV cache
2. 从 KV cache 中取出当前层可见的 K / V 全量历史
3. 用当前 Q 与缓存中的 K 做注意力分数计算
4. 用注意力权重对缓存中的 V 做加权求和，再经过输出投影

#### 输入

主输入包括：

- `Qcur`：`[64, 14, 当前 token 数]`
- `Kcur`：`[64, 2, 当前 token 数]`
- `Vcur`：`[64, 2, 当前 token 数]`

辅助输入包括：

- KV cache 写入索引
- attention mask
- 已缓存的历史 K / V 视图

#### 计算语义

从语义上，它做的是标准 decoder-only 自注意力：

- 当前 token 的 query 只能看见允许可见的历史 key
- 通过 mask 保证因果约束
- 用 `1 / sqrt(head_dim)` 做缩放，这里就是 `1 / sqrt(64)`

即：

- scale = `1 / 8`

#### 输出

attention 汇聚后的上下文向量会重新合并回隐藏维，随后经过输出投影 `wo`，得到：

- `[896, 当前 token 数]`

这就是 attention 子模块的输出 `cur`。

### 2.5.6 Attention 残差连接

每层在进入 attention 之前，会先保留一份原始输入：

- `inpSA = inpL`

attention 输出得到 `cur` 之后，不是直接进入 FFN，而是先与这份残差相加：

- `ffn_inp = attention_out + residual`

因此：

- attention 子模块输出和层输入必须在 shape 上对齐
- 得到的 `ffn_inp` 是 FFN 子模块真正的输入

通常它的 shape 是：

- `[896, n_tokens]`

但最后一层会在这一步前发生一次裁剪，下面单独说。

### 2.5.7 最后一层的输出 token 裁剪

这是 `llm_build_qwen2` 里很关键、也很容易被忽略的一个点。

在最后一层（也就是第 `23` 层）attention 完成后，图不会立刻把全部 token 都送入 FFN。相反，它会先根据 `inp_out_ids` 选出真正需要输出 logits 的 token。

被裁剪的有两路：

- attention 输出 `cur`
- residual 支路 `inpSA`

裁剪后再做残差相加，得到：

- `ffn_inp = selected_attention_out + selected_residual`

这样做的直接效果是：

- 前 23 层，所有 token 都完整经过 attention + FFN
- 最后一层，所有 token 先完整经过 attention
- 但只有 `n_outputs` 个 token 会继续经过最后一层 FFN、最终 norm 和 LM head

这可以明显减少最后阶段不必要的计算。

因此在最后一层里：

- 裁剪前：`cur`、`inpSA` 是 `[896, n_tokens]`
- 裁剪后：它们变成 `[896, n_outputs]`
- 之后的 `ffn_inp` 也是 `[896, n_outputs]`

### 2.5.8 FFN 前 RMSNorm

得到 `ffn_inp` 之后，再做一次 RMSNorm：

- 输入：`ffn_inp`
- 权重：该层 `ffn_norm`
- 输出：归一化后的 FFN 输入

shape 不变：

- 普通层：`[896, n_tokens]`
- 最后一层裁剪后：`[896, n_outputs]`

### 2.5.9 FFN 子模块（SwiGLU）

Qwen2.5-0.5B 的 FFN 走的是并行 gate 的 SwiGLU 风格路径。

主语义是：

1. 从 norm 后输入同时做两路线性映射：
   - up 投影
   - gate 投影
2. 对 gate 路做 SiLU
3. 将 `SiLU(gate)` 与 `up` 逐元素相乘
4. 再经过 down 投影回到 hidden size

#### up 路

- 输入：`[896, token_count]`
- 输出：`[4864, token_count]`

#### gate 路

- 输入：`[896, token_count]`
- 输出：`[4864, token_count]`

#### GLU 融合

逐元素融合之后，结果仍是：

- `[4864, token_count]`

其中 `token_count` 在不同场景下是：

- 普通层：`n_tokens`
- 最后一层裁剪后：`n_outputs`

#### down 投影

再通过 down 投影映射回：

- `[896, token_count]`

这就是 FFN 子模块输出。

### 2.5.10 FFN 残差连接

FFN 输出不会直接作为层输出，而是还要和进入 FFN 前的 `ffn_inp` 再做一次残差相加：

- `layer_out = ffn_out + ffn_inp`

得到的结果就是当前层最终输出 `cur`。

随后：

- 作为下一层输入 `inpL`

在“不考虑控制向量”的前提下，本层到这里就结束了。

## 2.6 模块 3：LM Head 模块

Transformer 全部结束后，进入最终输出模块。

### 1. 输入

它的输入来自：

- 最后一层 Transformer 输出

shape 为：

- `[896, n_outputs]`

### 2. 最终 RMSNorm

进入词表投影前，先做一次最终 RMSNorm：

- 输入：`[896, n_outputs]`
- 输出：`[896, n_outputs]`

这一步的结果同时也是最终语义 embedding，可视作模型最后的隐藏表征。

### 3. 输出投影

随后做 LM head 线性投影：

- 输入：`[896, n_outputs]`
- 输出：`[151936, n_outputs]`

这里的每一列就是一个输出 token 位置在整个词表上的 logits。

如果模型带输出 bias，则还会再加一次 bias；对你这个分析口径，可以把它视为：

- 最终 logits 的仿射变换收尾

### 4. 输出

最终得到：

- `t_logits`
- shape = `[vocab_size, n_outputs] = [151936, n_outputs]`

这就是后续采样阶段直接消费的结果。

---

# Step 3: `completion.cpp:main` 在当前场景下的执行流程

本节只按指定的场景来分析：

- 模型：`Qwen2.5-0.5B`，非 MoE，FP16 权重
- 启动参数口径：
  - `-m Qwen2.5-0.5B`
  - `-np 1`
  - `-ub 128`
  - `-b 256`
  - `-c 32768`
  - `-n 4`
  - `--no-warmup`
  - `--cont-batching`
- 输入请求数：`1`
- prompt 长度：`400 token`
- 普通文本生成
- 位置 id 连续递增
- 不考虑 YaRN，按标准 RoPE
- 生成行为：prefill 后先生成 `1` 个非 EOS token，再下一轮生成 EOS

由于请求数为 `1`，所以这一节统一采用：

- `n_parallel = 1`
- `stream = 1`

这意味着在 Step 3 里：

- 不存在多个请求之间的 KV cache 隔离问题
- 不存在多 stream 间的 mask 分拆问题
- ubatch 中的所有 token 都属于同一个 sequence / 同一个 stream

## 3.1 初始化

这一小节先只回答一个问题：

- **在 `completion.cpp:main` 里，进入真正的 prefill / decode 之前，初始化到底包含哪些内容？**

按当前单请求场景，初始化阶段可以拆成下面 9 项。

### 3.1.1 运行参数与全局运行环境初始化

这一部分负责把命令行参数落到 `common_params`，并初始化全局运行环境。

在当前场景下，这里会固定出后续流程最关键的运行约束：

- 上下文长度 `n_ctx = 32768`
- 逻辑 batch 上限 `n_batch = 256`
- 单次 ubatch 上限 `n_ubatch = 128`
- 并发请求数 `n_parallel = 1`
- warmup 关闭
- 连续 batching 开启
- 采样走默认 sampler 链

这一步的作用是：

- 把后续模型加载、上下文创建、batch 切分、采样策略所需的全局参数全部确定下来

### 3.1.2 backend / NUMA 运行时初始化

这一部分负责把 llama / ggml 的后端运行时拉起。

它的作用是：

- 初始化底层 backend 框架
- 初始化 NUMA 策略
- 为后续模型、context、graph 调度做好运行环境准备

这是进入任何真正模型计算前的 runtime 基础设施初始化。

### 3.1.3 模型、上下文、采样器初始化

这一部分是初始化阶段最核心的内容。

它会基于 `params` 创建出：

- `model`
- `context`
- `sampler`

对当前场景来说，这一步实质上完成了：

- 把 Qwen2.5-0.5B 权重加载进来
- 按 `n_ctx = 32768`、`n_batch = 256`、`n_ubatch = 128` 等参数创建推理上下文
- 构造默认采样链所需的 sampler

另外，由于指定了 `--no-warmup`，所以这里**不会**执行那次“空跑一次模型”的 warmup 过程。

### 3.1.4 context 级核心句柄获取

在 model / context 创建完成之后，`main` 会继续拿到后面循环一直要用的几个关键句柄：

- `memory`
- `vocab`
- chat template 相关对象

对当前普通文本生成场景来说，真正关键的是前两个：

- `memory`：后面做 KV cache / context shift / session 相关操作都要通过它
- `vocab`：后面做 token 判定、BOS/EOS/EOG 处理、token -> piece 转换都依赖它

chat template 这条线在你当前场景下不是主角，但初始化时仍然会建立相应能力。

### 3.1.5 CPU 线程池与计算执行环境初始化

这一部分负责把 context 和 CPU 线程池绑定起来。

它包括：

- 主线程池参数准备
- batch 线程池参数准备
- 线程池创建
- 将线程池附着到 `llama_context`

这一步的意义是：

- 把之后 `llama_decode` / graph compute 实际要跑在哪些 CPU worker 上这件事先确定下来

即使后端里有 device offload，这一层 CPU 调度环境仍然是整个推理流程的控制中心之一。

### 3.1.6 prompt / session 相关初始化

这一部分负责把“本次请求真正要送进模型的输入 token 序列”准备出来，并处理可选的 session 复用。

它包含：

- prompt 文本整理
- prompt tokenization
- session 文件加载（如果有）
- prompt 与 session token 前缀匹配
- 必要时清理“未来 token”残留状态
- 必要时回放最后一个 token 以恢复 logits

在你当前给定场景里，如果不考虑 session 文件，这一步最核心的结果就是：

- 得到长度为 `400` 的 `embd_inp`

也就是：

- 后续 prefill 要消费的完整 prompt token 序列

### 3.1.7 生成控制状态初始化

这一部分负责初始化后续生成循环会持续维护的状态变量。

包括但不限于：

- `n_past`
- `n_remain`
- `n_consumed`
- `n_session_consumed`
- 输入输出 token 缓冲
- 文本输出缓冲
- `embd`
- 反向提示词相关状态

这些变量在 Step 3 后面分析 prefill / decode 时都会持续出现。

对你当前场景，最关键的初始值可以先记住：

- `n_past = 0`      -> 还没有任何 token 进入 KV cache
- `n_remain = 4`    -> 最多还允许生成 4 个 token
- `n_consumed = 0`  -> prompt 还一个都没有被送去 decode

### 3.1.8 交互模式与显示相关初始化

`completion` 虽然是生成工具，但它同时兼容交互式模式、聊天模板模式、反向提示词模式、控制台彩色输出模式。

因此初始化阶段还会准备：

- console 显示方式
- 是否 interactive
- 是否 conversation mode
- 是否需要等待用户首轮输入
- 是否需要记录 assistant 当前回复片段

对你当前这个普通单请求、非交互讲解场景，这一块不是主线，但它会影响主循环里的若干分支，因此初始化里仍然要把它准备好。

### 3.1.9 进入生成主循环前的最终状态

初始化全部完成后，程序会处在这样一个状态：

- 模型和 context 已经准备好
- sampler 已经准备好
- 线程池已经挂到 context 上
- prompt 已经 tokenized 成 `400` 个 token
- `n_past = 0`
- `n_consumed = 0`
- `n_remain = 4`
- `stream = 1`
- warmup 没有执行

接下来主循环的第一件事就是：

- 先把 `embd_inp` 中尚未消费的 prompt token 分批送去做 **prefill**

也就是说，初始化阶段的终点，正好就是 prefill 阶段的起点。

## 3.1 小结：初始化阶段的 9 个组成部分

为了后面逐个展开，这里先把初始化阶段压缩成一个清单：

1. 运行参数与全局运行环境初始化
2. backend / NUMA 运行时初始化
3. 模型、上下文、采样器初始化
4. context 级核心句柄获取
5. CPU 线程池与计算执行环境初始化
6. prompt / session 相关初始化
7. 生成控制状态初始化
8. 交互模式与显示相关初始化
9. 进入生成主循环前的最终状态确认

这9部分的内容不再展开，有感兴趣的可以自己追代码。


## Step 4. 按 OP 拆开看这条 Qwen2.5 主链

这一节不再按 prefill / decode 的时间顺序讲，而是把前面反复出现的 `GGML_OP_*` 单独拆开讲。

分析依据主要来自：

- `docs/al_ops/**/*.md`
- `docs/al_ops/**/*.py`

并且仍然放在前面已经固定的场景里理解：

- 模型：`Qwen2.5-0.5B`
- 主链：`Word Embedding -> Transformer x 24 -> LM Head`
- 位置编码：默认主线先看**标准 RoPE**
- YaRN 单独作为扩展机制说明

下面按你要求的顺序，从 `softmax` 开始。

### 4.1 `GGML_OP_SOFT_MAX`

#### 4.1.1 它在这条图里的位置

`softmax` 出现在每一层 attention 的中间位置：

1. 先把 `Q` 与 `K^T` 做乘法，得到 attention score
2. 再加上 mask / 位置相关偏置
3. 然后对 key 维做 `softmax`
4. 最后再拿这个概率分布去乘 `V`

也就是说，`softmax` 负责把“未归一化的打分”变成“对所有可见 key 的概率分配”。

#### 4.1.2 输入、输出与归一化维度

输入有一下几个：

- `inp = Q @ K^T`
- dtype: `FP32`
- shape: `[n_kv, n_tokens / n_stream, n_heads, n_stream]`

可选输入还有：

- `attn_mask`
- `sink`

输出与输入 shape 一致，语义上是“沿 `ne[0] = n_kv` 这一维做 softmax 后的结果”。

因此从逻辑上看，`softmax` 不是对整个 4D 张量一起归一化，而是：

- 固定某个 `(token, head, stream)`
- 取出这一组对应的全部 `kv` 分数
- 在这条长度为 `n_kv` 的向量上做 softmax

#### 4.1.3 它真正做了什么

可以把 `softmax` 节点理解成 5 步：

1. 对 score 先乘 `scale = 1 / sqrt(head_dim)`
2. 若存在 mask，则把 mask 加进去
3. 若启用 alibi，还会把每个 head 的斜率乘到 mask 上
4. 为了数值稳定，先减去最大值 `m`
5. 再做 `exp`、求和、归一化

也就是常见的：

```text
prob_i = exp(z_i - m) / sum_j exp(z_j - m)
```

其中：

- `z_i` 是缩放后的 score，外加 mask / alibi
- `m = max(z)`

#### 4.1.4 为什么这里必须先减最大值

attention score 在长上下文里可能很大或很小，直接做 `exp(score)` 很容易溢出或下溢。

所以 `softmax` 实现里最关键的稳定化步骤不是 softmax 本身，而是：

```text
先求 m = max(z)
再算 exp(z_i - m)
```

这样不会改变最终概率分布，但能显著降低数值风险。

#### 4.1.5 `attn_mask` 在这里的真实作用

在前面的主链说明里，`inp_attn` 本质上是在告诉某个 query：

- 哪些历史 key 可见
- 哪些位置不可见

落到 `softmax` 这里，就是把不可见位置的 score 在归一化前打成一个极小值，常见写法就是：

- 可见位置加 `0`
- 不可见位置加 `-INF`

这样经过 `exp` 以后：

- 可见位置保留
- 不可见位置几乎变成 `0`

所以 `softmax` 是 attention mask 真正“生效”的地方。

#### 4.1.6 `sink` / `attn_sinks` 是什么

`softmax.md` 里还提到了一个可选输入 `sink`。

它不是普通的 query-key score，而是“额外并入分母的一项 logit”。它的作用是：

- 给每个 head 额外留一个可竞争的注意力槽位
- 在某些长上下文 / 滑窗 / 特定模型结构里，帮助注意力分配更稳定

对 Qwen2.5 这一条标准主线来说，可以先把它当成**可选扩展**，不是主逻辑的核心。

#### 4.1.7 `softmax_torch.py` 给出的关键信息

`softmax_torch.py` 把 CPU 版逻辑用 PyTorch 重写了一遍，里面最重要的结论有三个：

1. softmax 是沿 `dim=0` 做的，对应 ggml 的 `ne[0] = n_kv`
2. mask 会先提升到 `FP32` 再参与计算
3. 有 `sink` 时，`sink` 不是额外输出，而是并入分母

所以这份 `.py` 代码更像是对 `.md` 的“可执行定义”。

#### 4.1.8 在 Qwen2.5 attention 里的角色

如果只看注意力这段局部链路，`softmax` 前后的语义变化非常明确：

- `QK^T` 之前：还只是线性投影后的特征
- `QK^T` 之后：变成 query 对所有 key 的原始匹配分数
- `softmax` 之后：变成真正拿来加权 `V` 的概率权重

所以 attention 的“选择谁、忽略谁”，是在 `softmax` 这里完成的，而不是在 `mul_mat(Q, K^T)` 处完成的。

### 4.2 `RoPE` 与 `YaRN`

### 4.2.1 标准 `RoPE` 在图里的位置

标准 `RoPE` 发生在 attention 的 `Q`、`K` 上，而不是 `V` 上。

顺序可以概括成：

1. hidden state 经过线性投影，得到 `Q`
2. hidden state 经过线性投影，得到 `K`
3. 对 `Q`、`K` 做 RoPE
4. 再去算 `Q @ K^T`

所以 RoPE 的职责不是直接改 attention score，而是**先改 `Q`/`K` 的几何表示**，再让这种位置相关性自然进入 `QK^T`。

#### 4.2.2 标准 `RoPE` 的本质

RoPE 不是给每个标量简单加一个位置值，而是：

- 在每个 head 内
- 把相邻两维看成一个二维平面
- 对这对分量做一次二维旋转

也就是把：

- `(x[0], x[1])`
- `(x[2], x[3])`
- `(x[4], x[5])`

这样的维度对，分别乘上不同角度的旋转矩阵。

因此它的本质是“按频率分块的二维旋转位置编码”。

#### 4.2.3 为什么 RoPE 会把绝对位置变成相对位置信息

RoPE 最核心的性质，是：

- `Q` 在位置 `m` 上旋转
- `K` 在位置 `n` 上旋转

最后做内积时，结果里会自然出现与 `(n - m)` 相关的结构。

所以虽然实现里传进去的是“绝对位置 id”，但进入 attention score 以后，保留下来的更像是“相对位移关系”。

这就是 RoPE 比“直接加位置向量”更适合 attention 的地方。

#### 4.2.4 标准 `RoPE` 的计算步骤

可以把标准流程压成 3 步：

1. 先根据 `freq_base = rope_theta` 和 `n_dims` 算频率递推因子
2. 对每个位置 `m`、每个维度块 `k` 预计算 `cos` / `sin`
3. 用这对 `cos` / `sin` 去旋转 `(x[2k], x[2k+1])`

写成公式就是：

```text
x0' =  cos(theta) * x0 - sin(theta) * x1
x1' =  sin(theta) * x0 + cos(theta) * x1
```

这里旋转的是：

- `Q`
- `K`

而且只旋转前 `n_dims` 维；如果 `head_dim > n_dims`，后半段直接拷贝，不参与旋转。

#### 4.2.5 标准 `RoPE` 的缓存含义

 `cache`：

- 对每个位置，先算出一行 `cos/sin`
- 同一位置的这行缓存可以被多个 head 复用

所以 RoPE 的工程实现重点不是“现算一个旋转矩阵”，而是：

- 先把每个位置对应的 `cos/sin` 行计算并缓存起来
- 再用这些cache中的数据，批量的旋转不同 head / token 上的向量

这比每次临时调用三角函数更适合推理执行。

#### 4.2.6 `YaRN` 要解决的问题

`YaRN` 是对标准 `RoPE` 的长上下文扩展。

它要解决的问题不是“标准 RoPE 完全不能用”，而是：

- 训练上下文较短
- 推理时想外推到更长上下文
- 这时高频和低频维度对长外推的敏感性不同

如果简单只改一个全局缩放，往往不够稳。

所以 YaRN 的思路是：

- 某些频率块更偏向插值
- 某些频率块更偏向保留原始外推
- 中间用一个 ramp 平滑过渡

#### 4.2.7 `YaRN` 相比标准 `RoPE` 多做了什么

YaRN 相比RoPE多出来的是两类处理：

1. **改辐角**
   - 先有标准外推角 `theta_extrap`
   - 再有缩放后的插值角 `theta_interp`
   - 最后按块编号做线性混合，得到真正使用的 `theta`

2. **改幅度**
   - 给 `cos(theta)`、`sin(theta)` 再乘一个公共尺度 `mscale`

所以 YaRN 改的不只是“角速度”，还会改“旋转向量的整体幅度”。

#### 4.2.8 `corr_dims`、`ramp`、`ext_factor` 的含义

- `corr_dims`
  - 用来确定“从哪些维度块开始进入过渡区间”
  - 由 `beta_fast`、`beta_slow`、`n_ctx_orig`、`freq_base` 共同确定

- `ramp`
  - 是一个从 `1` 逐步降到 `0` 的线性系数
  - 低频块更偏一端，高频块更偏另一端

- `ext_factor`
  - 控制整个混合有多强
  - 为 `0` 时，YaRN 混合分支等价关闭

因此 YaRN 不是“所有维度一起缩放”，而是**沿 RoPE 频率块做分段、渐变式修正**。

#### 4.2.9 `YaRN_torch.py` 文件的作用

  - 用 PyTorch 复现 `corr_dims`、`rope_yarn`、cache 初始化和旋转

这个文件共同说明了一件事：

- 而在“标准 RoPE cache 初始化”这一步里，替换掉每个块的 `theta` 与 `mscale`

换句话说，**YaRN 仍然是 RoPE，只是把 `cache` 的生成规则改了**。

#### 4.2.10 对当前 Qwen2.5 主线如何理解

你前面已经要求 Step 2 / Step 3 的主线都按**标准 RoPE**来讲，因此对当前主链可以这样理解：

- 主线：`Q/K` 用标准 RoPE
- 扩展：若模型配置启用长上下文 YaRN，再把标准 cache 生成替换成 YaRN 规则

所以在图结构层面：

- 标准 RoPE 和 YaRN 并不是两张完全不同的 attention 图
- 差别主要在同一个 `rope` 节点内部，如何生成 `cos/sin`

### 4.3 `GGML_OP_RMS_NORM`

#### 4.3.1 它在 Qwen2.5 里的位置

`RMS_NORM` 在 Qwen2.5 里非常高频，至少出现在三类位置：

1. 每层 attention 前
2. 每层 FFN 前
3. 最终输出层前

所以它是整个 Transformer 主干里最典型的“稳定化节点”之一。

#### 4.3.2 它做的不是 LayerNorm

`rms_norm.md` 强调得很清楚：

- 它只做 RMS 归一化
- 不减均值
- 若还有 gamma / beta，一般由后续 `MUL` / `ADD` 节点再接上

所以这里不能把它理解成传统 LayerNorm。

#### 4.3.3 核心公式

对长度为 `d = ne[0]` 的向量 `x`：

```text
mean_sq = (1 / d) * sum_i x[i]^2
RMS     = sqrt(mean_sq + eps)
y[i]    = x[i] / RMS
```

也就是：

- 先算平方和
- 再开方得到 RMS
- 最后把整条向量按同一个系数缩放

#### 4.3.4 在主链里的意义

它的作用不是引入新信息，而是把当前 token 的 hidden state 在进入 attention / FFN 之前，重新压回一个更稳定的尺度范围。

因此它更像：

- 让后续线性层更稳定
- 让残差叠加不至于数值失控

### 4.4 `GGML_OP_MUL_MAT`

#### 4.4.1 它在图里几乎无处不在

`MUL_MAT` 是整条 Transformer 主链最核心的线性代数节点。

在 Qwen2.5 里至少会出现在：

- token embedding 后进入各层投影
- attention 的 `Q / K / V / O` 投影
- FFN 的 `gate / up / down` 投影
- 最后的 `LM head`
- attention 内部的 `Q @ K^T`
- attention 权重与 `V` 的乘法

所以如果从“算量大头”看，`MUL_MAT` 是最重要的 OP。

#### 4.4.2 这个 OP 统一承载了两类乘法

虽然名字叫 `mul_mat`，但在主链里它承担的是两类不同语义：

1. **参数线性层**
   - 权重矩阵乘激活
   - 例如 `Wq * x`、`Wk * x`、`Wv * x`、`Wo * x`

2. **attention 内部矩阵乘**
   - `Q @ K^T`
   - `P @ V`

所以在图里看到 `MUL_MAT`，不能直接断定它是“权重层”，还要看上下文。

#### 4.4.3 文档里给出的形状语义

`mul_mat.md` 给出的抽象是：

- `a`: `[K, M, ...]`
- `b`: `[K, N, ...]`
- 输出: `[M, N, ...]`

这说明 ggml 里的 `ne` 排列和很多线性代数教材的“行列直觉”不完全一致，读图时必须始终结合：

- `ne[0]`
- `ne[1]`
- 是否有 `transpose/view`

一起看，而不能只凭“矩阵乘”的名字猜 shape。

#### 4.4.4 在当前模型里的理解方式

对 Qwen2.5 主线，可以把 `MUL_MAT` 先粗分成三种角色：

1. **词向量/隐藏态投影**
2. **attention 打分或加权**
3. **logits 输出**

### 4.5 `GGML_OP_GLU`（SwiGLU）

#### 4.5.1 它在 FFN 里的位置

Qwen2.5 的 FFN 不是“单路激活 + 单路下投影”，而是典型的 SwiGLU 结构：

1. hidden state 分别经过 `gate_proj`
2. hidden state 分别经过 `up_proj`
3. 对 gate 路做 `SiLU`
4. 与 up 路逐元素相乘
5. 再经过 `down_proj`

其中第 3 步和第 4 步合起来，就是这里的 `GGML_OP_GLU` 子类型 `SWIGLU`。

#### 4.5.2 计算本质

文档里给出的公式非常直接：

```text
silu(t) = t * sigmoid(t)
out     = silu(gate) * up
```

注意这里没有跨通道归约，它是逐元素完成的。

#### 4.5.3 为什么它重要

SwiGLU 本质上是在 FFN 中做一层门控：

- `gate` 决定哪些通道更该被放大或抑制
- `up` 提供主值分支

所以它比传统 `GELU(Wx)` 多了一条并行控制支路。

### 4.6 `GGML_OP_GET_ROWS`

#### 4.6.1 它最典型的用途：词嵌入查表

`GET_ROWS` 的作用就是“按 token id 去 embedding 表里取行”。

这正好对应主链最开始的 Word Embedding：

- 输入：token id
- 表：`tok_embd`
- 输出：每个 token 对应的一行 embedding 向量

#### 4.6.2 为什么它不是普通矩阵乘

这里不是 one-hot 再乘 embedding matrix，而是直接 gather：

- 每个 token id 作为整数下标
- 直接去词表矩阵的第 1 维取对应行

所以从图上看，它更像“查表”，不是“计算密集型线性层”。

#### 4.6.3 在主链里的语义变化

`GET_ROWS` 完成的是：

- 离散 token id
- 变成连续向量表示

这是整条图里从“符号空间”进入“向量空间”的第一步。

### 4.7 `GGML_OP_SET_ROWS`

#### 4.7.1 它的语义

`SET_ROWS` 是 `GET_ROWS` 的反方向：

- 不是读表
- 而是按下标把某些行写回表中

文档把它描述成 scatter。

#### 4.7.2 在当前 Qwen2.5 单次前向里的地位

对你现在关心的这条标准 Qwen2.5 推理主线来说，`SET_ROWS` 不是 attention / FFN / logits 的常驻主角。

所以它更适合作为“补充型 OP”理解：

- 语义上它就是按索引写表
- 但不是这条纯前向主链的核心算子

### 4.8 `GGML_OP_ADD` 与 `GGML_OP_MUL`

#### 4.8.1 这两个都是广播友好的逐元素算子

`binary/add_mul.md` 里给出的要点很简单：

- 输出 shape 与 `a` 一致
- `b` 可以广播到 `a`
- dtype 继承 `a`

所以它们不是 reduction / matmul 类节点，而是最直接的逐元素节点。

#### 4.8.2 在 Qwen2.5 主链里的典型角色

- `ADD`
  - 残差连接
  - 某些 norm 后的 bias 叠加

- `MUL`
  - norm 后乘 gamma
  - 某些缩放项
  - attention 或 RoPE 相关的逐元素缩放辅助

#### 4.8.3 阅读图时怎么理解

如果看到一个 `ADD` 节点夹在大块 attention / FFN 之间，优先把它理解为：

- “把一条分支重新并回主残差”

如果看到 `MUL` 挨着 norm，优先把它理解为：

- “按通道做可学习缩放”

### 4.9 `GGML_OP_CONT`

#### 4.9.1 它不是数学变换，而是布局物化

`contiguous.md` 的重点不是公式，而是内存布局：

- 某些后端或算子要求输入连续
- 前面如果经过 view / transpose / permute，逻辑 shape 虽然对了，但物理布局可能不连续
- 这时就需要 `CONT`

所以 `GGML_OP_CONT` 的本质是：

- 按当前 shape / stride 把数据重新拷成一块连续内存

#### 4.9.2 为什么它重要

很多时候图上最容易忽略的不是“大算子”，而是这些布局整理节点。

没有 `CONT`，后面的高性能 kernel 即使数学上知道该怎么做，也可能因为 stride 不满足要求而不能直接执行。

因此 `CONT` 更接近“执行准备节点”，不是“模型语义节点”。

### 4.10 `VIEW / RESHAPE / TRANSPOSE / PERMUTE`

#### 4.10.1 它们都属于 shape-node

`view_reshape_transpose_premute.md` 把这些 OP 归到一类非常准确：`shape-node`。

共同点是：

- 它们主要改变看待同一块数据的方式
- 不一定真的搬数据
- 核心是改 `shape`、`stride`、`offset`、`axis` 解释方式

#### 4.10.2 这几类节点的区别

- `VIEW`
  - 从原张量取一个带 offset 的视图
- `RESHAPE`
  - 重新解释 shape，但要求底层连续
- `TRANSPOSE`
  - 交换某两维的 `ne/nb`
- `PERMUTE`
  - 更一般地重排四个轴

#### 4.10.3 为什么文档强调“前向中不应真正调 NPU 对应算子”

因为这些节点很多时候只是**逻辑视图变化**。

如果把它们错误地当成真实数据重排去执行，就会把后续依赖的 stride / offset 语义破坏掉。

所以这类节点的正确理解是：

- 它们主要服务于后继算子如何解释输入张量
- 本身通常不是推理算量重点

#### 4.10.4 在 attention 里最常见的用途

attention 前后经常要做：

- head 维拆分
- token/head 维交换
- 生成 `K^T` 视图

所以这类节点虽然“不算”，却是把 `Q/K/V` 张量组织成正确乘法形状的关键基础设施。

### 4.11 本节涉及 OP 的一张总表

为了后面继续下钻，这里把本节已经记录过的 OP 压成一张表：

| OP | 主要作用 | 在 Qwen2.5 主链中的典型位置 |
|----|-----------|------------------------------|
| `GGML_OP_GET_ROWS` | 按 token id 查 embedding 行 | Word Embedding |
| `GGML_OP_RMS_NORM` | 按向量做 RMS 归一化 | attention 前、FFN 前、最终输出前 |
| `GGML_OP_MUL_MAT` | 矩阵乘 / 线性投影 / attention 内乘法 | QKV/O、FFN、QK^T、PV、LM head |
| `GGML_OP_ROPE` | 对 `Q/K` 做旋转位置编码 | attention 中，Q/K 投影之后 |
| `GGML_OP_SOFT_MAX` | 把 score 变成注意力概率 | `QK^T + mask` 之后 |
| `GGML_OP_GLU(SWIGLU)` | `silu(gate) * up` | FFN 中间激活 |
| `GGML_OP_ADD` | 逐元素加法 / 残差并回 | attention、FFN 后残差 |
| `GGML_OP_MUL` | 逐元素乘法 / 通道缩放 | norm 权重、若干缩放分支 |
| `GGML_OP_CONT` | 物化连续内存布局 | 某些 kernel 前的布局整理 |
| `GGML_OP_VIEW` | 建视图 | attention / cache / shape 组织 |
| `GGML_OP_RESHAPE` | 改 shape 解释 | attention 张量整理 |
| `GGML_OP_TRANSPOSE` | 交换维度解释 | `K^T`、head/token 维整理 |
| `GGML_OP_PERMUTE` | 更一般的轴重排 | attention 张量重排 |
| `GGML_OP_SET_ROWS` | 按下标写回表 | 非当前主线核心，补充型 scatter |

### 4.12 小结

如果把这些 OP 再压缩成三类，会更容易抓主线：

1. **真正决定模型语义的算子**
   - `GET_ROWS`
   - `MUL_MAT`
   - `ROPE`
   - `SOFT_MAX`
   - `GLU`
   - `RMS_NORM`

2. **负责残差与缩放的逐元素算子**
   - `ADD`
   - `MUL`

3. **负责张量布局与执行条件的辅助算子**
   - `CONT`
   - `VIEW`
   - `RESHAPE`
   - `TRANSPOSE`
   - `PERMUTE`
   - `SET_ROWS`

因此读 Qwen2.5 这张图时，可以优先抓住一条主骨架：

`GET_ROWS -> RMS_NORM -> MUL_MAT(QKV) -> ROPE -> MUL_MAT(QK^T / PV) -> SOFT_MAX -> ADD -> RMS_NORM -> MUL_MAT(FFN) -> GLU -> MUL_MAT -> ADD -> ... -> RMS_NORM -> MUL_MAT(logits)`

这样后面如果继续展开某个具体 OP，就不会丢掉它在整张图里的位置。
