# LLaMA.cpp 计算图拆解

## 1. Input Embedding 模块

**对应代码**: `llama.cpp/src/llama-graph.cpp` 中的 `llm_graph_context::build_inp_embd(ggml_tensor * tok_embd)` 函数。
**调用位置**: `qwen2.cpp` 中的 `inpL = build_inp_embd(model.tok_embd);`

### 1.1 模块输入与输出
- **输入参数**:
  - `tok_embd`: 模型的词表 embedding 权重张量 (tensor)。
    - **dtype**: `GGML_TYPE_F16` 或量化类型 (取决于模型加载配置)
    - **shape**: `[n_embd_inp, n_vocab]`
- **构建的输入张量 (Input Tensors)**:
  - `inp->tokens`: 
    - **name**: `"inp_tokens"`
    - **dtype**: `GGML_TYPE_I32`
    - **shape**: `[ubatch.n_tokens]` (一维张量)
    - **作用**: 用于接收用户输入的 token IDs。
  - `inp->embd`: 
    - **name**: `"inp_embd"`
    - **dtype**: `GGML_TYPE_F32`
    - **shape**: `[n_embd_inp, ubatch.n_tokens]` (二维张量)
    - **作用**: 用于接收用户直接输入的向量 (vector embeddings)。
- **中间计算缓存 (Intermediate Tensors)**:
  - `inps[0]` (Token 分支中间张量): 
    - **dtype**: `GGML_TYPE_F32`
    - **shape**: `[n_embd, ubatch.n_tokens]` (Padding 前) 或 `[n_embd_inp, ubatch.n_tokens]` (Padding 后)
    - **作用**: 保存从 `tok_embd` 查表得到的 embeddings，并融合 LoRA 增量。
  - `inps[1]` (Vector 分支中间张量):
    - **dtype**: `GGML_TYPE_F32`
    - **shape**: `[n_embd_inp, ubatch.n_tokens]`
    - **作用**: 直接指向 `inp->embd`。
- **输出 (Return)**:
  - `cur`: 
    - **name**: `"embd"`
    - **dtype**: `GGML_TYPE_F32`
    - **shape**: `[n_embd, ubatch.n_tokens]`
    - **作用**: 经过分支选择、维度对齐和缩放处理后的最终 embedding 张量，作为后续 Transformer 层的输入 `inpL`。

### 1.2 计算逻辑流程图

```mermaid
graph TD
    Start([开始 build_inp_embd]) --> Init[创建输入张量: inp_tokens 和 inp_embd]
    
    Init --> Branch0[分支 0: Token Embeddings 路径]
    Init --> Branch1[分支 1: Vector Embeddings 路径]
    
    %% 分支 0 逻辑
    Branch0 --> GetRows[调用 GGML_OP_GET_ROWS]
    GetRows --> LoraCheck{是否有 LoRA 适配器?}
    
    LoraCheck -- 是 --> LoraLoop[遍历 LoRA 权重]
    LoraLoop --> LoraGetRows[调用 GGML_OP_GET_ROWS 获取 lora_a]
    LoraGetRows --> LoraMulMat[调用 GGML_OP_MUL_MAT 计算 lora_b * lora_a]
    LoraMulMat --> LoraScale[调用 GGML_OP_SCALE 缩放 delta]
    LoraScale --> LoraAdd[调用 GGML_OP_ADD 将 delta 加到 cur]
    LoraAdd --> LoraCheck
    
    LoraCheck -- 否 --> PadCheck0{n_embd_inp != n_embd ?}
    PadCheck0 -- 是 --> Pad[调用 GGML_OP_PAD 进行填充]
    Pad --> Branch0End[分支 0 结束: inps[0]]
    PadCheck0 -- 否 --> Branch0End
    
    %% 分支 1 逻辑
    Branch1 --> Branch1End[直接使用 inp_embd 作为 inps[1]]
    
    %% 路径选择
    Branch0End --> Select[调用 GGML_OP_FORWARD_SELECT]
    Branch1End --> Select
    
    Select --> ViewCheck{n_embd_inp != n_embd ?}
    
    %% 后处理
    ViewCheck -- 是 (解释: 统一维度) --> View[调用 GGML_OP_VIEW_2D 截取 n_embd 维度]
    View --> ScaleCheck
    ViewCheck -- 否 --> ScaleCheck{f_embedding_scale != 0.0f ?}
    
    ScaleCheck -- 是 (如 Granite 模型) --> Scale[调用 GGML_OP_SCALE 进行缩放]
    Scale --> End([返回 cur])
    ScaleCheck -- 否 --> End
```

### 1.3 调用的 OP 类型、参数及作用详解

在 `build_inp_embd` 函数中，依次调用了以下 GGML OP 来构建计算图：

#### 1. `GGML_OP_GET_ROWS` (基础查表)
- **调用场景**: 分支 0 (Token 输入路径) 的第一步。
- **参数**: `ctx0` (计算图上下文), `tok_embd` (词表权重张量), `inp->tokens` (输入的 token IDs 张量)。
- **作用**: 查表操作。根据 `inp->tokens` 中的每个 token ID，从 `tok_embd` 矩阵中提取对应行的词向量，生成初始的 embedding 张量 `cur`。

2 -> 5 是在存在lora的场景下的计算，暂时先忽略
    #### 2. `GGML_OP_GET_ROWS` (LoRA A 矩阵查表)
    - **调用场景**: 如果模型加载了 LoRA 适配器，在遍历 LoRA 时调用。
    - **参数**: `ctx0`, `lw->a` (LoRA 的 A 权重矩阵), `inp->tokens`。
    - **作用**: 对 LoRA 的 A 矩阵进行查表，获取当前输入 tokens 对应的低秩特征表示。

    #### 3. `GGML_OP_MUL_MAT` (LoRA B 矩阵乘法)
    - **调用场景**: LoRA 遍历中，紧接在获取 LoRA A 特征之后。
    - **参数**: `ctx0`, `lw->b` (LoRA 的 B 权重矩阵，非转置), 上一步 `GGML_OP_GET_ROWS` 的结果。
    - **作用**: 计算 `lora_b * lora_a_rows`，将低秩特征映射回原始的 embedding 维度，得到 LoRA 对当前 embedding 的增量 (delta) 的未缩放版本。

    #### 4. `GGML_OP_SCALE` (LoRA 增量缩放)
    - **调用场景**: LoRA 遍历中，计算完矩阵乘法后。
    - **参数**: `ctx0`, 上一步 `GGML_OP_MUL_MAT` 的结果, `scale` (LoRA 的缩放系数，由 alpha 和 adapter_scale 计算得出)。
    - **作用**: 对 LoRA 增量进行缩放，得到最终的 `inpL_delta`。

    #### 5. `GGML_OP_ADD` (融合 LoRA 增量)
    - **调用场景**: LoRA 遍历的最后一步。
    - **参数**: `ctx0`, `cur` (当前的 embedding 张量), `inpL_delta` (计算出的 LoRA 增量)。
    - **作用**: 将 LoRA 的增量加到基础的 embedding 上，实现 LoRA 权重的动态应用。

#### 6. `GGML_OP_PAD` (张量填充)
- **调用场景**: 分支 0 结束前，如果 `n_embd_inp != n_embd`（即输入 embedding 维度大于模型实际计算维度）。
- **参数**: `ctx0`, `cur`, `hparams.n_embd_inp() - n_embd` (需要填充的维度大小), `0, 0, 0` (其他维度不填充)。
- **作用**: 对查表得到的张量在特征维度上进行 padding 填充，使其维度与 `n_embd_inp` 对齐，以保证分支 0 和分支 1 的输出张量形状 (shape) 和步长 (stride) 完全一致，从而满足后续 `FORWARD_SELECT` OP 的严格要求。

#### 7. `GGML_OP_FORWARD_SELECT` (动态分支选择)
- **调用场景**: 汇合分支 0 和分支 1。
- **参数**: `gf` (计算图), `inps.data()` (包含两个分支输出张量的数组), `inps.size()` (数组大小为 2), `ubatch.token ? 0 : 1` (选择索引)。
- **作用**: 在图执行时，根据当前 batch 是基于 token 还是基于 embedding（由 `ubatch.token` 决定），动态选择 `inps[0]` 或 `inps[1]` 作为向后传递的张量 `cur`。

#### 8. `GGML_OP_VIEW_2D` (视图截取)
- **调用场景**: 分支选择完成后，如果 `n_embd_inp != n_embd`。
- **参数**: `ctx0`, `cur`, `n_embd` (目标宽度), `n_tokens` (目标高度), `cur->nb[1]` (原始步长), `0` (偏移量)。
- **作用**: **解释存在这个场景的原因**：在某些特殊模型或多模态场景下，输入的 embedding 维度 (`n_embd_inp`) 可能为了对齐等原因大于模型 Transformer 层实际需要的维度 (`n_embd`)。在分支合并时为了满足 `FORWARD_SELECT` 的同构要求，之前可能做了 Padding 或直接传入了较大的 `inp_embd`。此时通过 `VIEW_2D` 操作，在不发生内存拷贝的情况下，截取出实际需要的 `n_embd` 维度的子张量，供后续 Transformer 层使用。

9 为 Granite 架构模型专属逻辑，暂时忽略
    #### 9. `GGML_OP_SCALE` (全局 Embedding 缩放)
    - **调用场景**: 函数返回前，如果 `hparams.f_embedding_scale != 0.0f`（例如 Granite 架构模型）。
    - **参数**: `ctx0`, `cur`, `hparams.f_embedding_scale` (缩放因子)。
    - **作用**: 对最终的 embedding 张量进行全局的标量乘法缩放。

---

## 2. 辅助输入张量构建模块

在构建完基础的 `inpL` (Input Embedding) 后，计算图还会构建若干辅助的输入张量，用于位置编码、注意力机制缓存管理和输出提取等。

### 2.1 `build_inp_pos` (位置编码输入)

**对应代码**: `llama.cpp/src/llama-graph.cpp` 中的 `llm_graph_context::build_inp_pos()` 函数。
**调用位置**: `qwen2.cpp` 等模型构建文件中的 `ggml_tensor * inp_pos = build_inp_pos();`

- **作用与逻辑**:
  该函数用于构建位置编码（Positional Encoding，如 RoPE）所需的输入位置张量（Position IDs）。它会根据 `hparams.n_pos_per_embd()`（通常为 1，但对于 M-RoPE 等多维位置编码可能为 4）和当前 batch 的 token 数量 `n_tokens`，创建一个一维张量来存储每个 token 的绝对位置索引。
- **构建的输入张量 (Input Tensors)**:
  - `inp->pos` (返回的 `cur`):
    - **name**: 默认未显式命名，通常作为 `inp_pos` 传递。
    - **dtype**: `GGML_TYPE_I32`
    - **shape**: `[(int64_t)n_tokens * hparams.n_pos_per_embd()]` (一维张量)
    - **作用**: 接收用户传入的当前 batch 中每个 token 的位置 ID，供后续的 `GGML_OP_ROPE` 等算子使用，以注入序列的位置信息。

### 2.2 `build_attn_inp_kv` (注意力与 KV 缓存输入)

**对应代码**: `llama.cpp/src/llama-graph.cpp` 中的 `llm_graph_context::build_attn_inp_kv()` 函数及底层的 `build_attn_inp_kv_impl()`。
**调用位置**: `qwen2.cpp` 等模型构建文件中的 `auto * inp_attn = build_attn_inp_kv();`

- **作用与逻辑**:
  该函数用于构建自注意力机制（Self-Attention）和 KV Cache 相关的输入张量。它不直接参与矩阵运算，而是为注意力计算提供必要的辅助张量，包括 KV Cache 的存储索引（Indices）、注意力掩码（Attention Mask）以及可能的旋转矩阵（Rotation Matrices）。
- **构建的输入张量 (Input Tensors)**:
  - `inp->self_k_idxs`:
    - **name**: 未显式命名
    - **dtype**: `GGML_TYPE_I64`
    - **shape**: `[n_tokens]` (一维张量)
    - **作用**: 存储当前 batch 的 Key 向量在 KV Cache 中的目标物理槽位索引。
  - `inp->self_v_idxs`:
    - **name**: 未显式命名
    - **dtype**: `GGML_TYPE_I64`
    - **shape**: `[n_tokens]` 或 `[n_tokens * n_embd_v_gqa_max()]` (取决于 V 缓存是否转置)
    - **作用**: 存储当前 batch 的 Value 向量在 KV Cache 中的目标物理槽位索引。
  - `inp->self_kq_mask`:
    - **name**: `"attn_inp_kq_mask"`
    - **dtype**: `GGML_TYPE_F32`
    - **shape**: `[n_kv, n_tokens/n_stream, 1, n_stream]` (四维张量)
    - **作用**: 注意力掩码矩阵，用于在计算 Attention Score 时屏蔽掉无效的 token（如 padding tokens 或未来的 tokens，实现因果掩码 Causal Mask）。
  - `inp->self_kq_mask_cnv` (可选):
    - **dtype**: `GGML_TYPE_F16`
    - **作用**: 当启用 Flash Attention 时，对 `self_kq_mask` 进行类型转换（Cast）得到的半精度掩码张量。
  - `inp->self_k_rot` / `inp->self_v_rot` (可选):
    - **name**: `"attn_inp_k_rot"` / `"attn_inp_v_rot"`
    - **dtype**: `GGML_TYPE_F32`
    - **shape**: `[nrot, nrot]` (二维张量)
    - **作用**: 当模型需要对 K 或 V 应用特定的旋转变换时创建的旋转矩阵。

### 2.3 `build_inp_out_ids` (输出 Token 索引输入)

**对应代码**: `llama.cpp/src/llama-graph.cpp` 中的 `llm_graph_context::build_inp_out_ids()` 函数。
**调用位置**: `qwen2.cpp` 等模型构建文件中的 `ggml_tensor * inp_out_ids = build_inp_out_ids();`

- **作用与逻辑**:
  该函数用于构建输出 Token IDs 的张量。在推理时（特别是在 Prompt 处理阶段），通常只需要计算 batch 中最后一个（或特定几个）token 的 logits。该张量记录了需要计算 logits 的 token 索引，后续通过 `GGML_OP_GET_ROWS` 从最后一层的输出中提取对应行，从而避免对所有 token 进行全连接层（LM Head）计算，大幅节省算力。
- **构建的输入张量 (Input Tensors)**:
  - `inp->out_ids` (返回的 `cur`):
    - **name**: 未显式命名
    - **dtype**: `GGML_TYPE_I32`
    - **shape**: `[n_outputs]` (一维张量，`n_outputs` 为需要输出 logits 的 token 数量)
    - **作用**: 存储需要进行 LM Head 计算的 token 索引，用于在最后一层截取需要的特征行。
## 3. Transformer Layer 模块 (Qwen2)

**对应代码**: `llama.cpp/src/models/qwen2.cpp` 中的 `for (int il = 0; il < n_layer; ++il)` 循环部分。

该模块是 Qwen2 模型的核心主体，包含了 `n_layer` 层重复的 Transformer Block 计算。每一层包含 Self-Attention 和 Feed-Forward Network (FFN) 两个主要子模块，并使用了 Pre-RMSNorm 和残差连接。

### 3.1 模块输入与输出
- **输入参数**:
  - `inpL`: 上一层的输出张量（或第一层的 Input Embedding `cur`）。
    - **dtype**: `GGML_TYPE_F32`
    - **shape**: `[n_embd, n_tokens]`
  - `inp_pos`: 当前 batch 的绝对位置 ID。
  - `inp_attn`: 注意力机制辅助输入（包含 KV 缓存索引、掩码等）。
  - `inp_out_ids`: (仅在最后一层可能用到) 需要输出 logits 的 token 索引。
- **输出 (Return)**:
  - `inpL`: 当前层的输出张量，作为下一层的输入。
    - **dtype**: `GGML_TYPE_F32`
    - **shape**: `[n_embd, n_tokens]` (如果最后一层触发了 `inp_out_ids` 优化，则 shape 变为 `[n_embd, n_outputs]`)。

### 3.2 计算逻辑流程图

```mermaid
graph TD
    Start([Layer il 开始]) --> SaveResidual1[保存残差: inpSA = inpL]
    
    %% Attention Norm
    SaveResidual1 --> AttnNorm[build_norm: GGML_OP_RMS_NORM + GGML_OP_MUL]
    
    %% QKV Projections
    AttnNorm --> Q_Proj[Q 映射: GGML_OP_MUL_MAT + GGML_OP_ADD]
    AttnNorm --> K_Proj[K 映射: GGML_OP_MUL_MAT + GGML_OP_ADD]
    AttnNorm --> V_Proj[V 映射: GGML_OP_MUL_MAT + GGML_OP_ADD]
    
    %% Reshape & RoPE
    Q_Proj --> Q_Reshape[GGML_OP_RESHAPE_3D]
    K_Proj --> K_Reshape[GGML_OP_RESHAPE_3D]
    V_Proj --> V_Reshape[GGML_OP_RESHAPE_3D]
    
    Q_Reshape --> Q_RoPE[GGML_OP_ROPE_EXT]
    K_Reshape --> K_RoPE[GGML_OP_ROPE_EXT]
    
    %% Attention Core
    Q_RoPE --> AttnCore[build_attn: KV Cache 存储 + Flash Attention]
    K_RoPE --> AttnCore
    V_Reshape --> AttnCore
    
    %% Attention Output Projection
    AttnCore --> O_Proj[O 映射: GGML_OP_MUL_MAT + GGML_OP_ADD]
    
    %% Last Layer Optimization
    O_Proj --> LastLayerCheck{il == n_layer - 1 且有 inp_out_ids?}
    LastLayerCheck -- 是 --> GetRows[GGML_OP_GET_ROWS 提取目标 token]
    GetRows --> AddResidual1
    LastLayerCheck -- 否 --> AddResidual1[GGML_OP_ADD: cur + inpSA]
    
    %% FFN Norm
    AddResidual1 --> SaveResidual2[保存残差: ffn_inp = cur]
    SaveResidual2 --> FFNNorm[build_norm: GGML_OP_RMS_NORM + GGML_OP_MUL]
    
    %% FFN Core (SwiGLU)
    FFNNorm --> Gate_Proj[Gate 映射: GGML_OP_MUL_MAT]
    FFNNorm --> Up_Proj[Up 映射: GGML_OP_MUL_MAT]
    Gate_Proj --> SiLU[GGML_OP_SILU]
    SiLU --> FFN_Mul[GGML_OP_MUL: SiLU(Gate) * Up]
    Up_Proj --> FFN_Mul
    FFN_Mul --> Down_Proj[Down 映射: GGML_OP_MUL_MAT]
    
    %% FFN Residual & Output
    Down_Proj --> AddResidual2[GGML_OP_ADD: cur + ffn_inp]
    AddResidual2 --> CVec[build_cvec: 控制向量注入]
    CVec --> End([Layer il 结束, inpL = cur])
```

### 3.3 详细执行步骤与 OP 拆解

#### Step 1: Attention RMS Norm (`build_norm`)
- **输入**: `inpL` (`[n_embd, n_tokens]`, `F32`)
- **模型参数**: `model.layers[il].attn_norm` (RMSNorm 权重, `[n_embd]`, `F32`)
- **调用的 OP**:
  1. `GGML_OP_RMS_NORM`: 计算 RMS 归一化。
  2. `GGML_OP_MUL`: 将归一化结果与 `attn_norm` 权重逐元素相乘。
- **输出**: `cur` (命名为 `"attn_norm"`, `[n_embd, n_tokens]`, `F32`)

#### Step 2: Q, K, V 线性映射与偏置
- **输入**: 上一步的 `cur`
- **模型参数**: `wq`, `wk`, `wv` (权重矩阵), `bq`, `bk`, `bv` (偏置向量)
- **调用的 OP**:
  1. `GGML_OP_MUL_MAT`: 分别计算 `wq * cur`, `wk * cur`, `wv * cur`。
  2. `GGML_OP_ADD`: 分别加上偏置 `bq`, `bk`, `bv`。
- **中间变量**:
  - `Qcur`: `"Qcur"`, `[n_embd_head * n_head, n_tokens]`, `F32`
  - `Kcur`: `"Kcur"`, `[n_embd_head * n_head_kv, n_tokens]`, `F32`
  - `Vcur`: `"Vcur"`, `[n_embd_head * n_head_kv, n_tokens]`, `F32`

#### Step 3: Reshape 为多头格式
- **调用的 OP**: `GGML_OP_RESHAPE` (通过 `ggml_reshape_3d` 封装)
- **作用**: 将 2D 张量重塑为 3D，分离出 Head 维度。
- **输出**:
  - `Qcur`: `[n_embd_head, n_head, n_tokens]`
  - `Kcur`: `[n_embd_head, n_head_kv, n_tokens]`
  - `Vcur`: `[n_embd_head, n_head_kv, n_tokens]`

#### Step 4: RoPE 旋转位置编码
- **输入**: `Qcur`, `Kcur`, `inp_pos`
- **调用的 OP**: `GGML_OP_ROPE_EXT`
- **作用**: 根据 `inp_pos` 中的绝对位置，对 Q 和 K 的特征维度应用旋转矩阵，注入位置信息。
- **输出**: 注入位置信息后的 `Qcur` 和 `Kcur` (Shape 不变)。

#### Step 5: 注意力核心计算与输出映射 (`build_attn`)
- **输入**: `Qcur`, `Kcur`, `Vcur`, `inp_attn` (包含掩码和 KV 索引)
- **模型参数**: `wo` (输出映射权重), `bo` (输出偏置)
- **调用的 OP**:
  1. **KV Cache 写入**: `GGML_OP_CPY` (将 `Kcur` 和 `Vcur` 复制到 KV Cache 的指定物理槽位)。
  2. **Attention Score 计算**: `GGML_OP_FLASH_ATTN_EXT` (如果后端支持) 或组合 OP (`MUL_MAT` -> `SCALE` -> `ADD(mask)` -> `SOFT_MAX` -> `MUL_MAT(V)`)。
  3. **O 映射**: `GGML_OP_MUL_MAT` (`wo * attn_out`)。
  4. **O 偏置**: `GGML_OP_ADD` (`+ bo`)。
- **输出**: `cur` (`[n_embd, n_tokens]`, `F32`)

#### Step 6: 最后一层输出 Token 截取优化 (可选)
- **触发条件**: `il == n_layer - 1 && inp_out_ids != nullptr`
- **调用的 OP**: `GGML_OP_GET_ROWS`
- **作用**: 在最后一层，如果只需要部分 token 的 logits（如 prompt 阶段只取最后一个 token），则通过 `GET_ROWS` 提取 `cur` 和 `inpSA` 的对应行，大幅减少后续 FFN 和 LM Head 的计算量。
- **输出**: `cur` 和 `inpSA` 的 shape 变为 `[n_embd, n_outputs]`。

#### Step 7: Attention 残差连接
- **输入**: `cur` (Attention 输出), `inpSA` (Attention 输入)
- **调用的 OP**: `GGML_OP_ADD`
- **输出**: `ffn_inp` (`"ffn_inp"`, `[n_embd, n_tokens]`, `F32`)

#### Step 8: FFN RMS Norm (`build_norm`)
- **输入**: `ffn_inp`
- **模型参数**: `model.layers[il].ffn_norm` (`[n_embd]`, `F32`)
- **调用的 OP**: `GGML_OP_RMS_NORM`, `GGML_OP_MUL`
- **输出**: `cur` (`"ffn_norm"`, `[n_embd, n_tokens]`, `F32`)

#### Step 9: FFN 核心计算 (SwiGLU 结构, `build_ffn`)
- **输入**: 归一化后的 `cur`
- **模型参数**: `ffn_gate`, `ffn_up`, `ffn_down` (权重矩阵, shape 涉及 `n_ff`)
- **调用的 OP**:
  1. **Gate 映射**: `GGML_OP_MUL_MAT` (`ffn_gate * cur`) -> 得到 `[n_ff, n_tokens]`
  2. **SiLU 激活**: `GGML_OP_SILU` (作用于 Gate 映射结果)
  3. **Up 映射**: `GGML_OP_MUL_MAT` (`ffn_up * cur`) -> 得到 `[n_ff, n_tokens]`
  4. **SwiGLU 乘法**: `GGML_OP_MUL` (`SiLU(Gate) * Up`)
  5. **Down 映射**: `GGML_OP_MUL_MAT` (`ffn_down * SwiGLU_out`) -> 映射回 `[n_embd, n_tokens]`
- **输出**: `cur` (`"ffn_out"`, `[n_embd, n_tokens]`, `F32`)

#### Step 10: FFN 残差连接
- **输入**: `cur` (FFN 输出), `ffn_inp` (FFN 输入)
- **调用的 OP**: `GGML_OP_ADD`
- **输出**: `cur` (`[n_embd, n_tokens]`, `F32`)

#### Step 11: 控制向量注入 (`build_cvec`, 可选)
- **调用的 OP**: `GGML_OP_ADD` (如果模型加载了 Control Vector)
- **作用**: 将特定层的控制向量加到当前特征上，用于引导模型生成风格。
- **最终输出**: `cur` (`"l_out"`, `[n_embd, n_tokens]`, `F32`)，并赋值给 `inpL` 供下一层使用。
