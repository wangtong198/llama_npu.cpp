
# 算子名称： GGML_OP_RMS_NORM

要求： 考虑多 stream（若张量在更高维上携带 stream / batch 切片，则 `ne[2]`、`ne[3]` 可大于 1；**RMSNorm 仍对固定的 `(i01,i02,i03)` 沿 `ne[0]` 归一化**。Qwen2 主路径下 **`ne[2]=ne[3]=1`**，多序列并行体现在 **`ne[1]=n_tokens`** 的展开中。）

「下文以 Qwen2 / Qwen2.5 为例；每层与最后的 `build_norm(..., LLM_NORM_RMS, ...)` 会插入本算子」

## 计算图构造

`llm_graph_context::build_norm` 在 `type == LLM_NORM_RMS` 时调用 `ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps)`，得到归一化后的张量；若存在可学习的缩放（及偏置），再依次接 `GGML_OP_MUL`、`GGML_OP_ADD`，**不在** `GGML_OP_RMS_NORM` 单节点内融合。

- **本算子**：仅做按向量 RMS 缩放（无减均值）。
- **后续节点**：`mw` → 逐元素乘权重（Qwen 系列为各层的 `attn_norm`、`ffn_norm`、最后的 `output_norm`，即 gamma）；`mb` → 加偏置。

## 输入

记 `n_embd = hparams.n_embd`，`n_tokens = ubatch.n_tokens`，`n_outputs = params.n_outputs`。

attn_norm: 
  0 … n_layer - 1 层: [n_embd, n_tokens, 1, 1]

ffn_norm: 

  0 … n_layer - 2 层: [n_embd, n_tokens, 1, 1]

  n_layer - 1 层： [n_embd, n_outputs, 1, 1]  （最后一层的 ffn rms norm 保留需要算 logits 的行）

output_norm: 
  [n_embd, n_outputs, 1, 1]

## 输出：

  dtype： 和输入相同

  shape： 和输入相同

  如果是inplace，则和 input 共享存储空间

## 参数
  eps：配置选项

    dtype： 不限，
  
    scalar

---

## 实现原理

约定：对固定的 `(i03, i02, i01)`，取向量 `x[i00]`，`i00 = 0 … ne0-1`，`ne0` 即张量 `ne[0]`。

**1. 平方和**

```
S = x[0]^2 + x[1]^2 + … + x[ne0-1]^2
```

考虑 S 的溢出风险。

**2. 均方、缩放因子**

```
mean_sq = S / ne0
scale = 1 / sqrt(mean_sq + eps)
```

`eps` 来自 `dst->op_params`。实现中断言 `scale > 0`，便于尽早发现上游 `inf` 等。

**3. 输出向量**

先把该向量从 `src0` 拷到 `dst`，再逐元素缩放：

```
y[i00] = x[i00] * scale    （i00 = 0 … ne0-1）
```

---

## 公式

`d = ne[0]`，`x` 为长度 `d` 的向量，`eps` 为小常数。

```
mean_sq = (x[0]^2 + x[1]^2 + … + x[d-1]^2) / d

RMS = sqrt(mean_sq + eps)

y[i] = x[i] / RMS     （i = 0 … d-1）
```

