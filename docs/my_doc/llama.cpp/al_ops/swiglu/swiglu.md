
# 算子名称： GGML_OP_GLU（子类型： GGML_GLU_OP_SWIGLU）

要求： 考虑多 stream

## 输入

构图时有两个输入，落到算子上即 `src[0]` 与可选的 `src[1]`：第一路对应 `src[0]`，第二路对应 `src[1]`（无第二路时 `src[1]` 为空）。

### 两路输入（`src[1]` 存在）

- 形状：`src[0]` 与 `src[1]` 四维长度逐维相同。  
- 数据类型：两路相同。  
- 布局：两路均满足沿 GLU 所需方向的连续条件。  
- SwiGLU 语义：对 `src[0]` 这一路做 SiLU，再与 `src[1]` 逐元素相乘；`src[1]` 不再过 SiLU。  
- 在 FFN 中：`src[0]` 为 gate 投影（加偏置、缩放后的结果），`src[1]` 为 up 投影；构图上先得到 up、再得到 gate，接入本算子时第一路为 gate、第二路为 up。

模型配置信息中的 `intermediate_size` 即 FFN 中间通道维；并行 SwiGLU 两路为 gate、up 各自与输入做矩阵乘后的激活，典型二维图为：

- `src[0]`（gate）：[intermediate_size, n_tokens, 1, 1], `n_tokens` 即当前 `ubatch.n_tokens`。  
- `src[1]`（up）：与 `src[0]` 相同。  

最后一层若在 FFN 前对残差做过输出 token 截取（`n_outputs`），则两路均为: [intermediate_size, n_outputs, 1, 1]。多 stream / 更高维排布时，token 或序列可摊在 `ne[1]`～`ne[3]` 上，但两路仍须逐维相等。

### 单路输入（`src[1]` 为空，gate 与 up 同存于 `src[0]`）

- 仅 `src[0]`：`ne[0]` 须为偶数；在 `ne[0]` 上为前后两半，各长 `ne[0]/2`，逻辑上分别是 gate 与 up 的通道。  
- `swapped` 只在这种形式下起作用：决定前半、后半谁当作 gate、谁当作 up（数学仍为 `silu(gate) * up`）。  
- 其余维与两路情形一致，用于 batch / token / stream。


若把 gate、up 的激活拼进一张张量（等价于单路形式），则须令拼接发生在通道维 `ne[0]` 上：两半各长 `intermediate_size`，则

- `src[0]`：`ne[0] = 2 * intermediate_size`（前半与后半各长 `intermediate_size`，分别对应 gate、up 之一，具体由 `swapped` 与权重排布对齐），`ne[1] = n_tokens`（或最后一层截取后的 `n_outputs`），`ne[2] = ne[3] = 1`。  

此时不存在 `src[1]`

## 输出

- 数据类型：与 `src[0]` 相同。  
- 形状：  
  - 若 `src[1]` 存在：与 `src[0]`（及 `src[1]`）完全一致。
  - 若仅有 `src[0]` 且 gate/up 拼在 `ne[0]`：`dst->ne[0] = src[0]->ne[0] / 2 = intermediate_size`，`ne[1]`～`ne[3]` 与 `src[0]` 相同（例如 `ne[1] = n_tokens` 或 `n_outputs`）。  
- 新分配的结果张量，不与输入共享存储。

## 参数

- `op`：本文为 `GGML_GLU_OP_SWIGLU`。  
- `swapped`：仅单路、两半拼接时有效；两路输入时构图侧一般为假，gate/up 由两路张量本身区分，不再靠 `swapped` 交换半段。

## 限制

- `src[0]` 须 连续；若存在 `src[1]`，也需要连续。  
- 若 `src[1]` 存在：须与 `src[0]` 同形、同类型。  
- 若仅有 `src[0]`：`ne[0]` 须为偶数。  

## 计算原理与过程

对每个位置上的标量通道：

```
silu(t) = t * sigmoid(t)
out = silu(gate) * up    （逐元素相乘）
```

先算 gate 路的 SiLU，再与 up 路相乘；无跨通道归约。两路输入时 gate、up 为两张同形张量。单路输入时在每个行切片内从 `ne[0]` 两半取出 gate、up，由 `swapped` 定半段归属。

## 公式

```
silu(t) = t * sigmoid(t)
y[i] = silu(gate[i]) * up[i]
```

`i` 遍历通道长度：两路输入时为 `ne[0]`（Qwen2.5 下即 `intermediate_size`）；单路输入时为 `src[0]->ne[0] / 2`（即 `intermediate_size`）。
