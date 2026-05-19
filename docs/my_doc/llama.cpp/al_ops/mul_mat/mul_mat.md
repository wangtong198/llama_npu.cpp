
# GGML_OP_MUL_MAT

```c++
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b)
```

## 输入
- a: 
    - shape： [K, M, ne02, ne03]  (ne00=K, ne01=M)
    - type： /

- b: 
    - shape： [K, N, ne12, ne13]  (ne10=K, ne11=N)
    - type： /

## 输出
- shape： [ne01, ne11, ne12, ne13]
- dtype: FP32


## 限制

- `a` 不能是 transposed(即：nb00 必须 <= nb01)
- 当 `a` 的 shape 和 `b` 不一致时，要求 `a` 在ne2和ne3维度上都可以通过整段复制扩展到 `b` 的 shape