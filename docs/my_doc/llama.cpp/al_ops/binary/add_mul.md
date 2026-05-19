
# GGML_OP_ADD

```c++
static struct ggml_tensor * ggml_add_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        bool                  inplace)

static struct ggml_tensor * ggml_mul_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        bool                  inplace)
```

## 输入
- a:
    - shape： 四维

    - type： /

- b
    - shape： 四维

    - type： /

## 输出
- inplace == true:
    - `a`的view
    - shape 和 `a` 一致
    - dtype 和 `a` 一致

- inplace = fasle:
    - 新生成的tensor
    - shape 和 `a` 一致
    - dtype 和 `a` 一致


## 限制

当 `b` 的 shape 和 `a` 不一致时，要求 `b` 在每一个维度上都可以通过整段复制扩展到 `a` 的 shape
