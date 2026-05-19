# `GGML_OP_CONT`

### 硬件计算要求
硬件在执行计算的计算单元中要求操作数是连续的，如果操作数不连续，即使shape信息正确，也无法正确的计算。

Q：
- 硬件有没有 continue 算子的计算单元 ？
- DMA有没有按小块搬移的功能？

```c++
static struct ggml_tensor * ggml_cont_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a)
```
根据 node的 shape、dtype、stride等信息，将输入tensor `a`中**复制**到输出tensor中
