# shape 和视图相关的 OP

## `GGML_OP_VIEW`
**不要求**内存排布是连续的，这一点和torch中view的要求不一样。

### ggml_view_1d
```c++
struct ggml_tensor * ggml_view_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset)
```
- 输出tensor为**1**维，shape：[ne0]
- 输出tensor的数据类型和a一致
- 输出tensor将a作为view
- 输出tensor通过offset来获取view数据区域的子数据区域（offset前的数据会被舍弃）
- 将 offset 作为输出 tensor 的params

### ggml_view_2d
```c++
struct ggml_tensor * ggml_view_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset)
```
- 输出tensor为**2**维，shape：[ne0, ne1]
- 输出tensor的数据类型和a一致
- 输出tensor将a作为view
- 输出tensor通过offset来获取view数据区域的子数据区域（offset前的数据会被舍弃）
- 将 offset 作为输出 tensor 的params

### ggml_view_3d
```c++
struct ggml_tensor * ggml_view_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset)
```
- 输出tensor为**3**维，shape：[ne0, ne1, ne2]
- 输出tensor的数据类型和a一致
- 输出tensor将a作为view
- 输出tensor通过offset来获取view数据区域的子数据区域（offset前的数据会被舍弃）
- 将 offset 作为输出 tensor 的params

### ggml_view_4d
```c++
struct ggml_tensor * ggml_view_4d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset)
```
- 输出tensor为**4**维，shape：[ne0, ne1, ne2, ne3]
- 输出tensor的数据类型和a一致
- 输出tensor将a作为view
- 输出tensor通过offset来获取view数据区域的子数据区域（offset前的数据会被舍弃）
- 将 offset 作为输出 tensor 的params

----

## `GGML_OP_RESHAPE`
**要求**内存排布是连续的，这一点和torch中的reshape不一样

### ggml_reshape
```c++
struct ggml_tensor * ggml_reshape(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b)
```
- 输出tesnor将`a`作为view
- 输出tensor的type，和`a`的type一致
- 输出tensor的shape，与`b`的shape一致


### ggml_reshape_1d
```c++
struct ggml_tensor * ggml_reshape_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0)
```
- 输出tensor为**1**维，数据shape为[ne0](ne0 == `a`的元素总数)
- 输出tensor的type和`a`相同
- 输出tensor将a作为view

### ggml_reshape_2d
```c++
struct ggml_tensor * ggml_reshape_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1)
```
- 输出tensor为**2**维，数据shape为[ne0, ne1](ne0 * ne1 == `a`的元素总数)
- 输出tensor的type和`a`相同
- 输出tensor将a作为view

### ggml_reshape_3d
```c++
struct ggml_tensor * ggml_reshape_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2)
```
- 输出tensor为**3**维，数据shape为[ne0](ne0 * ne1 * ne2 == `a`的元素总数)
- 输出tensor的type和`a`相同
- 输出tensor将a作为view

### ggml_reshape_4d
```c++
struct ggml_tensor * ggml_reshape_4d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3)
```
- 输出tensor为**4**维，数据shape为[ne0](ne0 * ne1 * ne2 * ne3 == `a`的元素总数)
- 输出tensor的type和`a`相同
- 输出tensor将a作为view

## `GGML_OP_TRANSPOSE`
**无内存连续要求**

```c++
struct ggml_tensor * ggml_transpose(
        struct ggml_context * ctx,
        struct ggml_tensor  * a)
```

- 交换`a`的 ne[1] 和 ne[0]， nb[1] 和 nb[0]，并将`a`的view作为输出 tensor
- 输出tensor的数据类型和`a`一致
- 输出tensor的维度个数和`a`一致

## `GGML_OP_PERMUTE`
**无内存连续要求**
```c++
struct ggml_tensor * ggml_permute(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3)
```
- 作用同torch的permute算子
- 将{axis0, axis1, axis2, axis3}作为输出tensor的params

以上4个OP对应的 node 功能均**在构建计算图的过程中实现**，在**前向过程中都没有做任何操作**，这里记作 `shape-node`

- `shape-node` 的计算无需额外实现。
- 要求其后紧邻的node的计算过程要能够正确的使用其 shape 和 offset 等信息

在构建计算图的过程中，会将 `shape-node` 中构建的关于 源tensor 的新的 `view` 和 `offset` 传递给其紧邻的后边一个 node，这个 node 根据 `shape-node` 的`view`和`offset`信息进行计算

- 在具体推理实现的过程中，不应该调NPU上的对应算子
- 如果在推理过程中调用了NPU上的对应算子，将会改变实际的数据排布，引起后续tensor操作错误
