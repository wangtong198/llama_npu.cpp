# AWNPU 后端实现与推理部署说明

## 1. 概述

AWNPU 是 llama_npu.cpp 项目中为 NPU（神经网络处理器）设计的 ggml 后端插件。它遵循 ggml 标准后端架构，作为一等后端（first-class backend）集成到 llama.cpp 的推理流程中。当前阶段的实现以 **CPU 参考内核 + 模拟探测** 为主，为后续接入真实 NPU 硬件/运行时提供了完整的架构框架。

### 核心设计特点

- **主机共享内存模型**：所有张量分配在主机 DDR 中（`is_host = true`），CPU 可直接访问，预期 NPU 与 CPU 零拷贝协作。
- **同步执行**：计算图节点顺序执行，无异步调度。
- **自动 CPU 回退**：任何 NPU 内核执行失败或不支持的操作，自动通过 CPU 后端完成计算。
- **弱符号驱动探测**：通过 `awnpu_probe()` 弱符号函数实现真实 NPU 硬件的运行时探测，无需重新编译。
- **模拟模式**：通过环境变量 `SIM_PROBE=1` 可在没有 NPU 硬件的环境下完整运行推理流程。

---

## 2. 目录结构

```
ggml/
  include/
    ggml-awnpu.h                         # 公共 API（仅 ggml_backend_awnpu_reg）
  src/
    ggml-awnpu/
      CMakeLists.txt                     # 构建配置
      ggml-awnpu.cpp                     # 主后端实现：注册表、设备、缓冲区、图计算
      ggml-awnpu-graph-node.h            # 图节点分发声明
      ggml-awnpu-graph-node.cpp          # 图节点分发实现（op 支持检查 + 内核路由）
      ggml-awnpu-kernels.def             # 支持的计算操作清单（X-macro 定义，单一事实来源）
      ggml-awnpu-kernels-native.h        # 内核函数声明（X-macro 生成）
      ggml-awnpu-kernels-native.cpp      # 本机 CPU 参考内核实现（所有 op）
      ggml-awnpu-kernels-stub.cpp        # 桩内核（全部返回失败，强制 CPU 回退）
      ggml-awnpu-layer-map.h             # 张量/节点 NPU 放置逻辑声明
      ggml-awnpu-layer-map.cpp           # 张量/节点 NPU 放置逻辑实现
      ggml-awnpu-names.h                 # llama.cpp 张量命名约定常量
      test-awnpu-kernels.cpp             # 内核正确性测试

scripts/
  awnpu/
    run-completion-sim.sh                # 模拟模式运行 llama-completion 的 Shell 脚本
    run_completion_sim.py                # 模拟模式运行 llama-completion 的 Python 脚本
```

---

## 3. 构建配置

### CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `GGML_AWNPU` | `ON` | 启用 AWNPU 后端 |
| `NATIVE_KERNELS` | `OFF` | 使用本机 CPU 参考内核（ON）还是桩内核（OFF） |

### 构建命令

```bash
# 基础构建（桩内核模式：所有操作 CPU 回退）
cmake -B build
cmake --build build

# 本机内核模式（CPU 参考实现，可完整运行推理）
cmake -B build -DNATIVE_KERNELS=ON
cmake --build build

# 构建并运行内核正确性测试
cmake -B build -DNATIVE_KERNELS=ON
cmake --build build --target test-awnpu-kernels
./build/bin/test-awnpu-kernels
```

`NATIVE_KERNELS=OFF` 时，所有 NPU 内核返回 `GGML_STATUS_FAILED`，触发 CPU 回退链，适用于只有桩实现的场景。`NATIVE_KERNELS=ON` 时，使用完整 CPU 参考实现，可在模拟模式下端到端运行推理。

---

## 4. 核心模块

### 4.1 后端注册（Registry）

`ggml_backend_awnpu_reg()` 是唯一的公共 API 入口。调用流程：

1. 执行硬件探测（`awnpu_probe()` 或模拟探测）
2. 为每个发现的设备创建 `ggml_backend_awnpu_device_context`
3. 注册缓冲区类型和设备到 ggml 后端框架
4. 返回 `ggml_backend_reg_t`

在 llama.cpp 的设备选择流程中，AWNPU 的优先级排在 GPU/iGPU 之后、CPU 之前。

### 4.2 设备探测（Device Probing）

支持两种探测模式：

**A. 模拟探测**（`SIM_PROBE=1`）
- 返回 1 个设备，内存量等于主机总内存
- 用于无 NPU 硬件时的开发和测试

**B. 真实硬件探测**
- 调用弱符号函数 `awnpu_probe(awnpu_probe_result *result)`
- 该函数由 NPU 运行时共享库提供
- 若运行时未加载（弱符号为 `nullptr`），回退到模拟模式（需 `SIM_PROBE=1`）

### 4.3 缓冲区管理（Buffer）

- 内存通过 `posix_memalign`（Linux）或 `_aligned_malloc`（Windows）在主机共享 DDR 中分配
- `is_host = true`：张量直接 CPU 可访问
- 支持 `buffer_from_host_ptr`（包装外部内存）
- 所有缓冲区操作（init_tensor、memset、cpy_tensor）通过标准 `memcpy`/`memset` 完成

### 4.4 图层放置（Layer Map）

模块决定哪些操作和权重放置在 NPU 上：

- **`weight_on_npu(tensor)`**：判断张量是否在 NPU 缓冲区上。条件是缓冲区名称以 `"AWNPU"` 开头，且用途为 `WEIGHTS`（或 `ANY` + `op == NONE`）。
- **`device_supports_op(op)`**：判断操作是否适合 NPU 执行。满足以下任一条件即为支持：
  1. 布局操作（NONE、RESHAPE、VIEW、PERMUTE、TRANSPOSE）
  2. 操作数直接为 NPU 上的权重（如 `GET_ROWS` 权重、`MUL_MAT`/`MUL_MAT_ID` 的权重 src）
  3. 操作在图中可从 NPU 权重传递到达
- **`node_on_npu(node)`**：递归图遍历，沿着数据流从 NPU 权重传播，确定哪些中间节点也应在 NPU 上执行。

### 4.5 图计算（Graph Compute）

`ggml_backend_awnpu_graph_compute()` 顺序遍历计算图节点，对每个节点：

1. **布局操作**：直接返回成功（无需计算）
2. **不支持的操作**：记录一次警告，回退到 CPU
3. **NPU 内核执行**：调用对应的内核函数
4. **内核失败**：记录一次警告，回退到 CPU

CPU 回退基于单节点子图计算：创建只含一个节点的计算图视图，通过 `ggml_graph_plan` + `ggml_graph_compute` 在 CPU 后端上执行。工作缓冲区可复用增长，支持线程数配置和中止回调。

### 4.6 内核分发（Kernel Dispatch）

以 `ggml-awnpu-kernels.def` 为单一事实来源，通过 X-macro 机制驱动：

- **声明**（`ggml-awnpu-kernels-native.h`）：为每个操作生成内核函数声明
- **支持检查**（`ggml-awnpu-graph-node.cpp`）：生成 `op_supported()` 中的操作白名单
- **路由分发**（`ggml-awnpu-graph-node.cpp`）：生成 switch 语句将 ggml op 路由到对应内核

---

## 5. 支持的操作

### 计算操作（23 个）

| 操作 | 内核函数 | 说明 |
|------|---------|------|
| `GET_ROWS` | `get_rows` | 嵌入表查找 |
| `MUL_MAT` | `mul_mat` | 矩阵乘法 |
| `MUL_MAT_ID` | `mul_mat_id` | MoE 按专家索引的矩阵乘法 |
| `ADD` | `add` | 逐元素加法（支持广播） |
| `ADD_ID` | `add_id` | 按索引行加法 |
| `ADD1` | `add1` | 张量 + 标量 |
| `MUL` | `mul` | 逐元素乘法 |
| `DIV` | `div` | 逐元素除法 |
| `SUB` | `sub` | 逐元素减法 |
| `SQR` | `sqr` | 逐元素平方 |
| `SQRT` | `sqrt` | 逐元素平方根 |
| `LOG` | `log` | 逐元素自然对数 |
| `UNARY` | `unary` | 一元操作（RELU、SILU、GELU 等 20+ 种变体） |
| `NORM` | `norm` | 层归一化 |
| `RMS_NORM` | `rms_norm` | RMS 归一化 |
| `SOFT_MAX` | `soft_max` | Softmax（支持 mask 和 ALiBi） |
| `ROPE` | `rope` | 旋转位置编码（前向） |
| `ROPE_BACK` | `rope_back` | 旋转位置编码（反向） |
| `SCALE` | `scale` | dst = src0 * scale + bias |
| `CLAMP` | `clamp` | 值裁剪到 [min, max] |
| `SET_ROWS` | `set_rows` | 按索引设置行 |
| `GLU` | `glu` | 门控线性单元（REGLU、GEGLU、SWIGLU 等） |
| `FLASH_ATTN_EXT` | `flash_attn_ext` | Flash Attention（缩放点积注意力） |
| `CPY` | `cpy` | 复制（含类型转换） |
| `CONT` | `cont` | 连续化 |
| `DUP` | `dup` | 复制 |
| `PAD` | `pad` | 零填充 |

### 布局操作（5 个，无需计算直接返回）

`NONE`、`RESHAPE`、`VIEW`、`PERMUTE`、`TRANSPOSE`

### 支持的数据类型

- **F32**：所有操作支持 F32 输入和输出
- **F16 / BF16**：大多数逐元素操作通过加载时转 F32、存储时转回的机制支持
- **I32**：用于索引类操作（GET_ROWS、ADD_ID、ROPE 等）
- **量化类型**：GET_ROWS 可从量化嵌入表读取（通过 `ggml_type_traits::to_float`）

---

## 6. CPU 回退机制

AWNPU 后端实现了健壮的多层 CPU 回退保护：

```
foreach node in graph:
    if layout op        → 直接成功（零开销）
    if op not supported → 警告 + CPU 回退
    call NPU kernel     → 失败时警告 + CPU 回退
```

回退特点：
- **去重日志**：相同操作的失败警告只打印一次，避免日志泛滥
- **复用工作缓冲区**：CPU 回退的工作缓冲区随需增长，避免频繁分配
- **单节点子图**：每次回退只计算一个节点，开销可控
- **线程数配置**：通过 `GGML_AWNPU_THREADS` 环境变量或后端参数 `threads=N` 控制

---

## 7. 真实 NPU 硬件接入点

AWNPU 框架为接入真实 NPU 硬件预留了明确的扩展点：

1. **`awnpu_probe()` 弱符号**：NPU 运行时共享库提供此函数，返回设备数量、内存信息等。运行时通过 `dlopen` 加载后即可自动发现设备。

2. **内核替换**：将 `ggml-awnpu-kernels-stub.cpp`（或 `ggml-awnpu-kernels-native.cpp`）中的函数体替换为真实 NPU 运行时调用（如调用 NPU 驱动 API 下发计算任务）。

3. **内存管理**：若 NPU 有独立设备内存，需修改缓冲区分配逻辑（`ggml_backend_awnpu_buffer_alloc_buffer`），增加设备内存分配路径和数据传输逻辑。

---

## 8. 推理部署用法

### 8.1 命令行参数

使用 llama.cpp 标准参数选择 AWNPU 后端：

```bash
# 指定 AWNPU 设备
--device AWNPU0

# 控制卸载到 NPU 的 Transformer 层数
-ngl 99

# 示例：全部 99 层卸载到 AWNPU
llama-cli -m model.gguf --device AWNPU0 -ngl 99 -p "你好" -n 128
```

### 8.2 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SIM_PROBE` | `0` | 设为 `1` 启用模拟设备探测（无需 NPU 硬件） |
| `GGML_AWNPU_THREADS` | CPU 核心数 | CPU 回退时的线程数 |

### 8.3 模拟模式运行（无 NPU 硬件）

**Shell 脚本方式：**

```bash
# 默认参数运行（Qwen2.5-0.5B 模型）
AWNPU_LLM_SIM=1 ./scripts/awnpu/run-completion-sim.sh

# 自定义模型和提示词
MODEL=/path/to/model.gguf AWNPU_LLM_SIM=1 ./scripts/awnpu/run-completion-sim.sh -p "1+1等于几？" -n 64

# 指定设备和层数
DEVICE=AWNPU0 NGL=99 AWNPU_LLM_SIM=1 ./scripts/awnpu/run-completion-sim.sh

# 列出可用设备
AWNPU_LLM_SIM=1 ./scripts/awnpu/run-completion-sim.sh --list-devices
```

**Python 脚本方式：**

```bash
# 基础运行
python3 scripts/awnpu/run_completion_sim.py

# 自定义参数
python3 scripts/awnpu/run_completion_sim.py -p "今天天气怎么样？" -n 128 -c 4096

# 列出设备
python3 scripts/awnpu/run_completion_sim.py --list-devices
```

**直接运行 llama-cli：**

```bash
# 构建后
AWNPU_LLM_SIM=1 ./build/bin/llama-cli \
    -m /path/to/model.gguf \
    --device AWNPU0 \
    -ngl 99 \
    -c 8192 \
    -n 128 \
    -p "你好，请介绍一下你自己。"
```

### 8.4 后端参数

通过 `--device` 或 API 传递后端初始化参数：

| 参数 | 说明 |
|------|------|
| `threads=N` | 设置 CPU 回退线程数 |
| `ref=1` | 使用参考（慢速）CPU 实现 |

### 8.5 API 方式（C/C++ 程序集成）

```cpp
#include "ggml/include/ggml-awnpu.h"

// 显式按类型初始化
ggml_backend_t backend = ggml_backend_init_by_type(
    GGML_BACKEND_DEVICE_TYPE_AWNPU, nullptr);

// 或通过名称
ggml_backend_t backend = ggml_backend_init_by_name("AWNPU0", nullptr);

// 或自动选择最优后端（AWNPU 优先级在 GPU 之后、CPU 之前）
ggml_backend_t backend = ggml_backend_init_best();
```

---

## 9. 典型工作流

### 开发与调试流程

```
1. 构建（桩模式）
   cmake -B build && cmake --build build
   → 所有操作 CPU 回退，验证框架流程

2. 构建（本机内核模式）
   cmake -B build -DNATIVE_KERNELS=ON && cmake --build build
   → CPU 参考实现，可端到端推理

3. 运行正确性测试
   cmake --build build --target test-awnpu-kernels
   ./build/bin/test-awnpu-kernels
   → 验证所有内核与 CPU 参考的一致性

4. 模拟模式推理
   AWNPU_LLM_SIM=1 ./scripts/awnpu/run-completion-sim.sh
   → 验证端到端推理流程正确性

5. 接入真实 NPU 运行时
   → 提供 awnpu_probe() 实现
   → 替换内核函数体为 NPU API 调用
   → 去除 SIM_PROBE，直接运行
```

### 支持的模型

当前操作集覆盖了 Qwen2.5-0.5B 等典型 Transformer 类模型所需的所有计算操作。任何使用标准 Transformer 架构（Embedding → Attention → FFN → LM Head）且不涉及批量矩阵乘法（`MUL_MAT` 以外的不常见操作）的模型均可运行。MoE 架构通过 `MUL_MAT_ID` 和 `ADD_ID` 得到支持。

---

## 10. 架构总结

```
┌─────────────────────────────────────────────────────────┐
│                    llama.cpp 推理层                       │
│              (llama_supports_gpu_offload 等)              │
├─────────────────────────────────────────────────────────┤
│                 ggml 后端框架                              │
│         (backend init / device / buffer API)             │
├─────────────────────────────────────────────────────────┤
│               AWNPU 后端 (ggml-awnpu.cpp)                 │
│  ┌───────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ Registry  │  │  Device  │  │   Graph Compute       │  │
│  │ (probe)   │  │ (buffer) │  │   (node iteration)    │  │
│  └───────────┘  └──────────┘  └──────────┬───────────┘  │
│                                           │               │
│              ┌────────────────────────────┼──┐           │
│              │   Graph Node Dispatch      │  │           │
│              │   (op_supported / route)   │  │           │
│              └────────────────────────────┼──┘           │
│                                           │               │
│     ┌─────────────────────────────────────┼──────┐      │
│     │        Kernel Layer                 │      │      │
│     │  ┌──────────────────┐  ┌───────────┴────┐ │      │
│     │  │ native (CPU ref) │  │ stub (fallback) │ │      │
│     │  └──────────────────┘  └────────────────┘ │      │
│     │  ┌──────────────────────────────────────┐ │      │
│     │  │ future: real NPU runtime calls       │ │      │
│     │  └──────────────────────────────────────┘ │      │
│     └───────────────────────────────────────────┘      │
│                                                         │
│               Layer Map (weight/node placement)         │
│               CPU Fallback (single-node subgraph)       │
└─────────────────────────────────────────────────────────┘
```
