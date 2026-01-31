# Unsloth-Zoo NPU 适配总结与 Roadmap

> 本文档总结了 Unsloth-Zoo 项目中针对华为 Ascend NPU 的适配工作，基于代码中的 `Unsloth-PTO-XXX` 标记进行整理。

## 目录

- [修改标记统计](#修改标记统计)
- [已完成的 NPU 适配 (VERIFY 状态)](#已完成的-npu-适配-verify-状态)
- [待修复项 (FIXME 状态)](#待修复项-fixme-状态)
- [待办事项 (TODO 状态)](#待办事项-todo-状态)
- [GPT-OSS 模型适配说明](#gpt-oss-模型适配说明)
- [NPU 全面支持 Roadmap](#npu-全面支持-roadmap)
- [关键依赖关系](#关键依赖关系)
- [标记规范](#标记规范)

---

## 修改标记统计

根据代码库搜索，发现以下标记分布：

| 标记类型 | 数量 | 说明 |
|---------|------|------|
| `Unsloth-PTO-VERIFY` | ~15 | 需要验证的 NPU 适配代码 |
| `Unsloth-PTO-FIXME` | ~15 | 需要修复的 NPU 相关问题 |
| `Unsloth-PTO-TODO` | ~2 | 待完成的 NPU 功能 |

### 按文件分布

| 文件 | VERIFY | FIXME | TODO | 说明 |
|------|--------|-------|------|------|
| `device_type.py` | 2 | 0 | 1 | 设备检测核心 |
| `__init__.py` | 1 | 2 | 0 | 环境变量配置 |
| `gradient_checkpointing.py` | 5 | 0 | 0 | 梯度检查点 |
| `compiler.py` | 1 | 0 | 0 | 编译器配置 |
| `vllm_utils.py` | 4 | 0 | 0 | vLLM 推理 |
| `rl_replacements.py` | 1 | 6 | 0 | RL 训练替换 |
| `loss_utils.py` | 1 | 1 | 0 | 损失函数 |
| `fused_losses/cross_entropy_loss.py` | 2 | 0 | 0 | 融合交叉熵 |
| `temporary_patches/gpt_oss.py` | 1 | 1 | 0 | GPT-OSS 补丁 |
| `temporary_patches/__init__.py` | 0 | 1 | 0 | 补丁入口 |

---

## 已完成的 NPU 适配 (VERIFY 状态)

### 1. 设备检测与初始化

**文件**: `unsloth_zoo/device_type.py`

```python
# Line 43-44
elif hasattr(torch, "npu") and torch.npu.is_available(): # Unsloth-PTO-VERIFY: check npu devices
    return "npu"

# Line 69-70
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: check npu devices
    return torch.npu.device_count()
```

**状态**: ✅ 已实现，需验证

### 2. 梯度检查点

**文件**: `unsloth_zoo/gradient_checkpointing.py`

```python
# Line 65-68 - AMP 自定义前向/反向
# Unsloth-PTO-VERIFY: check torch_npu functions
if DEVICE_TYPE == "npu":
    torch_amp_custom_fwd = torch.npu.amp.custom_fwd
    torch_amp_custom_bwd = torch.npu.amp.custom_bwd

# Line 321-322 - GPU Stream
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: check npu devices
    torch_gpu_stream = torch.npu.stream

# Line 351-352 - BFloat16 支持
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: check npu devices
    SUPPORTS_BFLOAT16 = True

# Line 367-368 - 设备数量
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: check npu devices
    n_gpus = torch.npu.device_count()

# Line 383-387 - Extra Streams
# Unsloth-PTO-VERIFY: check npu devices
if DEVICE_TYPE in ("cuda", "hip"):
    EXTRA_STREAMS = tuple([torch.cuda.Stream() for i in range(n_gpus)])
elif DEVICE_TYPE == "npu":
    EXTRA_STREAMS = tuple([torch.npu.Stream() for i in range(n_gpus)])
```

**状态**: ✅ 已实现，需验证

### 3. 编译器配置

**文件**: `unsloth_zoo/compiler.py`

```python
# Line 90-91
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: check npu devices
    OLD_CUDA_ARCH_VERSION = False
```

**状态**: ✅ 已实现，需验证

### 4. vLLM 推理工具

**文件**: `unsloth_zoo/vllm_utils.py`

```python
# Line 88-89 - 内存信息
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: check npu devices
    free_memory, total_memory = torch.npu.mem_get_info()

# Line 1779-1780 - 数据类型
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: check npu devices
    _dtype = torch.bfloat16

# Line 1845-1846 - 前缀缓存
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: check npu devices
    enable_prefix_caching = True

# Line 1946-1948 - 设备信息
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: check npu devices
    platform = "Ascend NPU"
    npu_name = torch.npu.get_device_properties(0).name
```

**状态**: ✅ 已实现，需验证

### 5. 融合交叉熵损失

**文件**: `unsloth_zoo/fused_losses/cross_entropy_loss.py`

```python
# Line 124-126 - 内存查询
# Unsloth-PTO-VERIFY: check the torch_npu.mem_get_info()
if DEVICE_TYPE == "npu":
    free, total = torch.npu.mem_get_info(0)
```

**状态**: ✅ 已实现，需验证

### 6. RL 替换函数

**文件**: `unsloth_zoo/rl_replacements.py`

```python
# Line 938-940 - 设备同步
# Unsloth-PTO-VERIFY
if hasattr(torch, 'npu') and torch.npu.is_available():
    torch.npu.synchronize()
```

**状态**: ✅ 已实现，需验证

### 7. GPT-OSS 临时补丁

**文件**: `unsloth_zoo/temporary_patches/gpt_oss.py`

```python
# Line 543-544 - 设备内存
elif DEVICE_TYPE == "npu": # Unsloth-PTO-VERIFY: check npu devices
    device_memory = torch.npu.memory.mem_get_info(0)[-1]
```

**状态**: ✅ 已实现，需验证

---

## 待修复项 (FIXME 状态)

### 1. 环境变量配置

**文件**: `unsloth_zoo/__init__.py`

```python
# Line 19
# Unsloth-PTO-FIXME: set the environment variables for the *npu* device

# Line 175-177 - 内存分配配置
elif DEVICE_TYPE == "npu": # Unsloth-PTO-FIXME: check npu devices
    delete_key("PYTORCH_CUDA_ALLOC_CONF")
    delete_key("PYTORCH_HIP_ALLOC_CONF")

# Line 202-204 - GPT-OSS harmony 编码
# Unsloth-PTO-FIXME: support encode_conversations_with_harmony in gpt_oss
```

**待修复**:
- 添加 NPU 特定的环境变量 (如 `ASCEND_LAUNCH_BLOCKING`)
- 配置 NPU 内存分配策略

### 2. RL 训练函数

**文件**: `unsloth_zoo/rl_replacements.py`

```python
# Line 30-32 - torch.compile 支持
# Unsloth-PTO-FIXME: update the torch compile functions
# Unsloth-PTO-FIXME
# @torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)

# Line 45-47, 69-70 - 选择性 log softmax
# Unsloth-PTO-FIXME
# @torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_selective_log_softmax(...)
def chunked_hidden_states_selective_log_softmax(...)

# Line 174-175 - 稳定排序
# Unsloth-PTO-FIXME
# Must do stable=True since binary mark is unordered

# Line 249-251 - 内存信息
# Unsloth-PTO-FIXME
elif torch.npu.is_available():
    free_bytes, _ = torch.npu.mem_get_info()

# Line 436-438 - torch.compile 跳过
# Unsloth-PTO-FIXME
# Skip torch.compile on NPU as Triton is not supported

# Line 507-509 - torch.compile 跳过
# Unsloth-PTO-FIXME
# Skip torch.compile on NPU as it's not fully supported
_is_npu = hasattr(torch, 'npu') and torch.npu.is_available()

# Line 849 - GRPO 计算
# Unsloth-PTO-FIXME
```

**待修复**:
- 使用 `torchair` 替代 CUDA torch.compile
- 研究 NPU 上的 torch.compile 最佳实践
- 验证稳定排序在 NPU 上的行为

### 3. 损失函数

**文件**: `unsloth_zoo/loss_utils.py`

```python
# Line 68
elif DEVICE_TYPE == "npu": # Unsloth-PTO-FIXME: check torch_npu
    from cut_cross_entropy import linear_cross_entropy
```

**待修复**:
- 确认 `cut_cross_entropy` 在 NPU 上的兼容性
- 可能需要 NPU 原生实现

### 4. GPT-OSS 模型

**文件**: `unsloth_zoo/temporary_patches/gpt_oss.py`

```python
# Line 17
# Unsloth-PTO-FIXME: support gpt_oss with *cuda* and *xpu* devices
```

**待修复**:
- GPT-OSS 当前仅支持 CUDA
- 需要添加完整的 NPU 设备支持
- `triton_kernels` 依赖需要替换

**文件**: `unsloth_zoo/temporary_patches/__init__.py`

```python
# Line 22-23
# Unsloth-PTO-FIXME: support gpt_oss
# from .gpt_oss import *
```

**待修复**:
- GPT-OSS 补丁未启用
- 需要完成 NPU 适配后启用

---

## 待办事项 (TODO 状态)

### 1. bitsandbytes NPU 实现

**文件**: `unsloth_zoo/device_type.py`

```python
# Line 77
# Unsloth-PTO-TODO: Implement for NPU for BITSANDBYTES
```

**待办**:
- 研究 bitsandbytes NPU 兼容性
- 可能需要使用 PyPTO/PTO-ISA 实现量化算子
- 4-bit/8-bit 量化内核替换

---

## GPT-OSS 模型适配说明

### 当前状态

GPT-OSS 模型 (`gpt_oss.py`) 目前**仅支持 CUDA 设备**，主要依赖包括：

1. **`triton_kernels`** - OpenAI 提供的 Triton 内核
   - `matmul_ogs` - MXFP4 矩阵乘法
   - `swiglu` - SwiGLU 激活函数
   - `routing` - MoE 路由

2. **`torch.cuda.device`** - CUDA 设备上下文管理器

3. **`torch.compile`** - 需要 CUDA 后端

### NPU 适配需要的工作

| 组件 | 当前实现 | NPU 替代方案 | 优先级 |
|------|---------|-------------|--------|
| `triton_kernels.matmul_ogs` | Triton 内核 | PyPTO/PTO-ISA 实现 | P0 |
| `triton_kernels.swiglu` | Triton 内核 | 已有 PyTorch 回退 | P1 |
| `triton_kernels.routing` | Triton 内核 | PyPTO/PTO-ISA 实现 | P0 |
| `torch.cuda.device` | CUDA 上下文 | `torch.npu.device` | P1 |
| `torch.compile` 装饰器 | CUDA 后端 | `torchair` 后端 | P2 |
| Flex Attention | CUDA 实现 | NPU SDPA 或 eager | P1 |

### 文件中的关键位置

```python
# Line 47 - 设备上下文 (需要条件化)
torch_cuda_device = torch.cuda.device

# Line 89-93 - triton_kernels 导入 (需要 NPU 替代)
def patch_gpt_oss():
    try:
        import triton_kernels
    except Exception as e:
        return raise_error("Please install triton_kernels", e)

# Line 110-118 - matmul_ogs, swiglu 导入
from triton_kernels import matmul_ogs, swiglu
FnSpecs, FusedActivation, matmul_ogs = (...)
swiglu_fn = swiglu.swiglu_fn

# Line 311 - routing 导入
routing = triton_kernels.routing.routing
```

---

## NPU 全面支持 Roadmap

### Phase 1: 基础验证 ✅ (当前阶段)

```
torch_npu 基础集成
├── ✅ 设备检测 (device_type.py)
├── ✅ 设备数量获取
├── ✅ 内存信息查询
├── ✅ Stream 管理
├── ✅ BFloat16 支持检测
├── ✅ AMP 混合精度
└── ⚠️ 基础训练流程 (需验证)
```

**目标**: 确保 Unsloth-Zoo 可以在 NPU 上运行基础训练流程

**验证清单**:
- [ ] 运行简单的 LoRA 微调
- [ ] 验证梯度检查点工作正常
- [ ] 确认内存管理正确
- [ ] 检查 BF16/FP16 精度

### Phase 2: torch.compile 适配 🔧 (下一阶段)

```
torchair 集成
├── 🔧 研究 torchair torch.compile 后端
├── 🔧 rl_replacements.py 中的 compile 装饰器
├── 🔧 gradient_checkpointing.py 优化
├── 🔧 fused_losses 编译支持
└── 🔧 性能对比测试
```

**目标**: 通过 `torchair` 实现 NPU 上的 `torch.compile` 加速

**关键技术**:
- `torchair`: 华为的 torch.compile NPU 后端
- NPU Graph 编译优化
- 算子融合

**示例配置** (待验证):
```python
import torchair

# NPU torch.compile 选项
npu_compile_options = {
    "backend": "torchair",
    "dynamic": True,
    "fullgraph": False,  # NPU 可能需要 False
}
```

### Phase 3: 核心算子 NPU 实现 📋 (中期目标)

使用 **PyPTO / PTO-ISA** 实现关键算子：

```
PyPTO / PTO-ISA 算子实现
│
├── P0 (最高优先级) - GPT-OSS 关键
│   ├── 📋 matmul_ogs - MXFP4 矩阵乘法 (替换 triton_kernels)
│   ├── 📋 routing - MoE 路由算子
│   └── 📋 flash_attention - NPU Flash Attention
│
├── P1 (高优先级) - 训练核心
│   ├── 📋 rms_layernorm - RMS LayerNorm
│   ├── 📋 rope_embedding - RoPE 位置编码
│   ├── 📋 cross_entropy_loss - 交叉熵损失
│   └── 📋 swiglu/geglu - 激活函数
│
├── P2 (中优先级) - 量化支持
│   ├── 📋 nf4_quantize/dequantize - NF4 量化
│   ├── 📋 int8_matmul - INT8 矩阵乘法
│   └── 📋 cgemm_4bit_inference - 4-bit 推理
│
└── P3 (低优先级) - 推理优化
    ├── 📋 paged_attention - 分页注意力
    └── 📋 fused_moe - 融合 MoE
```

**技术路线**:
- **PyPTO**: Python 级别的 PTO 编程接口
- **PTO-ISA**: NPU 底层指令集架构

### Phase 4: GPT-OSS 完整支持 📋 (长期目标)

```
GPT-OSS NPU 适配
│
├── 🔧 替换 triton_kernels 依赖
│   ├── matmul_ogs → PyPTO 实现
│   ├── swiglu → PyTorch 原生 (已有)
│   └── routing → PyPTO 实现
│
├── 🔧 设备上下文适配
│   ├── torch_cuda_device → 条件化
│   └── with torch.npu.device(...)
│
├── 🔧 Flex Attention 适配
│   ├── 研究 NPU SDPA 接口
│   ├── 或使用 eager attention 回退
│   └── 性能优化
│
└── 🔧 启用 gpt_oss 补丁
    └── from .gpt_oss import *
```

### Phase 5: 量化与推理 📋 (长期目标)

```
bitsandbytes + vLLM-Ascend
│
├── bitsandbytes NPU 支持
│   ├── 📋 INT4/INT8 量化算子
│   ├── 📋 QLoRA 训练支持
│   └── 📋 GPTQ/AWQ 模型加载
│
└── vLLM-Ascend 集成
    ├── 📋 验证 vLLM-Ascend 兼容性
    ├── 📋 快速推理实现
    └── 📋 LoRA 推理优化
```

---

## 关键依赖关系

```
Unsloth-Zoo NPU 生态依赖图
│
├── torch_npu (基础层) ✅
│   ├── 已集成基础 API
│   ├── 版本要求: >= 2.4.0
│   └── 提供: 设备管理、张量操作、AMP
│
├── torchair (编译优化层) 🔧
│   ├── 需要验证和适配
│   ├── 依赖: torch_npu
│   └── 提供: torch.compile NPU 后端
│
├── PyPTO / PTO-ISA (算子加速层) 📋
│   ├── 待开发
│   ├── 依赖: torch_npu, CANN
│   ├── 提供: 高性能自定义算子
│   └── 替换范围:
│       ├── triton_kernels (GPT-OSS)
│       ├── bitsandbytes 量化算子
│       └── vLLM-Ascend 推理算子
│
├── vllm_ascend (推理加速层) 📋
│   ├── 需要集成和测试
│   ├── 依赖: torch_npu
│   └── 提供: 高效推理、LoRA 服务
│
└── bitsandbytes-npu (量化层) 📋
    ├── 待调研和实现
    ├── 依赖: torch_npu
    ├── 提供: 4-bit/8-bit 量化
    └── FP8: 暂不支持 (910B/C 硬件限制)
```

---

## 下一步行动建议

### 短期目标 (1-2 周)

1. **验证 VERIFY 标记**
   - 在 910B/C 上运行完整训练流程
   - 确认所有 `VERIFY` 标记的代码工作正常
   - 记录性能基准数据

2. **修复关键 FIXME**
   - 配置 NPU 环境变量
   - 验证 `cut_cross_entropy` 兼容性

### 中期目标 (1-2 月)

3. **torch.compile 适配**
   - 研究 `torchair` 对 Unsloth-Zoo 的支持
   - 测试 `torch.compile` 在 NPU 上的效果
   - 逐步启用 rl_replacements.py 中的装饰器

4. **GPT-OSS 初步支持**
   - 替换 `torch_cuda_device` 为条件化版本
   - 使用 PyTorch 原生实现替代 triton_kernels
   - 启用 `gpt_oss.py` 补丁

### 长期目标 (3-6 月)

5. **PyPTO/PTO-ISA 算子开发**
   - 优先实现 `matmul_ogs` (GPT-OSS 关键)
   - 实现 `rms_layernorm` (性能敏感)
   - 性能对标 CUDA Triton 内核

6. **完整 NPU 生态**
   - bitsandbytes NPU 量化支持
   - vLLM-Ascend 推理集成
   - 完整的测试套件

---

## 标记规范

### 标记格式

```
Unsloth-PTO-<TYPE>: <description>
```

### 标记类型

| 类型 | 含义 | 使用场景 |
|------|------|---------|
| `VERIFY` | 需要验证 | 已实现但未在真机测试 |
| `FIXME` | 需要修复 | 已知问题或缺失功能 |
| `TODO` | 待办事项 | 计划中的功能 |

### 示例

```python
# Unsloth-PTO-VERIFY: check npu devices
# Unsloth-PTO-FIXME: update the torch compile functions
# Unsloth-PTO-TODO: Implement for NPU for BITSANDBYTES
```

---

## 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-01-31 | v1.0 | 初始版本，基于 unsloth-zoo 代码库标记整理 |

---

*本文档由 Unsloth-Zoo 项目维护，如有问题请提交 Issue。*

