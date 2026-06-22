# LLaMA Factory 项目代码深度分析计划（Shadow-FT 专题）

## 概述

本计划旨在对 LLaMA Factory 项目进行全面的代码分析，**以 Shadow-FT（arXiv:2505.12716）为核心重点**，深入剖析其理论基础、实现细节、与 LLaMA Factory 的集成方式。所有文档保存在 `ReadCode` 文件夹中，采用"分-总-分"写作风格，每篇配图并说明，包含使用指南和案例。

---

## Shadow-FT 核心要点

**论文**：Shadow-FT: Tuning Instruct Model via Training on Paired Base Model (arXiv:2505.12716)

**核心思想**：直接微调 Instruct 模型往往效果有限甚至性能退化，因为 Base 模型和 Instruct 模型权重差异很小（Llama 3.1 8B 平均差异 < 2%）。Shadow-FT 的关键洞察是：先在 Base 模型上微调，然后将学到的权重增量（ΔW = W_tuned - W_base）直接嫁接到 Instruct 模型上（W_instruct + ΔW），无需额外参数，实现简单，效果显著优于直接微调。

**项目中的实现**：
- `src/shadow/weight_similarity.py`：计算 Base 与 Instruct 模型的权重相似度 σ
- `src/shadow/apply_diff.py`：全参数微调版的 Shadow-FT（ΔW 嫁接）
- `src/shadow/merge_lora.py`：LoRA 版的 Shadow-FT（LoRA 合并到 Instruct）
- `run.sh`：自动化训练脚本生成器
- `examples/model_pair.json`：Base-Instruct 模型对配置
- `data/dataset_info.json`：Shadow_2k 数据集配置

---

## 文档结构规划（共 10 篇）

| 序号 | 文档名称 | 文件路径 | 核心内容 |
|------|---------|---------|---------|
| 1 | 项目总览与Shadow-FT全景 | ReadCode/01_项目总览与Shadow-FT全景.md | 项目定位、Shadow-FT核心思想、架构总览 |
| 2 | Shadow-FT理论基础 | ReadCode/02_Shadow-FT理论基础.md | 论文原理、数学推导、关键洞察 |
| 3 | Shadow-FT代码实现 | ReadCode/03_Shadow-FT代码实现.md | 三个核心脚本的逐行解析 |
| 4 | Shadow-FT训练流程 | ReadCode/04_Shadow-FT训练流程.md | run.sh自动化、Base/Instruct训练、Delta嫁接 |
| 5 | 模型加载与适配 | ReadCode/05_模型加载与适配.md | 模型加载器、适配器(LoRA/Full/Freeze)、Patcher |
| 6 | 数据处理流水线 | ReadCode/06_数据处理流水线.md | 数据加载、格式转换、模板系统、Shadow_2k数据集 |
| 7 | 训练引擎 | ReadCode/07_训练引擎.md | SFT/DPO/KTO/PPO/RM/PT 与 Shadow-FT 的关系 |
| 8 | 推理引擎 | ReadCode/08_推理引擎.md | HF/vLLM/SGLang 推理与 Shadow-FT 模型 |
| 9 | API与WebUI | ReadCode/09_API与WebUI.md | API服务、WebUI界面 |
| 10 | Shadow-FT使用指南与案例 | ReadCode/10_Shadow-FT使用指南与案例.md | 完整使用流程、配置示例、实验复现 |

---

## 各文档详细内容规划

### 1. 项目总览与Shadow-FT全景 (ReadCode/01_项目总览与Shadow-FT全景.md)

**写作风格：分-总-分**
- **分**：先分别介绍项目各模块和 Shadow-FT 的定位
- **总**：总结 Shadow-FT 在整体架构中的核心地位
- **分**：深入 Shadow-FT 与各模块的交互关系

**内容要点：**
1. 项目定位
   - LLaMA Factory：统一的大语言模型微调框架
   - Shadow-FT：本项目的核心创新——通过 Base 模型间接微调 Instruct 模型

2. 目录结构全景图（配图：项目目录树状图，高亮 shadow/ 目录）
   ```
   /workspace/
   ├── src/
   │   ├── llamafactory/       # 核心框架
   │   │   ├── api/            # API服务
   │   │   ├── chat/           # 推理引擎
   │   │   ├── data/           # 数据处理
   │   │   ├── eval/           # 评估
   │   │   ├── extras/         # 工具
   │   │   ├── hparams/        # 超参数
   │   │   ├── model/          # 模型加载
   │   │   ├── train/          # 训练引擎
   │   │   └── webui/          # Web界面
   │   └── shadow/             # ★ Shadow-FT 核心实现
   │       ├── weight_similarity.py  # 权重相似度计算
   │       ├── apply_diff.py         # 全参数Delta嫁接
   │       └── merge_lora.py         # LoRA合并到Instruct
   ├── examples/
   │   └── model_pair.json     # ★ Base-Instruct模型对配置
   ├── data/
   │   └── dataset_info.json   # ★ Shadow_2k数据集配置
   ├── run.sh                  # ★ 自动化训练脚本
   └── ...
   ```

3. Shadow-FT 在架构中的位置（配图：Shadow-FT与LLaMA Factory交互图）
   - Shadow-FT 不是 llamafactory 的子模块，而是独立的 src/shadow/ 模块
   - 它调用 llamafactory-cli 进行训练和导出
   - 核心流程：训练Base → 提取ΔW → 嫁接到Instruct

4. 模块依赖关系（配图：模块依赖关系图）
   - shadow/ → llamafactory-cli (train, export)
   - shadow/ → safetensors (权重读写)
   - shadow/ → huggingface_hub (模型保存)

---

### 2. Shadow-FT理论基础 (ReadCode/02_Shadow-FT理论基础.md)

**写作风格：分-总-分**
- **分**：先分别介绍问题背景、核心观察、方法设计
- **总**：总结 Shadow-FT 的理论贡献
- **分**：深入数学原理与实验验证

**内容要点：**
1. 问题背景：为什么直接微调 Instruct 模型效果不好？
   - Instruct 模型已经过指令微调，继续微调容易过拟合
   - 微调方向可能与已有指令能力冲突
   - 实验表明：直接微调 Instruct 模型提升有限甚至退化

2. 核心观察：Base 与 Instruct 模型的权重相似性（配图：权重差异分布图）
   - σ(W_base, W_instruct) = Σ|W_base - W_instruct| / (Σ|W_base| + Σ|W_instruct|)
   - Llama 3.1 8B: σ < 2%
   - Qwen 系列同样具有高度相似性
   - 这意味着 Base 和 Instruct 的权重空间非常接近

3. Shadow-FT 方法（配图：Shadow-FT方法流程图）
   - Step 1: 在 Base 模型上微调 → W_tuned = W_base + ΔW
   - Step 2: 提取增量 ΔW = W_tuned - W_base
   - Step 3: 嫁接到 Instruct → W_new = W_instruct + ΔW
   - 关键：ΔW 捕获了任务特定的知识，而非指令格式的知识

4. 为什么 Shadow-FT 有效？（配图：损失景观示意图）
   - Base 模型是"好学习者但弱骨干"
   - Instruct 模型是"强骨干但差学习者"
   - Shadow-FT 结合两者优势：Base 学习 + Instruct 骨干
   - ΔW 在 Base 和 Instruct 之间具有可迁移性

5. LoRA 版 Shadow-FT
   - 在 Base 上用 LoRA 微调 → 合并到 Base → 再嫁接到 Instruct
   - 或直接将 LoRA 适配器合并到 Instruct 模型
   - 参数效率更高

6. 与 DPO 的结合
   - Shadow-FT + DPO：先用 Shadow-FT 获取 SFT 模型，再进行 DPO 对齐
   - 两阶段策略：Shadow-FT(SFT) → DPO

7. 实验结果概览
   - 19 个基准测试覆盖代码、推理、数学
   - 主流模型：Qwen3、Llama3 系列
   - 一致性优于全参数微调和参数高效微调
   - 可扩展到多模态大模型

---

### 3. Shadow-FT代码实现 (ReadCode/03_Shadow-FT代码实现.md)

**写作风格：分-总-分**
- **分**：逐一深入三个核心脚本
- **总**：总结代码架构与设计模式
- **分**：关键函数的逐行解析

**内容要点：**
1. weight_similarity.py 详解（配图：权重相似度计算流程图）
   - `parse_safetensors_index()`: 解析模型分片索引
     - 读取 model.safetensors.index.json
     - 按层分组权重键（model.layers.N.xxx）
     - 返回层数和层-键映射
   - `load_selected_weights_optimized()`: 按需加载权重
     - 逐分片加载，只读所需键
     - BF16 → FP32 上转换
     - 内存优化：处理完一层后释放
   - `calculate_sigma()`: σ 指标计算
     - σ = Σ|W_A - W_B| / (Σ|W_A| + Σ|W_B|)
   - `calculate_sparsity_ratio()`: 稀疏比计算
     - 两个权重差异小于阈值的比例
   - 使用方式：`python3 src/shadow/weight_similarity.py --B <base> --I <instruct>`

2. apply_diff.py 详解（配图：Delta嫁接流程图）
   - `is_linear_param()`: 识别线性层参数
     - 匹配模式：k_proj, q_proj, v_proj, o_proj, up_proj, gate_proj, down_proj
     - 只对线性层做 ΔW 嫁接，非线性格层直接保留 Instruct 权重
   - `load_weights()`: 多格式权重加载
     - 支持 safetensors 分片、单文件 pytorch_model.bin、分片 bin
   - `save_safetensor_weights()`: 分片保存
     - 使用 huggingface_hub.save_torch_state_dict
     - 自动处理共享张量（如 tied embeddings）
   - `process_single_model()`: 核心嫁接逻辑
     - ΔW = W_tuned - W_base（仅线性层）
     - W_new = W_instruct + ΔW（线性层）
     - 非线性层保持 W_instruct 原值
   - `copy_tokenizer_and_config()`: 复制 tokenizer 和配置
   - `print_debug_info()`: 调试信息输出

3. merge_lora.py 详解（配图：LoRA合并流程图）
   - 调用 llamafactory-cli export 合并 LoRA
   - 生成 merge_lora_config.yaml
   - 配置项：model_name_or_path, adapter_name_or_path, template, finetuning_type
   - 合并到目标 Base 或 Instruct 模型

4. 代码设计模式总结
   - 渐进式内存管理：逐层加载/释放
   - 格式兼容：支持多种权重存储格式
   - 命令行封装：通过 llamafactory-cli 复用框架能力
   - 配置驱动：YAML 配置文件控制合并行为

---

### 4. Shadow-FT训练流程 (ReadCode/04_Shadow-FT训练流程.md)

**写作风格：分-总-分**
- **分**：分别介绍 run.sh 的各个阶段
- **总**：总结完整的 Shadow-FT 实验流程
- **分**：深入关键配置与参数选择

**内容要点：**
1. run.sh 自动化脚本详解（配图：run.sh执行流程图）
   - 全局配置：模型列表、LoRA开关、学习率
   - 模型对解析：从 model_pair.json 读取 Base/Instruct 路径
   - 训练脚本生成：为每个模型对生成训练命令
   - Delta合并脚本生成：LoRA合并或全参数嫁接

2. 模型对配置 (model_pair.json)
   - 31 个 Base-Instruct 模型对
   - 涵盖：Qwen3, Qwen2.5, Llama, Falcon, Gemma, Mistral, Yi, Baichuan, InternLM, GLM-4
   - 格式：{"name": "...", "hf_base_path": "...", "hf_instruct_path": "..."}

3. 训练阶段详解（配图：Shadow-FT两阶段训练流程图）
   - **阶段一：在 Base 模型上训练**
     - LoRA 模式：`llamafactory-cli train --model_name_or_path <base> --finetuning_type lora`
     - Full SFT 模式：`llamafactory-cli train --model_name_or_path <base> --finetuning_type full`
     - 数据集：Shadow_2k (sharegpt 格式)
   - **阶段二：Delta 嫁接**
     - LoRA：`merge_lora.py --adapter_path <lora_output> --target_base <instruct>`
     - Full：`apply_diff.py --tuned_model <sft_output> --target_model <instruct> --base_model <base>`
   - **评估**：对 Instruct 直接训练结果和 Shadow-FT 结果分别评估

4. 训练参数配置
   - cutoff_len=4096, max_samples=2000
   - per_device_train_batch_size=2, gradient_accumulation_steps=16
   - learning_rate: LoRA=2e-4, Full=1e-5
   - DeepSpeed ZeRO-3, BF16, Flash Attention 2
   - LoRA rank=8, ratio=0.5

5. 两种 Shadow-FT 变体对比（配图：LoRA版 vs Full版对比图）
   - LoRA Shadow-FT: 训练快、参数少、需要 merge_lora.py
   - Full Shadow-FT: 训练慢、参数多、需要 apply_diff.py
   - 两者最终效果：将 Base 上学到的 ΔW 嫁接到 Instruct

---

### 5. 模型加载与适配 (ReadCode/05_模型加载与适配.md)

**写作风格：分-总-分**
- **分**：分别介绍加载器、适配器、Patcher
- **总**：总结模型初始化完整流程（含 Shadow-FT 视角）
- **分**：深入各适配器类型的实现细节

**内容要点：**
1. 模型加载流程（配图：模型加载完整流程图）
   - load_tokenizer() → load_config() → load_model() → init_adapter()
   - Shadow-FT 视角：Base 模型和 Instruct 模型使用相同的加载流程

2. 适配器系统（配图：适配器选择与初始化流程图）
   - init_adapter() 统一入口
   - Full Tuning：全参数微调，Shadow-FT 的 Full 版本使用
   - Freeze Tuning：冻结部分层
   - LoRA Tuning：Shadow-FT 的 LoRA 版本使用
   - 适配器合并：Shadow-FT 中 merge_lora.py 调用 export 命令

3. Patcher系统
   - patch_config/patch_model/patch_tokenizer/patch_processor/patch_valuehead_model
   - Shadow-FT 中 apply_diff.py 不使用 Patcher，直接操作权重

4. 模型工具集 (model_utils/)
   - 量化、注意力、RoPE、MoE、视觉模型等工具
   - 与 Shadow-FT 的关系：Shadow-FT 训练时可使用这些优化

---

### 6. 数据处理流水线 (ReadCode/06_数据处理流水线.md)

**写作风格：分-总-分**
- **分**：分别介绍数据加载、转换、模板、处理器
- **总**：总结端到端数据处理流程
- **分**：深入 Shadow_2k 数据集与 Shadow-FT 的数据需求

**内容要点：**
1. 数据处理总流程（配图：数据从原始文件到模型输入的完整流程图）
2. Shadow_2k 数据集详解
   - 格式：sharegpt (conversations 字段)
   - 文件：Shadow_2k.parquet
   - 用途：Shadow-FT 训练的默认数据集
3. 格式转换 (converter.py)
4. 模板系统 (template.py) - 60+ 模型模板
5. 数据整理器 (collator.py) - 多模态支持
6. 多模态插件 (mm_plugin.py)

---

### 7. 训练引擎 (ReadCode/07_训练引擎.md)

**写作风格：分-总-分**
- **分**：分别介绍6种训练方法
- **总**：总结训练流程的统一模式与 Shadow-FT 的关系
- **分**：深入 SFT 训练器（Shadow-FT 的核心训练阶段）

**内容要点：**
1. 训练引擎总览（配图：训练引擎选择与执行流程图）
   - Shadow-FT 使用 SFT 阶段在 Base 模型上训练
   - Shadow-FT + DPO 的两阶段策略
2. SFT (监督微调) - Shadow-FT 的核心
3. DPO (直接偏好优化) - Shadow-FT 的扩展
4. KTO/PPO/RM/PT 简介
5. 回调系统

---

### 8. 推理引擎 (ReadCode/08_推理引擎.md)

**写作风格：分-总-分**
- **分**：分别介绍三种推理引擎
- **总**：总结统一接口设计
- **分**：Shadow-FT 模型的推理方式

**内容要点：**
1. 推理引擎架构（配图：推理引擎类层次与接口图）
2. BaseEngine / ChatModel 设计
3. HuggingfaceEngine / VllmEngine / SGLangEngine
4. Shadow-FT 模型的推理：嫁接后的模型与普通模型无差异，可使用任意引擎

---

### 9. API与WebUI (ReadCode/09_API与WebUI.md)

**写作风格：分-总-分**
- **分**：分别介绍 API 和 WebUI
- **总**：总结用户交互层设计
- **分**：Shadow-FT 模型的部署方式

**内容要点：**
1. API服务（FastAPI + OpenAI兼容接口）
2. WebUI（Gradio界面）
3. Shadow-FT 模型的 API 部署

---

### 10. Shadow-FT使用指南与案例 (ReadCode/10_Shadow-FT使用指南与案例.md)

**写作风格：分-总-分**
- **分**：按使用场景分类介绍
- **总**：总结 Shadow-FT 最佳实践
- **分**：提供完整复现案例

**内容要点：**
1. 环境准备
   - 安装 LLaMA Factory
   - 下载 Base/Instruct 模型对
   - 配置 model_pair.json

2. Step-by-Step: Full Shadow-FT
   - Step 1: 在 Base 上全参数 SFT
   - Step 2: 用 apply_diff.py 嫁接 ΔW 到 Instruct
   - Step 3: 评估嫁接后的模型

3. Step-by-Step: LoRA Shadow-FT
   - Step 1: 在 Base 上 LoRA SFT
   - Step 2: 用 merge_lora.py 合并到 Instruct
   - Step 3: 评估

4. Step-by-Step: Shadow-FT + DPO
   - Step 1: Shadow-FT SFT 阶段
   - Step 2: DPO 对齐阶段

5. 权重相似度分析
   - 使用 weight_similarity.py 分析模型对

6. 使用 run.sh 自动化实验
   - 配置参数
   - 运行脚本
   - 解读结果

7. 多模态 Shadow-FT
   - 视觉语言模型的 Shadow-FT 流程

8. 最佳实践与调优建议
   - 学习率选择
   - LoRA rank 选择
   - 数据集规模

9. 常见问题与排查

---

## 配图规划

每篇文档包含 2-4 张 Mermaid 图表：

| 文档 | 图表 |
|------|------|
| 01 | 项目目录树状图（高亮shadow）、Shadow-FT与LLaMA Factory交互图、模块依赖图 |
| 02 | 权重差异分布图、Shadow-FT方法流程图、损失景观示意图、Base vs Instruct学习能力对比图 |
| 03 | 权重相似度计算流程图、Delta嫁接流程图、LoRA合并流程图 |
| 04 | run.sh执行流程图、Shadow-FT两阶段训练流程图、LoRA版vs Full版对比图 |
| 05 | 模型加载完整流程图、适配器选择流程图 |
| 06 | 数据处理全流程图、Shadow_2k数据格式图 |
| 07 | 训练引擎选择图、SFT训练流程图 |
| 08 | 推理引擎类层次图 |
| 09 | API架构图、WebUI组件图 |
| 10 | Shadow-FT端到端使用流程图 |

---

## 实施步骤

1. 创建 `ReadCode` 目录
2. 按顺序编写10篇文档
3. 每篇包含：分-总-分结构、Mermaid图表、代码引用、使用指南与案例
4. 所有代码引用基于实际文件内容
5. 重点突出 Shadow-FT 的理论、实现和使用
