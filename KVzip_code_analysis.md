# KVzip 代码分析文档

## 项目概述

**KVzip** 是一个 KV Cache 压缩工具，发表于 NeurIPS 2025 (Oral)。它通过计算每个 KV pair 的重要性分数，在推理时压缩 KV cache，实现 3-4x 内存减少和 2x 解码加速。

- **论文**: [KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](https://arxiv.org/abs/2505.23416)
- **GitHub**: https://github.com/snu-mllab/KVzip

## 代码结构

```
KVzip/
├── args.py                      # 命令行参数定义
├── eval.py                      # 评估脚本（多压缩 ratio 测试）
├── demo.py                      # 演示脚本
├── test.py                      # 测试脚本
│
├── model/                       # 模型相关
│   ├── __init__.py              # 导出 ModelKVzip
│   ├── load.py                  # 模型加载（支持 LLaMA3, Qwen2.5/3, Gemma3）
│   ├── wrapper.py               # ModelKVzip 主类（prefill, scoring, generate）
│   ├── monkeypatch.py           # Attention 层替换
│   ├── template.py              # Chat template 定义
│   └── quant_model/             # 量化模型支持 (QServe W8A8KV4)
│       ├── w8a8kv4_llama.py
│       ├── int4_kv.py
│       ├── attn.py
│       └── monkeypatch.py
│
├── attention/                   # Attention 相关
│   ├── __init__.py
│   ├── attn.py                  # 自定义 attention forward 实现
│   ├── kvcache.py               # KV cache 类 (EvictCache, RetainCache, RetainHybridCache)
│   └ score.py                   # KVScore 类（重要性分数计算）
│
├── data/                        # 数据加载
│   ├── __init__.py
│   ├── load.py                  # 数据集加载 (SQuAD, NIAH, GSM8K, SCBench)
│   ├── wrapper.py               # DataWrapper 类
│   └── needle/                  # Needle-in-a-Haystack 数据
│       ├── data.py
│       ├── utils.py
│       └── visualize.py
│
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── func.py                  # 时间戳、softmax 等工具
│   ├── tester.py                # 评估器
│   └ head_score/                # 预计算的 head-level importance scores
│
├── results/                     # 结果处理
│   ├── metric.py                # 评估指标
│   ├── parse.py                 # 结果解析
│   └ repo_qa_utils.py
│
└── csrc/                        # CUDA kernel 源码
    └ setup.py
```

## 核心执行流程

### 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        KVzip 执行流程                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load Model                                                  │
│     └─────────────────────────────────────────────              │
│     model/load.py:42-77                                         │
│     - 加载模型 + tokenizer                                       │
│     - Monkey patch attention 层                                  │
│                                                                 │
│                    ↓                                            │
│                                                                 │
│  2. Prefill Context                                             │
│     └─────────────────────────────────────────────              │
│     model/wrapper.py:169-195                                    │
│     - 分块处理 context，填充 KV cache                            │
│     - 初始化 EvictCache/RetainCache                             │
│                                                                 │
│                    ↓                                            │
│                                                                 │
│  3. Importance Scoring                                          │
│     ┌─────────────────────────────────────────────┐             │
│     │ model/wrapper.py:223-249                    │             │
│     │ - Context Reconstruction 任务               │             │
│     │ - 分块让模型 "repeat" 各块内容              │             │
│     │ - 通过 attention weights 计算分数           │             │
│     │ - 分数存储: kv.score[layer][head][position] │             │
│     └─────────────────────────────────────────────┘             │
│                    ↓                                            │
│                                                                 │
│  4. Prune KV Cache                                              │
│     ┌─────────────────────────────────────────────┐             │
│     │ attention/kvcache.py:123-138                │             │
│     │ - kv.prune(ratio=0.3)                       │             │
│     │ - 根据分数阈值裁剪                           │             │
│     │ - EvictCache: 真正删除 KV                   │             │
│     │ - RetainCache: 只标记 mask                  │             │
│     └─────────────────────────────────────────────┘             │
│                    ↓                                            │
│                                                                 │
│  5. Generate                                                    │
│     ┌─────────────────────────────────────────────┐             │
│     │ model/wrapper.py:251-284                    │             │
│     │ - 使用裁剪后的 KV cache 推理                │             │
│     │ - flash_attn_varlen_func 处理变长 KV        │             │
│     └─────────────────────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Start 示例 (demo.py)

```python
from model import ModelKVzip

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M")
context = "This is my basic profile. My name is Kim living in Seoul."

# Step 1+2: Prefill + Scoring
kv = model.prefill(context, load_score=False, do_score=True)

# Step 3: Prune
kv.prune(ratio=0.3)  # 保留 30% KV

# Step 4: Generate
query_ids = model.apply_template("What is my name?")
output = model.generate(query_ids, kv=kv, update_cache=False)
```

## 关键组件详解

### 1. ModelKVzip 类 (`model/wrapper.py`)

**主入口类，提供以下核心方法：**

| 方法 | 行号 | 功能 |
|------|------|------|
| `prefill()` | 169-195 | 分块预填充 KV cache |
| `scoring()` | 223-249 | 计算 KV 重要性分数 |
| `generate()` | 251-284 | 使用 KV cache 生成回答 |
| `self_task()` | 197-220 | 准备 context reconstruction 输入 |
| `apply_template()` | 115-118 | 应用 chat template |

**self_task 详解 (scoring 的核心)：**

```python
# model/wrapper.py:197-220
def self_task(self, ctx_ids, chunk_size=2000):
    """将 context 分块，每块生成一个 "repeat" 查询"""
    # 第一块: "Repeat the previous context exactly."
    # 后续块: "Repeat the part of the previous context exactly, starting with [前一块末尾]"
    # 这样通过 attention weights 可以计算每块 KV 的重要性
```

### 2. KV Cache 类 (`attention/kvcache.py`)

| 类名 | 行号 | 说明 |
|------|------|------|
| `EvictCache` | 14-214 | 真正从内存删除 KV 的 cache |
| `RetainCache` | 216-347 | 保留完整 KV，只标记 mask（用于评估） |
| `RetainHybridCache` | 350-599 | Gemma3 专用（Hybrid + Static cache） |

**核心方法：**

| 方法 | 位置 | 功能 |
|------|------|------|
| `update()` | EvictCache:41-80, RetainCache:244-266 | 更新 KV cache |
| `prune()` | EvictCache:123-138, RetainCache:284-298 | 裁剪 KV |
| `prepare()` | EvictCache:187-213, RetainCache:312-347 | 准备变长 attention 输入 |
| `slice()` | EvictCache:82-106, RetainCache:268-275 | 删除 query/generated tokens 的 KV |

### 3. KVScore 类 (`attention/score.py`)

**计算和管理重要性分数：**

| 方法 | 行号 | 功能 |
|------|------|------|
| `init_score()` | 25-31 | 初始化分数存储 |
| `_get_score()` | 36-65 | 计算 KV importance（通过 attention weights） |
| `_threshold()` | 88-102 | 根据 ratio 计算阈值 |
| `_threshold_uniform()` | 104-120 | 均匀分配每个 head 的预算 |

**_get_score 计算逻辑：**

```python
# attention/score.py:36-65
def _get_score(self, query_states, key_states, layer_idx):
    # 1. 取出 sink tokens + 当前 chunk KV + repeat query KV
    # 2. 计算 attention weights: Q @ K^T / sqrt(head_dim)
    # 3. 应用 causal mask
    # 4. Softmax
    # 5. 取 max over (group, query) 作为该位置的 importance score
    score = attn_weights.amax(dim=(-3, -2))  # max over group, q
```

### 4. Attention Forward (`attention/attn.py`)

**替换原始 attention，支持 scoring 和 pruned attention：**

| 函数 | 行号 | 适用模型 |
|------|------|----------|
| `llama_qwen_attn_forward` | 19-96 | LLaMA, Qwen2.5, Qwen3 |
| `gemma3_attn_forward` | 99-215 | Gemma3 |

**核心修改点：**

```python
# attention/attn.py:52-89
# Scoring 阶段
if getattr(past_key_value, "get_score", None):
    past_key_value._get_score(query_states, key_states, layer_idx)

# Pruned attention 阶段
if getattr(past_key_value, "pruned", None):
    # Subsample KV，使用 flash_attn_varlen_func 处理变长
    query_states, key_states, value_states, info = past_key_value.prepare(...)
    attn_output = flash_attn_varlen_func(
        query_states, key_states, value_states,
        cu_seqlens_q=info["cu_len_q"],
        cu_seqlens_k=info["cu_len_k"],
        ...
    )
```

### 5. Monkey Patch (`model/monkeypatch.py`)

**在模型加载时替换 attention 实现：**

```python
# model/monkeypatch.py
def replace_attn(model_id):
    if "llama" in model_id:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_qwen_attn_forward
    elif "qwen2.5" in model_id:
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = llama_qwen_attn_forward
    elif "qwen3" in model_id:
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = llama_qwen_attn_forward
    elif "gemma-3" in model_id:
        transformers.models.gemma3.modeling_gemma3.Gemma3Attention.forward = gemma3_attn_forward
```

### 6. 模型加载 (`model/load.py`)

**支持的模型：**

| 简称 | 完整 ID |
|------|---------|
| `llama3.1-8b` | meta-llama/Llama-3.1-8B-Instruct |
| `llama3.2-1b/3b` | meta-llama/Llama-3.2-{1/3}B-Instruct |
| `qwen2.5-7b/14b` | Qwen/Qwen2.5-{7/14}B-Instruct-1M |
| `qwen3-*b` | Qwen/Qwen3-{size}B |
| `gemma3-*b` | google/gemma-3-{size}b-it |

### 7. 数据加载 (`data/load.py`)

**支持的数据集：**

| 数据集 | 说明 |
|--------|------|
| `squad` | SQuAD QA |
| `needle` | Needle-in-a-Haystack |
| `gsm` | GSM8K 数学推理 |
| `scbench_*` | SCBench 长上下文任务 |

## 参数说明 (`args.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-r, --ratio` | 0.3 | 压缩比例（保留的 KV 占比） |
| `--kv_type` | `evict` | `evict`（真实删除）或 `retain`（标记 mask） |
| `--level` | `pair` | `pair`（context-dependent）或 `head`（context-independent） |
| `-m, --model` | - | 模型名称 |
| `-d, --data` | - | 数据集名称 |
| `--save_head_score` | False | 保存 head-level importance score |

## Context-Independent 模式

使用 `--level head` + `--ratio 0.6`：

- 预计算 head-level importance score（存储在 `utils/head_score/`）
- 无需 runtime scoring overhead
- 与 DuoAttention 兼容

```bash
# 计算新模型的 head score
python -B test.py -m [model] -d scbench_qa_eng --save_head_score
```

## VLM 兼容方案

### 需要修改的文件

| 文件 | 修改内容 |
|------|----------|
| `model/load.py` | 新增 VLM 模型 ID 映射 |
| `model/monkeypatch.py` | 新增 VLM attention patch |
| `attention/attn.py` | 新增 VLM attention forward（处理 2D/3D RoPE） |
| `attention/kvcache.py` | 新增 `vision_token_len` 参数，vision KV 保留不裁剪 |
| `attention/score.py` | 新增 `_get_score_vlm()` 方法 |
| `model/template.py` | 新增 VLM chat template（含 image placeholder） |
| `model/wrapper.py` | `prefill()` 方法支持 images 输入 |

### VLM Attention Forward 示例

```python
def vlm_attn_forward(self, hidden_states, position_embeddings, ...):
    # 1. 处理 vision tokens 的特殊 position encoding
    # 2. scoring 时区分 vision KV 和 text KV
    if getattr(past_key_value, "get_score", None):
        past_key_value._get_score_vlm(
            query_states, key_states, layer_idx,
            vision_token_len=self.vision_token_len
        )
    # 3. vision KV 通常全部保留
```

### VLM Scoring 建议

| KV 类型 | 处理方式 |
|---------|----------|
| Vision KV | 全部保留（数量少，对图像理解关键） |
| Text KV | 使用原有 context reconstruction scoring |

### 新增 VLM 模型示例

```python
# model/load.py
elif name.startswith("llava"):
    return "llava-hf/llava-1.5-7b-hf"
elif name.startswith("qwen2-vl"):
    return "Qwen/Qwen2-VL-7B-Instruct"
```

## 关键设计要点

1. **Context Reconstruction**: 通过让模型 "repeat" context 来计算 KV importance
2. **Causal Masking**: scoring 时只考虑可 attend 的位置
3. **Sink Tokens**: 系统提示的 KV 通常保留不裁剪
4. **Flash Attention Varlen**: 使用变长 attention 处理裁剪后的不均匀 KV
5. **Head-Level vs Pair-Level**: context-independent vs context-dependent 压缩策略