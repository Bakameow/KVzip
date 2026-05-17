# KVzip 修改说明 (Changes Documentation)

**日期:** 2026-05-15
**项目:** KVzip — Query-Agnostic KV Cache Compression with Context Reconstruction (NeurIPS 2025 Oral)

本文档记录了通过全自动研究流水线 (auto-review-loop) 发现并修复的所有问题，以及此前已做的修改。

---

## 一、关键 Bug 修复 (Critical Bug Fixes)

### 1. 无限循环修复 — `detect_multimodal_positions`
**文件:** `model/wrapper.py` 第317-325行

**问题:** 当 token 序列中存在 `<|vision_start|>` 但没有对应的 `<|vision_end|>` 时，内层 `for` 循环遍历完后没有更新 `i`，外层 `while` 循环的 `continue` 跳回同一位置，造成**死循环/程序挂起**。

**修复:** 为内层 `for` 循环添加 `else` 分支，当找不到匹配的 vision_end 时推进 `i += 1`。

---

### 2. 多轮对话 prefill_ids 指数增长 — `generate()`
**文件:** `model/wrapper.py` 第979-1002行

**问题:** `update_cache=True` 时：
```python
# line 983: input_ids 已包含 kv.prefill_ids
input_ids = torch.cat([kv.prefill_ids, raw_query], dim=1)
# line 1002: 再次拼接 input_ids (其中已有 prefill_ids)
kv.prefill_ids = torch.cat([kv.prefill_ids, input_ids, a_ids], dim=1)
# 结果: prefill内容重复 → 每次调用翻倍
```

**修复:** 保存原始 `raw_query`（不含 prefill_ids），在更新 `kv.prefill_ids` 时使用 `raw_query` 而非 `input_ids`。

---

### 3. `get_seq_length` 返回类型错误
**文件:** `attention/kvcache.py` 第504行

**问题:** `RetainHybridCache.get_seq_length()` 返回 `torch.tensor(...)` 而非 Python `int`。HuggingFace generate 期望标量整数，返回 tensor 可能导致类型比较异常和 CUDA 同步开销。

**修复:** 直接返回 `self._seen_tokens` 整数。

---

### 4. `OptimINT4KVCache` 缺少 `_seen_tokens` 初始化
**文件:** `model/quant_model/int4_kv.py` 第284行

**问题:** `OptimINT4KVCache` 多重继承 `StaticINT4KVCache` 和 `RetainCache`，但只调用了 `StaticINT4KVCache.__init__()`。`_seen_tokens` 由 `DynamicCache.__init__` 设置（RetainCache 的父类），从未被调用。调用 `get_seq_length()` 时会抛出 `AttributeError`。

**修复:** 显式添加 `self._seen_tokens = 0`。

---

### 5. 生成时总是丢弃最后一个 token
**文件:** `model/wrapper.py` 第986行

**问题:** `output[:, input_ids.shape[1]:-1]` 无条件丢弃最后一个 token，假设它总是 EOS。当 `max_new_tokens` 限制触发时（未生成 EOS），最后一个有意义的 token 会被错误丢掉。

**修复:** 检查最后一个 token 是否真的是 EOS token（从 `gen_kwargs["eos_token_id"]` 获取），仅在确认为 EOS 时才丢弃。

---

### 6. `_mem()` 在缓存未完全填充时崩溃
**文件:** `attention/kvcache.py` 第116-123行和第280-285行

**问题:** `EvictCache._mem()` 和 `RetainCache._mem()` 假设所有层都已填充。如果在 prefill 中途调用（如日志回调），会抛出 `IndexError`。

**修复:** 检查 `len(self.key_cache) == 0` 提前返回 0.0；`EvictCache._mem()` 只迭代已填充的层。

---

## 二、此前已做的修改 (Previously Committed/Staged Changes)

### 7. VLM 视频 Token 提取修复
**文件:** `data/wrapper.py` `_build_video_vlm_inputs` 函数

**问题:** 旧代码搜索 `<|im_start|>user\n` 的单个 token 字节码（最后一个 token），可能在视频 token 序列中误匹配。

**修复:** 
- 搜索完整的 `<|im_start|>user\n` token **序列**（使用 `torch.equal()` 匹配）
- 从序列末尾反向搜索 `<|im_end|>` 边界
- 仅提取 user content（不含 chat template 包装），让 `prefill()` 自行追加 sys_prompt
- 添加 token 数量限制（20000）跳过超大视频避免 OOM

---

### 8. Video MME 评分修复
**文件:** `results/metric.py` `evaluate_answer` 函数

**问题:** 旧代码对参考答案使用 `normalize_answer()`，该函数通过 `remove_articles` 正则 `\b(a|an|the)\b` 移除冠词。当正确答案为 "A" 时，`normalize_answer("A")` 返回空字符串 `""`，导致**所有答案为 A 的题目永远无法得分**。

**修复:**
- 跳过 `normalize_answer()`，直接 `ref.strip().upper()[:1]` 获取答案字母
- 使用正则表达式从模型输出中提取答案字母，支持多种格式：
  - `"A"` / `"(A)"` / `"A. xxx"` / `"A: xxx"` / `"Answer: A"`
  - 回退策略：搜索任意 A-D 字母

---

### 9. grid_thw 空张量处理
**文件:** `model/wrapper.py` 第636-638行

**问题:** 旧代码 `vlm_inputs.get("image_grid_thw") or vlm_inputs.get("video_grid_thw")` — Python 的 `or` 将空张量视为 falsy，会错误地 fall through 到 video_grid_thw。

**修复:** 使用显式 `if grid_thw is None` 检查。

---

### 10. multimodal_scoring 标志修正
**文件:** `attention/score.py` 第33行

**问题:** `enable_multimodal_scoring` 默认值为 `False`，导致多模态 token（图像/视频）被赋予 `float('inf')` 分数而永不裁剪。对于 VLM KV 压缩任务，应该允许对所有 token 进行评分。

**修复:** 将默认值改为 `True`，多模态 token 参与正常评分和裁剪。

---

### 11. print → logger 迁移
**文件:** `model/load.py`, `model/monkeypatch.py`

将所有 `print()` 语句替换为 `logger.info()` / `logger.warning()`，统一日志输出格式。

---

### 12. 其他代码质量改进

| 文件 | 修改 |
|------|------|
| `results/parse.py` | 移除未使用的 `from eval import set_ratios` 导入 |
| `eval.py` | 将变量 `eval` 重命名为 `evaluator`（避免覆盖内置函数） |
| `model/wrapper.py` | 修复 `vlm_chunk_fn` 中 `chunk_fn` 被调用两次的性能浪费 |
| `model/quant_model/int4_kv.py` | 移除重复的 `self.device = device` 赋值 |
| `demo_image_needle.py` | 重构图像文字渲染，提取 `_draw_text_on_background()` 函数（支持透明画布、文字描边、旋转/镜像/透视变换）；简化 hotpot 复合图像生成 |

---

## 三、涉及文件汇总

| 文件 | 修改类型 |
|------|---------|
| `attention/score.py` | multimodal_scoring 默认值修正 |
| `attention/kvcache.py` | get_seq_length 返回类型修正、_mem() 防御性检查 |
| `data/wrapper.py` | 视频 token 提取重写、token 限制、import 补充 |
| `demo_image_needle.py` | 重构图像渲染函数、注释部分 key |
| `model/load.py` | print → logger |
| `model/monkeypatch.py` | print → logger |
| `model/wrapper.py` | 无限循环修复、prefill_ids 修复、EOS 修复、grid_thw 修复、chunk_fn 性能修复 |
| `model/quant_model/int4_kv.py` | _seen_tokens 初始化、重复赋值清理 |
| `eval.py` | eval → evaluator 重命名 |
| `results/metric.py` | video_mme 评分重写 |
| `results/parse.py` | 移除未使用 import |

---

## 四、审阅流程

本次修复由 **auto-review-loop-llm** 全自动研究流水线驱动：
1. 对全部 14 个核心文件进行深度代码审查
2. 发现 4 个严重 Bug、5 个中等 Bug、9 个代码质量问题
3. 所有严重和中等 Bug 均已修复
4. 审阅详情见 `review-stage/AUTO_REVIEW.md`
