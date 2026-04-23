# ------------------------------------------------------------------------------
# Original Code developed by Jang-Hyun Kim
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
import torch
import glob
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
from transformers import DynamicCache, Gemma3ForCausalLM, Qwen3ForCausalLM

from attention.kvcache import RetainCache, EvictCache, RetainHybridCache
from utils.func import inplace_softmax
from model.load import load_model
from model.quant_model import OptimINT4KVCache, LlamaForCausalLMW8A8
from model.template import template


def vlm_chunk_fn(
    ctx_ids: torch.Tensor,
    chunk_size: int,
    multimodal_ranges: List[Tuple[int, int]] = []
) -> Tuple[List[torch.Tensor], List[bool]]:
    """将 token 序列分块处理，多模态 token 区域单独作为一个 chunk

    用于 VLM 模型的分块处理，确保图像/视频 token 区域作为一个独立的 chunk，
    这样 pixel_values 只传给包含多模态 token 的那个 chunk。

    Args:
        ctx_ids: 输入的 token ID 张量，形状为 [batch_size, seq_len]
        chunk_size: 每个分块的最大 token 数量
        multimodal_ranges: 多模态 token 的位置范围列表，每个元素为 (start, end)

    Returns:
        Tuple[List[torch.Tensor], List[bool]]:
            - 分块后的 token ID 列表
            - 每个分块是否需要 pixel_values 的标记列表

    Example:
        ctx_ids = [1, ..., 16200, <|image_pad|>..., 16350, ..., 32000]  # 32000 tokens
        multimodal_ranges = [(16200, 16350)]  # 图像在位置 16200-16350
        chunks, needs_vision = vlm_chunk_fn(ctx_ids, 16000, multimodal_ranges)
        # 返回:
        # chunks[0] = [0:16200] (纯文本)
        # chunks[1] = [16200:16350] (图像 token，needs_vision[1]=True)
        # chunks[2] = [16350:32000] (纯文本)
    """
    ctx_len = ctx_ids.shape[1]

    if not multimodal_ranges:
        # 无多模态 token，使用普通分块
        return chunk_fn(ctx_ids, chunk_size), [False] * len(chunk_fn(ctx_ids, chunk_size))

    # 构建分块边界点：包含 chunk 边界和多模态区域边界
    boundaries = set()
    boundaries.add(0)
    boundaries.add(ctx_len)

    # 添加 chunk_size 的边界（跳过落在多模态区域内部的边界）
    if ctx_len > chunk_size:
        for i in range(1, (ctx_len - 1) // chunk_size + 1):
            pos = i * chunk_size
            # 检查 pos 是否落在任何多模态区域内部
            in_multimodal = False
            for mm_start, mm_end in multimodal_ranges:
                if mm_start < pos < mm_end:
                    in_multimodal = True
                    break
            if not in_multimodal:
                boundaries.add(pos)

    # 添加多模态区域的边界
    for start, end in multimodal_ranges:
        boundaries.add(start)
        boundaries.add(end)

    # 排序边界点
    boundaries = sorted(boundaries)

    # 过滤掉空区间，生成 chunks
    chunks = []
    needs_vision = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue

        chunk = ctx_ids[:, start:end]
        if chunk.shape[1] == 0:
            continue

        # 检查这个 chunk 是否包含多模态 token
        # 通过检查 chunk 的位置是否与任何多模态区域重叠
        chunk_has_vision = False
        for mm_start, mm_end in multimodal_ranges:
            # chunk [start, end] 与多模态区域 [mm_start, mm_end] 有重叠
            if start < mm_end and end > mm_start:
                chunk_has_vision = True
                break

        chunks.append(chunk)
        needs_vision.append(chunk_has_vision)

    print(f"chunk inputs, size: {chunk_size} (num {len(chunks)}, {sum(needs_vision)} vision chunks)")
    return chunks, needs_vision


def chunk_fn(ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    """将 token 序列分块处理

    用于长 context 的分块处理，避免单次处理过长的序列导致内存溢出。

    Args:
        ctx_ids: 输入的 token ID 张量，形状为 [batch_size, seq_len]
        chunk_size: 每个分块的最大 token 数量

    Returns:
        List[torch.Tensor]: 分块后的 token ID 列表，每个元素形状为 [batch_size, chunk_size]

    Example:
        ctx_ids = [1, 2, 3, ..., 10000]  # 10000 tokens
        chunks = chunk_fn(ctx_ids, chunk_size=2000)  # 返回 5 个分块
    """
    ctx_len = ctx_ids.shape[1]

    # 如果序列长度超过分块大小，则进行分块
    if ctx_len > chunk_size:
        chunk_num = (ctx_len - 1) // chunk_size + 1  # 计算需要的分块数量
        print(f"chunk inputs, size: {chunk_size} (num {chunk_num})")

        input_ids = []
        for i in range(chunk_num):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            # 截取当前分块范围的 tokens
            a_ids = ctx_ids[:, start:end]
            if a_ids.shape[1] == 0:  # 跳过空分块
                continue
            input_ids.append(a_ids)
    else:
        # 序列长度不超过分块大小，直接返回原序列
        input_ids = [ctx_ids]

    return input_ids


def load_head_score(model_name, ctx_len):
    """从预存储文件加载 head-level importance scores

    用于 context-independent 的 KV 压缩模式（--level head）。
    预计算的 head scores 存储在 ./utils/head_score/ 目录下。

    Args:
        model_name: 模型名称，如 "Qwen2.5-7B-Instruct"
        ctx_len: context 的 token 长度

    Returns:
        torch.Tensor: head-level importance scores，形状为 [1, 1, n_layers, n_heads, ctx_len]

    Note:
        - head-level score 表示每个 attention head 的整体重要性，与具体位置无关
        - 分数会被扩展到所有 position，形成统一的 mask
        - 支持的模型: Qwen2.5-7B/14B, Llama-3.1-8B
    """
    # 标准化模型名称以匹配存储路径
    if model_name.startswith("Qwen2.5-7B"):
        model_name = "qwen2.5-7b"
    elif model_name.startswith("Qwen2.5-14B"):
        model_name = "qwen2.5-14b"
    elif model_name.startswith("Llama-3.1-8B"):
        model_name = "llama3.1-8b"

    # 加载所有匹配的 head score 文件（可能来自多个数据集）
    attn_ = []
    paths = f"./utils/head_score/{model_name}-*.pt"
    for path in glob.glob(paths):
        attn = torch.load(path).squeeze().cuda()  # 形状: [n_layers, n_heads]
        attn_.append(attn)
        print("Load head-score from", path)

    # 取多个数据集 score 的最大值（保守策略：重要的 head 在任何任务中都重要）
    attn = torch.stack(attn_, dim=0).amax(0)  # [n_layers, n_heads]

    # 扩展到所有 position：每个 head 的 score 对所有 token position 都相同
    score = attn.unsqueeze(-1).expand(-1, -1, ctx_len)  # [n_layers, n_heads, ctx_len]
    score = score.unsqueeze(1)  # [1, 1, n_layers, n_heads, ctx_len]
    return score


class ModelKVzip():
    """KVzip 主类：封装模型加载、KV cache 管理、prefill、scoring 和生成功能

    该类是 KVzip 的核心入口，提供完整的 KV cache 压缩推理流程：
    1. 加载模型并 monkey patch attention 层
    2. Prefill context 并填充 KV cache
    3. 计算每个 KV pair 的 importance score
    4. 根据压缩比例裁剪 KV cache
    5. 使用裁剪后的 KV cache 进行推理

    支持的模型: LLaMA3, Qwen2.5/3, Gemma3, Qwen2-VL, Qwen2.5-VL
    支持的 KV 类型: evict（真实删除）、retain（标记保留）、int4static（量化）、hybrid_static（Gemma3）
    """

    def __init__(self, model_name: str, kv_type: str = "evict"):
        """初始化 ModelKVzip 实例

        Args:
            model_name: 模型名称，可以是简称（如 "qwen2.5-7b"）或完整 ID（如 "Qwen/Qwen2.5-7B-Instruct-1M"）
            kv_type: KV cache 类型，可选值：
                - "evict": 真正从内存删除被裁剪的 KV（节省内存）
                - "retain": 只标记 mask，保留完整 KV（用于多压缩比评估）
                - "int4static": INT4 量化 KV cache（QServe 量化模型）
                - "hybrid_static": Gemma3 专用的 Hybrid cache
                - "original": 不使用 KVzip，保持原始 cache

        Attributes:
            model: 加载的 HuggingFace 模型
            tokenizer: 对应的 tokenizer
            name: 模型简称
            dtype: 模型数据类型（通常为 float16/bfloat16）
            device: 模型设备（cuda）
            config: 模型配置
            kv_type: 实际使用的 KV cache 类型
            is_vlm: 是否为视觉语言模型
            multimodal_token_ids: 多模态特殊 token 的 ID 映射
            gen_kwargs: 生成时的默认参数
            sys_prompt_ids: 系统提示的 token IDs
            postfix_ids: 响应后缀的 token IDs（如 "<|im_start|>assistant"）
        """
        self.model, self.tokenizer = load_model(model_name)

        self.name = self.model.name
        self.dtype = self.model.dtype
        self.device = self.model.device
        self.config = self.model.config

        # Check if this is a VLM model
        self.is_vlm = getattr(self.model, 'is_vlm', False)
        self.multimodal_token_ids = getattr(self.model, 'multimodal_token_ids', {})
        self.processor = getattr(self.model, 'processor', None)

        # 根据模型类型自动调整 KV cache 类型
        if isinstance(self.model, LlamaForCausalLMW8A8):
            # QServe 量化模型，使用 INT4 static cache
            self.kv_type = "int4static"
            print("[Note] Currently, only retain cache is available for QServe")
        elif isinstance(self.model, Gemma3ForCausalLM):
            # Gemma3 使用 Hybrid cache（交替的 sliding + static layers）
            self.kv_type = "hybrid_static"
            print("[Note] Currently, only retain cache is available for Gemma3")
        elif self.is_vlm and kv_type == "evict":
            # VLM 模型使用 retain cache，因为 evict cache 的 flatten 格式与 model.generate 不兼容
            self.kv_type = "retain"
            print("[Note] VLM models use retain cache for compatibility with model.generate")
        else:
            # 其他模型使用用户指定的 KV 类型
            self.kv_type = kv_type
        print(f"KV type: {self.kv_type}")

        if self.is_vlm:
            print(f"[VLM] Multimodal tokens will be preserved during KV pruning")

        # 设置生成参数：贪婪解码，最大 512 new tokens
        self.gen_kwargs = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1,
            "top_k": None,
            "max_new_tokens": 512,
        }

        # Gemma3 和 Qwen3 需要特殊的生成参数
        if isinstance(self.model, Gemma3ForCausalLM):
            self.gen_kwargs["cache_implementation"] = None
            self.gen_kwargs["use_model_defaults"] = False
            self.gen_kwargs["eos_token_id"] = [1, 106]  # Gemma3 的多个 EOS token
        elif isinstance(self.model, Qwen3ForCausalLM):
            self.gen_kwargs["cache_implementation"] = None
            self.gen_kwargs["use_model_defaults"] = False
            self.gen_kwargs["eos_token_id"] = 151645  # Qwen3 的 EOS token

        # 设置 chat template（系统提示 + 响应后缀）
        self.set_chat_template()

    def detect_multimodal_positions(self, token_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """检测多模态 token（如图像/视频 token）的位置范围。

        对于 VLM 模型，多模态内容（图像/视频）会被编码为特殊的 token 序列，
        这些 token 不应该参与 KV cache 的 scoring 和 pruning。

        Args:
            token_ids: Token ID 张量，形状为 [batch_size, seq_len]

        Returns:
            List[Tuple[int, int]]: 多模态 token 的位置范围列表
                每个元素为 (start_pos, end_pos)，表示一个连续的多模态 token 区块

        Example:
            # Qwen-VL 的图像 token 结构:
            # <|vision_start|> ... image tokens ... <|vision_end|>
            # 返回: [(10, 500)] 表示 position 10-500 是图像 token
        """
        if not self.is_vlm or not self.multimodal_token_ids:
            return []

        ranges = []
        token_ids_flat = token_ids[0].cpu().tolist()

        # Get vision-related token IDs
        vision_start_id = self.multimodal_token_ids.get('vision_start')
        vision_end_id = self.multimodal_token_ids.get('vision_end')
        image_pad_id = self.multimodal_token_ids.get('image_pad')
        video_pad_id = self.multimodal_token_ids.get('video_pad')

        # Find multimodal token ranges
        i = 0
        while i < len(token_ids_flat):
            # Check for vision_start marker (Qwen-VL format)
            if vision_start_id and token_ids_flat[i] == vision_start_id:
                start_pos = i
                # Find the matching vision_end
                for j in range(i + 1, len(token_ids_flat)):
                    if vision_end_id and token_ids_flat[j] == vision_end_id:
                        ranges.append((start_pos, j + 1))  # Include vision_end
                        i = j + 1
                        break
                continue

            # Check for continuous image_pad tokens (alternative format)
            if image_pad_id and token_ids_flat[i] == image_pad_id:
                start_pos = i
                while i < len(token_ids_flat) and token_ids_flat[i] == image_pad_id:
                    i += 1
                ranges.append((start_pos, i))
                continue

            # Check for continuous video_pad tokens
            if video_pad_id and token_ids_flat[i] == video_pad_id:
                start_pos = i
                while i < len(token_ids_flat) and token_ids_flat[i] == video_pad_id:
                    i += 1
                ranges.append((start_pos, i))
                continue

            i += 1

        if ranges:
            print(f"[VLM] Detected {len(ranges)} multimodal token ranges: {ranges}")

        return ranges

    def encode(self, text: str) -> torch.Tensor:
        """将文本编码为 token IDs

        Args:
            text: 输入文本字符串

        Returns:
            torch.Tensor: token IDs，形状为 [1, seq_len]，已移到 CUDA 设备

        Note:
            不添加特殊 tokens（add_special_tokens=False），因为特殊 tokens 由 chat template 管理
        """
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").cuda()

    def decode(self, input_ids: torch.Tensor) -> str:
        """将 token IDs 解码为文本

        Args:
            input_ids: token IDs 张量，形状可以是 [seq_len] 或 [batch_size, seq_len]

        Returns:
            str: 解码后的文本字符串
        """
        if len(input_ids.shape) == 2:
            input_ids = input_ids[0]  # 取第一个 batch 的序列
        return self.tokenizer.decode(input_ids)

    def set_chat_template(self, task: str = "qa"):
        """设置模型的 chat template（系统提示和响应后缀）

        根据模型类型和任务类型，设置相应的对话格式模板。

        Args:
            task: 任务类型，影响系统提示的内容
                - "qa": 问答任务，"Given the context, answer to the following question..."
                - "gsm": 数学推理任务，包含额外的推理引导提示

        Attributes 设置:
            sys_prompt_ids: 系统提示部分的 token IDs（会保留在 KV cache 中不被裁剪）
            postfix_ids: 响应后缀部分的 token IDs（如 "<|im_start|>assistant"）

        Example:
            LLaMA3 template:
            - sys_prompt: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n..."
            - postfix: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        """
        prefix, postfix = template(self.name, task)
        self.sys_prompt_ids, self.postfix_ids = self.encode(prefix), self.encode(postfix)

    def apply_template(self, query: str) -> torch.Tensor:
        """将用户查询应用 chat template 格式

        将查询文本转换为模型的输入格式，添加必要的格式标记。

        Args:
            query: 用户查询文本

        Returns:
            torch.Tensor: 格式化后的 token IDs，包含查询和响应后缀

        Example:
            query = "What is my name?"
            返回: "\n\nWhat is my name?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        Note:
            - 查询前添加 "\n\n" 以与前面的 context 分隔
            - 响应后缀由 set_chat_template 设置的 postfix_ids 提供
        """
        query = f"\n\n{query.strip()}"
        query_ids = torch.cat([self.encode(query), self.postfix_ids], dim=1)
        return query_ids

    def __call__(
        self,
        input_ids: torch.Tensor,
        kv: Union[RetainCache, EvictCache],
        update_cache: bool = False,
        return_logits: bool = False,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        """执行模型的 forward pass

        这是模型推理的核心方法，用于：
        1. Prefill 阶段填充 KV cache
        2. Scoring 阶段计算 importance scores（不更新 cache）
        3. 获取 logits 用于概率计算

        Args:
            input_ids: 输入 token IDs，形状为 [batch_size, seq_len]
            kv: KV cache 实例（EvictCache 或 RetainCache）
            update_cache: 是否将当前输入的 KV 写入 cache
                - True: Prefill 时使用，填充 KV cache
                - False: Scoring 时使用，只计算 attention，不写入 cache
            return_logits: 是否返回输出 logits
                - True: 用于概率计算（_prob 方法）
                - False: 只执行 forward，不返回结果（节省内存）
            *args, **kwargs: 传递给模型的其他参数

        Returns:
            模型输出（如果 return_logits=True）或 None

        Note:
            - 当 update_cache=False 时，会使用 kv.slice() 撤销对 cache 的任何修改
            - 这样可以在 scoring 时复用同一份 KV cache，而不污染原始 cache
        """
        seen_token_prev = kv._seen_tokens  # 记录当前 cache 的 token 数量

        # Gemma3 的 Hybrid cache 需要特殊处理：备份 sliding window 部分
        if isinstance(kv, RetainHybridCache) and not update_cache:
            kv.backup_sliding_cache()

        # 构建视觉参数（仅在首次 prefill 时传入）
        vision_kwargs = {}
        if pixel_values is not None:
            vision_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            vision_kwargs["image_grid_thw"] = image_grid_thw
        if pixel_values_videos is not None:
            vision_kwargs["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            vision_kwargs["video_grid_thw"] = video_grid_thw

        if return_logits:
            # 执行完整 forward，返回 logits（用于概率计算）
            outputs = self.model(input_ids, past_key_values=kv, *args, **vision_kwargs, **kwargs)
        else:
            # 只执行 decoder forward，不计算 logits（节省计算）
            if vision_kwargs:
                # VLM 首次 prefill：需要走完整 model.forward 以处理 pixel_values
                _ = self.model(input_ids, past_key_values=kv, *args, **vision_kwargs, **kwargs)
            else:
                _ = self.model.model(input_ids, past_key_values=kv, *args, **kwargs)
            outputs = None

        if not update_cache:
            # 撤销对 cache 的修改：删除本次 forward 产生的 KV
            # 这样可以在多次 scoring 中复用同一份 prefill 的 KV cache
            kv.slice(seen_token_prev)

        return outputs

    def _init_kv(self, kv=None, evict_range=(0, 0)):
        """初始化 KV cache 实例

        根据模型和配置创建合适的 KV cache 类型。

        Args:
            kv: 如果提供，直接返回该 kv；如果为 None，则创建新的 cache
            evict_range: 可被裁剪的 token 范围 (start, end)
                - start: 系统提示结束位置（这些 KV 不参与裁剪，作为 sink 保留）
                - end: context 结束位置

        Returns:
            KV cache 实例，类型取决于 kv_type:
                - "retain": RetainCache（保留完整 KV，标记 mask）
                - "evict": EvictCache（真实删除 KV）
                - "int4static": OptimINT4KVCache（INT4 量化 cache）
                - "hybrid_static": RetainHybridCache（Gemma3 Hybrid cache）
                - "original": DynamicCache（原始 HuggingFace cache）

        Note:
            evict_range 的设计确保系统提示（如 "You are a helpful assistant"）的 KV
            不会被裁剪，因为它们对所有后续查询都很重要。
        """
        if kv is None:
            if self.kv_type == "retain":
                kv = RetainCache(self.model, evict_range)
            elif self.kv_type == "evict":
                kv = EvictCache(self.model, evict_range)
            elif self.kv_type == "int4static":
                kv = OptimINT4KVCache(self.model.model, evict_range)
            elif self.kv_type == "hybrid_static":
                max_size = 190000  # Gemma3 的最大 cache 长度
                kv = RetainHybridCache(self.model.model, evict_range, max_size)
            elif self.kv_type == "original":
                kv = DynamicCache()
                # 标记属性，兼容 KVzip 的检查逻辑
                kv.pruned, kv.get_score = False, False
            else:
                raise NotImplementedError(f"type {self.kv_type} is not implemented")
        return kv

    @torch.inference_mode()
    def prefill(
        self,
        ctx_ids: Union[str, torch.Tensor],
        prefill_chunk_size: int = 16000,
        load_score=False,
        do_score=True,
        vlm_inputs: Optional[dict] = None,
    ) -> Union[RetainCache, EvictCache]:
        """分块预填充 KV cache 并计算 importance scores

        这是 KVzip 的核心方法之一，完成两个关键任务：
        1. 将 context tokens 分块处理，填充到 KV cache
        2. 计算每个 KV pair 的 importance score（用于后续裁剪）

        Args:
            ctx_ids: Context 内容，可以是字符串或已编码的 token IDs
            prefill_chunk_size: Prefill 时的分块大小（默认 16000 tokens）
                - 长 context 需要分块处理以避免内存溢出
                - 每个分块独立进行 forward pass
            load_score: 是否从预存储文件加载 head-level scores
                - True: 使用预计算的 head-level scores（context-independent 模式）
                - False: 在当前 context 上实时计算 scores（context-dependent 模式）
            do_score: 是否执行 importance scoring
                - True: 完成预填充后立即计算 scores
                - False: 只预填充，不计算 scores（用于快速测试）
            vlm_inputs: VLM 视觉输入字典（可选），包含：
                - pixel_values: 图像 patch 特征张量
                - image_grid_thw: 图像网格信息（时间/高/宽）
                仅在首个 prefill chunk 时使用（含图像 token 的那部分）。
                如果为 None，走纯文本 prefill 路径。

        Returns:
            Union[RetainCache, EvictCache]: 填充好的 KV cache 实例
                - 包含 context 的所有 KV pairs
                - 包含每个 KV position 的 importance score

        流程详解:
            1. 编码 context（如果是字符串）
            2. 添加系统提示，构建完整的 prefill_ids
            3. 初始化 KV cache，设置 evict_range（系统提示不参与裁剪）
            4. 分块执行 forward，填充 KV cache
            5. 执行 importance scoring（如果 do_score=True）

        Example:
            context = "My name is Kim. I live in Seoul."
            kv = model.prefill(context)
            # kv 包含:
            #   - key_cache[layer]: 所有 key vectors
            #   - value_cache[layer]: 所有 value vectors
            #   - score[layer][head][position]: 每个 KV 的 importance
        """
        # 如果输入是字符串，先编码为 token IDs
        if type(ctx_ids) == str:
            ctx_ids = self.encode(ctx_ids)

        # 构建完整的预填充输入：系统提示 + context
        prefill_ids = torch.cat([self.sys_prompt_ids, ctx_ids], dim=1)

        # 设置裁剪范围：系统提示部分不参与裁剪（作为 sink 保留）
        # evict_range = (sys_prompt_len, total_len)
        evict_range = (self.sys_prompt_ids.shape[1], prefill_ids.shape[1])

        # 初始化 KV cache
        kv = self._init_kv(evict_range=evict_range)
        kv.ctx_ids = ctx_ids  # 保存原始 context（用于 scoring）
        kv.prefill_ids = prefill_ids  # 保存完整预填充 IDs（用于 generate）

        # VLM: 检测多模态 token 位置并设置到 KV cache
        multimodal_ranges_ctx = []  # 基于 ctx_ids 的位置
        if self.is_vlm:
            # 检测 context 中的多模态 token 范围
            multimodal_ranges_ctx = self.detect_multimodal_positions(ctx_ids)
            if multimodal_ranges_ctx:
                # 设置到 KV cache，这些位置的 token 将不会被 scoring/pruning
                kv.set_multimodal_ranges(multimodal_ranges_ctx)

        # 分块预填充：避免长 context 导致内存溢出
        # 将多模态 token 区域转换为 prefill_ids 的位置（加上系统提示长度）
        sys_prompt_len = self.sys_prompt_ids.shape[1]
        multimodal_ranges_prefill = [(s + sys_prompt_len, e + sys_prompt_len) for s, e in multimodal_ranges_ctx]

        # 使用 VLM 分块函数，确保多模态 token 单独处理
        chunks, needs_vision = vlm_chunk_fn(prefill_ids, prefill_chunk_size, multimodal_ranges_prefill)

        for i, input_ids in enumerate(tqdm(chunks, desc="Prefill")):
            if needs_vision[i] and vlm_inputs is not None:
                # 包含多模态 token 的 chunk：传入视觉参数
                self.__call__(input_ids, kv, update_cache=True, **vlm_inputs)
            else:
                # 纯文本 chunk：不传入视觉参数
                self.__call__(input_ids, kv, update_cache=True)

        if do_score:
            # 计算 KV importance scores
            # VLM 时传入 grid_thw，用于 vision chunk 的 patch 行级二次分块
            grid_thw = None
            if vlm_inputs:
                grid_thw = vlm_inputs.get("image_grid_thw") or vlm_inputs.get("video_grid_thw")
            self.scoring(kv, ctx_ids, load_score=load_score, image_grid_thw=grid_thw)

        return kv

    def _split_vision_chunk(
        self,
        vision_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        scoring_patch_rows: int,
    ) -> List[torch.Tensor]:
        """将单个 vision chunk 按 patch 行进行二次分块

        对于大图像，直接对整个 vision chunk 做 scoring 会导致内存溢出。
        根据 image_grid_thw 中的 (T, H, W) 信息，按 patch 行将 vision chunk
        切分为若干 sub-chunk，每个 sub-chunk 包含 scoring_patch_rows 行的 patch token。

        每行的 token 数 = W // spatial_merge_size（经过 PatchMerger 合并后）。
        spatial_merge_size 从模型 config 中读取（Qwen2.5-VL 默认为 2）。

        Args:
            vision_ids: 当前 vision chunk 的 token IDs，形状 [1, n_vision_tokens]
            image_grid_thw: 图像网格信息，形状 [n_images, 3]，每行为 (T, H, W)
                            如果为 None，则不分块，直接返回原 chunk
            scoring_patch_rows: 每个 sub-chunk 包含的 patch 行数

        Returns:
            List[torch.Tensor]: 分块后的 vision sub-chunk 列表
        """
        if image_grid_thw is None or scoring_patch_rows <= 0:
            return [vision_ids]

        # 获取 spatial_merge_size（默认 2，适用于 Qwen2-VL / Qwen2.5-VL）
        spatial_merge_size = getattr(
            getattr(self.config, 'vision_config', None), 'spatial_merge_size', 2
        )

        n_vision = vision_ids.shape[1]
        sub_chunks = []
        offset = 0  # 在 vision_ids 中的当前偏移

        for thw in image_grid_thw:
            t, h, w = thw[0].item(), thw[1].item(), thw[2].item()
            # 经过 PatchMerger 后，每行 token 数
            tokens_per_row = w // spatial_merge_size
            # 该图像在 LLM 中的总 token 数
            total_img_tokens = t * (h // spatial_merge_size) * tokens_per_row

            if offset >= n_vision:
                break

            img_end = min(offset + total_img_tokens, n_vision)
            img_ids = vision_ids[:, offset:img_end]

            # 按 scoring_patch_rows 行切分
            tokens_per_sub = scoring_patch_rows * tokens_per_row
            if tokens_per_sub <= 0 or img_ids.shape[1] <= tokens_per_sub:
                sub_chunks.append(img_ids)
            else:
                pos = 0
                while pos < img_ids.shape[1]:
                    sub_chunks.append(img_ids[:, pos:pos + tokens_per_sub])
                    pos += tokens_per_sub

            offset = img_end

        # 若有剩余（如 vision_start/vision_end 特殊 token），作为单独 chunk
        if offset < n_vision:
            sub_chunks.append(vision_ids[:, offset:])

        if not sub_chunks:
            return [vision_ids]

        n_sub = len(sub_chunks)
        if n_sub > 1:
            print(f"  vision chunk split into {n_sub} sub-chunks (scoring_patch_rows={scoring_patch_rows})")
        return sub_chunks

    def self_task(
        self,
        ctx_ids: torch.Tensor,
        chunk_size: int = 2000,
        prev_postfix_size=8,
        multimodal_ranges: List[Tuple[int, int]] = [],
        image_grid_thw: Optional[torch.Tensor] = None,
        scoring_patch_rows: int = 4,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """构建 Context Reconstruction 任务的分块输入

        这是 KVzip importance scoring 的核心设计：通过让模型"复述" context 的各个部分，
        来观察模型在生成时 attend 到了哪些 KV pairs，从而确定每个 KV 的重要性。

        对于 VLM 模型：
        1. 使用多模态感知分块（vlm_chunk_fn），确保多模态 token 区域单独成 chunk
        2. 对大 vision chunk 按 patch 行进行二次分块（_split_vision_chunk），
           避免超大图像的 scoring 导致内存溢出

        Args:
            ctx_ids: Context 的 token IDs，形状为 [batch_size, ctx_len]
            chunk_size: Scoring 时的分块大小（默认 2000 tokens）
            prev_postfix_size: 前一块末尾保留的 token 数量（默认 8）
            multimodal_ranges: 多模态 token 的位置范围列表（相对于 ctx_ids）
            image_grid_thw: 图像网格信息 [n_images, 3]，用于 vision chunk 的二次分块
            scoring_patch_rows: vision chunk 二次分块时每个 sub-chunk 的 patch 行数

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: 每个分块的任务输入
                每个元素是 (chunk_ids, repeat_ids_p)
        """
        # 使用多模态感知分块，确保多模态区域单独成 chunk
        chunked_inputs, is_multimodal_chunk = vlm_chunk_fn(ctx_ids, chunk_size, multimodal_ranges)

        input_ids = []
        prev_chunk = None
        first_chunk = True
        # 跟踪已处理的图像索引（用于 _split_vision_chunk 时按序分配 grid_thw）
        img_idx = 0

        for i, a_ids in enumerate(chunked_inputs):
            if is_multimodal_chunk[i] and image_grid_thw is not None:
                # 对 vision chunk 按 patch 行进行二次分块
                # 计算当前 vision chunk 对应哪些图像（按顺序分配）
                n_vision_tokens = a_ids.shape[1]
                spatial_merge_size = getattr(
                    getattr(self.config, 'vision_config', None), 'spatial_merge_size', 2
                )
                # 找出属于当前 vision chunk 的 image_grid_thw 条目
                chunk_grids = []
                consumed = 0
                j = img_idx
                while j < image_grid_thw.shape[0] and consumed < n_vision_tokens:
                    t, h, w = image_grid_thw[j][0].item(), image_grid_thw[j][1].item(), image_grid_thw[j][2].item()
                    img_tokens = t * (h // spatial_merge_size) * (w // spatial_merge_size)
                    chunk_grids.append(image_grid_thw[j:j+1])
                    consumed += img_tokens
                    j += 1
                img_idx = j

                grids_for_chunk = torch.cat(chunk_grids, dim=0) if chunk_grids else None
                sub_chunks = self._split_vision_chunk(a_ids, grids_for_chunk, scoring_patch_rows)
            else:
                sub_chunks = [a_ids]

            for sub in sub_chunks:
                if first_chunk:
                    prompt = f"\n\nRepeat the previous context exactly."
                    q_ids = self.encode(prompt)
                    first_chunk = False
                else:
                    prompt = f"\n\nRepeat the part of the previous context exactly, starting with "
                    q_ids = self.encode(prompt)
                    postfix_prev = prev_chunk[:, -prev_postfix_size:]
                    q_ids = torch.cat([q_ids, postfix_prev], dim=1)

                input_ids.append((
                    sub,
                    torch.cat([q_ids, self.postfix_ids, sub], dim=1)
                ))
                prev_chunk = sub

        return input_ids

    @torch.inference_mode()
    def scoring(
        self,
        kv: Union[RetainCache, EvictCache],
        ctx_ids: torch.Tensor,
        load_score=False,
        image_grid_thw: Optional[torch.Tensor] = None,
        scoring_patch_rows: int = 4,
    ):
        """计算 KV cache 的重要性分数

        通过 Context Reconstruction 任务计算每个 KV pair 的 importance score。
        核心思想：让模型尝试复述 context 各部分，通过观察 attention weights 确定重要性。

        Args:
            kv: KV cache 实例（已完成 prefill）
            ctx_ids: Context 的 token IDs
            load_score: 是否加载预计算的 head-level scores
                - True: context-independent 模式，使用预存储的 scores
                - False: context-dependent 模式，在当前 context 上实时计算

        实现流程（context-dependent 模式）:
            1. 初始化 score 存储结构 kv.score[layer][head][position] = 0
            2. 构建分块的 self_task（每个 chunk 一个 repeat 任务）
            3. 对每个 chunk:
               a. 设置 scoring 范围: start_idx ~ end_idx
               b. 执行 forward pass（在 attention 层触发 _get_score）
               c. _get_score 计算 query 对当前 chunk KV 的 attention weights
               d. 取 max 作为 importance score
            4. 最终 kv.score 包含每个 position 的分数

        Scoring 范围控制（关键机制）:
            ┌─────────────────────────────────────────────────────────────────┐
            │ KV Cache 结构:                                                  │
            │ [sink(系统提示) | chunk0_kv | chunk1_kv | chunk2_kv | ...]      │
            │                                                                 │
            │ Scoring chunk 1 时:                                            │
            │ start_idx = 2000, end_idx = 4000                               │
            │ _get_score 只计算 chunk1_kv 范围内的 attention weights         │
            │ 其他 chunks 的 KV 被忽略（不参与 attention 计算）               │
            └─────────────────────────────────────────────────────────────────┘

        为什么这样设计?
            - 每个 chunk 独立 scoring，确保每个 position 都有机会作为 query 被 attend
            - 只关注当前 chunk 的 KV，避免 attention 被 long context 稀释
            - 分块处理提高效率，避免一次处理超长序列

        context-independent 模式（load_score=True）:
            - 使用预计算的 head-level importance scores
            - 无需 runtime scoring overhead
            - 分数来自多个数据集的 max，保证泛化性
            - 存储路径: ./utils/head_score/{model_name}-*.pt

        输出:
            kv.score 更新为 [n_layers][n_heads][ctx_len] 的 importance 分数
            kv.get_score 设为 False（完成 scoring 标记）
        """
        if not load_score:
            # Context-dependent 模式：在当前 context 上实时计算 scores

            # Step 1: 初始化 score 存储结构
            kv.init_score()  # 创建 score[layer] = zeros([1, n_heads_kv, ctx_len])

            # Step 2: 保存原始 start_idx（scoring 后会恢复）
            start_idx_tmp = kv.start_idx  # 通常是系统提示长度（如 50）

            # Step 3: 构建分块的 repeat 任务（VLM 时传入多模态范围和图像网格信息）
            kv.end_idx = 0  # 初始化 end_idx
            input_ids = self.self_task(
                ctx_ids,
                multimodal_ranges=kv.multimodal_ranges,
                image_grid_thw=image_grid_thw,
                scoring_patch_rows=scoring_patch_rows,
            )  # 返回 List[(chunk_ids, repeat_ids)]

            # Step 4: 对每个 chunk 执行 forward，计算 score
            for i, (prefill_ids_p, repeat_ids_p) in enumerate(
                tqdm(input_ids, desc=f"Importance scoring")
            ):
                # 设置当前 chunk 的 scoring 范围
                kv.end_idx = kv.start_idx + prefill_ids_p.shape[1]
                # 例如 chunk1: start_idx=2000, end_idx=4000

                # 执行 forward pass
                # 在 attention forward 中，如果 kv.get_score=True，会调用 _get_score()
                # _get_score() 计算 query 对 [start_idx:end_idx] 范围 KV 的 attention weights
                self.__call__(repeat_ids_p, kv, update_cache=False)  # 不更新 cache

                # 移动到下一个 chunk
                kv.start_idx = kv.end_idx

            # Step 5: 恢复原始 start_idx
            kv.start_idx = start_idx_tmp

            # Step 6: 验证 score 维度正确
            assert kv.score[0].shape[-1] == kv.ctx_len
        else:
            # Context-independent 模式：加载预计算的 head-level scores
            kv.score = load_head_score(self.name, kv.ctx_len)

        # 完成 scoring，关闭 get_score 标记
        kv.get_score = False

    @torch.inference_mode()
    def generate(
        self,
        query: Union[str, torch.Tensor],
        kv: Optional[Union[RetainCache, EvictCache]] = None,
        update_cache: bool = False,
    ) -> str:
        """使用 KV cache 生成模型响应

        在预填充的 context KV cache 上，生成对用户查询的回答。
        支持 KV cache 的裁剪（pruned）和完整（full）两种模式。

        Args:
            query: 用户查询，可以是字符串或已编码的 token IDs
            kv: KV cache 实例（通常来自 prefill）
                - 如果为 None，会创建新的空 cache（不推荐，会丢失 context）
                - 如果是已裁剪的 cache（kv.pruned=True），使用压缩后的 KV 推理
            update_cache: 是否在生成后保留 query 和回答的 KV
                - False: 生成后删除 query 和回答的 KV（默认，支持多查询评估）
                - True: 保留所有 KV，支持多轮对话（multi-turn）

        Returns:
            str: 模型生成的回答文本（不包含 query）

        生成流程:
            1. 初始化/验证 KV cache
            2. 构建完整输入（prefill_ids + query_ids）
            3. 调用 model.generate() 进行自回归生成
            4. 解码生成的 token IDs
            5. 根据 update_cache 决定是否清理 cache

        KV Cache 状态管理:
            ┌─────────────────────────────────────────────────────────────────┐
            │ update_cache=False (默认):                                     │
            │   生成前: cache = [sys_prompt | context_kv]                    │
            │   生成中: cache += [query_kv | answer_kv]                      │
            │   生成后: cache = [sys_prompt | context_kv] (slice 撤销修改)    │
            │   → 支持对同一 context 进行多个 query 的评估                    │
            ├─────────────────────────────────────────────────────────────────┤
            │ update_cache=True (多轮对话):                                  │
            │   生成前: cache = [sys_prompt | context_kv | prev_qa_kv]       │
            │   生成后: cache = [sys_prompt | context_kv | prev_qa_kv |      │
            │                    new_query_kv | new_answer_kv]               │
            │   → 支持连续对话，保留完整历史                                  │
            └─────────────────────────────────────────────────────────────────┘

        HuggingFace generate 的特殊处理:
            - model.generate() 需要完整的 input_ids（包含已 cache 的部分）
            - 内部会自动切片，只处理新 tokens（input_ids[:, -kv.get_seq_length():]）
            - 这是为了兼容 HuggingFace 的 cache 机制

        Example:
            kv = model.prefill(context)
            kv.prune(ratio=0.3)  # 裁剪 70% KV

            # 第一个 query
            answer1 = model.generate("What is my name?", kv)
            # cache 状态保持 [sys_prompt | context_kv(30%)]

            # 第二个 query（复用同一份裁剪后的 cache）
            answer2 = model.generate("Where do I live?", kv)
        """
        # 初始化 KV cache（如果 kv 为 None，创建空 cache）
        kv = self._init_kv(kv=kv)
        seen_token_prev = kv._seen_tokens  # 记录当前 cache 的 token 数量

        # Gemma3 的 Hybrid cache 需要特殊处理
        if isinstance(kv, RetainHybridCache) and not update_cache:
            kv.backup_sliding_cache()

        # 准备输入
        input_ids = query
        if type(query) == str:
            input_ids = self.encode(query)

        # HuggingFace 的 model.generate 需要完整的 input_ids（包含已 cache 的部分）
        # 内部会根据 kv.get_seq_length() 自动切片，只处理新 tokens
        if kv.prefill_ids is not None:
            input_ids = torch.cat([kv.prefill_ids, input_ids], dim=1)

        # 执行自回归生成
        output = self.model.generate(input_ids, past_key_values=kv, **self.gen_kwargs)

        # 解析生成的回答（去掉输入部分和最后的 EOS token）
        a_ids = output[:, input_ids.shape[1]:-1]
        a = self.decode(a_ids)

        # Cache 状态管理
        if not update_cache:
            # 删除本次 query 和 answer 的 KV，恢复到 prefill 状态
            # 支持对同一 context 进行多个 query 的评估
            kv.slice(seen_token_prev)
        else:
            # 保留本次 query 和 answer 的 KV，更新 prefill_ids
            # 支持多轮对话
            # 注意：此时 kv.prefill_ids 需要更新为包含 query 和 answer
            if kv.prefill_ids is not None:
                kv.prefill_ids = torch.cat([kv.prefill_ids, input_ids, a_ids], dim=1)
            else:
                kv.prefill_ids = torch.cat([input_ids, a_ids], dim=1)

        return a

    @torch.inference_mode()
    def _prob(self, input_ids, kv=None, device="cuda") -> torch.Tensor:
        """获取下一个 token 的预测概率分布

        用于评估模型在给定 query-answer 对时的预测概率。
        主要用于计算 perplexity、准确率等评估指标。

        Args:
            input_ids: 输入 token IDs（query + answer）
            kv: KV cache 实例（可选）
            device: 输出设备，"cuda" 或 "cpu"

        Returns:
            torch.Tensor: 每个位置的 next token 概率分布
                形状为 [seq_len, vocab_size]
                softmax 已应用，值域 [0, 1]

        使用场景:
            - 评估模型在裁剪 KV cache 后的预测能力
            - 计算生成的 perplexity
            - 分析模型对特定 token 的置信度

        Example:
            # 计算 answer 的概率
            kv = model.prefill(context)
            kv.prune(ratio=0.3)

            query_ids = model.apply_template("What is my name?")
            answer_ids = model.encode("Kim")
            input_ids = torch.cat([query_ids, answer_ids], dim=1)

            prob = model._prob(input_ids, kv)
            # prob[-1] 是 "Kim" 第一个 token 的预测概率
        """
        kv = self._init_kv(kv=kv)

        # 获取 logits
        if isinstance(self.model, LlamaForCausalLMW8A8):
            # QServe 量化模型有特殊的输出格式
            output = self.__call__(
                input_ids,
                kv,
                update_cache=False,
                return_logits=True,
                is_prompt=False
            )
            output = output[0]
        else:
            # 标准模型的 logits 输出
            output = self.__call__(input_ids, kv, update_cache=False, return_logits=True)
            output = output.logits[0]

        # 应用 softmax 得到概率分布
        output = inplace_softmax(output).squeeze()

        # 根据需要移动到指定设备
        if device == "cpu":
            return output.cpu()
        return output
