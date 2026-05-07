import os

# 强制离线模式：不从网络下载，只使用本地缓存
# 必须在导入 transformers 之前设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor


def get_model_id(name: str):
    """ We support abbreviated model names such as:
        llama3.1-8b, llama3.2-*b, qwen2.5-*b, qwen3-*b, and gemma3-*b.
        The full model ID, such as "meta-llama/Llama-3.1-8B-Instruct", is also supported.
    """

    size = name.split("-")[-1].split("b")[0]  # xx-14b -> 14

    if name == "llama3.1-8b":
        return "meta-llama/Llama-3.1-8B-Instruct"
    elif name == "llama3.0-8b":
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    elif name == "duo":
        return "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
    elif name == "llama3-8b-4m-w8a8kv4":
        return "mit-han-lab/Llama-3-8B-Instruct-Gradient-4194k-per-channel"

    elif name.startswith("llama3.2-"):
        assert size in ["1", "3"], "Model is not supported!"
        return f"meta-llama/Llama-3.2-{size}B-Instruct"

    elif name.startswith("qwen2.5-vl-"):
        # Qwen2.5-VL vision-language models (must come before qwen2.5-)
        assert size in ["3", "7", "72"], "Model is not supported!"
        return f"Qwen/Qwen2.5-VL-{size}B-Instruct"

    elif name.startswith("qwen2.5-"):
        assert size in ["7", "14"], "Model is not supported!"
        return f"Qwen/Qwen2.5-{size}B-Instruct-1M"

    elif name.startswith("qwen2-vl-"):
        # Qwen2-VL vision-language models
        assert size in ["2", "7", "72"], "Model is not supported!"
        return f"Qwen/Qwen2-VL-{size}B-Instruct"

    elif name.startswith("qwen3-"):
        assert size in ["0.6", "1.7", "4", "8", "14", "32"], "Model is not supported!"
        return f"Qwen/Qwen3-{size}B"

    elif name.startswith("qwen3-vl-"):
        # Future Qwen3-VL (placeholder, will use Qwen2.5-VL as fallback)
        logger.warning("Qwen3-VL may not be released yet. Using Qwen2.5-VL as fallback.")
        assert size in ["3", "7", "72"], "Model is not supported!"
        return f"Qwen/Qwen2.5-VL-{size}B-Instruct"

    elif name.startswith("gemma3-"):
        assert size in ["1", "4", "12", "27"], "Model is not supported!"
        return f"google/gemma-3-{size}b-it"

    else:
        return name  # Warning: some models might not be compatible and cause errors


def is_vlm_model(model_id: str) -> bool:
    """Check if the model is a vision-language model (VLM).

    Args:
        model_id: Model identifier string

    Returns:
        True if the model is a VLM, False otherwise
    """
    model_id_lower = model_id.lower()
    vlm_keywords = ["vl", "vision", "qwen2-vl", "qwen2.5-vl", "qwen3-vl",
                    "llava", "idefics", "mllama"]
    return any(kw in model_id_lower for kw in vlm_keywords)


def get_multimodal_token_ids(tokenizer) -> dict:
    """Get special token IDs for multimodal tokens.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Dictionary with token IDs for vision-related special tokens
    """
    special_tokens = {}

    # Qwen-VL and other VLM special tokens
    token_names = [
        "vision_start", "vision_end", "vision_pad", "image_pad", "video_pad",
    ]

    for name in token_names:
        token_str = f"<|{name}|>"
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                special_tokens[name] = token_id
        except:
            pass

    return special_tokens


def load_model(model_name: str, **kwargs):
    model_id = get_model_id(model_name)
    is_vlm = is_vlm_model(model_id)

    if not ("w8a8kv4" in model_name):
        from model.monkeypatch import replace_attn
        replace_attn(model_id)

        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        if "Qwen3-" in model_id:
            config.rope_scaling = {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768
            }
            config.max_position_embeddings = 131072

        # VL models require AutoModelForVision2Seq, not AutoModelForCausalLM
        if is_vlm:
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation='flash_attention_2',
                config=config,
                local_files_only=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation='flash_attention_2',
                config=config,
                local_files_only=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

        if "llama" in model_id.lower():
            model.generation_config.pad_token_id = tokenizer.pad_token_id = 128004

        if "gemma-3" in model_id.lower():
            model = model.language_model
    else:
        model, tokenizer = load_quant_model(quant_model_id=model_id)

    model.eval()
    model.name = model_name.split("/")[-1]
    model.is_vlm = is_vlm  # Mark if this is a VLM model

    # Store multimodal token IDs for VLM models
    if is_vlm:
        model.multimodal_token_ids = get_multimodal_token_ids(tokenizer)
        logger.info(f"VLM model detected. Multimodal tokens: {model.multimodal_token_ids}")
        model.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    else:
        model.processor = None

    logger.info(f"Load {model_id} with {model.dtype}")
    return model, tokenizer


def load_quant_model(quant_model_id: str):
    from model.quant_model.w8a8kv4_llama import LlamaForCausalLM as LlamaForCausalLMW8A8
    from model.quant_model.monkeypatch import replace_attn, replace_quantized_wrapper
    replace_attn()
    replace_quantized_wrapper()

    model = LlamaForCausalLMW8A8.from_quantized(quant_model_id)
    tokenizer = LlamaForCausalLMW8A8.get_tokenizer()

    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="llama3-8b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.name)
    print(model)

    messages = [{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(input_text)

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=30)
    print(tokenizer.decode(outputs[0]))