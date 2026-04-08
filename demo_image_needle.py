"""
图像大海捞针任务：在长文本中插入图像，测试模型能否从压缩后的 KV cache 中检索图像内容
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model import ModelKVzip
from utils.func import TimeStamp
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont


def generate_haystack(tokenizer, target_length: int = 100000) -> str:
    """生成haystack文本（重复的无关内容）

    Args:
        tokenizer: tokenizer用于计算token数量
        target_length: 目标token数量（约100K）

    Returns:
        str: haystack文本
    """
    base_text = """
The solar system consists of the Sun and the objects that orbit it.
The Sun is a star located at the center of the solar system.
Earth is the third planet from the Sun and the only astronomical object known to harbor life.
The Moon is Earth's only natural satellite.
Mars is the fourth planet from the Sun, often called the Red Planet.
Jupiter is the largest planet in the solar system.
Saturn is known for its prominent ring system.
Uranus and Neptune are ice giants located in the outer solar system.
The solar system formed approximately 4.6 billion years ago from a giant interstellar molecular cloud.
"""

    # 计算需要重复的次数以达到目标长度
    base_tokens = len(tokenizer.encode(base_text, add_special_tokens=False))
    repeat_count = (target_length // base_tokens) + 1

    haystack = base_text * repeat_count
    return haystack


def create_test_image(text: str, size=(256, 256), bg_color=(255, 200, 200), text_color=(0, 0, 0)):
    """创建包含特定文本的测试图像

    Args:
        text: 要在图像中显示的文本
        size: 图像尺寸
        bg_color: 背景颜色 (R, G, B)
        text_color: 文字颜色 (R, G, B)

    Returns:
        PIL.Image: 生成的图像
    """
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # 尝试使用默认字体，如果失败则使用内置字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()

    # 计算文本位置（居中）
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)

    draw.text(position, text, fill=text_color, font=font)
    return img


def insert_image_in_context(
    processor,
    haystack_before: str,
    haystack_after: str,
    image: Image.Image,
    image_description: str = "Here is an important image:"
) -> dict:
    """在文本中插入图像，返回 processor 处理后的输入

    Args:
        processor: VLM processor
        haystack_before: 图像前的文本
        haystack_after: 图像后的文本
        image: 要插入的图像
        image_description: 图像前的描述文本

    Returns:
        dict: 包含 input_ids, pixel_values, image_grid_thw 等的字典
    """
    # 构建多模态消息
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': haystack_before + "\n\n" + image_description + "\n"},
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': "\n" + haystack_after}
        ]
    }]

    # 使用 processor 的 chat template（不添加 assistant 提示）
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # 处理图像和文本
    inputs = processor(text=[text], images=[image], return_tensors='pt')

    return inputs


def run_image_needle_experiment(
    model_name: str = "qwen2.5-vl-3b",
    context_length: int = 10000,
    depth_percent: int = 50,
    mode: str = "kvzip",
    compression_ratio: float = 0.3,
):
    """运行图像大海捞针实验

    Args:
        model_name: 模型名称
        context_length: 目标上下文长度（tokens）
        depth_percent: 图像插入深度（百分比）
        mode: KVzip模式（kvzip, kvzip_head, no, full）
        compression_ratio: KV压缩保留比例
    """
    stamp = TimeStamp(verbose=True, unit="ms")
    model = ModelKVzip(model_name)

    if not model.is_vlm:
        print(f"Error: {model_name} is not a VLM model!")
        return False

    # 创建包含特定数字的图像
    secret_number = "847291"
    test_image = create_test_image(secret_number, size=(4096, 4096))
    test_image.save("test_image.png")  # 保存图像以供检查

    # 定义问题和答案
    question = "What number is shown in the image that was provided earlier in the context?"
    answer = secret_number

    # 生成haystack
    print(f"Generating haystack (~{context_length} tokens)...")
    haystack = generate_haystack(model.tokenizer, context_length)

    # 计算插入位置
    haystack_tokens = model.tokenizer.encode(haystack, add_special_tokens=False)
    insertion_point = int(len(haystack_tokens) * (depth_percent / 100))

    # 找到句子边界
    period_tokens = model.tokenizer.encode('.', add_special_tokens=False) + \
                   model.tokenizer.encode('.\n', add_special_tokens=False)
    while insertion_point > 0 and haystack_tokens[insertion_point] not in period_tokens:
        insertion_point -= 1

    # 分割 haystack
    haystack_before = model.tokenizer.decode(haystack_tokens[:insertion_point])
    haystack_after = model.tokenizer.decode(haystack_tokens[insertion_point:])

    print(f"Image inserted at position {insertion_point} ({depth_percent}% depth)")

    # 使用 processor 构建多模态输入
    inputs = insert_image_in_context(
        model.processor,
        haystack_before,
        haystack_after,
        test_image,
        image_description="Here is an important number shown in an image:"
    )

    # 提取 context 部分的 input_ids（去掉系统提示）
    # processor 返回的 input_ids 包含完整的 chat template
    full_input_ids = inputs['input_ids'].cuda()

    # 找到 user content 的起始位置（跳过系统提示）
    # Qwen2.5-VL 的格式: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...
    user_start_token = model.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)[-1]
    user_start_idx = (full_input_ids[0] == user_start_token).nonzero(as_tuple=True)[0][0].item()

    # 提取 context 部分（从 user 开始到结束，不包括 assistant 提示）
    ctx_ids = full_input_ids[:, user_start_idx:]

    actual_length = ctx_ids.shape[1]
    print(f"Actual context length: {actual_length} tokens (including image tokens)")

    # 准备 VLM 输入（pixel_values 和 image_grid_thw）
    vlm_inputs = {
        'pixel_values': inputs['pixel_values'].cuda(),
        'image_grid_thw': inputs['image_grid_thw'].cuda(),
    }

    stamp("Before Prefill")

    # Prefill KV cache（传入 vlm_inputs）
    kv = model.prefill(
        ctx_ids,
        load_score=(mode == "kvzip_head"),
        do_score=(mode in ["kvzip", "kvzip_head"]),
        vlm_inputs=vlm_inputs,
    )
    stamp(f"KV cache size: {kv._mem()} GB. After Prefill")

    # KV compression
    if mode in ["kvzip", "kvzip_head"]:
        ratio = compression_ratio if mode == "kvzip" else 0.6
        kv.prune(ratio=ratio)
        stamp(f"KV cache size: {kv._mem()} GB. After Compression (ratio={ratio})")

    # 生成回答
    print("-" * 100)
    print(f"Question: {question}")
    query_ids = model.apply_template(question)
    output = model.generate(query_ids, kv=kv, update_cache=False)
    print(f"Model Output: {output}")
    print(f"Ground-truth: {answer}")

    # 检查答案是否正确
    success = answer in output
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")

    num_tokens = query_ids.shape[1] + model.encode(output).shape[1] + 1
    stamp(f"After Generation", denominator=num_tokens)
    print("-" * 100)

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Needle-in-a-Haystack experiment for KVzip VLM")
    parser.add_argument("-m", "--model", default="qwen2.5-vl-3b", help="Model name")
    parser.add_argument("-c", "--context_length", type=int, default=10000, help="Target context length (tokens)")
    parser.add_argument("-d", "--depth_percent", type=int, default=50, help="Image insertion depth (0-100)")
    parser.add_argument("--mode", default="kvzip", choices=["kvzip", "kvzip_head", "no", "full"], help="KVzip mode")
    parser.add_argument("-r", "--ratio", type=float, default=0.3, help="Compression ratio (keep ratio)")
    args = parser.parse_args()

    run_image_needle_experiment(
        model_name=args.model,
        context_length=args.context_length,
        depth_percent=args.depth_percent,
        mode=args.mode,
        compression_ratio=args.ratio,
    )
