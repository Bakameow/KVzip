"""
图像大海捞针任务：在长文本中插入一张包含多种视觉 key 的复合图像，
测试模型能否从压缩后的 KV cache 中分别检索每个 key。

复合图像布局（2×3 网格）：
  ┌──────────────┬──────────────┬──────────────┐
  │  plain       │  colored     │  rotated     │
  │  (粉底黑字)  │  (彩色背景)  │  (旋转35°)   │
  ├──────────────┼──────────────┼──────────────┤
  │  mirrored    │  perspective │  (空白)      │
  │  (镜像翻转)  │  (透视变形)  │              │
  └──────────────┴──────────────┴──────────────┘

每个格子显示不同的 6 位数字，用不同的 query 分别询问。
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from model import ModelKVzip
from utils.func import TimeStamp
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────
# 图像生成工具
# ──────────────────────────────────────────────

def _get_font(size=60):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_centered_text(draw, text, font, region, text_color):
    """在指定区域（x0,y0,x1,y1）内居中绘制文字"""
    x0, y0, x1, y1 = region
    w, h = x1 - x0, y1 - y0
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x0 + (w - tw) // 2, y0 + (h - th) // 2), text, fill=text_color, font=font)


def _find_perspective_coeffs(src, dst):
    """计算透视变换系数（PIL.Image.transform 所需的 8 个系数）"""
    matrix = []
    for (x, y), (X, Y) in zip(dst, src):
        matrix.extend([X, Y, 1, 0, 0, 0, -x * X, -x * Y])
        matrix.extend([0, 0, 0, X, Y, 1, -y * X, -y * Y])
    A = np.array(matrix, dtype=float).reshape(8, 8)
    b = np.array([coord for pt in dst for coord in pt], dtype=float)
    return tuple(np.linalg.solve(A, b))


def _make_cell(text, style, cell_size):
    """生成单个格子图像（cell_size × cell_size）"""
    w, h = cell_size, cell_size
    font = _get_font(h // 6)

    if style == "plain":
        img = Image.new("RGB", (w, h), (255, 200, 200))
        draw = ImageDraw.Draw(img)
        _draw_centered_text(draw, text, font, (0, 0, w, h), (0, 0, 0))

    elif style == "colored":
        rng = random.Random(sum(ord(c) for c in text) + 1)
        bg = (rng.randint(30, 200), rng.randint(30, 200), rng.randint(30, 200))
        brightness = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        fg = (10, 10, 10) if brightness > 128 else (245, 245, 245)
        img = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)
        _draw_centered_text(draw, text, font, (0, 0, w, h), fg)

    elif style == "rotated":
        # 先画在 2× 画布上再旋转裁剪
        canvas = Image.new("RGB", (w * 2, h * 2), (240, 240, 240))
        draw = ImageDraw.Draw(canvas)
        font_big = _get_font(h // 4)
        bbox = draw.textbbox((0, 0), text, font=font_big)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((canvas.width // 2 - tw // 2, canvas.height // 2 - th // 2),
                  text, fill=(20, 20, 180), font=font_big)
        rotated = canvas.rotate(35, resample=Image.Resampling.BICUBIC)
        left = (rotated.width - w) // 2
        top  = (rotated.height - h) // 2
        img = rotated.crop((left, top, left + w, top + h))

    elif style == "mirrored":
        img = Image.new("RGB", (w, h), (200, 240, 200))
        draw = ImageDraw.Draw(img)
        _draw_centered_text(draw, text, font, (0, 0, w, h), (150, 0, 0))
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    elif style == "perspective":
        img = Image.new("RGB", (w, h), (220, 220, 255))
        draw = ImageDraw.Draw(img)
        _draw_centered_text(draw, text, font, (0, 0, w, h), (0, 80, 0))
        skew = w // 5
        coeffs = _find_perspective_coeffs(
            [(0, 0), (w, 0), (w, h), (0, h)],
            [(skew, 0), (w - skew, 0), (w, h), (0, h)]
        )
        img = img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs,
                             Image.Resampling.BICUBIC)
    else:
        img = Image.new("RGB", (w, h), (230, 230, 230))

    return img


# ──────────────────────────────────────────────
# 复合图像：所有 key 拼在一张图里
# ──────────────────────────────────────────────

# 每个 key 的配置：(样式, 标签描述, query模板, 数字位数, 旋转角度)
# query 模板中 {label} 会被替换为标签描述
KEYS = [
    ("plain",       "3-digit-number",          "What is the 3-digit number shown in the image?", 3, 0),
    ("colored",     "6-digit-number",          "What 6-digit number appears in the image?", 6, 0),
    ("rotated",     "rotated-35-degree",       "What number is shown rotated by approximately 35 degrees?", 6, 35),
    ("mirrored",    "horizontally-flipped",    "What number appears horizontally flipped in the image?", 5, 0),
    ("perspective", "perspective-transformed", "What number is displayed with perspective transformation?", 7, 0),
    # 新增问题：关于图像内容的问题
    ("plain",       "bowls-count",             "How many bowls are there in the image?", 0, 0),
    ("plain",       "hotpot-flavor",           "What flavor is the hotpot in the image?", 0, 0),
    ("plain",       "tissue-presence",         "Are there tissues in the image?", 0, 0),
]

GRID_COLS = 3  # 每行格子数


def create_composite_image(secrets: list[str], cell_size: int = 400) -> Image.Image:
    """将所有 key 拼成一张网格图像

    Args:
        secrets: 每个 key 对应的秘密数字列表，长度须与 KEYS 一致
        cell_size: 每个格子的像素尺寸

    Returns:
        拼合后的 PIL Image
    """
    n = len(KEYS)
    cols = GRID_COLS
    rows = (n + cols - 1) // cols
    border = 4  # 格子间分隔线宽度

    total_w = cols * cell_size + (cols + 1) * border
    total_h = rows * cell_size + (rows + 1) * border

    canvas = Image.new("RGB", (total_w, total_h), (80, 80, 80))  # 深灰分隔线

    for idx, ((style, label, _), secret) in enumerate(zip(KEYS, secrets)):
        row, col = divmod(idx, cols)
        x0 = border + col * (cell_size + border)
        y0 = border + row * (cell_size + border)

        cell = _make_cell(secret, style, cell_size)

        # 在格子左上角加标签
        draw = ImageDraw.Draw(cell)
        label_font = _get_font(max(14, cell_size // 20))
        draw.rectangle([0, 0, cell_size, cell_size // 8], fill=(0, 0, 0, 160))
        draw.text((4, 2), label, fill=(255, 255, 0), font=label_font)

        canvas.paste(cell, (x0, y0))

    return canvas


def create_hotpot_composite_image(secrets: list[str], background_path: str = "hotpot.jpg") -> Image.Image:
    """使用 hotpot.jpg 作为背景，在不同位置放置不同格式的 key 数字

    Args:
        secrets: 每个 key 对应的秘密数字列表，长度须与 KEYS 一致
        background_path: 背景图像路径

    Returns:
        合成后的 PIL Image
    """
    # 加载背景图像
    background = Image.open(background_path).convert("RGB")

    # 获取背景图像尺寸
    width, height = background.size

    # 定义数字放置的位置
    positions = [
        (0.15, 0.15),   # 左上
        (0.85, 0.15),   # 右上
        (0.15, 0.85),   # 左下
        (0.85, 0.85),   # 右下
        (0.50, 0.50),   # 中心
        (0.30, 0.50),   # 左中
        (0.70, 0.50),   # 右中
        (0.50, 0.30),   # 上中
    ]

    # 创建绘图对象
    draw = ImageDraw.Draw(background)
    font_size = min(width, height) // 10
    font = _get_font(font_size)

    # 在指定位置绘制不同格式的数字（仅前5个）
    for idx in range(min(5, len(KEYS))):
        if secrets[idx]:
            style, label, question, num_digits, rotation_angle = KEYS[idx]
            secret = secrets[idx]
            x_ratio, y_ratio = positions[idx]
            x = int(width * x_ratio)
            y = int(height * y_ratio)

            # 根据样式设置颜色和变换
            if style == "plain":
                text_color = (50, 50, 50)  # 深灰文字，无特殊变换
                # 直接绘制
                bbox = draw.textbbox((0, 0), secret, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = x - text_w // 2
                text_y = y - text_h // 2
                draw.text((text_x, text_y), secret, fill=text_color, font=font)

            elif style == "colored":
                # 根据数字生成随机颜色
                rng = random.Random(sum(ord(c) for c in secret) + 1)
                text_color = (rng.randint(30, 200), rng.randint(30, 200), rng.randint(30, 200))
                # 直接绘制彩色数字
                bbox = draw.textbbox((0, 0), secret, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = x - text_w // 2
                text_y = y - text_h // 2
                draw.text((text_x, text_y), secret, fill=text_color, font=font)

            elif style == "rotated":
                # 旋转数字
                text_color = (20, 20, 180)  # 蓝色文字
                # 在大画布上绘制后旋转裁剪
                canvas_size = max(font_size * 4, int(font_size * len(secret) * 1.5))
                canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
                canvas_draw = ImageDraw.Draw(canvas)
                font_big = _get_font(font_size)
                bbox = canvas_draw.textbbox((0, 0), secret, font=font_big)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                canvas_draw.text(
                    (canvas_size // 2 - text_w // 2, canvas_size // 2 - text_h // 2),
                    secret, fill=text_color, font=font_big
                )
                # 旋转
                rotated_canvas = canvas.rotate(rotation_angle, resample=Image.Resampling.BICUBIC, expand=True)
                # 粘贴到背景，中心对齐
                paste_x = x - rotated_canvas.width // 2
                paste_y = y - rotated_canvas.height // 2
                background.paste(rotated_canvas, (paste_x, paste_y), rotated_canvas)

            elif style == "mirrored":
                # 镜像翻转（水平）
                text_color = (150, 0, 0)  # 红色文字
                # 创建临时图像
                temp_size = int(font_size * len(secret) * 1.2)
                temp = Image.new("RGB", (temp_size, font_size * 2), (0, 0, 0))
                temp_draw = ImageDraw.Draw(temp)
                bbox = temp_draw.textbbox((0, 0), secret, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                temp_draw.text((temp_size // 2 - text_w // 2, font_size - text_h // 2), secret, fill=text_color, font=font)
                # 水平翻转
                temp = temp.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                # 粘贴到背景
                paste_x = x - temp.width // 2
                paste_y = y - temp.height // 2
                background.paste(temp, (paste_x, paste_y))

            elif style == "perspective":
                # 透视变换
                text_color = (0, 80, 0)  # 绿色文字
                # 创建临时图像
                temp_size = int(font_size * len(secret) * 1.5)
                temp = Image.new("RGB", (temp_size, temp_size), (0, 0, 0))
                temp_draw = ImageDraw.Draw(temp)
                bbox = temp_draw.textbbox((0, 0), secret, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                temp_draw.text((temp_size // 2 - text_w // 2, temp_size // 2 - text_h // 2), secret, fill=text_color, font=font)
                # 应用透视变换
                w, h = temp_size, temp_size
                skew = w // 5
                coeffs = _find_perspective_coeffs(
                    [(0, 0), (w, 0), (w, h), (0, h)],  # 原始四个角
                    [(skew, 0), (w - skew, 0), (w, h), (0, h)]  # 变换后的四个角（上边倾斜）
                )
                temp = temp.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
                # 粘贴到背景
                paste_x = x - temp.width // 2
                paste_y = y - temp.height // 2
                background.paste(temp, (paste_x, paste_y))

    return background


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def generate_haystack(tokenizer, target_length: int = 100000) -> str:
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
    base_tokens = len(tokenizer.encode(base_text, add_special_tokens=False))
    repeat_count = (target_length // base_tokens) + 1
    return base_text * repeat_count


def insert_image_in_context(processor, haystack_before, haystack_after, image, image_description):
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': haystack_before + "\n\n" + image_description + "\n"},
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': "\n" + haystack_after}
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], return_tensors='pt')
    return inputs


# ──────────────────────────────────────────────
# 主实验函数
# ──────────────────────────────────────────────

def run_image_needle_experiment(
    model_name: str = "qwen2.5-vl-3b",
    context_length: int = 10000,
    depth_percent: int = 50,
    mode: str = "kvzip",
    compression_ratio: float = 0.3,
    cell_size: int = 400,
    use_hotpot_background: bool = False,
):
    """运行图像大海捞针实验（复合图像版）

    一张图包含所有视觉 key，每个 key 显示不同的 6 位数字，
    prefill 一次后用不同的 query 分别询问每个 key 的数字。

    Args:
        model_name: 模型名称
        context_length: 目标上下文长度（tokens）
        depth_percent: 图像插入深度（百分比）
        mode: KVzip 模式（kvzip, kvzip_head, no, full）
        compression_ratio: KV 压缩保留比例
        cell_size: 复合图像中每个格子的像素尺寸
        use_hotpot_background: 是否使用 hotpot.jpg 作为背景图像
    """
    stamp = TimeStamp(verbose=True, unit="ms")
    model = ModelKVzip(model_name)

    if not model.is_vlm:
        print(f"Error: {model_name} is not a VLM model!")
        return

    # 为每个 key 分配不同位数的秘密数字（仅前5个需要数字）
    rng = random.Random(42)
    secrets = []
    for idx, (style, label, question, num_digits, rotation_angle) in enumerate(KEYS):
        if idx < 5 and num_digits > 0:
            # 根据位数生成随机数字
            min_val = 10 ** (num_digits - 1)
            max_val = 10 ** num_digits - 1
            secret = str(rng.randint(min_val, max_val))
            secrets.append(secret)
        elif idx < 5:
            # 如果没有指定位数，使用默认6位
            secrets.append(str(rng.randint(100000, 999999)))
        else:
            # 后3个是关于图像内容的问题，不需要数字
            secrets.append("")

    # 对于 hotpot 图像内容的问题，设置预期的答案
    # 根据图像内容：有大约 4-5 个碗，火锅看起来是红油麻辣味，有纸巾
    expected_answers = {
        "bowls-count": "4-5",  # 或者 "four", "five"
        "hotpot-flavor": "spicy",  # 或者 "麻辣", "红油"
        "tissue-presence": "yes",  # 或者 "有"
    }

    print("Secret numbers per key:")
    for idx, (style, label, question, num_digits, rotation_angle) in enumerate(KEYS):
        if secrets[idx]:
            print(f"  {label:30s}: {secrets[idx]} ({num_digits} digits, rotation: {rotation_angle}°)")
        else:
            print(f"  {label:30s}: (image content question)")

    # 生成复合图像
    if use_hotpot_background:
        composite = create_hotpot_composite_image(secrets, background_path="hotpot.jpg")
        print(f"\nUsing hotpot.jpg as background")
    else:
        composite = create_composite_image(secrets, cell_size=cell_size)
    composite.save("test_image.png")
    print(f"Composite image saved: test_image.png  ({composite.width}×{composite.height}px)")

    # 生成 haystack
    haystack = generate_haystack(model.tokenizer, context_length)
    haystack_tokens = model.tokenizer.encode(haystack, add_special_tokens=False)
    insertion_point = int(len(haystack_tokens) * (depth_percent / 100))
    period_tokens = (model.tokenizer.encode('.', add_special_tokens=False) +
                     model.tokenizer.encode('.\n', add_special_tokens=False))
    while insertion_point > 0 and haystack_tokens[insertion_point] not in period_tokens:
        insertion_point -= 1

    haystack_before = model.tokenizer.decode(haystack_tokens[:insertion_point])
    haystack_after  = model.tokenizer.decode(haystack_tokens[insertion_point:])
    print(f"Image inserted at {insertion_point} tokens ({depth_percent}% depth)")

    if use_hotpot_background:
        image_description = (
            "Here is a hotpot image with numbers displayed at different positions. "
            "The image shows a hotpot with various ingredients and dining setup. "
            "There are also questions about the image content including bowl count, hotpot flavor, and tissue presence."
        )
    else:
        image_description = (
            "Here is an image containing multiple numbers displayed in different visual styles "
            "(plain, colorful, rotated, mirror-flipped, and perspective-distorted). "
            "Each region is labeled with its style name."
        )
    inputs = insert_image_in_context(
        model.processor, haystack_before, haystack_after, composite, image_description
    )

    full_input_ids = inputs['input_ids'].cuda()
    user_start_token = model.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)[-1]
    user_start_idx = (full_input_ids[0] == user_start_token).nonzero(as_tuple=True)[0][0].item()
    ctx_ids = full_input_ids[:, user_start_idx:]
    print(f"Context length: {ctx_ids.shape[1]} tokens (including image tokens)")

    vlm_inputs = {
        'pixel_values':   inputs['pixel_values'].cuda(),
        'image_grid_thw': inputs['image_grid_thw'].cuda(),
    }

    stamp("Before Prefill")
    kv = model.prefill(
        ctx_ids,
        load_score=(mode == "kvzip_head"),
        do_score=(mode in ["kvzip", "kvzip_head"]),
        vlm_inputs=vlm_inputs,
    )
    stamp(f"KV={kv._mem()}GB. After Prefill")

    if mode in ["kvzip", "kvzip_head"]:
        ratio = compression_ratio if mode == "kvzip" else 0.6
        kv.prune(ratio=ratio)
        stamp(f"KV={kv._mem()}GB. After Compression (ratio={ratio})")

    # 对每个 key 分别提问，复用同一份 KV cache
    print(f"\n{'='*70}")
    results = {}
    for idx, ((style, label, question), secret) in enumerate(zip(KEYS, secrets)):
        print(f"\n[{label}]")
        print(f"  Q: {question}")
        query_ids = model.apply_template(question)
        output = model.generate(query_ids, kv=kv, update_cache=False)

        # 对于数字类问题，检查数字是否在答案中
        if idx < 5:
            success = secret in output
            gt = secret
        else:
            # 对于图像内容问题，检查答案是否包含关键词
            if label == "bowls-count":
                success = any(kw in output.lower() for kw in ["4", "5", "four", "five", "四", "五"])
                gt = expected_answers[label]
            elif label == "hotpot-flavor":
                success = any(kw in output.lower() for kw in ["spicy", "麻辣", "红油", "辣", "hot"])
                gt = expected_answers[label]
            elif label == "tissue-presence":
                success = any(kw in output.lower() for kw in ["yes", "有", "存在", "present", "tissue", "纸巾"])
                gt = expected_answers[label]
            else:
                success = False
                gt = "unknown"

        print(f"  A: {output.strip()}")
        print(f"  GT: {gt}  →  {'✓ SUCCESS' if success else '✗ FAILED'}")
        results[label] = success
        num_tokens = query_ids.shape[1] + model.encode(output).shape[1] + 1
        stamp(f"[{label}]", denominator=num_tokens)

    print(f"\n{'='*70}")
    print("Summary:")
    for label, ok in results.items():
        print(f"  {label:30s}: {'✓' if ok else '✗'}")
    print(f"  Total: {sum(results.values())}/{len(results)} passed")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Needle-in-a-Haystack (composite image) for KVzip VLM")
    parser.add_argument("-m", "--model",          default="qwen2.5-vl-3b")
    parser.add_argument("-c", "--context_length", type=int,   default=10000)
    parser.add_argument("-d", "--depth_percent",  type=int,   default=50)
    parser.add_argument("--mode",                 default="kvzip",
                        choices=["kvzip", "kvzip_head", "no", "full"])
    parser.add_argument("-r", "--ratio",          type=float, default=0.3)
    parser.add_argument("--cell_size",            type=int,   default=400,
                        help="每个格子的像素尺寸（默认 400）")
    parser.add_argument("--use_hotpot",           action="store_true",
                        help="使用 hotpot.jpg 作为背景图像")
    args = parser.parse_args()

    run_image_needle_experiment(
        model_name=args.model,
        context_length=args.context_length,
        depth_percent=args.depth_percent,
        mode=args.mode,
        compression_ratio=args.ratio,
        cell_size=args.cell_size,
        use_hotpot_background=args.use_hotpot,
    )

