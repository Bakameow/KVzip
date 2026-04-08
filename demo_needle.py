"""
简单的大海捞针任务，用于KVzip实验
上下文长度约100K tokens，测试模型从长文本中检索特定信息的能力
"""

from model import ModelKVzip
from utils.func import TimeStamp
import argparse
import torch


def generate_haystack(tokenizer, target_length: int = 100000) -> str:
    """生成haystack文本（重复的无关内容）

    Args:
        tokenizer: tokenizer用于计算token数量
        target_length: 目标token数量（约100K）

    Returns:
        str: haystack文本
    """
    # 使用重复的无关文本作为haystack
    # 这样可以快速生成大量内容，不需要外部文件
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


def insert_needle(context: str, needle: str, depth_percent: int, tokenizer) -> str:
    """在haystack中插入needle

    Args:
        context: haystack文本
        needle: 需要隐藏的关键信息
        depth_percent: needle插入位置的百分比（0-100）
        tokenizer: tokenizer用于计算位置

    Returns:
        str: 包含needle的完整context
    """
    tokens = tokenizer.encode(context, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)

    # 计算插入位置
    insertion_point = int(len(tokens) * (depth_percent / 100))

    # 找到句子边界（以.结尾）
    context_tokens = tokens[:insertion_point]
    period_tokens = tokenizer.encode('.', add_special_tokens=False) + tokenizer.encode('.\n', add_special_tokens=False)

    while context_tokens and context_tokens[-1] not in period_tokens:
        insertion_point -= 1
        context_tokens = tokens[:insertion_point]

    print(f"Needle inserted at position {insertion_point} ({depth_percent}% depth)")

    # 构建新context
    new_tokens = context_tokens + needle_tokens + tokens[insertion_point:]
    return tokenizer.decode(new_tokens)


def run_needle_haystack_experiment(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    context_length: int = 100000,
    depth_percent: int = 50,
    mode: str = "kvzip",
    compression_ratio: float = 0.3,
):
    """运行大海捞针实验

    Args:
        model_name: 模型名称
        context_length: 目标上下文长度（tokens）
        depth_percent: needle插入深度（百分比）
        mode: KVzip模式（kvzip, kvzip_head, no, full）
        compression_ratio: KV压缩保留比例
    """
    stamp = TimeStamp(verbose=True, unit="ms")
    model = ModelKVzip(model_name)

    # 定义needle和问题
    needle = "\nThe special number stored in the document is 847291. Remember this number.\n"
    question = "What is the special number mentioned in the document?"
    answer = "847291"

    # 生成haystack并插入needle
    print(f"Generating haystack (~{context_length} tokens)...")
    haystack = generate_haystack(model.tokenizer, context_length)
    context = insert_needle(haystack, needle, depth_percent, model.tokenizer)

    # 确保context长度接近目标
    actual_length = len(model.tokenizer.encode(context, add_special_tokens=False))
    print(f"Actual context length: {actual_length} tokens")

    # 格式化查询
    query = question + "\nAnswer with just the number, no explanation."

    stamp("Before Prefill")

    # Prefill KV cache
    kv = model.prefill(
        context,
        load_score=(mode == "kvzip_head"),
        do_score=(mode in ["kvzip", "kvzip_head"]),
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
    query_ids = model.apply_template(query)
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
    parser = argparse.ArgumentParser(description="Simple Needle-in-a-Haystack experiment for KVzip")
    parser.add_argument("-m", "--model", default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("-c", "--context_length", type=int, default=100000, help="Target context length (tokens)")
    parser.add_argument("-d", "--depth_percent", type=int, default=50, help="Needle insertion depth (0-100)")
    parser.add_argument("--mode", default="kvzip", choices=["kvzip", "kvzip_head", "no", "full"], help="KVzip mode")
    parser.add_argument("-r", "--ratio", type=float, default=0.3, help="Compression ratio (keep ratio)")
    args = parser.parse_args()

    run_needle_haystack_experiment(
        model_name=args.model,
        context_length=args.context_length,
        depth_percent=args.depth_percent,
        mode=args.mode,
        compression_ratio=args.ratio,
    )