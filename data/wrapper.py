import os
import torch
from typing import List, Tuple, Union, Optional
from collections import defaultdict
from loguru import logger

from attention.kvcache import RetainCache, EvictCache
from model import ModelKVzip


def get_query(task, q=None):
    if task == "repeat":
        query = f"Repeat the previous context exactly."
    elif task == "qa":
        if q is None:
            query = f"Q: Answer the question based on the previous context."
        else:
            query = f"Q: {q}"
    elif task == "reason":
        query = f"Reason and answer the question. You must say the answer in the last sentence beginning with 'The answer is'. Q: {q}"
    elif task == "summarize":
        query = f"Please summarize the previous context."
    else:
        raise ValueError(f"Invalid task: {task}")

    return query


def _ensure_video(data) -> str:
    """Download video if not present locally. Returns local path or empty string on failure.

    Uses yt-dlp Python API (import yt_dlp) so no external binary is required.
    Install with: pip install yt-dlp
    """
    video_path = data.get("video_path", "")
    if video_path and os.path.exists(video_path):
        return video_path

    # Dataset field 'videoID' holds the YouTube video ID (e.g. "fFjv93ACGo8")
    # 'video_id' is the sequential index ("001") — not the YouTube ID
    yt_id = data.get("videoID") or data.get("video_id", "")
    url = data.get("url", "")
    if not yt_id and not url:
        return ""

    os.makedirs("data/video_mme", exist_ok=True)
    if not video_path:
        video_path = f"data/video_mme/{yt_id}.mp4"

    yt_url = url if url else f"https://www.youtube.com/watch?v={yt_id}"
    print(f"[video_mme] Downloading {yt_id} -> {video_path} ...")

    try:
        import yt_dlp
    except ImportError:
        print("[video_mme] yt-dlp not installed. Run: pip install yt-dlp")
        return ""

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": video_path,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt_url])
    except Exception as e:
        print(f"[video_mme] Download failed: {e}")
        return ""

    if not os.path.exists(video_path):
        print(f"[video_mme] File not found after download: {video_path}")
        return ""

    print(f"[video_mme] Saved to {video_path}")
    return video_path


def _build_video_vlm_inputs(data, model: ModelKVzip):
    """Build ctx_ids and vlm_inputs from a video_mme sample using the VLM processor.

    Returns (ctx_ids, vlm_inputs) where vlm_inputs may be None for subtitle-only mode.
    Downloads the video via yt-dlp if not present locally.

    The returned ctx_ids contains ONLY the user message content (video tokens + text),
    WITHOUT the chat template wrappers (<|im_start|>user\\n and <|im_end|>).
    This allows prefill() to prepend its own sys_prompt_ids and build a single
    coherent user turn: system + user(instruction + video + text).
    """
    if not model.is_vlm:
        return None, None

    video_path = _ensure_video(data)
    if not video_path:
        return None, None

    import av
    import torch

    processor = model.processor
    max_frames = 32
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total = stream.frames or 0
        step = max(1, total // max_frames)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i % step == 0:
                frames.append(frame.to_image())
            if len(frames) >= max_frames:
                break
        container.close()
    except Exception as e:
        print(f"[video_mme] Failed to decode {video_path}: {e}")
        return None, None

    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames},
        {"type": "text", "text": "Watch the video carefully."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], videos=[frames], return_tensors="pt")

    full_ids = inputs["input_ids"].cuda()

    # Extract user content between <|im_start|>user\\n and the trailing <|im_end|>
    # Processor output format:
    #   <|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n[CONTENT]<|im_end|>
    # We need only the [CONTENT] part so that prefill can prepend its own
    # sys_prompt_ids and form a single coherent user turn.
    user_marker_ids = model.tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    im_end_ids = model.tokenizer.encode("<|im_end|>", add_special_tokens=False)

    # Find <|im_start|>user\\n position
    user_content_start = -1
    for i in range(full_ids.shape[1] - len(user_marker_ids) + 1):
        if torch.equal(full_ids[0, i:i + len(user_marker_ids)],
                       torch.tensor(user_marker_ids, device=full_ids.device)):
            user_content_start = i + len(user_marker_ids)
            break

    if user_content_start < 0:
        print("[video_mme] Failed to locate user marker in processor output")
        return None, None

    # Find trailing <|im_end|> (search from end)
    user_content_end = full_ids.shape[1]
    for i in range(full_ids.shape[1] - 1, -1, -1):
        if full_ids[0, i].item() in im_end_ids:
            user_content_end = i
            break

    ctx_ids = full_ids[:, user_content_start:user_content_end]

    vlm_inputs = {
        "pixel_values_videos": inputs["pixel_values_videos"].cuda(),
        "video_grid_thw": inputs["video_grid_thw"].cuda(),
    }
    return ctx_ids, vlm_inputs


class DataWrapper():

    def __init__(self, dataname, dataset, model: ModelKVzip):
        self.name, self.dataset, self.model = dataname, dataset, model
        model.set_chat_template(dataname)

    def __len__(self):
        return len(self.dataset)

    def prefill_context(self, idx: int, load_score=False) -> Union[RetainCache, EvictCache]:
        """ Prefill and scoring KV importance
        """
        data = self.dataset[idx]

        vlm_inputs = None
        if "video_mme_video" in self.name:
            ctx_ids, vlm_inputs = _build_video_vlm_inputs(data, self.model)
            if ctx_ids is None:
                # Fallback to empty context
                ctx_ids = self.model.encode("(Video unavailable)")
            # TODO: 后续需要恢复此处逻辑，改用 processor max_pixels 限制分辨率（方案 A）
            # 当前临时跳过 token 数超限的视频，避免 OOM
            elif ctx_ids.shape[1] > 20000:
                logger.warning(f"[skip] video token count {ctx_ids.shape[1]} exceeds limit 20000, falling back to empty context")
                ctx_ids = self.model.encode("(Video too large, skipped)")
                vlm_inputs = None
        else:
            ctx_ids = self.model.encode(data['context'])

        kv = self.model.prefill(ctx_ids, load_score=load_score, vlm_inputs=vlm_inputs)

        print(f"# prefill {self.model.name} {self.name}-{idx}:", end=" ")
        logger.info(f"prefill {self.model.name} {self.name}-{idx}: "
                    f"{ctx_ids.shape[1] if hasattr(ctx_ids, 'shape') else len(ctx_ids[0])} tokens, "
                    f"KV cache {kv._mem()} GB, {kv.key_cache[0].dtype}")
        return kv

    def _prepare_query(self, data, kv, inputs: dict, task: str):
        """ Generate answers of each task for evaluation.
            For each task, we store (query, answer, grount_truth) in inputs
        """
        if task in ["qa", "reason"]:
            logger.info("Generated output | Ground truth")
            for i, (q, gt) in enumerate(zip(data['question'], data['answers'])):
                q = get_query(task, q)
                q_ids = self.model.apply_template(q)

                a = self.model.generate(q_ids, kv=kv)

                a_ids = self.model.encode(a)
                gt_ids = self.model.encode(gt)

                tag = f"qa-{i}" if i > 0 else "qa"
                inputs[tag] = {"q": q_ids, "a": a_ids, "gt": gt_ids}
                inputs["eval_task"].append(tag)

                logger.info(f"[QA {i}] Q: {q.strip()}")
                logger.info(f"[QA {i}] pred: {a} | gt: {gt}")

        else:
            q = get_query(task)
            q_ids = self.model.apply_template(q)

            if task == "repeat":
                a_ids = kv.ctx_ids
            else:
                a = self.model.generate(q_ids, kv=kv)
                a_ids = self.model.encode(a)

            gt_ids = a_ids  # no ground truth
            inputs[task] = {"q": q_ids, "a": a_ids, "gt": gt_ids}
            if "scbench" not in self.name and a_ids.shape[-1] < 512:
                inputs["eval_task"].append(task)

    @torch.inference_mode()
    def generate_answer(self, idx: int, kv: Union[RetainCache, EvictCache]):
        """ Prepare inputs, answers, and prediction probabilities (with full KV cache) for evaluation.
        """
        data = self.dataset[idx]

        eval_task = ["qa"]
        if "gsm" in self.name:
            eval_task = ["reason"]
        # # Add new eval tasks if needed
        # if "squad" in self.name:
        #     eval_task += ["summarize", "repeat"]

        inputs = defaultdict(list)
        for task in eval_task:
            self._prepare_query(data, kv, inputs, task)

        info = defaultdict(dict)
        for fmt in inputs["eval_task"]:
            input_ids = torch.cat([inputs[fmt][k] for k in ["q", "a"]], dim=1)
            info[fmt]["prob"] = self.model._prob(input_ids, kv, device="cpu")

        return inputs, info
