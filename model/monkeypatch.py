import transformers
from loguru import logger
from attention.attn import llama_qwen_attn_forward, gemma3_attn_forward, qwen_vl_attn_forward


def patch_qwen2_5_vl_prepare_inputs_for_generation():
    """Patch Qwen2.5-VL's prepare_inputs_for_generation to fix position_ids issue.

    The issue: When using KV cache with long context, Qwen2_5_VLForConditionalGeneration.forward
    calls get_rope_index with the full input_ids instead of just the new tokens, causing
    position_embeddings shape mismatch.

    Additionally, attention_mask is not sliced when using KV cache, causing get_rope_index
    to compute position_ids based on the full sequence length.

    The fix:
    1. Slice attention_mask to match cache_position when past_key_values exists
    2. Compute position_ids from cache_position when past_key_values exists
    """
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

    original_prepare = modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation

    def patched_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Call original prepare
        model_inputs = original_prepare(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # If we have past_key_values with content, fix the inputs
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            # Fix 1: Slice attention_mask to match the new tokens only
            if model_inputs.get('attention_mask') is not None and cache_position is not None:
                attn_mask = model_inputs['attention_mask']
                # attention_mask shape is [batch, full_seq_len], slice to [batch, new_tokens_len]
                if attn_mask.shape[-1] != len(cache_position):
                    model_inputs['attention_mask'] = attn_mask[:, cache_position]

            # Fix 2: Compute position_ids from cache_position
            if cache_position is not None:
                # Get the input length from either inputs_embeds or input_ids
                if model_inputs.get('inputs_embeds') is not None:
                    batch_size = model_inputs['inputs_embeds'].shape[0]
                elif model_inputs.get('input_ids') is not None:
                    batch_size = model_inputs['input_ids'].shape[0]
                else:
                    batch_size = 1

                # Compute position_ids from cache_position for the new tokens only
                # Shape: [3, batch_size, new_tokens_length]
                new_position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
                model_inputs['position_ids'] = new_position_ids

        return model_inputs

    modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation = patched_prepare_inputs_for_generation


def replace_attn(model_id):
    model_id = model_id.lower()
    if "llama" in model_id:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_qwen_attn_forward
        print("Replace llama attention with KVzip")
        logger.info("Replace llama attention with KVzip")

    elif "qwen2.5-vl" in model_id or "qwen2-vl" in model_id or "qwen3-vl" in model_id:
        # Qwen-VL vision-language models
        # Need to patch all attention variants (FlashAttention2, Sdpa, eager)
        patched = False

        # Try Qwen2.5-VL first
        try:
            from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
            # Patch all attention classes to ensure the correct one is used
            modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_vl_attn_forward
            modeling_qwen2_5_vl.Qwen2_5_VLSdpaAttention.forward = qwen_vl_attn_forward
            modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen_vl_attn_forward
            # Patch prepare_inputs_for_generation to fix position_ids issue with KV cache
            patch_qwen2_5_vl_prepare_inputs_for_generation()
            print("Replace Qwen2.5-VL attention with KVzip (multimodal-aware)")
            logger.info("Replace Qwen2.5-VL attention with KVzip (multimodal-aware)")
            patched = True
        except ImportError:
            pass

        # Try Qwen2-VL as fallback
        if not patched:
            try:
                from transformers.models.qwen2_vl import modeling_qwen2_vl
                # Patch all attention classes
                modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_vl_attn_forward
                modeling_qwen2_vl.Qwen2VLSdpaAttention.forward = qwen_vl_attn_forward
                modeling_qwen2_vl.Qwen2VLAttention.forward = qwen_vl_attn_forward
                print("Replace Qwen2-VL attention with KVzip (multimodal-aware)")
                logger.info("Replace Qwen2-VL attention with KVzip (multimodal-aware)")
                patched = True
            except ImportError:
                print("Warning: Qwen-VL model classes not found, using standard attention")
                logger.warning("Qwen-VL model classes not found, using standard attention")

        if not patched:
            print("Warning: Qwen-VL model classes not found, using standard attention")
            logger.warning("Qwen-VL model classes not found, using standard attention")

    elif "qwen2.5" in model_id:
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = llama_qwen_attn_forward
        print("Replace qwen2.5 attention with KVzip")
        logger.info("Replace qwen2.5 attention with KVzip")

    elif "qwen3" in model_id:
        transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = llama_qwen_attn_forward
        print("Replace qwen3 attention with KVzip")
        logger.info("Replace qwen3 attention with KVzip")

    elif "gemma-3" in model_id:
        transformers.models.gemma3.modeling_gemma3.Gemma3Attention.forward = gemma3_attn_forward
        print("Replace gemma3 with KVzip attention")
        logger.info("Replace gemma3 with KVzip attention")
