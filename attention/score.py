# ------------------------------------------------------------------------------
# Original Code developed by Jang-Hyun Kim
# Licensed under The MIT License
# GitHub Repository: https://github.com/snu-mllab/KVzip
# ------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional


class KVScore():
    """ Functions to compute the score for the KV features. (kvcache.py)"""

    def __init__(self):
        self.n_heads_kv = None
        self.dtype = None
        self.device = None
        self.get_score = True
        self.causal_mask_score = None
        self.score = None
        self.sink = None
        self.start_idx, self.end_idx = None, None

        # Multimodal token support for VLM
        # Stores ranges of multimodal tokens (e.g., image tokens) that should not be scored/pruned
        # Format: List[Tuple[int, int]] where each tuple is (start_pos, end_pos) relative to context
        self.multimodal_ranges: List[Tuple[int, int]] = []
        # Flag to enable/disable multimodal scoring
        self.enable_multimodal_scoring: bool = True

    def set_multimodal_ranges(self, ranges: List[Tuple[int, int]]):
        """Set the ranges of multimodal tokens in the context.

        Args:
            ranges: List of (start_pos, end_pos) tuples indicating multimodal token positions.
                    Positions are relative to the context (after system prompt).

        Example:
            # Image tokens at positions 100-500 in context
            kv.set_multimodal_ranges([(100, 500)])
        """
        self.multimodal_ranges = ranges

    def add_multimodal_range(self, start: int, end: int):
        """Add a single multimodal token range.

        Args:
            start: Start position of multimodal tokens (relative to context)
            end: End position of multimodal tokens (relative to context)
        """
        self.multimodal_ranges.append((start, end))

    def clear_multimodal_ranges(self):
        """Clear all multimodal token ranges."""
        self.multimodal_ranges = []

    def _is_multimodal_position(self, pos: int) -> bool:
        """Check if a position falls within any multimodal token range.

        Args:
            pos: Position to check (relative to context start, i.e., after sink)

        Returns:
            True if position is within a multimodal range, False otherwise
        """
        for start, end in self.multimodal_ranges:
            if start <= pos < end:
                return True
        return False

    def _get_multimodal_mask(self, ctx_len: int) -> torch.Tensor:
        """Generate a mask indicating multimodal positions.

        Args:
            ctx_len: Length of the context

        Returns:
            Boolean tensor of shape [ctx_len], True for multimodal positions
        """
        mask = torch.zeros(ctx_len, dtype=torch.bool, device=self.device)
        for start, end in self.multimodal_ranges:
            if end <= ctx_len:
                mask[start:end] = True
        return mask

    def init_score(self):
        self.get_score = True
        self.causal_mask_score = None
        self.score = [
            torch.zeros((1, self.n_heads_kv, 0), dtype=self.dtype, device=self.device)
            for _ in range(self.n_layers)
        ]

    def _update_score(self, layer_idx: int, score: torch.Tensor, multimodal_mask: Optional[torch.Tensor] = None):
        """Update score for a layer, handling multimodal tokens.

        Args:
            layer_idx: Layer index to update
            score: Score tensor to append
            multimodal_mask: Optional mask indicating multimodal positions to skip scoring
        """
        if multimodal_mask is not None and not self.enable_multimodal_scoring:
            # Set multimodal positions to max score to prevent pruning
            # This ensures multimodal tokens are always retained
            score = score.clone()
            score[:, :, multimodal_mask] = float('inf')

        self.score[layer_idx] = torch.cat([self.score[layer_idx], score], dim=-1)

    def _get_score(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_idx: int):
        """ Compute KV importance scores.
            # key_states: bsz x head_kv x k x dim, query_states: bsz x head x q x dim

            For VLM compatibility, multimodal tokens (e.g., image tokens) are handled specially:
            - By default, multimodal tokens are NOT scored and assigned infinite importance
            - This prevents them from being pruned during KV cache compression
            - Future: enable_multimodal_scoring=True will allow scoring multimodal tokens
        """

        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv = key_states.size(1)

        query_states = query_states.view(bsz, num_kv, -1, q_len, head_dim)
        key_states = torch.cat(
            [
                key_states[:, :, :self.sink],  # sink tokens (generally system prompt)
                key_states[:, :, self.start_idx:self.end_idx],  # KV chunk in the cache
                key_states[:, :, -q_len:],  # KV repeat chunk
            ],
            dim=2)

        # bsz, head, 1, dim, k
        key_states = key_states.unsqueeze(2).transpose(-2, -1).contiguous()
        ctx_len = self.end_idx - self.start_idx

        attn_weights = torch.matmul(query_states, key_states) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)

        # bsz, head, group, q, ctx_len
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # not fp32
        attn_weights = attn_weights[..., self.sink:self.sink + ctx_len]
        score = attn_weights.amax(dim=(-3, -2))  # max over group, q

        # Handle multimodal tokens: generate mask for current scoring range
        multimodal_mask = None
        if self.multimodal_ranges and not self.enable_multimodal_scoring:
            # Check if any multimodal range falls within current scoring window
            relative_start = self.start_idx - self.sink  # position relative to context start
            relative_end = self.end_idx - self.sink

            # Generate mask for positions within current window that are multimodal
            local_mask = torch.zeros(ctx_len, dtype=torch.bool, device=self.device)
            for mm_start, mm_end in self.multimodal_ranges:
                # Calculate overlap with current scoring window
                overlap_start = max(mm_start, relative_start)
                overlap_end = min(mm_end, relative_end)
                if overlap_start < overlap_end:
                    # Convert to local indices within current score tensor
                    local_start = overlap_start - relative_start
                    local_end = overlap_end - relative_start
                    local_mask[local_start:local_end] = True

            if local_mask.any():
                multimodal_mask = local_mask

        self._update_score(layer_idx, score, multimodal_mask)

    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):
        """ Define causal mask shared across layers
        """
        mask = torch.full((window_size, window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        self.causal_mask_score = mask[None, None, None, :, :]

    def _mask_causal(self, attn_weights: torch.Tensor, window_size: int):
        """ Apply causal maksing
        """
        if self.causal_mask_score is None:
            self._make_mask(attn_weights, window_size)
        elif self.causal_mask_score.size(-1) != window_size:
            self._make_mask(attn_weights, window_size)

        attn_weights[..., -window_size:, -window_size:] += self.causal_mask_score

    ##################################################################################################
    def _threshold(self, score: Union[torch.Tensor, List[torch.Tensor]], ratio: float):
        """ Apply thresholding to KV importance scores

            Multimodal tokens (with infinite score) are always retained regardless of ratio.
        """
        if type(score) == list:
            score = torch.stack(score, dim=0)

        # Separate multimodal tokens (inf score) from regular tokens
        inf_mask = score == float('inf')
        regular_mask = ~inf_mask

        if ratio < 1:
            # Only consider regular tokens for threshold calculation
            regular_scores = score[regular_mask]
            if regular_scores.numel() > 0:
                score_sort = torch.sort(regular_scores.reshape(-1), descending=True).values
                # Adjust ratio to account for always-retained multimodal tokens
                n_regular = regular_scores.numel()
                n_multimodal = inf_mask.sum().item()
                n_total = score.numel()
                # Calculate how many regular tokens to keep to achieve target ratio
                n_keep = max(int(n_total * ratio) - n_multimodal, 0)
                if n_keep > 0 and n_keep < n_regular:
                    thres = score_sort[n_keep].item()
                else:
                    thres = 0.0 if n_keep >= n_regular else float('inf')
            else:
                thres = float('inf')  # All tokens are multimodal

            # Apply threshold: keep multimodal tokens and tokens above threshold
            valids = (score > thres) | inf_mask
            valids = valids.bool()
        else:
            valids = torch.ones_like(score, dtype=bool)
            thres = 0.

        return valids, thres

    def _threshold_uniform(self, scores: Union[torch.Tensor, List[torch.Tensor]], ratio: float):
        """ Apply thresholding to KV importance scores with uniform head budgets

            Multimodal tokens (with infinite score) are always retained per head.
        """
        valids = []
        for nl, score in enumerate(scores):
            # Identify multimodal tokens
            inf_mask = score == float('inf')
            n_multimodal = inf_mask.sum(dim=-1)  # Count per head

            if ratio < 1:
                n_seq = score.size(-1)
                # Calculate how many regular tokens to keep per head
                k_per_head = max(int(n_seq * ratio) - n_multimodal.min().item(), 0)

                # For each head, select top-k regular tokens plus all multimodal
                valid = inf_mask.clone()  # Start with multimodal tokens marked as valid
                for h in range(score.size(1)):  # Iterate over heads
                    head_score = score[0, h]  # Score for this head
                    head_inf_mask = inf_mask[0, h]
                    regular_scores = head_score[~head_inf_mask]
                    if k_per_head > 0 and regular_scores.numel() > 0:
                        _, topk_indices = torch.topk(regular_scores, min(k_per_head, len(regular_scores)))
                        # Get absolute indices for regular tokens
                        regular_indices = (~head_inf_mask).nonzero().squeeze()
                        selected_indices = regular_indices[topk_indices]
                        valid[0, h, selected_indices] = True
            else:
                valid = torch.ones_like(score, dtype=bool)

            valids.append(valid)

        valids = torch.stack(valids)
        return valids, 0


class HybridKVScore(KVScore):
    """Extended KVScore for Hybrid cache models (e.g., Gemma3).

    Inherits multimodal token support from KVScore.
    """

    def init_score(self):
        self.get_score = True
        self.causal_mask_score = None

        self.score = [
            torch.zeros((1, self.n_heads_kv, 0), dtype=self.dtype, device=self.device)
            for _ in range(self.num_static_layers)
        ]


    def _get_score(self, query_states, key_states, layer_idx):
        if layer_idx in self.layer_id_to_static_id:
            static_layer_idx = self.layer_id_to_static_id[layer_idx]
            super()._get_score(query_states, key_states, static_layer_idx)

