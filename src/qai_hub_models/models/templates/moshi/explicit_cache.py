# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Pure inference graphs for Moshi's Temporal and DepFormer transformers.

The upstream implementation stores KV tensors in Python streaming-state objects.
The modules below expose that state as explicit tensors so each graph is a pure
function: one Temporal layer with a fixed-capacity KV ring, the head, and one
DepFormer codebook step with its per-frame depth cache.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


def _weight_index(module: nn.Module, step: int) -> int:
    """Resolve an upstream weights-per-step schedule at export time."""
    schedule = module.weights_per_step_schedule
    return schedule[step] if schedule is not None else step


def _cache_capacity(attention: nn.Module) -> int:
    """Match upstream StreamingMultiheadAttention cache allocation."""
    if attention.context is not None:
        return attention.context
    if attention.weights_per_step:
        return attention.weights_per_step
    raise ValueError("Attention has neither a context nor weights-per-step capacity")


class TemporalBlock(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        attention = layer.self_attn
        if (
            layer.skip_self_attn
            or layer.cross_attention is not None
            or layer.weights_per_step
        ):
            raise ValueError(
                "Only unconditional Temporal self-attention blocks are supported"
            )
        if not attention.causal or attention.context is None:
            raise ValueError("A finite causal attention context is required")
        if len(attention.in_projs) != 1 or len(attention.out_projs) != 1:
            raise ValueError("Per-step attention projections are unsupported")
        self.layer = layer
        self.capacity = attention.context
        self.heads = attention.num_heads
        self.kv_heads = attention.num_heads // attention.kv_repeat
        self.head_dim = attention.embed_dim // attention.num_heads

    def empty_cache(self) -> torch.Tensor:
        parameter = next(self.parameters())
        return parameter.new_zeros((2, 1, self.kv_heads, self.capacity, self.head_dim))

    def forward(
        self, hidden: torch.Tensor, cache: torch.Tensor, position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position = position.to(torch.long)
        attention = self.layer.self_attn
        projected = attention.in_projs[0](self.layer.norm1(hidden))
        query_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        query, key, value = projected.split((query_width, kv_width, kv_width), dim=-1)
        query = query.reshape(1, 1, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(1, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(1, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        if attention.rope is not None:
            query, key = attention.rope(query, key, position, time_before_heads=False)
        slots = torch.arange(self.capacity, device=position.device)
        write_mask = (slots == position % self.capacity).reshape(1, 1, self.capacity, 1)
        keys = torch.where(write_mask, key, cache[0])
        values = torch.where(write_mask, value, cache[1])
        updated = torch.stack((keys, values))
        delta = slots - (position.reshape(1, 1) % self.capacity)
        key_positions = torch.where(
            delta <= 0, position + delta, position + delta - self.capacity
        )
        valid = (key_positions >= 0) & ((position - key_positions) < self.capacity)
        if attention.kv_repeat > 1:
            keys = keys.repeat_interleave(attention.kv_repeat, dim=1)
            values = values.repeat_interleave(attention.kv_repeat, dim=1)
        update = functional.scaled_dot_product_attention(
            query, keys, values, valid.reshape(1, 1, 1, self.capacity), dropout_p=0.0
        )
        update = attention.out_projs[0](
            update.transpose(1, 2).reshape(1, 1, query_width)
        )
        hidden = hidden.to(update) + self.layer.layer_scale_1(update)
        normalized = self.layer.norm2(hidden)
        if self.layer.gating is None:
            update = self.layer.linear2(
                self.layer.activation(self.layer.linear1(normalized))
            )
        else:
            update = self.layer.gating(normalized)
        return hidden.to(update) + self.layer.layer_scale_2(update), updated


class TemporalFrontend(nn.Module):
    """Inference-only sum of Moshi's text and audio-code embeddings."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.audio_embeddings = lm.emb
        self.text_embedding = lm.text_emb
        self.audio_offset = lm.audio_offset

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.audio_embeddings[0](sequence[:, self.audio_offset])
        for index in range(1, len(self.audio_embeddings)):
            hidden = hidden + self.audio_embeddings[index](
                sequence[:, index + self.audio_offset]
            )
        return hidden + self.text_embedding(sequence[:, 0])


class TemporalHead(nn.Module):
    """Inference-only Temporal normalization and text projection."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.out_norm = lm.out_norm
        self.text_linear = lm.text_linear

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.out_norm is not None:
            hidden = self.out_norm(hidden)
        return hidden, self.text_linear(hidden)[:, None]


class DepFormerStepBlock(nn.Module):
    """One real DepFormer layer specialized to one constant codebook index."""

    def __init__(self, layer: nn.Module, step: int) -> None:
        super().__init__()
        attention = layer.self_attn
        projection_index = _weight_index(attention, step)
        self.norm1 = layer.norm1
        self.in_proj = attention.in_projs[projection_index]
        self.out_proj = attention.out_projs[projection_index]
        self.norm2 = layer.norm2
        self.layer_scale_1 = layer.layer_scale_1
        self.layer_scale_2 = layer.layer_scale_2
        self.activation = layer.activation
        self.linear1 = layer.linear1
        self.linear2 = layer.linear2
        self.gating = (
            None if layer.gating is None else layer.gating[_weight_index(layer, step)]
        )
        self.step = step
        self.context = _cache_capacity(attention)
        self.heads = attention.num_heads
        self.kv_heads = attention.num_heads // attention.kv_repeat
        self.kv_repeat = attention.kv_repeat
        self.head_dim = attention.embed_dim // attention.num_heads

    def empty_cache(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (hidden.shape[0], self.kv_heads, self.context, self.head_dim)
        return hidden.new_zeros(shape), hidden.new_zeros(shape)

    def forward(
        self,
        hidden: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.in_proj(self.norm1(hidden))
        query_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        query, key, value = projected.split((query_width, kv_width, kv_width), dim=-1)
        query = query.reshape(hidden.shape[0], 1, self.heads, self.head_dim).transpose(
            1, 2
        )
        key = key.reshape(hidden.shape[0], 1, self.kv_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.reshape(
            hidden.shape[0], 1, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        slots = torch.arange(self.context, device=hidden.device)
        write_mask = (slots == self.step).reshape(1, 1, self.context, 1)
        keys = torch.where(write_mask, key, key_cache)
        values = torch.where(write_mask, value, value_cache)
        attention_keys, attention_values = keys, values
        if self.kv_repeat > 1:
            attention_keys = attention_keys.repeat_interleave(self.kv_repeat, dim=1)
            attention_values = attention_values.repeat_interleave(self.kv_repeat, dim=1)
        # Slots after this constant step are always masked; slicing them away
        # avoids a -inf mask, which ruins fixed-point activation ranges.
        visible = self.step + 1
        update = functional.scaled_dot_product_attention(
            query,
            attention_keys[:, :, :visible],
            attention_values[:, :, :visible],
            dropout_p=0.0,
        )
        update = self.out_proj(
            update.transpose(1, 2).reshape(hidden.shape[0], 1, query_width)
        )
        hidden = hidden.to(update) + self.layer_scale_1(update)
        normalized = self.norm2(hidden)
        if self.gating is None:
            update = self.linear2(self.activation(self.linear1(normalized)))
        else:
            update = self.gating(normalized)
        return hidden.to(update) + self.layer_scale_2(update), keys, values


class ExplicitDepFormerStep(nn.Module):
    """One codebook step with explicit per-layer DepFormer K/V tensors."""

    def __init__(self, lm: nn.Module, step: int) -> None:
        super().__init__()
        if not 0 <= step < lm.dep_q:
            raise ValueError(f"DepFormer step must be within 0:{lm.dep_q}")
        linear_index = step
        if lm.depformer_weights_per_step_schedule is not None:
            linear_index = lm.depformer_weights_per_step_schedule[step]
        if not lm.depformer_multi_linear:
            linear_index = 0
        self.input_linear = lm.depformer_in[linear_index]
        self.token_embedding = (
            lm.depformer_text_emb if step == 0 else lm.depformer_emb[step - 1]
        )
        self.blocks = nn.ModuleList(
            DepFormerStepBlock(layer, step) for layer in lm.depformer.layers
        )
        self.output_norm = lm.depformer_norms[step]
        self.output_linear = lm.linears[step]

    def empty_caches(self, temporal: torch.Tensor) -> list[torch.Tensor]:
        return [value for block in self.blocks for value in block.empty_cache(temporal)]

    def forward(
        self,
        previous_token: torch.Tensor,
        temporal: torch.Tensor,
        *caches: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if len(caches) != 2 * len(self.blocks):
            raise ValueError("Expected one key and one value cache per DepFormer layer")
        hidden = self.input_linear(temporal) + self.token_embedding(
            previous_token[:, None]
        )
        updated = []
        for index, block in enumerate(self.blocks):
            hidden, keys, values = block(
                hidden, caches[2 * index], caches[2 * index + 1]
            )
            updated.extend((keys, values))
        logits = self.output_linear(self.output_norm(hidden))[:, None]
        token = logits.argmax(dim=-1)[:, 0, 0]
        return token, logits, *updated


class ExplicitDepFormer(nn.Module):
    """Greedy, single-audio-frame DepFormer: the ``dep_q`` codebook steps chained.

    The depth cache is local to one Temporal step (upstream ``LMGen.depformer_step``),
    so it starts empty every frame.
    """

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        if (
            lm.depformer is None
            or lm.depformer_text_emb is None
            or lm.depformer_emb is None
        ):
            raise ValueError("The checkpoint has no DepFormer")
        for layer in lm.depformer.layers:
            attention = layer.self_attn
            if (
                layer.skip_self_attn
                or layer.cross_attention is not None
                or not attention.causal
            ):
                raise ValueError("Only causal DepFormer self-attention is supported")
            if attention.rope is not None:
                raise ValueError(
                    "This export expects the checkpoint's position-free DepFormer"
                )
            if _cache_capacity(attention) < lm.dep_q:
                raise ValueError("DepFormer context is shorter than dep_q")
            if (
                attention.weights_per_step != lm.dep_q
                or layer.weights_per_step != lm.dep_q
            ):
                raise ValueError("Expected one DepFormer weight set per codebook")
        self.steps = nn.ModuleList(
            ExplicitDepFormerStep(lm, step) for step in range(lm.dep_q)
        )

    def forward(
        self, text_token: torch.Tensor, temporal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``text_token`` [B], ``temporal`` [B, 1, D] -> tokens [B, dep_q, 1], logits [B, dep_q, 1, card]."""
        caches = self.steps[0].empty_caches(temporal)
        previous, tokens, logits = text_token, [], []
        for step in self.steps:
            previous, step_logits, *caches = step(previous, temporal, *caches)
            tokens.append(previous[:, None, None])
            logits.append(step_logits)
        return torch.cat(tokens, dim=1), torch.cat(logits, dim=1)
