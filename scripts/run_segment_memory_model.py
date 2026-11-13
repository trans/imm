from __future__ import annotations

import argparse
import collections
import csv
import gc
import hashlib
import math
import json
import resource
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imm.data import CharVocab, read_text
from imm.row_prefetch import RowPrefetcher, rows_to_device
from imm.eval import split_text
from imm.count_gate import (
    CountModel,
    MarisaCountModel,
    PackedCountModel,
    expand_extra_features,
    select_positions,
    train_gate,
)
from imm.run_ids import prefixed_path, resolve_run_id
from imm.segments import SEGMENT_RULES, rule_segments
from imm.typo_noise import substitute, unigram_probs
from imm.skip_horizon_features import (
    FEATURE_NAMES as SKIP_HORIZON_FEATURE_NAMES,
    blend_log_prob_rows,
    compile_skip_horizon_channel_rows,
    compile_skip_horizon_distribution_rows,
    compile_skip_horizon_features,
    default_skip_horizon_cache_dir,
    ensure_skip_horizon_artifact,
    ensure_skip_horizon_count_artifact,
)
from imm.compact_evidence import CompactEvidenceRows, load_compact
from imm.vectorized_count_prior import (
    compile_cross_fit_depth_evidence_mmap,
    compile_depth_evidence_mmap,
    build_depth_prior_tables,
    compile_cross_fit_rows_numpy_mmap,
    compile_prior_rows_numpy_mmap,
)


MemoryEntry = tuple[torch.Tensor, int]
StreamMemory = tuple[torch.Tensor, torch.Tensor]  # records [B, M, H], their positions [B, M]
PriorRows = torch.Tensor | np.memmap | CompactEvidenceRows


@dataclass
class EpochProfile:
    zero_sec: float = 0.0
    rows_sec: float = 0.0  # gathering prior rows (from memmaps) and copying them to the device
    rows_wait_sec: float = 0.0  # of rows_sec: waiting for the background reader (--prefetch-rows)
    forward_sec: float = 0.0
    wait_sec: float = 0.0  # loss.item(): waiting for the device to finish the forward pass
    backward_sec: float = 0.0
    clip_sec: float = 0.0
    optimizer_sec: float = 0.0
    detach_sec: float = 0.0
    eval_sec: float = 0.0
    checkpoint_sec: float = 0.0
    updates: int = 0
    tokens: int = 0

    def add(self, other: "EpochProfile") -> None:
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name) + getattr(other, field.name))

    @property
    def train_sec(self) -> float:
        return (
            self.zero_sec + self.rows_sec + self.forward_sec + self.wait_sec + self.backward_sec + self.clip_sec
            + self.optimizer_sec + self.detach_sec
        )


def profile_summary(profile: EpochProfile) -> str:
    train_sec = max(profile.train_sec, 1e-9)
    return (
        f"profile updates={profile.updates} tokens={profile.tokens} train_sec={train_sec:.2f} "
        f"tok_per_sec={profile.tokens / train_sec if profile.tokens else 0.0:.1f} "
        f"zero={profile.zero_sec:.2f}s/{100.0 * profile.zero_sec / train_sec:.1f}% "
        f"rows={profile.rows_sec:.2f}s/{100.0 * profile.rows_sec / train_sec:.1f}% "
        f"rows_wait={profile.rows_wait_sec:.2f}s/{100.0 * profile.rows_wait_sec / train_sec:.1f}% "
        f"forward={profile.forward_sec:.2f}s/{100.0 * profile.forward_sec / train_sec:.1f}% "
        f"wait={profile.wait_sec:.2f}s/{100.0 * profile.wait_sec / train_sec:.1f}% "
        f"backward={profile.backward_sec:.2f}s/{100.0 * profile.backward_sec / train_sec:.1f}% "
        f"clip={profile.clip_sec:.2f}s/{100.0 * profile.clip_sec / train_sec:.1f}% "
        f"optim={profile.optimizer_sec:.2f}s/{100.0 * profile.optimizer_sec / train_sec:.1f}% "
        f"detach={profile.detach_sec:.2f}s/{100.0 * profile.detach_sec / train_sec:.1f}% "
        f"eval={profile.eval_sec:.2f}s checkpoint={profile.checkpoint_sec:.2f}s"
    )


def empty_stats() -> dict[str, float]:
    return {
        "gate_sum": 0.0,
        "gate_count": 0.0,
        "context_norm_sum": 0.0,
        "delta_norm_sum": 0.0,
        "token_count": 0.0,
        "no_memory_loss": 0.0,
        "context_only_loss": 0.0,
        "diagnostic_token_count": 0.0,
        "attn_entropy_sum": 0.0,
        "attn_top_weight_sum": 0.0,
        "attn_char_distance_sum": 0.0,
        "attn_token_count": 0.0,
        "attn_prev_weight_sum": 0.0,
        "attn_current_weight_sum": 0.0,
        "attn_top_is_prev_sum": 0.0,
        "first_token_loss": 0.0,
        "first_token_count": 0.0,
        "later_token_loss": 0.0,
        "later_token_count": 0.0,
        "retrieval_loss": 0.0,
        "retrieval_correct": 0.0,
        "retrieval_count": 0.0,
        "utility_loss": 0.0,
        "utility_count": 0.0,
        "utility_target_entropy_sum": 0.0,
        "residual_alpha_sum": 0.0,
        "residual_alpha_count": 0.0,
        "skip_prior_alpha_sum": 0.0,
        "skip_prior_alpha_count": 0.0,
    }


def merge_stats(total: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        total[key] += value


def mean_or_none(total: float, count: float) -> float | None:
    if count <= 0:
        return None
    return total / count


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        state = self.norm_attn(state + context)
        return self.norm_mlp(state + self.mlp(state))


class AttentionDecisionBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.state_to_context = nn.Linear(hidden_size, hidden_size)
        self.norm_state = nn.LayerNorm(hidden_size)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.norm_update = nn.LayerNorm(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        state_signal = self.state_to_context(self.norm_state(state))
        context_norm = self.norm_context(context)
        gate = torch.sigmoid(self.gate(torch.cat([state_signal, context_norm], dim=1)))
        decision = self.norm_update(context + gate * state_signal)
        return self.norm_mlp(decision + self.mlp(decision))


class LMATokenBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm_attn(tokens + attn_out)
        return self.norm_mlp(tokens + self.mlp(tokens))


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self, hidden_size: int, heads: int, memory_size: int | None = None, mlp_mult: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        memory_size = memory_size or hidden_size
        # As the transformer baseline places it: on the attention weights, the
        # attention output, inside the MLP and on its output. Applied in
        # forward, so the parameter names (and old checkpoints) don't change.
        self.dropout = dropout
        if hidden_size % heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by heads {heads}")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.rope_dim = self.head_dim - self.head_dim % 2
        self.ln_q = nn.LayerNorm(hidden_size)
        self.ln_kv = nn.LayerNorm(memory_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(memory_size, hidden_size)
        self.v_proj = nn.Linear(memory_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / max(1, self.rope_dim))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.ln_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_mult * hidden_size),
            nn.GELU(),
            nn.Linear(mlp_mult * hidden_size, hidden_size),
        )
        self.gate_mlp = nn.Parameter(torch.zeros(1))

    def drop(self, x: torch.Tensor) -> torch.Tensor:
        # Skipped at 0, so a run without dropout draws no random numbers for it.
        return F.dropout(x, self.dropout, self.training) if self.dropout > 0.0 else x

    def apply_rope(self, x: torch.Tensor, positions: torch.Tensor, use_rope: bool) -> torch.Tensor:
        if not use_rope or self.rope_dim == 0:
            return x
        rope_part = x[..., : self.rope_dim]
        pass_part = x[..., self.rope_dim :]
        # positions: [T], shared by the batch, or [B, T], one row per stream
        shape = (1, 1, -1, 1) if positions.dim() == 1 else (positions.shape[0], 1, positions.shape[1], 1)
        angles = positions.to(device=x.device, dtype=x.dtype).view(*shape) * self.rope_inv_freq.to(
            dtype=x.dtype
        ).view(1, 1, 1, -1)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        even = rope_part[..., 0::2]
        odd = rope_part[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        if pass_part.numel() == 0:
            return rotated
        return torch.cat([rotated, pass_part], dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        query_positions: torch.Tensor | None = None,
        memory_positions: torch.Tensor | None = None,
        use_rope: bool = True,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        stats = empty_stats()
        if memory.shape[1] > 0:
            batch_size, query_count, _ = hidden.shape
            memory_count = memory.shape[1]
            query_norm = self.ln_q(hidden)
            memory_norm = self.ln_kv(memory)
            query = self.q_proj(query_norm).view(batch_size, query_count, self.heads, self.head_dim).transpose(1, 2)
            key = self.k_proj(memory_norm).view(batch_size, memory_count, self.heads, self.head_dim).transpose(1, 2)
            value = self.v_proj(memory_norm).view(batch_size, memory_count, self.heads, self.head_dim).transpose(1, 2)
            if query_positions is None:
                query_positions = torch.zeros(query_count, device=hidden.device)
            if memory_positions is None:
                memory_positions = torch.arange(-memory_count, 0, device=hidden.device)
            query = self.apply_rope(query, query_positions, use_rope)
            key = self.apply_rope(key, memory_positions, use_rope)
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
            weights = torch.softmax(scores, dim=-1)
            if return_diagnostics:
                mean_weights = weights.mean(dim=1).squeeze(0)
                eps = torch.finfo(mean_weights.dtype).eps
                entropy = -(mean_weights * mean_weights.clamp_min(eps).log()).sum(dim=1)
                top_weight = mean_weights.max(dim=1).values
                top_indices = mean_weights.argmax(dim=1)
                distances = (query_positions[:, None] - memory_positions[None, :]).abs().to(dtype=mean_weights.dtype)
                mean_distance = (mean_weights * distances).sum(dim=1)
                stats["attn_entropy_sum"] = float(entropy.detach().sum().item())
                stats["attn_top_weight_sum"] = float(top_weight.detach().sum().item())
                stats["attn_char_distance_sum"] = float(mean_distance.detach().sum().item())
                stats["attn_token_count"] = float(query_count)
                stats["attn_prev_weight_sum"] = float(mean_weights.detach().sum().item())
                stats["attn_top_is_prev_sum"] = float(torch.ones_like(top_indices, dtype=mean_weights.dtype).sum().item())
            attn_out = self.drop(weights) @ value
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, query_count, self.hidden_size)
            attn_out = self.drop(self.out_proj(attn_out))
        else:
            attn_out = torch.zeros_like(hidden)
        hidden = hidden + torch.tanh(self.gate_attn) * attn_out
        up, act, down = self.mlp
        hidden = hidden + torch.tanh(self.gate_mlp) * self.drop(down(self.drop(act(up(self.ln_mlp(hidden))))))
        if return_diagnostics:
            return hidden, stats
        return hidden


class SegmentMemoryLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        use_rope: bool,
        rope_positions: str,
        memory_state: str,
        memory_record: str,
        rnn_core: str,
        attn_interface_norm: str,
        mixing: str,
        attention_layers: int,
        attention_heads: int,
        input_rope: bool,
        feedback_gate_bias: float,
        feedback_delta_cap: float,
        prior_strength: float,
        prior_residual_scale: float,
        prior_residual_l2: float,
        prior_residual_gate: str,
        prior_residual_gate_max: float,
        prior_residual_gate_features: int,
        prior_residual_state_features: list[int],
        skip_prior_gate: str,
        skip_prior_gate_max: float,
        skip_prior_gate_bias: float,
        skip_prior_channels: int,
        record_aux_weight: float,
        terminal_record_aux_weight: float,
        retrieval_aux_weight: float,
        utility_aux_weight: float,
        utility_temperature: float,
        prior_input: str = "none",
        prior_evidence_width: int = 0,
        dropout: float = 0.0,
        carry_bound: str = "none",
        carry_scale: float = 1.0,
        prior_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= prior_dropout < 1.0:
            raise ValueError(f"prior_dropout must be in [0, 1), got {prior_dropout}")
        # Prior dropout, in training on the batched path only: each stream
        # loses the prior with this probability, drawn anew for every call
        # (one update's segments), from its input and its output. A stream
        # without it predicts from its own head alone (gru_head of the
        # final state), with no gate or state features either, since those
        # are the prior's too. Held-out scoring always has the prior. @air's
        # --prior-dropout for the transformer baseline does the same per
        # sequence (air 47b1711).
        self.prior_dropout = prior_dropout
        if carry_bound not in ("none", "tanh", "norm", "gru"):
            raise ValueError(f"unknown carry_bound: {carry_bound}")
        if not carry_scale > 0.0:
            raise ValueError(f"carry_scale must be positive, got {carry_scale}")
        # What a stream carries into its next segment's GRU. "none": the state
        # after the attention block, as before; the block adds to it, so a
        # stream whose GRU keeps passing it on grows it without bound.
        # "tanh": that state squashed, c * tanh(x / c) with c = carry_scale
        # (1: the GRU's own range). "norm": its length capped at
        # carry_scale, direction kept. "gru": the GRU's own last state; the
        # block reaches the next segment only through the memory records.
        self.carry_bound = carry_bound
        self.carry_scale = carry_scale
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if dropout > 0.0 and mixing != "gated-xattn":
            raise ValueError("dropout is built for --mixing gated-xattn only")
        # Zaremba et al.'s placement for recurrent LMs, on the connections that
        # aren't recurrent: the GRU's input (embedding plus prior inputs) and
        # the state the head reads; inside the attention blocks as the
        # transformer baseline has it. Not on the carried state or the memory
        # records. The prior's own log-probs are never dropped.
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.use_rope = use_rope
        self.rope_positions = rope_positions
        self.memory_state = memory_state
        self.memory_record = memory_record
        self.rnn_core = rnn_core
        self.attn_interface_norm = attn_interface_norm
        self.mixing = mixing
        self.attention_heads = attention_heads
        self.input_rope = input_rope
        self.feedback_delta_cap = feedback_delta_cap
        self.prior_strength = prior_strength
        self.prior_residual_scale = prior_residual_scale
        self.prior_residual_l2 = prior_residual_l2
        self.prior_residual_gate = prior_residual_gate
        self.prior_residual_gate_max = prior_residual_gate_max
        self.skip_prior_gate = skip_prior_gate
        self.skip_prior_gate_max = skip_prior_gate_max
        self.skip_prior_channels = skip_prior_channels
        self.record_aux_weight = record_aux_weight
        self.terminal_record_aux_weight = terminal_record_aux_weight
        self.retrieval_aux_weight = retrieval_aux_weight
        self.utility_aux_weight = utility_aux_weight
        self.utility_temperature = utility_temperature
        self.rope_dim = hidden_size - hidden_size % 2
        self.embed_rope_dim = embedding_size - embedding_size % 2
        self.embed = nn.Embedding(vocab_size, embedding_size)
        if rnn_core == "gru":
            self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
            self.cell = nn.GRUCell(embedding_size, hidden_size)
        elif rnn_core == "tanh":
            self.rnn = nn.RNN(embedding_size, hidden_size, nonlinearity="tanh", batch_first=True)
            self.cell = nn.RNNCell(embedding_size, hidden_size, nonlinearity="tanh")
        else:
            raise ValueError(f"unknown rnn_core: {rnn_core}")
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_rnn = nn.Linear(hidden_size, hidden_size, bias=False)
        if attn_interface_norm == "layer":
            self.query_interface_norm = nn.LayerNorm(hidden_size)
            self.memory_interface_norm = nn.LayerNorm(hidden_size)
            self.head_state_interface_norm = nn.LayerNorm(hidden_size)
            self.head_context_interface_norm = nn.LayerNorm(hidden_size)
        elif attn_interface_norm == "context":
            self.query_interface_norm = nn.LayerNorm(hidden_size)
            self.memory_interface_norm = nn.LayerNorm(hidden_size)
            self.head_state_interface_norm = nn.Identity()
            self.head_context_interface_norm = nn.LayerNorm(hidden_size)
        elif attn_interface_norm == "attention":
            self.query_interface_norm = nn.LayerNorm(hidden_size)
            self.memory_interface_norm = nn.LayerNorm(hidden_size)
            self.head_state_interface_norm = nn.Identity()
            self.head_context_interface_norm = nn.Identity()
        elif attn_interface_norm == "none":
            self.query_interface_norm = nn.Identity()
            self.memory_interface_norm = nn.Identity()
            self.head_state_interface_norm = nn.Identity()
            self.head_context_interface_norm = nn.Identity()
        else:
            raise ValueError(f"unknown attn_interface_norm: {attn_interface_norm}")
        self.memory_writer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.cross_blocks = nn.ModuleList([CrossAttentionBlock(hidden_size) for _ in range(attention_layers)])
        self.decision_blocks = nn.ModuleList([AttentionDecisionBlock(hidden_size) for _ in range(attention_layers)])
        self.lma_blocks = nn.ModuleList([LMATokenBlock(hidden_size, attention_heads) for _ in range(attention_layers)])
        self.gated_xattn_blocks = nn.ModuleList(
            [GatedCrossAttentionBlock(hidden_size, attention_heads, dropout=dropout) for _ in range(attention_layers)]
        )
        self.gate_update = nn.Sequential(
            nn.Linear(2 * hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        self.feedback_update = nn.Sequential(
            nn.Linear(2 * hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.feedback_gate = nn.Linear(2 * hidden_size, hidden_size)
        nn.init.zeros_(self.feedback_gate.weight)
        nn.init.constant_(self.feedback_gate.bias, feedback_gate_bias)
        self.cap_fuse = nn.Linear(2 * hidden_size, hidden_size)
        self.cap_norm = nn.LayerNorm(hidden_size)
        self.gru_head = nn.Linear(hidden_size, vocab_size)
        self.late_head = nn.Linear(2 * hidden_size, vocab_size)
        self.attn_head = nn.Linear(hidden_size, vocab_size)
        self.dual_attn_head = nn.Linear(2 * hidden_size, vocab_size)
        self.record_head = nn.Linear(hidden_size, vocab_size)
        self.gated_head = nn.Linear(3 * hidden_size, vocab_size)
        self.state_head = nn.Linear(hidden_size, vocab_size)
        if prior_residual_gate == "context":
            if prior_residual_gate_features <= 0:
                raise ValueError("context residual gate requires feature rows")
            self.residual_trust_gate = nn.Linear(prior_residual_gate_features, 1, bias=False)
            nn.init.zeros_(self.residual_trust_gate.weight)
        elif prior_residual_gate == "none":
            self.residual_trust_gate = None
        else:
            raise ValueError(f"unknown prior_residual_gate: {prior_residual_gate}")
        if skip_prior_gate == "context":
            if prior_residual_gate_features <= 0:
                raise ValueError("context skip prior gate requires feature rows")
            self.skip_prior_trust_gate = nn.Linear(prior_residual_gate_features, 1)
            nn.init.zeros_(self.skip_prior_trust_gate.weight)
            nn.init.constant_(self.skip_prior_trust_gate.bias, skip_prior_gate_bias)
        elif skip_prior_gate == "horizon-context":
            if prior_residual_gate_features <= 0:
                raise ValueError("horizon-context skip prior gate requires feature rows")
            if skip_prior_channels <= 0:
                raise ValueError("horizon-context skip prior gate requires positive channels")
            self.skip_prior_trust_gate = nn.Linear(prior_residual_gate_features, skip_prior_channels + 1)
            nn.init.zeros_(self.skip_prior_trust_gate.weight)
            nn.init.zeros_(self.skip_prior_trust_gate.bias)
            nn.init.constant_(self.skip_prior_trust_gate.bias[1:], skip_prior_gate_bias)
        elif skip_prior_gate == "none":
            self.skip_prior_trust_gate = None
        else:
            raise ValueError(f"unknown skip_prior_gate: {skip_prior_gate}")
        if prior_residual_state_features:
            self.register_buffer(
                "prior_residual_state_feature_indices",
                torch.tensor(prior_residual_state_features, dtype=torch.long),
                persistent=False,
            )
            self.prior_state_residual = nn.Sequential(
                nn.Linear(len(prior_residual_state_features), hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, vocab_size),
            )
            nn.init.zeros_(self.prior_state_residual[-1].weight)
            nn.init.zeros_(self.prior_state_residual[-1].bias)
        else:
            self.register_buffer(
                "prior_residual_state_feature_indices",
                torch.empty(0, dtype=torch.long),
                persistent=False,
            )
            self.prior_state_residual = None
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / max(1, self.rope_dim))
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        input_inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.embed_rope_dim, 2, dtype=torch.float32) / max(1, self.embed_rope_dim))
        )
        self.register_buffer("input_rope_inv_freq", input_inv_freq, persistent=False)
        if prior_input not in ("none", "log-probs", "depth-evidence", "both"):
            raise ValueError(f"unknown prior_input: {prior_input}")
        if prior_input in ("depth-evidence", "both") and prior_evidence_width <= 0:
            raise ValueError("depth evidence needs a positive prior_evidence_width")
        # Built under a forked RNG and zeroed, so every other parameter starts
        # as it would without them and the GRU's first inputs are unchanged.
        with torch.random.fork_rng(devices=[]):
            log_probs_proj = evidence_proj = None
            if prior_input in ("log-probs", "both"):
                log_probs_proj = nn.Sequential(nn.LayerNorm(vocab_size), nn.Linear(vocab_size, embedding_size))
            if prior_input in ("depth-evidence", "both"):
                evidence_proj = nn.Sequential(
                    nn.LayerNorm(prior_evidence_width), nn.Linear(prior_evidence_width, embedding_size)
                )
        for proj in (log_probs_proj, evidence_proj):
            if proj is not None:
                nn.init.zeros_(proj[-1].weight)
                nn.init.zeros_(proj[-1].bias)
        self.prior_input_proj = log_probs_proj
        self.prior_evidence_proj = evidence_proj
        # Scoring switches for the take-the-prior-away test (prior_ablation),
        # read by segment_loss_streams only: "none" scores the model as
        # trained; "output" scores its residual logits alone, with the prior
        # still in the input; "all" also leaves the prior out of the input.
        # state_sink, a list, collects (final state, target) for each scored
        # character. Neither is a parameter or saved.
        self.prior_ablation = "none"
        self.state_sink: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    def zero_prediction_heads(self) -> None:
        for head in [
            self.gru_head,
            self.late_head,
            self.attn_head,
            self.dual_attn_head,
            self.gated_head,
            self.state_head,
        ]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def zero_state(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def drop(self, x: torch.Tensor) -> torch.Tensor:
        # Skipped at 0, so a run without dropout draws no random numbers for it.
        return F.dropout(x, self.dropout, self.training) if self.dropout > 0.0 else x

    def bound_carry(self, next_state: torch.Tensor, gru_last: torch.Tensor) -> torch.Tensor:
        if self.carry_bound == "tanh":
            c = self.carry_scale
            return torch.tanh(next_state) if c == 1.0 else c * torch.tanh(next_state / c)
        if self.carry_bound == "norm":
            norm = next_state.norm(dim=-1, keepdim=True)
            return next_state * (self.carry_scale / norm.clamp_min(self.carry_scale))
        if self.carry_bound == "gru":
            return gru_last
        return next_state

    def apply_prior_residual(
        self,
        logits: torch.Tensor,
        prior_log_probs: torch.Tensor | None,
        prior_gate_features: torch.Tensor | None,
        skip_prior_log_probs: torch.Tensor | None = None,
        skip_prior_strength: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if prior_log_probs is None:
            return logits, None, None, None
        skip_alpha = None
        if skip_prior_log_probs is not None:
            if prior_gate_features is None or self.skip_prior_trust_gate is None:
                raise ValueError("learned skip prior requires prior_gate_features and a skip prior gate")
            if skip_prior_strength is None:
                raise ValueError("learned skip prior requires skip_prior_strength")
            if self.skip_prior_gate == "horizon-context":
                if skip_prior_log_probs.dim() != 3 or skip_prior_strength.dim() != 2:
                    raise ValueError("horizon-context skip prior expects [N,H,V] log probs and [N,H] strengths")
                channel_logits = self.skip_prior_trust_gate(prior_gate_features.to(dtype=logits.dtype))
                strength = torch.clamp(skip_prior_strength.to(dtype=logits.dtype), min=1.0e-8, max=1.0)
                horizon_logits = channel_logits[:, 1:] + torch.log(strength)
                weights = torch.softmax(torch.cat([channel_logits[:, :1], horizon_logits], dim=1), dim=1)
                count_weight = weights[:, :1]
                horizon_weights = weights[:, 1:]
                count_probs = torch.exp(prior_log_probs.to(dtype=torch.float32))
                skip_probs = torch.exp(skip_prior_log_probs.to(dtype=torch.float32))
                prior_probs = count_weight.to(dtype=torch.float32) * count_probs + (
                    horizon_weights.to(dtype=torch.float32).unsqueeze(2) * skip_probs
                ).sum(dim=1)
                skip_alpha = horizon_weights.sum(dim=1, keepdim=True)
            else:
                skip_gate = self.skip_prior_gate_max * torch.sigmoid(
                    self.skip_prior_trust_gate(prior_gate_features.to(dtype=logits.dtype))
                )
                skip_alpha = torch.clamp(skip_gate * skip_prior_strength.to(dtype=logits.dtype), 0.0, 1.0)
                prior_probs = (1.0 - skip_alpha) * torch.exp(prior_log_probs.to(dtype=torch.float32)) + skip_alpha * torch.exp(
                    skip_prior_log_probs.to(dtype=torch.float32)
                )
            prior_log_probs = torch.log(torch.clamp(prior_probs, min=1.0e-8)).to(dtype=prior_log_probs.dtype)
        if self.prior_state_residual is not None:
            if prior_gate_features is None:
                raise ValueError("prior residual state features require prior_gate_features")
            selected_features = prior_gate_features.index_select(
                1,
                self.prior_residual_state_feature_indices.to(prior_gate_features.device),
            )
            logits = logits + self.prior_state_residual(selected_features.to(dtype=logits.dtype))
        if self.prior_residual_gate == "context":
            if prior_gate_features is None or self.residual_trust_gate is None:
                raise ValueError("context residual gate requires prior_gate_features")
            alpha = self.prior_residual_gate_max * torch.sigmoid(self.residual_trust_gate(prior_gate_features))
            residual_logits = alpha * logits
        else:
            alpha = None
            residual_logits = self.prior_residual_scale * logits
        return self.prior_strength * prior_log_probs + residual_logits, residual_logits, alpha, skip_alpha

    def write_memory(self, state: torch.Tensor) -> torch.Tensor:
        if self.memory_record == "raw":
            return state
        if self.memory_record == "written":
            return self.memory_writer(state)
        raise ValueError(f"unknown memory_record: {self.memory_record}")

    def late_logits(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        state_for_head = self.head_state_interface_norm(state)
        context_for_head = self.head_context_interface_norm(context)
        return self.late_head(torch.cat([state_for_head, context_for_head], dim=1))

    def previous_record_loss(
        self,
        query_state: torch.Tensor,
        query_position: int,
        memory: list[MemoryEntry],
        max_memory: int,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        stats = empty_stats()
        window = memory[-max_memory:]
        if not window:
            return None, stats
        mem = self.memory_interface_norm(torch.cat([entry[0] for entry in window], dim=0))
        query = self.query(self.query_interface_norm(query_state))
        keys = self.key(mem)
        if self.rope_positions == "char":
            query_positions = torch.tensor([query_position], device=query_state.device)
            key_positions = torch.tensor([entry[1] for entry in window], device=query_state.device)
        else:
            query_positions = torch.zeros(1, device=query_state.device)
            key_positions = torch.arange(-mem.shape[0], 0, device=query_state.device)
        query = self.apply_rope(query, query_positions)
        keys = self.apply_rope(keys, key_positions)
        scores = query @ keys.T / math.sqrt(self.hidden_size)
        target = torch.tensor([mem.shape[0] - 1], device=query_state.device)
        loss = F.cross_entropy(scores, target, reduction="sum")
        prediction = scores.argmax(dim=1)
        stats["retrieval_loss"] = float(loss.detach().item())
        stats["retrieval_correct"] = float((prediction == target).detach().sum().item())
        stats["retrieval_count"] = 1.0
        return loss, stats

    def utility_attention_loss(
        self,
        weights: torch.Tensor | None,
        values: torch.Tensor | None,
        allowed: torch.Tensor | None,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        stats = empty_stats()
        if weights is None or values is None or allowed is None:
            return None, stats
        if weights.numel() == 0 or values.numel() == 0:
            return None, stats
        candidate_logits = self.attn_head(values)
        candidate_log_probs = F.log_softmax(candidate_logits, dim=1)
        candidate_nll = -candidate_log_probs[:, targets].T
        quality = -candidate_nll
        quality = quality.masked_fill(~allowed, -torch.inf)
        valid = allowed.any(dim=1)
        if not bool(valid.any()):
            return None, stats
        temperature = max(self.utility_temperature, 1e-6)
        target_distribution = torch.softmax(quality[valid] / temperature, dim=1).detach()
        eps = torch.finfo(weights.dtype).eps
        log_weights = weights[valid].clamp_min(eps).log()
        loss_by_token = -(target_distribution * log_weights).sum(dim=1)
        target_entropy = -(target_distribution * target_distribution.clamp_min(eps).log()).sum(dim=1)
        loss = loss_by_token.sum()
        stats["utility_loss"] = float(loss.detach().item())
        stats["utility_count"] = float(loss_by_token.numel())
        stats["utility_target_entropy_sum"] = float(target_entropy.detach().sum().item())
        return loss, stats

    def apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if not self.use_rope or self.rope_dim == 0:
            return x
        rope_part = x[:, : self.rope_dim]
        pass_part = x[:, self.rope_dim :]
        angles = positions.to(device=x.device, dtype=x.dtype).unsqueeze(1) * self.rope_inv_freq.to(
            dtype=x.dtype
        ).unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        even = rope_part[:, 0::2]
        odd = rope_part[:, 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(1)
        if pass_part.numel() == 0:
            return rotated
        return torch.cat([rotated, pass_part], dim=1)

    def apply_input_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if not self.input_rope or self.embed_rope_dim == 0:
            return x
        rope_part = x[:, : self.embed_rope_dim]
        pass_part = x[:, self.embed_rope_dim :]
        angles = positions.to(device=x.device, dtype=x.dtype).unsqueeze(1) * self.input_rope_inv_freq.to(
            dtype=x.dtype
        ).unsqueeze(0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        even = rope_part[:, 0::2]
        odd = rope_part[:, 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(1)
        if pass_part.numel() == 0:
            return rotated
        return torch.cat([rotated, pass_part], dim=1)

    def attend(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        context, _ = self.attend_with_diagnostics(hidden, memory, query_positions)
        return context

    def attend_with_diagnostics(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        query_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        stats = empty_stats()
        if not memory:
            return torch.zeros_like(hidden), stats
        mem = self.memory_interface_norm(torch.cat([entry[0] for entry in memory], dim=0))
        mem_count = mem.shape[0]
        query = self.query(self.query_interface_norm(hidden))
        keys = self.key(mem)
        values = self.value(mem)
        if self.rope_positions == "char":
            query_rope_positions = query_positions
            key_positions = torch.tensor([entry[1] for entry in memory], device=hidden.device)
        else:
            query_rope_positions = torch.zeros(hidden.shape[0], device=hidden.device)
            key_positions = torch.arange(-mem_count, 0, device=hidden.device)
        query = self.apply_rope(query, query_rope_positions)
        keys = self.apply_rope(keys, key_positions)
        scores = query @ keys.T / math.sqrt(self.hidden_size)
        weights = torch.softmax(scores, dim=1)
        eps = torch.finfo(weights.dtype).eps
        entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=1)
        top_weight = weights.max(dim=1).values
        top_indices = weights.argmax(dim=1)
        if self.rope_positions == "char":
            distances = (query_positions[:, None] - key_positions[None, :]).abs().to(dtype=weights.dtype)
        else:
            distances = torch.arange(mem_count, 0, -1, device=hidden.device, dtype=weights.dtype).unsqueeze(0)
        mean_distance = (weights * distances).sum(dim=1)
        stats["attn_entropy_sum"] = float(entropy.detach().sum().item())
        stats["attn_top_weight_sum"] = float(top_weight.detach().sum().item())
        stats["attn_char_distance_sum"] = float(mean_distance.detach().sum().item())
        stats["attn_token_count"] = float(hidden.shape[0])
        stats["attn_prev_weight_sum"] = float(weights.detach().sum().item())
        stats["attn_top_is_prev_sum"] = float(torch.ones_like(top_indices, dtype=weights.dtype).sum().item())
        return weights @ values, stats

    def attend_dual_value(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        query_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        context_pred, stats = self.attend_with_diagnostics(hidden, memory, query_positions)
        if not memory:
            return context_pred, torch.zeros_like(hidden), stats
        mem = self.memory_interface_norm(torch.cat([entry[0] for entry in memory], dim=0))
        mem_count = mem.shape[0]
        query = self.query(self.query_interface_norm(hidden))
        keys = self.key(mem)
        values_rnn = self.value_rnn(mem)
        if self.rope_positions == "char":
            query_rope_positions = query_positions
            key_positions = torch.tensor([entry[1] for entry in memory], device=hidden.device)
        else:
            query_rope_positions = torch.zeros(hidden.shape[0], device=hidden.device)
            key_positions = torch.arange(-mem_count, 0, device=hidden.device)
        query = self.apply_rope(query, query_rope_positions)
        keys = self.apply_rope(keys, key_positions)
        scores = query @ keys.T / math.sqrt(self.hidden_size)
        weights = torch.softmax(scores, dim=1)
        return context_pred, weights @ values_rnn, stats

    def attend_causal_current_full(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        query_positions: torch.Tensor,
        include_self: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        stats = empty_stats()
        if hidden.numel() == 0:
            return torch.zeros_like(hidden), stats, None, None, None
        current_entries = [
            (self.write_memory(hidden[index : index + 1]), int(query_positions[index].item()))
            for index in range(hidden.shape[0])
        ]
        all_entries = memory + current_entries
        mem = self.memory_interface_norm(torch.cat([entry[0] for entry in all_entries], dim=0))
        mem_count = len(memory)
        total_count = mem.shape[0]
        query = self.query(self.query_interface_norm(hidden))
        keys = self.key(mem)
        values = self.value(mem)
        if self.rope_positions == "char":
            query_rope_positions = query_positions
            key_positions = torch.tensor([entry[1] for entry in all_entries], device=hidden.device)
        else:
            query_rope_positions = torch.zeros(hidden.shape[0], device=hidden.device)
            key_positions = torch.arange(-total_count, 0, device=hidden.device)
        query = self.apply_rope(query, query_rope_positions)
        keys = self.apply_rope(keys, key_positions)
        scores = query @ keys.T / math.sqrt(self.hidden_size)
        current_indices = torch.arange(hidden.shape[0], device=hidden.device)
        key_indices = torch.arange(total_count, device=hidden.device)
        current_limit = mem_count + current_indices[:, None] + (1 if include_self else 0)
        allowed = key_indices[None, :] < current_limit
        no_allowed = ~allowed.any(dim=1)
        scores = scores.masked_fill(~allowed, -torch.inf)
        if no_allowed.any():
            scores = scores.clone()
            scores[no_allowed] = 0.0
        weights = torch.softmax(scores, dim=1)
        if no_allowed.any():
            weights = weights.masked_fill(no_allowed[:, None], 0.0)
        eps = torch.finfo(weights.dtype).eps
        entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=1)
        top_weight = weights.max(dim=1).values
        top_indices = weights.argmax(dim=1)
        if self.rope_positions == "char":
            distances = (query_positions[:, None] - key_positions[None, :]).abs().to(dtype=weights.dtype)
        else:
            distances = torch.arange(total_count, 0, -1, device=hidden.device, dtype=weights.dtype).unsqueeze(0)
        mean_distance = (weights * distances).sum(dim=1)
        if mem_count > 0:
            prev_weight = weights[:, :mem_count].sum(dim=1)
            top_is_prev = (top_indices < mem_count).to(dtype=weights.dtype)
        else:
            prev_weight = torch.zeros(hidden.shape[0], device=hidden.device, dtype=weights.dtype)
            top_is_prev = torch.zeros(hidden.shape[0], device=hidden.device, dtype=weights.dtype)
        current_weight = 1.0 - prev_weight
        stats["attn_entropy_sum"] = float(entropy.detach().sum().item())
        stats["attn_top_weight_sum"] = float(top_weight.detach().sum().item())
        stats["attn_char_distance_sum"] = float(mean_distance.detach().sum().item())
        stats["attn_token_count"] = float(hidden.shape[0])
        stats["attn_prev_weight_sum"] = float(prev_weight.detach().sum().item())
        stats["attn_current_weight_sum"] = float(current_weight.detach().sum().item())
        stats["attn_top_is_prev_sum"] = float(top_is_prev.detach().sum().item())
        return weights @ values, stats, weights, values, allowed

    def attend_causal_current(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        query_positions: torch.Tensor,
        include_self: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        context, stats, _, _, _ = self.attend_causal_current_full(
            hidden,
            memory,
            query_positions,
            include_self=include_self,
        )
        return context, stats

    def lma_token_decision(
        self,
        hidden: torch.Tensor,
        memory: list[MemoryEntry],
        max_memory: int,
    ) -> torch.Tensor:
        if hidden.numel() == 0:
            return hidden
        mem_entries = memory[-max_memory:]
        if mem_entries:
            mem = self.memory_interface_norm(torch.cat([entry[0] for entry in mem_entries], dim=0))
        else:
            mem = hidden.new_zeros((0, self.hidden_size))
        current = self.query_interface_norm(hidden).unsqueeze(1)
        if mem.shape[0] > 0:
            memory_tokens = mem.unsqueeze(0).expand(hidden.shape[0], -1, -1)
            tokens = torch.cat([current, memory_tokens], dim=1)
        else:
            tokens = current
        for block in self.lma_blocks:
            tokens = block(tokens)
        return tokens[:, 0, :]

    def segment_loss_token_feedback(
        self,
        ids: torch.Tensor,
        segments: list[tuple[int, int]],
        memory: list[MemoryEntry],
        max_memory: int,
        collect_diagnostics: bool,
        carried_hidden: torch.Tensor | None,
        carry_hidden: bool,
        token_feedback: str,
        prior_log_probs: torch.Tensor | None,
        prior_gate_features: torch.Tensor | None,
        skip_prior_log_probs: torch.Tensor | None,
        skip_prior_strength: torch.Tensor | None,
        prior_offset: int,
    ) -> tuple[torch.Tensor, int, list[MemoryEntry], torch.Tensor | None, dict[str, float]]:
        losses: list[torch.Tensor] = []
        token_count = 0
        stats = empty_stats()
        next_carried_hidden = carried_hidden
        for start, end in segments:
            prev_state = (
                next_carried_hidden.squeeze(0)
                if carry_hidden and next_carried_hidden is not None
                else torch.zeros(1, self.hidden_size, device=ids.device)
            )
            current_entries: list[MemoryEntry] = []
            final_state = prev_state
            final_context = torch.zeros_like(prev_state)
            for pos in range(start, end):
                token = ids[pos].view(1)
                emb = self.embed(token)
                emb = self.apply_input_rope(emb, torch.tensor([pos], device=ids.device))
                raw_state = self.cell(emb, prev_state)
                query_positions = torch.tensor([pos], device=ids.device)
                history = memory[-max_memory:] + current_entries
                if token_feedback == "dual-gated":
                    context, context_rnn, attn_stats = self.attend_dual_value(raw_state, history, query_positions)
                else:
                    context, attn_stats = self.attend_with_diagnostics(raw_state, history, query_positions)
                    context_rnn = context
                has_context = bool(history)
                if self.mixing == "attn-only":
                    logits = self.attn_head(context)
                else:
                    logits = self.late_logits(raw_state, context)
                logits, residual_logits, residual_alpha, skip_prior_alpha = self.apply_prior_residual(
                    logits,
                    prior_log_probs[pos - prior_offset : pos + 1 - prior_offset]
                    if prior_log_probs is not None
                    else None,
                    prior_gate_features[pos - prior_offset : pos + 1 - prior_offset]
                    if prior_gate_features is not None
                    else None,
                    skip_prior_log_probs[pos - prior_offset : pos + 1 - prior_offset]
                    if skip_prior_log_probs is not None
                    else None,
                    skip_prior_strength[pos - prior_offset : pos + 1 - prior_offset]
                    if skip_prior_strength is not None
                    else None,
                )
                if residual_alpha is not None:
                    stats["residual_alpha_sum"] += float(residual_alpha.detach().sum().item())
                    stats["residual_alpha_count"] += float(residual_alpha.numel())
                if skip_prior_alpha is not None:
                    stats["skip_prior_alpha_sum"] += float(skip_prior_alpha.detach().sum().item())
                    stats["skip_prior_alpha_count"] += float(skip_prior_alpha.numel())
                target = ids[pos + 1].view(1)
                token_loss = F.cross_entropy(logits, target, reduction="none")
                token_segment_loss = token_loss.sum()
                if self.training and residual_logits is not None and self.prior_residual_l2 > 0.0:
                    token_segment_loss = token_segment_loss + self.prior_residual_l2 * residual_logits.square().mean()
                losses.append(token_segment_loss)
                token_count += 1
                merge_stats(stats, attn_stats)
                stats["context_norm_sum"] += float(context.detach().norm(dim=1).sum().item())
                stats["token_count"] += 1.0
                if collect_diagnostics:
                    if pos == start:
                        stats["first_token_loss"] += float(token_loss.detach().sum().item())
                        stats["first_token_count"] += 1.0
                    else:
                        stats["later_token_loss"] += float(token_loss.detach().sum().item())
                        stats["later_token_count"] += 1.0

                if token_feedback == "context":
                    next_state = context if has_context else raw_state
                else:
                    feedback_input = torch.cat([raw_state, context_rnn], dim=1)
                    feedback_update = self.feedback_update(feedback_input)
                    feedback_gate = torch.sigmoid(self.feedback_gate(feedback_input))
                    delta = feedback_gate * feedback_update
                    delta_norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-6)
                    delta = delta * (self.feedback_delta_cap / delta_norm).clamp_max(1.0)
                    next_state = raw_state + delta
                    stats["gate_sum"] += float(feedback_gate.detach().sum().item())
                    stats["gate_count"] += float(feedback_gate.numel())
                    stats["delta_norm_sum"] += float(delta.detach().norm(dim=1).sum().item())
                current_entries.append((self.write_memory(next_state), pos))
                final_state = next_state
                final_context = context
                prev_state = next_state
            if self.memory_state == "fused":
                cap_input = torch.cat([final_state, final_context], dim=1)
                cap_state = self.cap_norm(torch.tanh(self.cap_fuse(cap_input)))
            else:
                cap_state = final_state
            memory.append((self.write_memory(cap_state), end - 1))
            if len(memory) > max_memory:
                memory = memory[-max_memory:]
            next_carried_hidden = final_state.unsqueeze(0)
        if not losses:
            return torch.zeros((), device=ids.device), 0, memory, next_carried_hidden, stats
        return torch.stack(losses).sum(), token_count, memory, next_carried_hidden, stats

    def segment_loss(
        self,
        ids: torch.Tensor,
        segments: list[tuple[int, int]],
        memory: list[MemoryEntry],
        max_memory: int,
        collect_diagnostics: bool = False,
        carried_hidden: torch.Tensor | None = None,
        carry_hidden: bool = False,
        feedback_state: str = "gru",
        token_feedback: str = "none",
        prior_log_probs: torch.Tensor | None = None,
        prior_gate_features: torch.Tensor | None = None,
        skip_prior_log_probs: torch.Tensor | None = None,
        skip_prior_strength: torch.Tensor | None = None,
        prior_offset: int = 0,
        prior_evidence: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, list[MemoryEntry], torch.Tensor | None, dict[str, float]]:
        """input_ids, if given, is what the model reads (a noisy copy of ids,
        say); the targets are always ids."""
        if token_feedback != "none":
            if input_ids is not None:
                raise ValueError("input_ids isn't supported with token feedback")
            if self.dropout > 0.0:
                raise ValueError("dropout isn't supported with token feedback")
            return self.segment_loss_token_feedback(
                ids,
                segments,
                memory,
                max_memory,
                collect_diagnostics,
                carried_hidden,
                carry_hidden,
                token_feedback,
                prior_log_probs,
                prior_gate_features,
                skip_prior_log_probs,
                skip_prior_strength,
                prior_offset,
            )
        losses: list[torch.Tensor] = []
        token_count = 0
        stats = empty_stats()
        next_carried_hidden = carried_hidden
        source = ids if input_ids is None else input_ids
        for start, end in segments:
            tokens = source[start:end]
            initial_hidden = next_carried_hidden if carry_hidden and next_carried_hidden is not None else self.zero_state(ids.device)
            embeddings = self.embed(tokens)
            if self.prior_input_proj is not None:
                if prior_log_probs is None:
                    raise ValueError("--prior-input requires count-prior rows")
                rows = prior_log_probs[start - prior_offset : end - prior_offset]
                embeddings = embeddings + self.prior_input_proj(rows.to(dtype=embeddings.dtype).clamp(min=-18.0))
            if self.prior_evidence_proj is not None:
                if prior_evidence is None:
                    raise ValueError("--prior-input depth-evidence requires depth-evidence rows")
                evidence = prior_evidence[start - prior_offset : end - prior_offset]
                embeddings = embeddings + self.prior_evidence_proj(
                    evidence.reshape(evidence.shape[0], -1).to(dtype=embeddings.dtype)
                )
            embeddings = self.drop(embeddings)
            embeddings = self.apply_input_rope(embeddings, torch.arange(start, end, device=ids.device))
            hidden_seq, hidden = self.rnn(embeddings.unsqueeze(0), initial_hidden)
            hidden_seq = hidden_seq.squeeze(0)
            query_positions = torch.arange(start, end, device=ids.device)
            utility_weights: torch.Tensor | None = None
            utility_values: torch.Tensor | None = None
            utility_allowed: torch.Tensor | None = None
            if self.mixing == "none":
                context = torch.zeros_like(hidden_seq)
                attn_stats = empty_stats()
                state = hidden_seq
                logits = self.gru_head(hidden_seq)
            elif self.mixing == "cross":
                state = hidden_seq
                for block in self.cross_blocks:
                    state = block(state, self.attend(state, memory[-max_memory:], query_positions))
                context, attn_stats = self.attend_with_diagnostics(state, memory[-max_memory:], query_positions)
                logits = self.state_head(state)
            elif self.mixing == "gated":
                context, attn_stats = self.attend_with_diagnostics(hidden_seq, memory[-max_memory:], query_positions)
                gate_input = torch.cat([hidden_seq, context], dim=1)
                update = self.gate_update(gate_input)
                gate = torch.sigmoid(self.gate(gate_input))
                delta = gate * update
                state = hidden_seq + delta
                logits = self.gated_head(torch.cat([hidden_seq, state, context], dim=1))
                stats["gate_sum"] += float(gate.detach().sum().item())
                stats["gate_count"] += float(gate.numel())
                stats["delta_norm_sum"] += float(delta.detach().norm(dim=1).sum().item())
            elif self.mixing == "attn-only":
                context, attn_stats = self.attend_with_diagnostics(hidden_seq, memory[-max_memory:], query_positions)
                state = hidden_seq
                logits = self.attn_head(context)
            elif self.mixing == "current-attn":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    memory[-max_memory:],
                    query_positions,
                )
                state = hidden_seq
                logits = self.late_logits(hidden_seq, context)
            elif self.mixing == "current-only-attn":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    [],
                    query_positions,
                )
                state = hidden_seq
                logits = self.late_logits(hidden_seq, context)
            elif self.mixing == "prefix-attn":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    memory[-max_memory:],
                    query_positions,
                    include_self=False,
                )
                state = hidden_seq
                logits = self.late_logits(hidden_seq, context)
            elif self.mixing == "prefix-attn-only":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    memory[-max_memory:],
                    query_positions,
                    include_self=False,
                )
                state = hidden_seq
                logits = self.attn_head(context)
            elif self.mixing == "current-attn-only-final":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    memory[-max_memory:],
                    query_positions,
                    include_self=True,
                )
                state = hidden_seq
                logits = self.attn_head(context)
            elif self.mixing == "local-attn-final":
                context, attn_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    [],
                    query_positions,
                    include_self=True,
                )
                state = hidden_seq
                logits = self.attn_head(context)
            elif self.mixing == "dual-attn-final":
                local_context, local_stats, utility_weights, utility_values, utility_allowed = self.attend_causal_current_full(
                    hidden_seq,
                    [],
                    query_positions,
                    include_self=True,
                )
                long_context, long_stats = self.attend_with_diagnostics(
                    hidden_seq,
                    memory[-max_memory:],
                    query_positions,
                )
                context = long_context
                attn_stats = long_stats
                state = hidden_seq
                logits = self.dual_attn_head(torch.cat([local_context, long_context], dim=1))
            elif self.mixing == "late-block":
                context, attn_stats = self.attend_with_diagnostics(hidden_seq, memory[-max_memory:], query_positions)
                state = hidden_seq
                for block in self.cross_blocks:
                    state = block(state, context)
                logits = self.state_head(state)
            elif self.mixing == "attention-decision":
                context, attn_stats = self.attend_with_diagnostics(hidden_seq, memory[-max_memory:], query_positions)
                state = context
                for block in self.decision_blocks:
                    state = block(hidden_seq, state)
                logits = self.state_head(state)
            elif self.mixing == "lma-token":
                context = torch.zeros_like(hidden_seq)
                attn_stats = empty_stats()
                state = self.lma_token_decision(hidden_seq, memory, max_memory)
                logits = self.state_head(state)
            elif self.mixing == "gated-xattn":
                mem_entries = memory[-max_memory:]
                if mem_entries:
                    mem = self.memory_interface_norm(torch.cat([entry[0] for entry in mem_entries], dim=0)).unsqueeze(0)
                    if self.rope_positions == "char":
                        memory_positions = torch.tensor([entry[1] for entry in mem_entries], device=ids.device)
                    else:
                        memory_positions = torch.arange(-len(mem_entries), 0, device=ids.device)
                else:
                    mem = hidden_seq.new_zeros((1, 0, self.hidden_size))
                    memory_positions = torch.empty(0, device=ids.device)
                state_batch = hidden_seq.unsqueeze(0)
                attn_stats = empty_stats()
                for block in self.gated_xattn_blocks:
                    block_out = block(
                        state_batch,
                        mem,
                        query_positions=query_positions if self.rope_positions == "char" else torch.zeros_like(query_positions),
                        memory_positions=memory_positions,
                        use_rope=self.use_rope,
                        return_diagnostics=collect_diagnostics,
                    )
                    if collect_diagnostics:
                        state_batch, block_stats = block_out
                        merge_stats(attn_stats, block_stats)
                    else:
                        state_batch = block_out
                state = state_batch.squeeze(0)
                context = state - hidden_seq
                logits = self.gru_head(self.drop(state))
                for block in self.gated_xattn_blocks:
                    stats["gate_sum"] += float(torch.tanh(block.gate_attn).detach().abs().item())
                    stats["gate_sum"] += float(torch.tanh(block.gate_mlp).detach().abs().item())
                    stats["gate_count"] += 2.0
            else:
                context, attn_stats = self.attend_with_diagnostics(hidden_seq, memory[-max_memory:], query_positions)
                state = hidden_seq
                logits = self.late_logits(hidden_seq, context)
                if collect_diagnostics:
                    zero_context = torch.zeros_like(context)
                    zero_hidden = torch.zeros_like(hidden_seq)
                    no_memory_logits = self.late_logits(hidden_seq, zero_context)
                    context_only_logits = self.late_logits(zero_hidden, context)
                    targets = ids[start + 1 : end + 1]
                    stats["no_memory_loss"] += float(
                        F.cross_entropy(no_memory_logits, targets, reduction="sum").detach().item()
                    )
                    stats["context_only_loss"] += float(
                        F.cross_entropy(context_only_logits, targets, reduction="sum").detach().item()
                    )
                    stats["diagnostic_token_count"] += float(tokens.numel())
            logits, residual_logits, residual_alpha, skip_prior_alpha = self.apply_prior_residual(
                logits,
                prior_log_probs[start - prior_offset : end - prior_offset] if prior_log_probs is not None else None,
                prior_gate_features[start - prior_offset : end - prior_offset]
                if prior_gate_features is not None
                else None,
                skip_prior_log_probs[start - prior_offset : end - prior_offset]
                if skip_prior_log_probs is not None
                else None,
                skip_prior_strength[start - prior_offset : end - prior_offset]
                if skip_prior_strength is not None
                else None,
            )
            if residual_alpha is not None:
                stats["residual_alpha_sum"] += float(residual_alpha.detach().sum().item())
                stats["residual_alpha_count"] += float(residual_alpha.numel())
            if skip_prior_alpha is not None:
                stats["skip_prior_alpha_sum"] += float(skip_prior_alpha.detach().sum().item())
                stats["skip_prior_alpha_count"] += float(skip_prior_alpha.numel())
            merge_stats(stats, attn_stats)
            stats["context_norm_sum"] += float(context.detach().norm(dim=1).sum().item())
            stats["token_count"] += float(tokens.numel())
            targets = ids[start + 1 : end + 1]
            token_losses = F.cross_entropy(logits, targets, reduction="none")
            segment_loss = token_losses.sum()
            if self.training and residual_logits is not None and self.prior_residual_l2 > 0.0:
                segment_loss = segment_loss + self.prior_residual_l2 * residual_logits.square().mean(dim=1).sum()
            terminal = state[-1:].contiguous()
            if self.training and self.record_aux_weight > 0.0:
                record_seq = self.write_memory(hidden_seq)
                record_logits = self.record_head(record_seq)
                record_losses = F.cross_entropy(record_logits, targets, reduction="none")
                segment_loss = segment_loss + self.record_aux_weight * record_losses.sum()
            if self.training and self.retrieval_aux_weight > 0.0:
                retrieval_loss, retrieval_stats = self.previous_record_loss(
                    terminal,
                    end - 1,
                    memory,
                    max_memory,
                )
                merge_stats(stats, retrieval_stats)
                if retrieval_loss is not None:
                    segment_loss = segment_loss + self.retrieval_aux_weight * retrieval_loss
            if self.training and self.utility_aux_weight > 0.0:
                utility_loss, utility_stats = self.utility_attention_loss(
                    utility_weights,
                    utility_values,
                    utility_allowed,
                    targets,
                )
                merge_stats(stats, utility_stats)
                if utility_loss is not None:
                    segment_loss = segment_loss + self.utility_aux_weight * utility_loss
            losses.append(segment_loss)
            if collect_diagnostics and token_losses.numel() > 0:
                stats["first_token_loss"] += float(token_losses[:1].detach().sum().item())
                stats["first_token_count"] += 1.0
                if token_losses.numel() > 1:
                    stats["later_token_loss"] += float(token_losses[1:].detach().sum().item())
                    stats["later_token_count"] += float(token_losses.numel() - 1)
            token_count += int(tokens.numel())
            if self.memory_state == "fused":
                cap_input = torch.cat([hidden_seq[-1:].contiguous(), context[-1:].contiguous()], dim=1)
                cap_state = self.cap_norm(torch.tanh(self.cap_fuse(cap_input)))
            else:
                cap_state = terminal
            memory_record = self.write_memory(cap_state)
            if self.training and self.terminal_record_aux_weight > 0.0:
                terminal_record_logits = self.record_head(memory_record)
                terminal_record_target = ids[end : end + 1]
                terminal_record_loss = F.cross_entropy(terminal_record_logits, terminal_record_target, reduction="sum")
                losses.append(self.terminal_record_aux_weight * terminal_record_loss)
            memory.append((memory_record, end - 1))
            if len(memory) > max_memory:
                memory = memory[-max_memory:]
            if feedback_state == "context":
                next_state = context[-1:].contiguous()
            elif feedback_state == "gated":
                feedback_input = torch.cat([terminal, context[-1:].contiguous()], dim=1)
                feedback_update = self.feedback_update(feedback_input)
                feedback_gate = torch.sigmoid(self.feedback_gate(feedback_input))
                next_state = terminal + feedback_gate * feedback_update
            else:
                next_state = terminal
            next_state = self.bound_carry(next_state, hidden_seq[-1:].contiguous())
            next_carried_hidden = next_state.unsqueeze(0)
        if not losses:
            return torch.zeros((), device=ids.device), 0, memory, next_carried_hidden, stats
        return torch.stack(losses).sum(), token_count, memory, next_carried_hidden, stats

    def check_streams_supported(self, token_feedback: str) -> None:
        """segment_loss_streams covers gated-xattn without token feedback,
        auxiliary losses or a skip-horizon prior; refuse anything else."""
        unsupported = []
        if self.mixing != "gated-xattn":
            unsupported.append(f"--mixing {self.mixing}")
        if token_feedback != "none":
            unsupported.append(f"--token-feedback {token_feedback}")
        for name in ("record_aux_weight", "terminal_record_aux_weight", "retrieval_aux_weight", "utility_aux_weight"):
            if getattr(self, name) > 0.0:
                unsupported.append("--" + name.replace("_", "-"))
        if self.skip_prior_trust_gate is not None:
            unsupported.append("--skip-horizon-prior")
        if unsupported:
            raise ValueError("--train-streams > 1 supports only --mixing gated-xattn without " + ", ".join(unsupported))

    def segment_loss_streams(
        self,
        ids: torch.Tensor,
        stream_segments: list[list[tuple[int, int]]],
        memory: StreamMemory | None,
        max_memory: int,
        carried_hidden: torch.Tensor | None = None,
        carry_hidden: bool = False,
        feedback_state: str = "gru",
        token_feedback: str = "none",
        prior_log_probs: torch.Tensor | None = None,
        prior_gate_features: torch.Tensor | None = None,
        prior_evidence: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        score_mask: np.ndarray | None = None,
        collect_stats: bool = False,
    ) -> tuple[torch.Tensor, int, StreamMemory, torch.Tensor, dict[str, float]]:
        """segment_loss for B independent streams at once. Stream b runs the
        segments stream_segments[b], in order, with its own memory and
        carried state, as segment_loss would run them alone; the loss is the
        sum over streams. Each segment index is one batched GRU call, padded to
        the longest of the B segments.

        Every stream has the same number of segments. Prior rows hold each
        stream's span (its first segment's start to its last segment's end),
        the streams' spans one after another (prior_rows_for_spans). memory is
        (records [B, M, H], positions [B, M]) or None for empty; the carried
        state is [1, B, H].

        score_mask ([B, K] bool), if given, scores only the segments it marks;
        the rest are run, feeding memory and the carried state, unscored (the
        burn-in of evaluate's streams). collect_stats adds what evaluate's
        rows report and this path can give: the residual gate, the gates of
        the attention blocks, first-token and later-token losses."""
        self.check_streams_supported(token_feedback)
        device = ids.device
        streams = len(stream_segments)
        count = len(stream_segments[0]) if streams else 0
        if streams == 0 or count == 0 or any(len(segs) != count for segs in stream_segments):
            raise ValueError("segment_loss_streams needs streams with the same, nonzero number of segments")
        starts = np.array([[start for start, _ in segs] for segs in stream_segments], dtype=np.int64)
        ends = np.array([[end for _, end in segs] for segs in stream_segments], dtype=np.int64)
        lengths = ends - starts  # [B, K]
        if (lengths <= 0).any():
            raise ValueError("segment_loss_streams got an empty segment")
        longest = int(lengths.max())
        offsets = np.arange(longest)
        valid = offsets[None, None, :] < lengths[:, :, None]  # [B, K, L]: real positions, not padding
        positions = starts[:, :, None] + offsets[None, None, :]
        span_starts = starts[:, 0]
        span_lengths = ends[:, -1] - span_starts
        row_starts = np.concatenate([[0], np.cumsum(span_lengths)[:-1]])
        rows = row_starts[:, None, None] + positions - span_starts[:, None, None]
        valid_t = torch.from_numpy(valid).to(device)
        flat_positions = torch.from_numpy(positions[valid]).to(device)  # in (stream, segment, offset) order
        flat_rows = torch.from_numpy(rows[valid]).to(device)

        embeddings = self.embed((ids if input_ids is None else input_ids)[flat_positions])
        if self.prior_ablation not in ("none", "output", "all"):
            raise ValueError(f"unknown prior_ablation: {self.prior_ablation}")
        input_prior = self.prior_ablation != "all"
        # prior_kept [N, 1]: whether each character's stream keeps the prior
        # this call (prior dropout); None when every stream does. No random
        # draw at 0 or in eval mode.
        prior_kept = None
        if self.training and self.prior_dropout > 0.0 and prior_log_probs is not None:
            stream_kept = torch.rand(streams, device=device) >= self.prior_dropout
            flat_streams = np.broadcast_to(np.arange(streams)[:, None, None], valid.shape)[valid]
            prior_kept = stream_kept[torch.from_numpy(flat_streams).to(device)].unsqueeze(1)

        def kept(x: torch.Tensor) -> torch.Tensor:
            return x if prior_kept is None else x * prior_kept.to(x.dtype)

        if self.prior_input_proj is not None and input_prior:
            if prior_log_probs is None:
                raise ValueError("--prior-input requires count-prior rows")
            embeddings = embeddings + kept(
                self.prior_input_proj(prior_log_probs[flat_rows].to(dtype=embeddings.dtype).clamp(min=-18.0))
            )
        if self.prior_evidence_proj is not None and input_prior:
            if prior_evidence is None:
                raise ValueError("--prior-input depth-evidence requires depth-evidence rows")
            evidence = prior_evidence[flat_rows]
            embeddings = embeddings + kept(
                self.prior_evidence_proj(evidence.reshape(evidence.shape[0], -1).to(dtype=embeddings.dtype))
            )
        embeddings = self.drop(embeddings)
        embeddings = self.apply_input_rope(embeddings, flat_positions)
        grid = embeddings.new_zeros((streams, count, longest, embeddings.shape[-1]))
        grid[valid_t] = embeddings
        # One view per segment index. Slicing grid[:, k] inside the loop
        # instead made the backward pass build a full-size zero gradient of
        # grid for every k and add them up (64 x 67 MB at h256: about 100 ms
        # of GPU time an update); unbind's backward stacks the K gradients
        # once. The same values.
        grid_steps = grid.unbind(1)

        if memory is None:
            records = embeddings.new_zeros((streams, 0, self.hidden_size))
            record_positions = torch.zeros((streams, 0), dtype=torch.long, device=device)
        else:
            records, record_positions = memory
        zero_hidden = embeddings.new_zeros((1, streams, self.hidden_size))
        next_hidden = carried_hidden if carried_hidden is not None else zero_hidden
        lengths_t = torch.from_numpy(lengths).to(device)
        positions_t = torch.from_numpy(positions).to(device)
        record_ends_t = torch.from_numpy(ends - 1).to(device)
        stream_index = torch.arange(streams, device=device)
        states = []
        context_norm_sum = 0.0
        scored_k = None if score_mask is None else torch.from_numpy(np.ascontiguousarray(score_mask)).to(device)
        for k in range(count):
            width = int(lengths[:, k].max())
            initial_hidden = next_hidden if carry_hidden else zero_hidden
            hidden_seq, _ = self.rnn(grid_steps[k][:, :width], initial_hidden.contiguous())
            if self.rope_positions == "char":
                query_positions, memory_positions = positions_t[:, k, :width], record_positions
            else:
                query_positions = torch.zeros(width, device=device)
                memory_positions = torch.arange(-records.shape[1], 0, device=device)
            mem = self.memory_interface_norm(records)
            state = hidden_seq
            for block in self.gated_xattn_blocks:
                state = block(
                    state, mem, query_positions=query_positions, memory_positions=memory_positions, use_rope=self.use_rope
                )
            if collect_stats:
                counted = valid_t[:, k, :width] if scored_k is None else valid_t[:, k, :width] & scored_k[:, k, None]
                context_norm_sum += float((state - hidden_seq).norm(dim=-1)[counted].sum().item())
            last = lengths_t[:, k] - 1
            terminal = state[stream_index, last]  # [B, H]
            terminal_context = terminal - hidden_seq[stream_index, last]
            if self.memory_state == "fused":
                cap_input = torch.cat([hidden_seq[stream_index, last], terminal_context], dim=1)
                cap_state = self.cap_norm(torch.tanh(self.cap_fuse(cap_input)))
            else:
                cap_state = terminal
            records = torch.cat([records, self.write_memory(cap_state).unsqueeze(1)], dim=1)[:, -max_memory:]
            record_positions = torch.cat([record_positions, record_ends_t[:, k : k + 1]], dim=1)[:, -max_memory:]
            if feedback_state == "context":
                next_state = terminal_context
            elif feedback_state == "gated":
                feedback_input = torch.cat([terminal, terminal_context], dim=1)
                feedback_update = self.feedback_update(feedback_input)
                next_state = terminal + torch.sigmoid(self.feedback_gate(feedback_input)) * feedback_update
            else:
                next_state = terminal
            next_state = self.bound_carry(next_state, hidden_seq[stream_index, last])
            next_hidden = next_state.unsqueeze(0)
            states.append(F.pad(state, (0, 0, 0, longest - width)))

        final_states = torch.stack(states, dim=1)[valid_t]  # [N, H], (stream, segment, offset) order
        own_logits = self.gru_head(self.drop(final_states))  # [N, V]
        logits, residual_logits, residual_alpha, _ = self.apply_prior_residual(
            own_logits,
            prior_log_probs[flat_rows] if prior_log_probs is not None else None,
            prior_gate_features[flat_rows] if prior_gate_features is not None else None,
        )
        if prior_kept is not None:
            # Streams without the prior score their own head; the residual's
            # L2 leash is for streams with it.
            logits = torch.where(prior_kept, logits, own_logits)
            residual_logits = kept(residual_logits)
        if self.prior_ablation != "none" and residual_logits is not None:
            logits = residual_logits
        targets = ids[flat_positions + 1]
        if self.state_sink is not None:
            if score_mask is None:
                sink_keep = slice(None)
            else:
                sink_keep = torch.from_numpy(np.broadcast_to(score_mask[:, :, None], valid.shape)[valid]).to(device)
            self.state_sink.append((final_states[sink_keep].detach().to("cpu", torch.float16), targets[sink_keep].cpu()))
        stats = empty_stats()
        if score_mask is None and not collect_stats:
            loss = F.cross_entropy(logits, targets, reduction="sum")
            if self.training and residual_logits is not None and self.prior_residual_l2 > 0.0:
                loss = loss + self.prior_residual_l2 * residual_logits.square().mean(dim=1).sum()
            stats["token_count"] = float(flat_positions.numel())
            return loss, int(flat_positions.numel()), (records, record_positions), next_hidden, stats

        if score_mask is None:
            keep = torch.ones(flat_positions.numel(), dtype=torch.bool, device=device)
        else:
            if score_mask.shape != (streams, count):
                raise ValueError(f"score_mask must be {(streams, count)}, got {score_mask.shape}")
            keep = torch.from_numpy(np.broadcast_to(score_mask[:, :, None], valid.shape)[valid]).to(device)
        first = torch.from_numpy(np.broadcast_to(offsets[None, None, :] == 0, valid.shape)[valid]).to(device)[keep]
        token_losses = F.cross_entropy(logits[keep], targets[keep], reduction="none")
        loss = token_losses.sum()
        if self.training and residual_logits is not None and self.prior_residual_l2 > 0.0:
            loss = loss + self.prior_residual_l2 * residual_logits[keep].square().mean(dim=1).sum()
        scored = int(keep.sum().item())
        stats["token_count"] = float(scored)
        if collect_stats:
            stats["context_norm_sum"] = context_norm_sum
            stats["first_token_loss"] = float(token_losses[first].sum().item())
            stats["first_token_count"] = float(first.sum().item())
            stats["later_token_loss"] = float(token_losses[~first].sum().item())
            stats["later_token_count"] = float((~first).sum().item())
            if residual_alpha is not None:
                stats["residual_alpha_sum"] = float(residual_alpha[keep].sum().item())
                stats["residual_alpha_count"] = float(residual_alpha[keep].numel())
            for block in self.gated_xattn_blocks:
                stats["gate_sum"] += float(torch.tanh(block.gate_attn).abs().item() + torch.tanh(block.gate_mlp).abs().item())
                stats["gate_count"] += 2.0
        return loss, scored, (records, record_positions), next_hidden, stats


def split_streams(segments: list[tuple[int, int]], streams: int) -> list[list[tuple[int, int]]]:
    """Cut the segment list into `streams` contiguous runs of equal length;
    the last len(segments) % streams segments are left out."""
    per_stream = len(segments) // streams
    if per_stream == 0:
        raise ValueError(f"{len(segments)} segments can't make {streams} streams")
    return [segments[b * per_stream : (b + 1) * per_stream] for b in range(streams)]


def prior_rows_for_spans(
    rows: PriorRows | None, spans: list[tuple[int, int]], device: torch.device
) -> torch.Tensor | None:
    """Rows start..end - 1 of each span, the spans one after another, on device."""
    if rows is None:
        return None
    if isinstance(rows, torch.Tensor):
        return torch.cat([rows[start:end] for start, end in spans]).to(device)
    return torch.from_numpy(np.concatenate([np.asarray(rows[start:end]) for start, end in spans])).to(device)


@dataclass(frozen=True)
class SegmentMemoryRow:
    epoch: int
    step: int
    train_nll: float | None
    val_nll: float | None
    val_ppl: float | None
    val_bpc: float | None
    segment_count: int
    mean_segment_len: float
    max_segment_len: int
    mean_gate: float | None
    mean_context_norm: float | None
    mean_delta_norm: float | None
    no_memory_ppl: float | None
    context_only_ppl: float | None
    mean_attn_entropy: float | None
    mean_attn_top_weight: float | None
    mean_attn_char_distance: float | None
    mean_attn_prev_weight: float | None
    mean_attn_current_weight: float | None
    mean_attn_top_is_prev: float | None
    first_token_ppl: float | None
    later_token_ppl: float | None
    runtime_sec: float
    peak_rss_kb: int
    mean_residual_alpha: float | None = None
    mean_skip_prior_alpha: float | None = None


def _parser_config_keys(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    keys: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if action.dest == argparse.SUPPRESS:
            continue
        keys[action.dest] = action
        for option in action.option_strings:
            if option.startswith("--"):
                keys[option[2:]] = action
                keys[option[2:].replace("-", "_")] = action
    return keys


def _coerce_config_value(parser: argparse.ArgumentParser, action: argparse.Action, value: object) -> object:
    if value is None:
        return None
    if isinstance(action, argparse._StoreTrueAction) or isinstance(action, argparse._StoreFalseAction):
        if not isinstance(value, bool):
            parser.error(f"config value for {action.dest!r} must be true or false")
        return value
    if action.type is not None and not isinstance(value, action.type):
        try:
            value = action.type(value)
        except Exception as error:
            parser.error(f"could not convert config value for {action.dest!r}: {error}")
    if action.choices is not None and value not in action.choices:
        parser.error(f"config value for {action.dest!r} must be one of {sorted(action.choices)!r}")
    return value


def apply_config_defaults(parser: argparse.ArgumentParser, config_path: Path) -> None:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        parser.error(f"could not read config {config_path}: {error}")
    except json.JSONDecodeError as error:
        parser.error(f"could not parse config {config_path}: {error}")
    if not isinstance(config, dict):
        parser.error(f"config {config_path} must contain a JSON object")

    actions = _parser_config_keys(parser)
    defaults: dict[str, object] = {}
    for raw_key, raw_value in config.items():
        key = raw_key[2:] if raw_key.startswith("--") else raw_key
        key = key.replace("-", "_")
        action = actions.get(raw_key) or actions.get(key)
        if action is None:
            parser.error(f"unknown config key {raw_key!r} in {config_path}")
        if action.dest == "config":
            continue
        defaults[action.dest] = _coerce_config_value(parser, action, raw_value)
    parser.set_defaults(**defaults)


def scheduled_lr(update: int, total: int, lr: float, min_lr: float, warmup: int) -> float:
    """--lr-schedule cosine's learning rate for update `update` (0-based) of
    `total`: a linear rise over the first `warmup` updates, then a cosine from lr
    down toward min_lr over the rest. transformer_baseline.py's lr_at, so imm and
    the transformer decay alike."""
    if update < warmup:
        return lr * (update + 1) / (warmup + 1)
    frac = min(1.0, (update - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * frac)) * (lr - min_lr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a char LM with trie-derived variable segments and attention over segment states.")
    parser.add_argument("--config", type=Path, default=None, help="Load JSON defaults before parsing CLI overrides.")
    parser.add_argument("--input", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--train-input", type=Path, default=None)
    parser.add_argument("--eval-input", type=Path, default=None)
    parser.add_argument(
        "--prior-rows",
        type=Path,
        default=None,
        help="PREFIX of prior rows another program wrote (PREFIX.json, mobius_prior_rows_v2): its log-probs, and "
        "gate features if it has them, replace imm's count prior; the --count-prior-* build and cache aren't used",
    )
    parser.add_argument("--vocab-input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/segment_memory_model.csv"))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--unique-threshold", type=int, default=1)
    parser.add_argument(
        "--segment-rule",
        choices=list(SEGMENT_RULES),
        default="unique",
        help="Where segments end. unique: once their text occurs at most --unique-threshold times in the "
        "training text. word: after each word and the whitespace that follows it. hybrid: the unique length, "
        "extended to the next word boundary. All capped at --max-depth.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--rnn-core", choices=["gru", "tanh"], default="gru")
    parser.add_argument("--attn-interface-norm", choices=["none", "attention", "context", "layer"], default="none")
    parser.add_argument("--max-memory", type=int, default=32)
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--input-rope", action="store_true")
    parser.add_argument("--rope-positions", choices=["segment", "char"], default="segment")
    parser.add_argument("--memory-state", choices=["terminal", "fused"], default="terminal")
    parser.add_argument("--memory-record", choices=["raw", "written"], default="raw")
    parser.add_argument("--record-aux-weight", type=float, default=0.0)
    parser.add_argument("--terminal-record-aux-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-aux-weight", type=float, default=0.0)
    parser.add_argument("--utility-aux-weight", type=float, default=0.0)
    parser.add_argument("--utility-temperature", type=float, default=1.0)
    parser.add_argument("--carry-hidden", action="store_true")
    parser.add_argument("--feedback-state", choices=["gru", "context", "gated"], default="gru")
    parser.add_argument(
        "--carry-bound",
        choices=["none", "tanh", "norm", "gru"],
        default="none",
        help="what a stream carries into its next segment's GRU: none, the state after the attention block as "
        "before (unbounded: the block adds to it every segment); tanh, that state squashed, c * tanh(x / c) with "
        "c = --carry-scale; norm, its length capped at --carry-scale; gru, the GRU's own last state",
    )
    parser.add_argument(
        "--carry-scale",
        type=float,
        default=1.0,
        help="for --carry-bound tanh, the bound c of c * tanh(x / c) (1: the GRU's range); for norm, the longest "
        "carried state (a healthy stream's runs about 8-26 at hidden 256)",
    )
    parser.add_argument("--token-feedback", choices=["none", "context", "gated", "dual-gated"], default="none")
    parser.add_argument("--feedback-gate-bias", type=float, default=-6.0)
    parser.add_argument("--feedback-delta-cap", type=float, default=1.0)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--prior-residual-scale", type=float, default=1.0)
    parser.add_argument("--prior-residual-l2", type=float, default=0.0)
    parser.add_argument("--prior-residual-gate", choices=["none", "context"], default="none")
    parser.add_argument(
        "--prior-residual-state-features",
        default="",
        help="Comma-separated prior feature columns to feed to an auxiliary residual-state MLP, e.g. 0,3.",
    )
    parser.add_argument(
        "--prior-input",
        choices=["none", "log-probs", "depth-evidence", "both"],
        default="none",
        help="Show the residual what the prior knows, through zero-initialized projections into the GRU's input "
        "(training starts from the model without them). log-probs: each position's prior log-probabilities. "
        "depth-evidence: what each context length's counts say, before blending (cached next to the prior "
        "cache). both: the two together.",
    )
    parser.add_argument(
        "--prior-evidence-rows",
        type=Path,
        default=None,
        help="Depth evidence written elsewhere (mobius, say), used by --prior-input depth-evidence or both in "
        "place of imm's own: PREFIX.json, PREFIX.train_depth_evidence.dat and PREFIX.eval_depth_evidence.dat, "
        "float16, shape (rows, depth, vocab + 1) in depth_evidence_chunk's layout. Any depth; the json's "
        "digests must match this run's ids and vocabulary, and its depth_evidence section may name the files.",
    )
    parser.add_argument(
        "--prior-residual-gate-max",
        type=float,
        default=0.0,
        help="Maximum alpha for context residual gate. Defaults to 2 * --prior-residual-scale.",
    )
    parser.add_argument(
        "--mixing",
        choices=[
            "none",
            "late",
            "cross",
            "gated",
            "attn-only",
            "current-attn",
            "current-only-attn",
            "prefix-attn",
            "prefix-attn-only",
            "current-attn-only-final",
            "local-attn-final",
            "dual-attn-final",
            "late-block",
            "attention-decision",
            "lma-token",
            "gated-xattn",
        ],
        default="late",
    )
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--segments-per-update", type=int, default=64)
    parser.add_argument(
        "--train-streams",
        type=int,
        default=1,
        help="Train on this many streams at once: the training segments are cut into equal contiguous runs, "
        "each with its own memory and carried state, batched through one GRU call per segment index; an update "
        "takes --segments-per-update segments from every stream. The last (segments mod streams) segments are "
        "left out. Evaluation stays one stream. 1: one stream, as before.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
        help="constant: --lr throughout, as before. cosine: --lr-warmup updates rising to --lr, then a cosine "
        "fall toward --min-lr, reached at the end of --epochs (transformer_baseline.py's schedule). It follows "
        "the update count, so a resumed run continues it when given the same --epochs",
    )
    parser.add_argument("--min-lr", type=float, default=1e-4, help="where --lr-schedule cosine ends")
    parser.add_argument("--lr-warmup", type=int, default=0, help="updates of linear warm-up for --lr-schedule cosine")
    parser.add_argument(
        "--prior-dropout",
        type=float,
        default=0.0,
        help="with --train-streams > 1: in training, each stream loses the prior, from its input and its output, "
        "with this probability, drawn anew for each update; it then predicts from its own head alone. Held-out "
        "scoring always has the prior",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout in the residual (gated-xattn only): its GRU input, the state its head reads, and inside "
        "the attention blocks; never on the carried state, the memory records or the prior",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-chars", type=int, default=None)
    parser.add_argument("--eval-max-chars", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-initial", action="store_true")
    parser.add_argument(
        "--eval-streams",
        type=int,
        default=1,
        help="score the held-out text as this many batched streams, each warmed up on --eval-burn-in unscored "
        "segments first (1: one stream, with every diagnostic)",
    )
    parser.add_argument("--eval-burn-in", type=int, default=256, help="unscored segments before each stream's own")
    parser.add_argument(
        "--prior-ablation-output",
        type=Path,
        default=None,
        help="with --eval-only and --eval-streams > 1: the take-the-prior-away test. Score the checkpoint in full, "
        "on its residual logits alone, and also without the prior in its input; then train a fresh output layer "
        "on its frozen final state (no prior in the output) with the prior in the input and without, and score "
        "those. Writes this JSON and stops",
    )
    parser.add_argument(
        "--probe-train-chars",
        type=int,
        default=1_000_000,
        help="training characters whose final states the fresh output layer learns from, read as --train-streams "
        "streams spread over the training text",
    )
    parser.add_argument("--probe-steps", type=int, default=3000)
    parser.add_argument("--probe-batch", type=int, default=4096)
    parser.add_argument("--probe-lr", type=float, default=3e-3)
    parser.add_argument("--probe-seed", type=int, default=7)
    parser.add_argument(
        "--prefetch-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with --train-streams > 1, read the next update's prior rows in a background thread (same rows)",
    )
    parser.add_argument(
        "--prefetch-ahead",
        type=int,
        default=2,
        help="with --prefetch-rows, how many updates' rows are read ahead, that many at once (same rows; 1 was the "
        "behaviour before 2026-11-13)",
    )
    parser.add_argument(
        "--row-gather",
        choices=("vector", "spans"),
        default="vector",
        help="with --prefetch-rows, read each source's spans in a few numpy calls (vector) or a slice per span "
        "(spans, the behaviour before 2026-11-13); the same rows",
    )
    parser.add_argument(
        "--pin-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with --prefetch-rows on CUDA and --row-gather vector, read rows into pinned memory and copy them to "
        "the GPU without blocking (same rows)",
    )
    parser.add_argument(
        "--drop-evidence-cache",
        action="store_true",
        help="with --prefetch-rows, drop the training depth evidence's pages from the page cache once read: for "
        "an evidence file too big to stay cached, so reading it each epoch doesn't push other memory into swap",
    )
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--best-checkpoint-output", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="with --resume-checkpoint: score the checkpoint on the held-out text once, write one row, and stop",
    )
    parser.add_argument(
        "--input-noise",
        type=float,
        default=0.0,
        help="substitute this fraction of the residual's training input characters, fresh each epoch "
        "(unigram replacements); targets and prior rows stay clean",
    )
    parser.add_argument(
        "--eval-clean-targets",
        type=Path,
        default=None,
        help="score held-out targets from this text (the clean original of a typo'd --eval-input); "
        "the model, prior and segments read --eval-input",
    )
    parser.add_argument(
        "--eval-residual-input",
        type=Path,
        default=None,
        help="the residual's held-out characters come from this text (a typo'd copy of --eval-input); "
        "the prior, evidence, segments and targets stay on --eval-input: a prior that repairs every typo",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--progress-every-steps", type=int, default=0)
    parser.add_argument("--count-prior", choices=["none", "frozen"], default="none")
    parser.add_argument("--count-prior-depth", type=int, default=8)
    parser.add_argument("--count-prior-valid-ratio", type=float, default=0.05)
    parser.add_argument("--count-prior-epochs", type=int, default=100)
    parser.add_argument("--count-prior-lr", type=float, default=0.05)
    parser.add_argument("--count-prior-max-fit-positions", type=int, default=50000)
    parser.add_argument("--count-prior-extra-features", default="entropy_delta,suffix_stats")
    parser.add_argument("--count-prior-cache", type=Path, default=None)
    parser.add_argument("--count-prior-impl", choices=["python", "packed", "marisa"], default="python")
    parser.add_argument("--count-prior-theta-json", type=Path, default=None)
    parser.add_argument("--count-prior-precompute", choices=["python", "numpy"], default="python")
    parser.add_argument("--count-prior-storage", choices=["device", "cpu", "mmap"], default="device")
    parser.add_argument("--count-prior-log-prob-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument(
        "--count-prior-train-folds",
        type=int,
        default=0,
        help="cross-fit the training rows' prior over this many folds, so no training position's prior "
        "counts its own passage (0: off, one count model for every row; 20 gives each row counts over "
        "95%% of the training text, like the held-out prior)",
    )
    parser.add_argument(
        "--count-prior-fold-chunk",
        type=int,
        default=0,
        help="cross-fitting chunk size in characters; use the split's held-out chunk size",
    )
    parser.add_argument(
        "--count-prior-precompute-cache-clear-every",
        type=int,
        default=0,
        help="Clear count-prior memoization caches every N precompute rows. 0 keeps caches until the model is released.",
    )
    parser.add_argument(
        "--transition-category-features",
        choices=["none", "svd"],
        default="none",
        help="Append distributional category features derived from count-prior successor distributions.",
    )
    parser.add_argument("--transition-category-dim", type=int, default=8)
    parser.add_argument("--transition-category-sample-rows", type=int, default=50000)
    parser.add_argument(
        "--skip-horizon-features",
        choices=["none", "profile"],
        default="none",
        help="Append learned gate features from a train-built SQLite skip-horizon artifact.",
    )
    parser.add_argument(
        "--skip-horizon-depth",
        type=int,
        default=0,
        help="Context depth for skip horizon features. 0 uses --count-prior-depth.",
    )
    parser.add_argument("--skip-horizon-max", type=int, default=16)
    parser.add_argument("--skip-horizon-db", type=Path, default=None)
    parser.add_argument("--skip-horizon-cache", type=Path, default=None)
    parser.add_argument(
        "--skip-horizon-prior",
        choices=["none", "distribution", "learned-distribution", "learned-horizon-distribution"],
        default="none",
        help="Use token-distribution skip-horizon evidence. 'distribution' hand-blends it; learned modes keep it separate and learn a gate.",
    )
    parser.add_argument(
        "--skip-horizon-prior-weight",
        type=float,
        default=0.25,
        help="Maximum probability-space mixture weight for --skip-horizon-prior distribution.",
    )
    parser.add_argument(
        "--skip-horizon-prior-gate-max",
        type=float,
        default=0.5,
        help="Maximum learned skip-prior gate before multiplying by skip strength.",
    )
    parser.add_argument(
        "--skip-horizon-prior-gate-bias",
        type=float,
        default=0.0,
        help="Initial logit bias for learned skip-prior gate. 0 means gate starts at half of --skip-horizon-prior-gate-max.",
    )
    parser.add_argument("--skip-horizon-count-db", type=Path, default=None)
    config_args, _ = parser.parse_known_args()
    if config_args.config is not None:
        apply_config_defaults(parser, config_args.config)
    return parser.parse_args()


def parse_feature_indices(raw: str, names: list[str] | None = None) -> list[int]:
    """Comma-separated feature columns, each an index or, when the rows name
    their features, a name."""
    out = []
    for part in [part.strip() for part in raw.split(",") if part.strip()]:
        if part.lstrip("-").isdigit():
            out.append(int(part))
        elif names is not None and part in names:
            out.append(names.index(part))
        else:
            raise ValueError(f"feature {part!r} is neither an index nor one of the rows' features {names}")
    return out


def segment_stats(segments: list[tuple[int, int]]) -> tuple[float, int]:
    lengths = [end - start for start, end in segments]
    return sum(lengths) / max(1, len(lengths)), max(lengths, default=0)


def count_split(n_train: int, depth: int, valid_ratio: float) -> int:
    """Where the prior's counted text ends and the gate's fitting text starts."""
    split = int(n_train * (1.0 - valid_ratio))
    return max(depth + 1, min(split, n_train - depth - 1))


def fit_count_gate_prior(
    train_ids: list[int],
    vocab_size: int,
    depth: int,
    valid_ratio: float,
    epochs: int,
    lr: float,
    max_fit_positions: int,
    extra_features: str,
    seed: int,
    impl: str,
    theta_json: Path | None,
) -> tuple[CountModel, list[float], list[dict[str, float]]]:
    split = count_split(len(train_ids), depth, valid_ratio)
    count_train = bytes(train_ids[:split])
    valid = bytes(train_ids[split:])
    model_classes = {
        "python": CountModel,
        "packed": PackedCountModel,
        "marisa": MarisaCountModel,
    }
    model_cls = model_classes[impl]
    model = model_cls(count_train, vocab_size, depth, expand_extra_features(extra_features))
    model.build()
    if theta_json is not None:
        payload = json.loads(theta_json.read_text(encoding="utf-8"))
        feature_names = model.feature_names()
        if payload.get("feature_names") != feature_names:
            raise ValueError(
                f"theta feature mismatch: expected {feature_names}, got {payload.get('feature_names')}"
            )
        theta_map = payload["theta"]
        theta = [float(theta_map[name]) for name in feature_names]
        return model, theta, payload.get("history", [])
    fit_positions = select_positions(len(valid), max_fit_positions, 1, seed)
    theta, history = train_gate(model, valid, depth, fit_positions, epochs, lr)
    return model, theta, history


def precompute_count_prior_log_probs(
    ids: list[int],
    model: CountModel,
    theta: list[float],
    depth: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if len(ids) < 2:
        return torch.empty((0, model.vocab_size), dtype=dtype, device=device)
    tokens = bytes(ids)
    rows = torch.empty((len(ids) - 1, model.vocab_size), dtype=dtype)
    for pos in range(len(ids) - 1):
        target_pos = pos + 1
        ctx = tokens[max(0, target_pos - depth) : target_pos]
        rows[pos] = torch.tensor(
            [math.log(prob) for prob in model.gated_distribution(ctx, theta)],
            dtype=dtype,
        )
    return rows.to(device)


def precompute_count_prior_feature_rows(
    ids: list[int],
    model: CountModel,
    depth: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    feature_count = len(model.feature_names())
    if len(ids) < 2:
        return torch.empty((0, feature_count), dtype=dtype, device=device)
    tokens = bytes(ids)
    rows = torch.zeros((len(ids) - 1, feature_count), dtype=dtype)
    if feature_count > 0:
        rows[:, -1] = 1.0
    for pos in range(len(ids) - 1):
        target_pos = pos + 1
        ctx = tokens[max(0, target_pos - depth) : target_pos]
        for d in range(min(depth, len(ctx)), 0, -1):
            feats = model.features(ctx[-d:])
            if feats is not None:
                rows[pos] = torch.tensor(feats, dtype=dtype)
                break
    return rows.to(device)


def numpy_dtype_for_torch(dtype: torch.dtype) -> np.dtype:
    if dtype == torch.float16:
        return np.dtype("float16")
    if dtype == torch.float32:
        return np.dtype("float32")
    raise ValueError(f"unsupported dtype for memmap prior cache: {dtype}")


def precompute_count_prior_log_probs_mmap(
    ids: list[int],
    model: CountModel,
    theta: list[float],
    depth: int,
    path: Path,
    dtype: torch.dtype,
    clear_cache_every: int = 0,
) -> np.memmap:
    np_dtype = numpy_dtype_for_torch(dtype)
    shape = (max(0, len(ids) - 1), model.vocab_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.memmap(path, dtype=np_dtype, mode="w+", shape=shape)
    if shape[0] == 0:
        rows.flush()
        return np.memmap(path, dtype=np_dtype, mode="r", shape=shape)
    tokens = bytes(ids)
    for pos in range(len(ids) - 1):
        target_pos = pos + 1
        ctx = tokens[max(0, target_pos - depth) : target_pos]
        rows[pos] = [math.log(prob) for prob in model.gated_distribution(ctx, theta)]
        if clear_cache_every > 0 and pos > 0 and pos % clear_cache_every == 0:
            model.clear_runtime_caches()
    rows.flush()
    del rows
    return np.memmap(path, dtype=np_dtype, mode="r", shape=shape)


def precompute_count_prior_feature_rows_mmap(
    ids: list[int],
    model: CountModel,
    depth: int,
    path: Path,
    dtype: torch.dtype = torch.float32,
    clear_cache_every: int = 0,
) -> np.memmap:
    np_dtype = numpy_dtype_for_torch(dtype)
    feature_count = len(model.feature_names())
    shape = (max(0, len(ids) - 1), feature_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.memmap(path, dtype=np_dtype, mode="w+", shape=shape)
    rows[:] = 0.0
    if feature_count > 0:
        rows[:, -1] = 1.0
    if shape[0] == 0:
        rows.flush()
        return np.memmap(path, dtype=np_dtype, mode="r", shape=shape)
    tokens = bytes(ids)
    for pos in range(len(ids) - 1):
        target_pos = pos + 1
        ctx = tokens[max(0, target_pos - depth) : target_pos]
        for d in range(min(depth, len(ctx)), 0, -1):
            feats = model.features(ctx[-d:])
            if feats is not None:
                rows[pos] = feats
                break
        if clear_cache_every > 0 and pos > 0 and pos % clear_cache_every == 0:
            model.clear_runtime_caches()
    rows.flush()
    del rows
    return np.memmap(path, dtype=np_dtype, mode="r", shape=shape)


def precompute_count_prior_rows_mmap(
    ids: list[int],
    model: CountModel,
    theta: list[float],
    depth: int,
    log_path: Path,
    feature_path: Path,
    log_dtype: torch.dtype,
    feature_dtype: torch.dtype = torch.float32,
    clear_cache_every: int = 0,
) -> tuple[np.memmap, np.memmap]:
    log_np_dtype = numpy_dtype_for_torch(log_dtype)
    feature_np_dtype = numpy_dtype_for_torch(feature_dtype)
    row_count = max(0, len(ids) - 1)
    log_shape = (row_count, model.vocab_size)
    feature_count = len(model.feature_names())
    feature_shape = (row_count, feature_count)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    log_rows = np.memmap(log_path, dtype=log_np_dtype, mode="w+", shape=log_shape)
    feature_rows = np.memmap(feature_path, dtype=feature_np_dtype, mode="w+", shape=feature_shape)
    feature_rows[:] = 0.0
    if feature_count > 0:
        feature_rows[:, -1] = 1.0
    if row_count == 0:
        log_rows.flush()
        feature_rows.flush()
        return (
            np.memmap(log_path, dtype=log_np_dtype, mode="r", shape=log_shape),
            np.memmap(feature_path, dtype=feature_np_dtype, mode="r", shape=feature_shape),
        )
    tokens = bytes(ids)
    for pos in range(row_count):
        target_pos = pos + 1
        ctx = tokens[max(0, target_pos - depth) : target_pos]
        log_rows[pos] = [math.log(prob) for prob in model.gated_distribution(ctx, theta)]
        for d in range(min(depth, len(ctx)), 0, -1):
            feats = model.features(ctx[-d:])
            if feats is not None:
                feature_rows[pos] = feats
                break
        if clear_cache_every > 0 and pos > 0 and pos % clear_cache_every == 0:
            model.clear_runtime_caches()
    log_rows.flush()
    feature_rows.flush()
    del log_rows, feature_rows
    return (
        np.memmap(log_path, dtype=log_np_dtype, mode="r", shape=log_shape),
        np.memmap(feature_path, dtype=feature_np_dtype, mode="r", shape=feature_shape),
    )


def load_prior_mmap(path: Path, dtype: str, shape: tuple[int, int]) -> np.memmap:
    return np.memmap(path, dtype=np.dtype(dtype), mode="r", shape=shape)


def prior_rows_to_device(rows: PriorRows | None, start: int, end: int, device: torch.device) -> torch.Tensor | None:
    if rows is None:
        return None
    if isinstance(rows, torch.Tensor):
        return rows[start:end].to(device)
    return torch.tensor(np.asarray(rows[start:end]), device=device)


def prior_mmap_sidecar_path(cache_path: Path, name: str) -> Path:
    return cache_path.with_suffix(f"{cache_path.suffix}.{name}.dat")


def prior_prob_rows_numpy(rows: PriorRows, start: int, end: int) -> np.ndarray:
    if isinstance(rows, torch.Tensor):
        return torch.exp(rows[start:end].detach().to(device="cpu", dtype=torch.float32)).numpy()
    return np.exp(np.asarray(rows[start:end], dtype=np.float32))


@dataclass
class TransitionCategoryProjection:
    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray


def fit_transition_category_projection(
    train_log_probs: PriorRows,
    dim: int,
    sample_rows: int,
    seed: int,
) -> TransitionCategoryProjection:
    row_count, vocab_size = train_log_probs.shape
    if row_count <= 0 or dim <= 0:
        return TransitionCategoryProjection(
            mean=np.empty((vocab_size,), dtype=np.float32),
            components=np.empty((vocab_size, 0), dtype=np.float32),
            scale=np.empty((0,), dtype=np.float32),
        )
    take = min(row_count, sample_rows if sample_rows > 0 else row_count)
    if take < row_count:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(row_count, size=take, replace=False))
        if isinstance(train_log_probs, torch.Tensor):
            sample_log_probs = train_log_probs[indices].detach().to(device="cpu", dtype=torch.float32).numpy()
            sample = np.exp(sample_log_probs)
        else:
            sample = np.exp(np.asarray(train_log_probs[indices], dtype=np.float32))
    else:
        sample = prior_prob_rows_numpy(train_log_probs, 0, row_count)
    mean = sample.mean(axis=0).astype(np.float32)
    centered = sample - mean
    _u, _s, vh = np.linalg.svd(centered.astype(np.float32, copy=False), full_matrices=False)
    k = min(dim, vh.shape[0])
    components = vh[:k].T.astype(np.float32, copy=False)
    projected = centered @ components
    scale = projected.std(axis=0).astype(np.float32, copy=False)
    scale = np.maximum(scale, 1.0e-6)
    return TransitionCategoryProjection(mean=mean, components=components, scale=scale)


def project_transition_category_chunk(
    log_probs: PriorRows,
    start: int,
    end: int,
    projection: TransitionCategoryProjection,
) -> np.ndarray:
    if projection.components.shape[1] == 0:
        return np.empty((end - start, 0), dtype=np.float32)
    probs = prior_prob_rows_numpy(log_probs, start, end)
    return ((probs - projection.mean) @ projection.components / projection.scale).astype(np.float32, copy=False)


def append_transition_category_features(
    feature_rows: PriorRows,
    log_probs: PriorRows,
    projection: TransitionCategoryProjection,
    device: torch.device,
    mmap_path: Path | None = None,
    chunk_size: int = 100_000,
) -> PriorRows:
    category_dim = projection.components.shape[1]
    if category_dim == 0:
        return feature_rows
    row_count, feature_count = feature_rows.shape
    if isinstance(feature_rows, torch.Tensor):
        category = torch.tensor(
            project_transition_category_chunk(log_probs, 0, row_count, projection),
            dtype=feature_rows.dtype,
            device=feature_rows.device,
        )
        return torch.cat([feature_rows, category], dim=1)
    if mmap_path is None:
        raise ValueError("mmap feature rows require a transition category mmap path")
    mmap_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(row_count, feature_count + category_dim))
    for start in range(0, row_count, chunk_size):
        end = min(row_count, start + chunk_size)
        out[start:end, :feature_count] = np.asarray(feature_rows[start:end], dtype=np.float32)
        out[start:end, feature_count:] = project_transition_category_chunk(log_probs, start, end, projection)
    out.flush()
    del out
    return np.memmap(mmap_path, dtype=np.float32, mode="r", shape=(row_count, feature_count + category_dim))


def add_transition_category_features(
    train_log_probs: PriorRows,
    eval_log_probs: torch.Tensor,
    train_features: PriorRows,
    eval_features: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PriorRows, torch.Tensor, int]:
    if args.transition_category_features == "none":
        return train_features, eval_features, int(train_features.shape[1])
    if args.transition_category_features != "svd":
        raise ValueError(f"unknown transition category feature mode: {args.transition_category_features}")
    base_feature_count = int(train_features.shape[1])
    projection = fit_transition_category_projection(
        train_log_probs,
        args.transition_category_dim,
        args.transition_category_sample_rows,
        args.seed,
    )
    suffix = f"transition_category_{args.transition_category_features}_d{projection.components.shape[1]}"
    if args.count_prior_cache is not None:
        train_feature_path = prior_mmap_sidecar_path(args.count_prior_cache, f"train_gate_features.{suffix}")
    else:
        train_feature_path = Path(tempfile.gettempdir()) / f"imm_{suffix}_train_features.dat"
    train_features = append_transition_category_features(
        train_features,
        train_log_probs,
        projection,
        device,
        mmap_path=train_feature_path if isinstance(train_features, np.memmap) else None,
    )
    eval_category = torch.tensor(
        project_transition_category_chunk(eval_log_probs, 0, eval_log_probs.shape[0], projection),
        dtype=eval_features.dtype,
        device=device,
    )
    eval_features = torch.cat([eval_features, eval_category], dim=1)
    print(
        f"transition_category_features={args.transition_category_features} "
        f"base_features={base_feature_count} category_dim={projection.components.shape[1]} "
        f"category_feature_start={base_feature_count} total_features={train_features.shape[1]}",
        flush=True,
    )
    return train_features, eval_features, int(train_features.shape[1])


def append_numpy_features(
    feature_rows: PriorRows,
    extra_rows: np.ndarray | np.memmap,
    mmap_path: Path | None = None,
    chunk_size: int = 100_000,
) -> PriorRows:
    row_count, feature_count = feature_rows.shape
    if extra_rows.shape[0] != row_count:
        raise ValueError(f"feature row mismatch: base rows={row_count} extra rows={extra_rows.shape[0]}")
    extra_count = int(extra_rows.shape[1])
    if isinstance(feature_rows, torch.Tensor):
        extra = torch.tensor(np.asarray(extra_rows), dtype=feature_rows.dtype, device=feature_rows.device)
        return torch.cat([feature_rows, extra], dim=1)
    if mmap_path is None:
        raise ValueError("mmap feature rows require an output mmap path")
    mmap_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(row_count, feature_count + extra_count))
    for start in range(0, row_count, chunk_size):
        end = min(row_count, start + chunk_size)
        out[start:end, :feature_count] = np.asarray(feature_rows[start:end], dtype=np.float32)
        out[start:end, feature_count:] = np.asarray(extra_rows[start:end], dtype=np.float32)
    out.flush()
    del out
    return np.memmap(mmap_path, dtype=np.float32, mode="r", shape=(row_count, feature_count + extra_count))


def add_skip_horizon_features(
    train_ids: list[int],
    eval_ids: list[int],
    vocab: CharVocab,
    train_features: PriorRows,
    eval_features: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PriorRows, torch.Tensor, int]:
    if args.skip_horizon_features == "none":
        return train_features, eval_features, int(train_features.shape[1])
    if args.skip_horizon_features != "profile":
        raise ValueError(f"unknown skip horizon feature mode: {args.skip_horizon_features}")

    base_feature_count = int(train_features.shape[1])
    depth = args.skip_horizon_depth if args.skip_horizon_depth > 0 else args.count_prior_depth
    horizon = args.skip_horizon_max
    cache_dir = args.skip_horizon_cache or default_skip_horizon_cache_dir(args.count_prior_cache)
    repo_root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    artifact = ensure_skip_horizon_artifact(
        train_ids,
        vocab.chars,
        depth,
        horizon,
        repo_root,
        cache_dir,
        db_path=args.skip_horizon_db,
    )
    suffix = f"skip_horizon_profile_d{depth}_w{horizon}"
    if args.count_prior_cache is not None:
        train_skip_path = prior_mmap_sidecar_path(args.count_prior_cache, f"train_gate_features.{suffix}.extra")
        train_augmented_path = prior_mmap_sidecar_path(args.count_prior_cache, f"train_gate_features.{suffix}")
    else:
        train_skip_path = cache_dir / f"train_{suffix}.extra.dat"
        train_augmented_path = cache_dir / f"train_{suffix}.features.dat"
    eval_skip_path = cache_dir / f"eval_{suffix}.extra.dat"

    train_skip_features = compile_skip_horizon_features(
        train_ids,
        artifact.db_path,
        depth,
        horizon,
        vocab.size,
        len(train_ids),
        output_path=train_skip_path,
    )
    eval_skip_features = compile_skip_horizon_features(
        eval_ids,
        artifact.db_path,
        depth,
        horizon,
        vocab.size,
        len(train_ids),
        output_path=eval_skip_path,
    )
    train_features = append_numpy_features(
        train_features,
        train_skip_features,
        mmap_path=train_augmented_path if isinstance(train_features, np.memmap) else None,
    )
    eval_extra = torch.tensor(np.asarray(eval_skip_features), dtype=eval_features.dtype, device=device)
    eval_features = torch.cat([eval_features, eval_extra], dim=1)
    print(
        f"skip_horizon_features={args.skip_horizon_features} depth={depth} horizon={horizon} "
        f"base_features={base_feature_count} skip_feature_start={base_feature_count} "
        f"skip_feature_count={len(SKIP_HORIZON_FEATURE_NAMES)} total_features={train_features.shape[1]} "
        f"feature_names={','.join(SKIP_HORIZON_FEATURE_NAMES)} "
        f"artifact={artifact.db_path} artifact_cache_loaded={artifact.cache_loaded} "
        f"build_precompute_sec={time.perf_counter() - started:.2f}",
        flush=True,
    )
    return train_features, eval_features, int(train_features.shape[1])


def prior_rows_numpy_dtype(rows: PriorRows) -> np.dtype:
    if isinstance(rows, torch.Tensor):
        return numpy_dtype_for_torch(rows.dtype)
    return np.dtype(rows.dtype)


def blend_log_probs_tensor(
    base_log_probs: torch.Tensor,
    skip_log_probs: np.ndarray | np.memmap,
    skip_strength: np.ndarray | np.memmap,
    skip_weight: float,
    device: torch.device,
) -> torch.Tensor:
    skip = torch.tensor(np.asarray(skip_log_probs), dtype=base_log_probs.dtype, device=device)
    strength = torch.tensor(np.asarray(skip_strength), dtype=torch.float32, device=device)
    alpha = torch.clamp(strength * float(skip_weight), 0.0, 1.0).to(dtype=torch.float32)
    probs = (1.0 - alpha) * torch.exp(base_log_probs.to(dtype=torch.float32)) + alpha * torch.exp(
        skip.to(dtype=torch.float32)
    )
    return torch.log(torch.clamp(probs, min=1.0e-8)).to(dtype=base_log_probs.dtype)


def add_skip_horizon_prior(
    train_ids: list[int],
    eval_ids: list[int],
    vocab: CharVocab,
    train_log_probs: PriorRows,
    eval_log_probs: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PriorRows, torch.Tensor, PriorRows | None, torch.Tensor | None, PriorRows | None, torch.Tensor | None]:
    if args.skip_horizon_prior == "none":
        return train_log_probs, eval_log_probs, None, None, None, None
    if args.skip_horizon_prior not in {"distribution", "learned-distribution", "learned-horizon-distribution"}:
        raise ValueError(f"unknown skip horizon prior mode: {args.skip_horizon_prior}")
    if args.skip_horizon_prior_weight < 0.0 or args.skip_horizon_prior_weight > 1.0:
        raise ValueError("--skip-horizon-prior-weight must be in [0, 1]")

    depth = args.skip_horizon_depth if args.skip_horizon_depth > 0 else args.count_prior_depth
    horizon = args.skip_horizon_max
    cache_dir = args.skip_horizon_cache or default_skip_horizon_cache_dir(args.count_prior_cache)
    repo_root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    artifact = ensure_skip_horizon_count_artifact(
        train_ids,
        vocab.chars,
        depth,
        horizon,
        repo_root,
        cache_dir,
        db_path=args.skip_horizon_count_db,
    )
    suffix = f"skip_horizon_{args.skip_horizon_prior}_d{depth}_w{horizon}"
    if args.skip_horizon_prior == "distribution":
        suffix = f"{suffix}_a{args.skip_horizon_prior_weight:g}"
    if args.count_prior_cache is not None:
        train_skip_path = prior_mmap_sidecar_path(args.count_prior_cache, f"train_log_probs.{suffix}.skip")
        train_strength_path = prior_mmap_sidecar_path(args.count_prior_cache, f"train_log_probs.{suffix}.strength")
        train_blend_path = prior_mmap_sidecar_path(args.count_prior_cache, f"train_log_probs.{suffix}")
    else:
        train_skip_path = cache_dir / f"train_{suffix}.skip_log_probs.dat"
        train_strength_path = cache_dir / f"train_{suffix}.strength.dat"
        train_blend_path = cache_dir / f"train_{suffix}.log_probs.dat"
    eval_skip_path = cache_dir / f"eval_{suffix}.skip_log_probs.dat"
    eval_strength_path = cache_dir / f"eval_{suffix}.strength.dat"

    if args.skip_horizon_prior == "learned-horizon-distribution":
        train_skip_log_probs, train_skip_strength = compile_skip_horizon_channel_rows(
            train_ids,
            artifact.db_path,
            depth,
            horizon,
            vocab.size,
            len(train_ids),
            log_output_path=train_skip_path,
            strength_output_path=train_strength_path,
            dtype=np.dtype("float32"),
        )
        eval_skip_log_probs, eval_skip_strength = compile_skip_horizon_channel_rows(
            eval_ids,
            artifact.db_path,
            depth,
            horizon,
            vocab.size,
            len(train_ids),
            log_output_path=eval_skip_path,
            strength_output_path=eval_strength_path,
            dtype=np.dtype("float32"),
        )
    else:
        train_skip_log_probs, train_skip_strength = compile_skip_horizon_distribution_rows(
            train_ids,
            artifact.db_path,
            depth,
            horizon,
            vocab.size,
            len(train_ids),
            log_output_path=train_skip_path,
            strength_output_path=train_strength_path,
            dtype=np.dtype("float32"),
        )
        eval_skip_log_probs, eval_skip_strength = compile_skip_horizon_distribution_rows(
            eval_ids,
            artifact.db_path,
            depth,
            horizon,
            vocab.size,
            len(train_ids),
            log_output_path=eval_skip_path,
            strength_output_path=eval_strength_path,
            dtype=np.dtype("float32"),
        )
    train_skip_log_probs_out: PriorRows | None = None
    eval_skip_log_probs_out: torch.Tensor | None = None
    train_skip_strength_out: PriorRows | None = None
    eval_skip_strength_out: torch.Tensor | None = None
    if isinstance(train_log_probs, torch.Tensor):
        if args.skip_horizon_prior == "distribution":
            train_log_probs = blend_log_probs_tensor(
                train_log_probs,
                train_skip_log_probs,
                train_skip_strength,
                args.skip_horizon_prior_weight,
                train_log_probs.device,
            )
        else:
            train_skip_log_probs_out = torch.tensor(
                np.asarray(train_skip_log_probs),
                dtype=train_log_probs.dtype,
                device=train_log_probs.device,
            )
            train_skip_strength_out = torch.tensor(
                np.asarray(train_skip_strength),
                dtype=torch.float32,
                device=train_log_probs.device,
            )
    else:
        if args.skip_horizon_prior == "distribution":
            train_log_probs = blend_log_prob_rows(
                train_log_probs,
                train_skip_log_probs,
                train_skip_strength,
                args.skip_horizon_prior_weight,
                train_blend_path,
                dtype=prior_rows_numpy_dtype(train_log_probs),
            )
        else:
            train_skip_log_probs_out = train_skip_log_probs
            train_skip_strength_out = train_skip_strength
    if args.skip_horizon_prior == "distribution":
        eval_log_probs = blend_log_probs_tensor(
            eval_log_probs,
            eval_skip_log_probs,
            eval_skip_strength,
            args.skip_horizon_prior_weight,
            device,
        )
    else:
        eval_skip_log_probs_out = torch.tensor(
            np.asarray(eval_skip_log_probs),
            dtype=eval_log_probs.dtype,
            device=device,
        )
        eval_skip_strength_out = torch.tensor(
            np.asarray(eval_skip_strength),
            dtype=torch.float32,
            device=device,
        )
    print(
        f"skip_horizon_prior={args.skip_horizon_prior} depth={depth} horizon={horizon} "
        f"weight={args.skip_horizon_prior_weight:g} artifact={artifact.db_path} "
        f"artifact_cache_loaded={artifact.cache_loaded} "
        f"train_strength_mean={float(np.asarray(train_skip_strength).mean()) if train_skip_strength.shape[0] else 0.0:.4f} "
        f"eval_strength_mean={float(np.asarray(eval_skip_strength).mean()) if eval_skip_strength.shape[0] else 0.0:.4f} "
        f"build_precompute_sec={time.perf_counter() - started:.2f}",
        flush=True,
    )
    return (
        train_log_probs,
        eval_log_probs,
        train_skip_log_probs_out,
        eval_skip_log_probs_out,
        train_skip_strength_out,
        eval_skip_strength_out,
    )


def ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(bytes(ids)).hexdigest()


def count_prior_cache_meta(args: argparse.Namespace, train_ids: list[int], eval_ids: list[int], vocab: CharVocab) -> dict[str, object]:
    theta_json_digest = None
    if args.count_prior_theta_json is not None:
        theta_json_digest = hashlib.sha256(args.count_prior_theta_json.read_bytes()).hexdigest()
    meta = {
        "train_digest": ids_digest(train_ids),
        "eval_digest": ids_digest(eval_ids),
        "vocab_chars": vocab.chars,
        "depth": args.count_prior_depth,
        "valid_ratio": args.count_prior_valid_ratio,
        "epochs": args.count_prior_epochs,
        "lr": args.count_prior_lr,
        "max_fit_positions": args.count_prior_max_fit_positions,
        "extra_features": args.count_prior_extra_features,
        "seed": args.seed,
        "impl": args.count_prior_impl,
        "theta_json_digest": theta_json_digest,
        "precompute": args.count_prior_precompute,
        "log_prob_dtype": args.count_prior_log_prob_dtype,
        "storage": args.count_prior_storage,
        "schema": "log_probs_and_feature_rows_v1",
    }
    if args.count_prior_train_folds > 0:  # only then, so existing caches still match
        meta["train_folds"] = args.count_prior_train_folds
        meta["fold_chunk"] = args.count_prior_fold_chunk
    return meta


def depth_evidence_meta(args: argparse.Namespace, train_ids: list[int], eval_ids: list[int], vocab: CharVocab) -> dict:
    return {
        "schema": "depth_evidence_v1",
        "train_digest": ids_digest(train_ids),
        "eval_digest": ids_digest(eval_ids),
        "vocab_chars": list(vocab.chars),
        "depth": args.count_prior_depth,
        "valid_ratio": args.count_prior_valid_ratio,
        "extra_features": args.count_prior_extra_features,
        "train_folds": args.count_prior_train_folds,
        "fold_chunk": args.count_prior_fold_chunk if args.count_prior_train_folds > 0 else None,
        "dtype": "float16",
    }


def load_or_build_depth_evidence(
    args: argparse.Namespace,
    train_ids: list[int],
    eval_ids: list[int],
    vocab: CharVocab,
    device: torch.device,
) -> tuple[np.memmap, torch.Tensor]:
    """Per-depth evidence rows (vectorized_count_prior.depth_evidence_chunk),
    cached beside the prior cache. Held-out rows count the same text as the
    held-out prior; training rows are cross-fitted as the prior's are, or
    counted like the held-out rows when there are no folds."""
    if args.count_prior_cache is None:
        raise ValueError("--prior-input depth-evidence requires --count-prior-cache")
    meta = depth_evidence_meta(args, train_ids, eval_ids, vocab)
    meta_path = prior_mmap_sidecar_path(args.count_prior_cache, "depth_evidence").with_suffix(".json")
    train_path = prior_mmap_sidecar_path(args.count_prior_cache, "train_depth_evidence")
    eval_path = prior_mmap_sidecar_path(args.count_prior_cache, "eval_depth_evidence")
    width = vocab.size + 1
    train_shape = (max(0, len(train_ids) - 1), args.count_prior_depth, width)
    eval_shape = (max(0, len(eval_ids) - 1), args.count_prior_depth, width)
    started = time.perf_counter()
    if (
        meta_path.exists()
        and train_path.exists()
        and eval_path.exists()
        and json.loads(meta_path.read_text()) == meta
    ):
        train_rows = np.memmap(train_path, dtype=np.float16, mode="r", shape=train_shape)
        eval_rows = np.memmap(eval_path, dtype=np.float16, mode="r", shape=eval_shape)
        built = False
    else:
        meta_path.unlink(missing_ok=True)  # written last, once both files are whole
        extra = expand_extra_features(args.count_prior_extra_features)
        split = count_split(len(train_ids), args.count_prior_depth, args.count_prior_valid_ratio)
        model = PackedCountModel(bytes(train_ids[:split]), vocab.size, args.count_prior_depth, extra)
        model.build()
        eval_rows = compile_depth_evidence_mmap(eval_ids, model, eval_path)
        if args.count_prior_train_folds > 0:
            del model
            train_rows = compile_cross_fit_depth_evidence_mmap(
                train_ids,
                vocab.size,
                args.count_prior_depth,
                extra,
                args.count_prior_train_folds,
                args.count_prior_fold_chunk,
                train_path,
                progress=lambda fold, folds: print(
                    f"depth_evidence_cross_fit fold={fold + 1}/{folds} sec={time.perf_counter() - started:.1f}",
                    flush=True,
                ),
            )
        else:
            train_rows = compile_depth_evidence_mmap(train_ids, model, train_path)
            del model
        meta_path.write_text(json.dumps(meta))
        built = True
    print(
        f"depth_evidence built={built} train_shape={train_rows.shape} eval_shape={eval_rows.shape} "
        f"sec={time.perf_counter() - started:.1f} train_path={train_path}",
        flush=True,
    )
    return train_rows, prior_rows_to_device(eval_rows, 0, eval_rows.shape[0], device)


def external_evidence_entry(meta: dict, split: str) -> dict:
    """What the json says about split's ("train" or "eval") evidence file:
    meta["depth_evidence"][split] (mobius_prior_rows_v2), else {}. Top-level
    "train"/"eval" objects aren't read: in mobius's v1 json they named the
    log-prob files."""
    section = meta.get("depth_evidence")
    entry = section.get(split) if isinstance(section, dict) else None
    return entry if isinstance(entry, dict) else {}


def external_evidence_path(prefix: Path, meta: dict, split: str) -> Path:
    """The .dat for split: its "path" in the json, relative to the json, if
    there is one, else PREFIX.<split>_depth_evidence.dat."""
    named = external_evidence_entry(meta, split).get("path")
    if named:
        return prefix.parent / named
    return prefix.parent / f"{prefix.name}.{split}_depth_evidence.dat"


def check_rows_meta(meta: dict, meta_path: Path, train_ids: list[int], eval_ids: list[int], vocab: CharVocab) -> None:
    """Refuses rows made for other text or another vocabulary."""
    for split, ids in (("train", train_ids), ("eval", eval_ids)):
        digest = meta.get(f"{split}_digest")
        if digest != ids_digest(ids):
            raise ValueError(f"{meta_path}: {split}_digest {digest} doesn't match this run's {split} ids")
    if meta.get("vocab_chars") != list(vocab.chars):
        raise ValueError(f"{meta_path}: vocab_chars don't match this run's vocabulary")


def rows_section_memmaps(
    prefix: Path, meta: dict, meta_path: Path, section: str, rows: dict[str, int], width: int | None
) -> dict[str, np.memmap]:
    """meta[section]["train"/"eval"] = {path, shape, dtype}: the two files as
    read-only memmaps, after checking each file's size against its shape,
    the shape against the row count, and (if given) the width. A relative
    path is relative to the json."""
    out = {}
    for split, row_count in rows.items():
        entry = meta[section].get(split)
        if not isinstance(entry, dict) or not {"path", "shape", "dtype"} <= set(entry):
            raise ValueError(f"{meta_path}: {section}.{split} needs path, shape and dtype")
        path = prefix.parent / entry["path"]
        shape, dtype = tuple(int(n) for n in entry["shape"]), np.dtype(entry["dtype"])
        if shape[0] != row_count or (width is not None and shape[1:] != (width,)):
            raise ValueError(f"{meta_path}: {section}.{split} shape {shape} isn't ({row_count}, {width})")
        if path.stat().st_size != int(np.prod(shape)) * dtype.itemsize:
            raise ValueError(f"{path}: {path.stat().st_size} bytes isn't {shape} {dtype}")
        out[split] = np.memmap(path, dtype=dtype, mode="r", shape=shape)
    return out


def load_prior_rows(
    prefix: Path,
    train_ids: list[int],
    eval_ids: list[int],
    vocab: CharVocab,
    device: torch.device,
) -> tuple[np.memmap, torch.Tensor, np.memmap | None, torch.Tensor | None, list[str] | None, dict]:
    """The frozen prior's rows from another program (--prior-rows), in place
    of imm's count prior: PREFIX.json (mobius_prior_rows_v2's layout) names
    log-prob files (log_probs: row r predicts ids[r + 1], natural log, over
    the vocabulary) and, optionally, gate-feature files (gate_features, with
    feature_names: what the residual's trust gate reads). Training rows stay
    memmaps; held-out rows go to the device. Returns train and eval log-probs,
    train and eval gate features (None without them), the feature names, and
    the json."""
    meta_path = prefix.parent / f"{prefix.name}.json"
    meta = json.loads(meta_path.read_text())
    check_rows_meta(meta, meta_path, train_ids, eval_ids, vocab)
    rows = {"train": max(0, len(train_ids) - 1), "eval": max(0, len(eval_ids) - 1)}
    if not isinstance(meta.get("log_probs"), dict):
        raise ValueError(f"{meta_path}: no log_probs section")
    log_probs = rows_section_memmaps(prefix, meta, meta_path, "log_probs", rows, vocab.size)
    features, names = None, None
    if meta.get("gate_features") is not None:
        section = meta["gate_features"]
        names = section.get("feature_names", section.get("names"))  # mobius's first port wrote "names"
        if not isinstance(names, list) or not names:
            raise ValueError(f"{meta_path}: gate_features needs feature_names")
        features = rows_section_memmaps(prefix, meta, meta_path, "gate_features", rows, len(names))
    return (
        log_probs["train"],
        prior_rows_to_device(log_probs["eval"], 0, rows["eval"], device),
        None if features is None else features["train"],
        None if features is None else prior_rows_to_device(features["eval"], 0, rows["eval"], device),
        names,
        meta,
    )


def load_external_depth_evidence(
    prefix: Path,
    train_ids: list[int],
    eval_ids: list[int],
    vocab: CharVocab,
    device: torch.device,
) -> tuple[np.memmap, torch.Tensor]:
    """Depth evidence rows another program wrote (--prior-evidence-rows), in
    depth_evidence_chunk's layout: row r predicts ids[r + 1]; [k - 1, :V] is
    the next-ID distribution after the k IDs ending at r, [k - 1, V] is
    log(1 + total). The depth is read from the files, so it may differ from
    --count-prior-depth. Refuses rows made for other text or another
    vocabulary."""
    meta_path = prefix.parent / f"{prefix.name}.json"
    meta = json.loads(meta_path.read_text())
    for split, ids in (("train", train_ids), ("eval", eval_ids)):
        digest = meta.get(f"{split}_digest")
        if digest != ids_digest(ids):
            raise ValueError(f"{meta_path}: {split}_digest {digest} doesn't match this run's {split} ids")
    if meta.get("vocab_chars") != list(vocab.chars):
        raise ValueError(f"{meta_path}: vocab_chars don't match this run's vocabulary")
    width = vocab.size + 1
    if "compact_evidence" in meta:
        # imm/compact_evidence.py's layout: the same rows, rebuilt dense per slice
        compact = load_compact(prefix, meta)
        for split, ids in (("train", train_ids), ("eval", eval_ids)):
            rows_expected = max(0, len(ids) - 1)
            if compact[split].shape[0] != rows_expected or compact[split].shape[2] != width:
                raise ValueError(
                    f"{meta_path}: compact {split} shape {compact[split].shape} isn't {rows_expected} rows x depth x {width}"
                )
        if compact["train"].shape[1] != compact["eval"].shape[1]:
            raise ValueError(f"{prefix}: train depth {compact['train'].shape[1]} != eval depth {compact['eval'].shape[1]}")
        section = meta["compact_evidence"]
        print(
            f"depth_evidence external={prefix} layout=compact shallow={section.get('shallow')} "
            f"train_shape={compact['train'].shape} eval_shape={compact['eval'].shape} "
            f"builder={meta.get('builder')!r} cross_fit={json.dumps(meta.get('cross_fit'))}",
            flush=True,
        )
        return compact["train"], prior_rows_to_device(compact["eval"], 0, compact["eval"].shape[0], device)
    rows = {}
    for split, ids in (("train", train_ids), ("eval", eval_ids)):
        path = external_evidence_path(prefix, meta, split)
        row_count = max(0, len(ids) - 1)
        itemsize = np.dtype(np.float16).itemsize
        size = path.stat().st_size
        if row_count == 0 or size % (row_count * width * itemsize) != 0:
            raise ValueError(f"{path}: {size} bytes isn't {row_count} rows x depth x {width} float16")
        depth = size // (row_count * width * itemsize)
        shape = (row_count, depth, width)
        named = external_evidence_entry(meta, split).get("shape")
        if named is not None and tuple(named) != shape:
            raise ValueError(f"{meta_path}: {split} shape {tuple(named)} doesn't match the file's {shape}")
        rows[split] = np.memmap(path, dtype=np.float16, mode="r", shape=shape)
    if rows["train"].shape[1] != rows["eval"].shape[1]:
        raise ValueError(f"{prefix}: train depth {rows['train'].shape[1]} != eval depth {rows['eval'].shape[1]}")
    named_depth = meta.get("depth_evidence", {}).get("depth") if isinstance(meta.get("depth_evidence"), dict) else None
    if named_depth is not None and named_depth != rows["train"].shape[1]:
        raise ValueError(f"{meta_path}: depth_evidence.depth {named_depth} != the files' {rows['train'].shape[1]}")
    print(
        f"depth_evidence external={prefix} train_shape={rows['train'].shape} eval_shape={rows['eval'].shape} "
        f"builder={meta.get('builder')!r} cross_fit={json.dumps(meta.get('cross_fit'))}",
        flush=True,
    )
    return rows["train"], prior_rows_to_device(rows["eval"], 0, rows["eval"].shape[0], device)


def check_cross_fit_args(args: argparse.Namespace) -> None:
    if args.count_prior_train_folds <= 0:
        return
    if args.count_prior_train_folds < 2 or args.count_prior_fold_chunk <= 0:
        raise ValueError("--count-prior-train-folds needs at least 2 folds and a positive --count-prior-fold-chunk")
    if (args.count_prior_storage, args.count_prior_precompute, args.count_prior_impl) != ("mmap", "numpy", "packed"):
        raise ValueError("cross-fitting is implemented for --count-prior-storage mmap --count-prior-precompute numpy --count-prior-impl packed")
    # These count the training text themselves and aren't cross-fitted
    if args.skip_horizon_features != "none" or args.skip_horizon_prior != "none":
        raise ValueError("cross-fitting doesn't cover skip-horizon features or priors")


def count_prior_cache_matches(cache: dict, meta: dict[str, object]) -> bool:
    stored = cache.get("meta")
    if not isinstance(stored, dict):
        return False
    if meta.get("theta_json_digest") is not None:
        # The seed only picks the positions the gate is fitted on, and with
        # theta from a file there's no fitting: every seed builds the same rows.
        stored = {k: v for k, v in stored.items() if k != "seed"}
        meta = {k: v for k, v in meta.items() if k != "seed"}
    return stored == meta


@torch.no_grad()
def evaluate(
    model: SegmentMemoryLM,
    ids: torch.Tensor,
    segments: list[tuple[int, int]],
    max_memory: int,
    carry_hidden: bool,
    feedback_state: str,
    token_feedback: str,
    prior_log_probs: torch.Tensor | None = None,
    prior_gate_features: torch.Tensor | None = None,
    skip_prior_log_probs: torch.Tensor | None = None,
    skip_prior_strength: torch.Tensor | None = None,
    prior_evidence: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    streams: int = 1,
    burn_in: int = 0,
    chunk: int = 64,
) -> tuple[float, float, float, dict[str, float]]:
    """Score the held-out segments. streams > 1 runs them as that many
    batched streams (eval_stream_windows), each warmed up on burn_in unscored
    segments before its own; with burn_in at least the segment count every
    stream reads the whole text and the score is the one-stream score. The
    batched rows leave the attention and no-memory diagnostics empty."""
    model.eval()
    if streams > 1:
        return evaluate_streams(
            model, ids, segments, max_memory, carry_hidden, feedback_state, token_feedback, streams, burn_in, chunk,
            prior_log_probs=prior_log_probs, prior_gate_features=prior_gate_features, prior_evidence=prior_evidence,
            input_ids=input_ids, skip_prior=skip_prior_log_probs is not None or skip_prior_strength is not None,
        )
    memory: list[MemoryEntry] = []
    loss, tokens, _, _, stats = model.segment_loss(
        ids,
        segments,
        memory,
        max_memory,
        collect_diagnostics=True,
        carry_hidden=carry_hidden,
        feedback_state=feedback_state,
        token_feedback=token_feedback,
        prior_log_probs=prior_log_probs,
        prior_gate_features=prior_gate_features,
        skip_prior_log_probs=skip_prior_log_probs,
        skip_prior_strength=skip_prior_strength,
        prior_evidence=prior_evidence,
        input_ids=input_ids,
    )
    if tokens == 0:
        raise ValueError("cannot evaluate no tokens")
    nll = float(loss.item()) / tokens
    return nll, math.exp(nll) if nll <= 700.0 else float("inf"), nll / math.log(2.0), stats


def eval_stream_windows(count: int, streams: int, burn_in: int) -> list[tuple[int, int, int, int]]:
    """(lo, hi, score_lo, score_hi) for each stream: it runs segments lo..hi-1
    and scores score_lo..score_hi-1. The scored ranges cut 0..count into
    `streams` contiguous runs (lengths differing by at most one); every window
    has the same length, the longest run plus burn_in (at most count), and
    starts at least burn_in segments before its run, or at 0."""
    if count <= 0 or streams < 1 or burn_in < 0:
        raise ValueError("eval_stream_windows needs segments, at least one stream and burn_in >= 0")
    streams = min(streams, count)
    base, extra = divmod(count, streams)
    bounds = np.concatenate([[0], np.cumsum([base + (i < extra) for i in range(streams)])])
    width = min(count, base + (extra > 0) + burn_in)
    windows = []
    for i in range(streams):
        score_lo, score_hi = int(bounds[i]), int(bounds[i + 1])
        lo = max(0, score_lo - burn_in)
        if lo + width > count:
            lo = count - width
        windows.append((lo, lo + width, score_lo, score_hi))
    return windows


@torch.no_grad()
def evaluate_streams(
    model: SegmentMemoryLM,
    ids: torch.Tensor,
    segments: list[tuple[int, int]],
    max_memory: int,
    carry_hidden: bool,
    feedback_state: str,
    token_feedback: str,
    streams: int,
    burn_in: int,
    chunk: int,
    prior_log_probs: PriorRows | None = None,
    prior_gate_features: PriorRows | None = None,
    prior_evidence: PriorRows | None = None,
    input_ids: torch.Tensor | None = None,
    skip_prior: bool = False,
) -> tuple[float, float, float, dict[str, float]]:
    if skip_prior:
        raise ValueError("--eval-streams > 1 doesn't support the skip prior")
    if chunk < 1:
        raise ValueError("evaluate_streams needs chunk >= 1")
    model.check_streams_supported(token_feedback)
    windows = eval_stream_windows(len(segments), streams, burn_in)
    width = windows[0][1] - windows[0][0]
    index = np.arange(width)
    score_mask = np.stack([(lo + index >= s_lo) & (lo + index < s_hi) for lo, _, s_lo, s_hi in windows])
    device = ids.device
    memory: StreamMemory | None = None
    hidden: torch.Tensor | None = None
    total_loss = 0.0
    stats = empty_stats()
    gates = (0.0, 0.0)
    for c0 in range(0, width, chunk):
        c1 = min(width, c0 + chunk)
        stream_segments = [segments[lo + c0 : lo + c1] for lo, _, _, _ in windows]
        spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
        loss, _, memory, hidden, chunk_stats = model.segment_loss_streams(
            ids,
            stream_segments,
            memory,
            max_memory,
            carried_hidden=hidden,
            carry_hidden=carry_hidden,
            feedback_state=feedback_state,
            token_feedback=token_feedback,
            prior_log_probs=prior_rows_for_spans(prior_log_probs, spans, device),
            prior_gate_features=prior_rows_for_spans(prior_gate_features, spans, device),
            prior_evidence=prior_rows_for_spans(prior_evidence, spans, device),
            input_ids=input_ids,
            score_mask=score_mask[:, c0:c1],
            collect_stats=True,
        )
        total_loss += float(loss.item())
        gates = (chunk_stats.pop("gate_sum"), chunk_stats.pop("gate_count"))
        merge_stats(stats, chunk_stats)
    stats["gate_sum"], stats["gate_count"] = gates  # the blocks' gates are parameters: the same every chunk
    tokens = stats["token_count"]
    want = sum(end - start for start, end in segments)
    if tokens != want:
        raise RuntimeError(f"evaluate_streams scored {tokens} characters, expected {want}")
    nll = total_loss / tokens
    return nll, math.exp(nll) if nll <= 700.0 else float("inf"), nll / math.log(2.0), stats


@torch.no_grad()
def collect_train_states(
    model: SegmentMemoryLM,
    ids: torch.Tensor,
    segments: list[tuple[int, int]],
    streams: int,
    chunk: int,
    chars: int,
    max_memory: int,
    carry_hidden: bool,
    feedback_state: str,
    token_feedback: str,
    prior_log_probs: PriorRows | None,
    prior_gate_features: PriorRows | None,
    prior_evidence: PriorRows | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Final states and targets of about `chars` training characters, read as
    `streams` streams that start evenly through the training text (the
    training loop's split_streams), each warmed up on one unscored chunk of
    segments first. Run in eval mode, as the model scores held-out text."""
    model.eval()
    stream_segments = split_streams(segments, streams)
    mean_len = sum(end - start for start, end in segments) / len(segments)
    count = min(len(stream_segments[0]), chunk + math.ceil(chars / streams / mean_len))
    memory: StreamMemory | None = None
    hidden: torch.Tensor | None = None
    sink: list[tuple[torch.Tensor, torch.Tensor]] = []
    device = ids.device
    try:
        for c0 in range(0, count, chunk):
            c1 = min(count, c0 + chunk)
            batch = [segs[c0:c1] for segs in stream_segments]
            spans = [(segs[0][0], segs[-1][1]) for segs in batch]
            model.state_sink = sink if c0 > 0 else None
            _, _, memory, hidden, _ = model.segment_loss_streams(
                ids,
                batch,
                memory,
                max_memory,
                carried_hidden=hidden,
                carry_hidden=carry_hidden,
                feedback_state=feedback_state,
                token_feedback=token_feedback,
                prior_log_probs=prior_rows_for_spans(prior_log_probs, spans, device),
                prior_gate_features=prior_rows_for_spans(prior_gate_features, spans, device),
                prior_evidence=prior_rows_for_spans(prior_evidence, spans, device),
            )
    finally:
        model.state_sink = None
    if not sink:
        raise ValueError("--probe-train-chars read no characters past the warm-up chunk")
    return torch.cat([s for s, _ in sink]), torch.cat([t for _, t in sink])


def probe_nll(head: nn.Linear, states: torch.Tensor, targets: torch.Tensor, batch: int = 65536) -> float:
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(states), batch):
            logits = head(states[i : i + batch].float())
            total += float(F.cross_entropy(logits, targets[i : i + batch], reduction="sum"))
    return total / len(states)


def train_probe(
    trained_head: nn.Linear,
    train_states: torch.Tensor,
    train_targets: torch.Tensor,
    eval_states: torch.Tensor,
    eval_targets: torch.Tensor,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
) -> dict:
    """A fresh output layer on frozen final states, with no prior in the
    output: started from the trained head, Adam with cosine lr to 1e-4 (as
    @air's probe.py does for the transformers), scored on the held-out
    states."""
    head = nn.Linear(trained_head.in_features, trained_head.out_features)
    with torch.no_grad():
        head.weight.copy_(trained_head.weight.detach().float().cpu())
        head.bias.copy_(trained_head.bias.detach().float().cpu())
    start = probe_nll(head, eval_states, eval_targets)
    curve = [{"step": 0, "heldout_nll": start, "heldout_ppl": math.exp(start)}]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for step in range(1, steps + 1):
        for group in opt.param_groups:
            group["lr"] = 1e-4 + 0.5 * (1 + math.cos(math.pi * step / steps)) * (lr - 1e-4)
        pick = torch.randint(0, len(train_states), (batch,), generator=gen)
        loss = F.cross_entropy(head(train_states[pick].float()), train_targets[pick])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps:
            nll = probe_nll(head, eval_states, eval_targets)
            curve.append({"step": step, "train_loss": float(loss), "heldout_nll": nll, "heldout_ppl": math.exp(nll)})
            print(f"probe step={step} train_loss={float(loss):.4f} heldout_ppl={math.exp(nll):.4f}", flush=True)
    final = curve[-1]["heldout_nll"]
    return {
        "heldout_ppl": math.exp(final),
        "heldout_bpc": final / math.log(2.0),
        "trained_head_alone_ppl": math.exp(start),
        "train_states": len(train_states),
        "curve": curve,
    }


def run_prior_ablation(
    model: SegmentMemoryLM,
    args: argparse.Namespace,
    eval_ids: torch.Tensor,
    eval_segments: list[tuple[int, int]],
    eval_rows: dict[str, PriorRows | None],
    eval_input_ids: torch.Tensor | None,
    train_ids: torch.Tensor,
    train_segments: list[tuple[int, int]],
    train_rows: dict[str, PriorRows | None],
) -> dict:
    """The take-the-prior-away test (the user, 2026-11-11, via @air): how much
    has the residual learned of the text itself, and how much does it lean on
    the prior? Scores the held-out text three ways (prior_ablation none,
    output, all), then a fresh output layer on the frozen final states with
    the prior in the input and without."""
    if args.eval_streams < 2:
        raise ValueError("--prior-ablation-output needs --eval-streams > 1")
    if eval_rows["prior_log_probs"] is None:
        raise ValueError("--prior-ablation-output needs a model with a prior")
    common = dict(
        max_memory=args.max_memory,
        carry_hidden=args.carry_hidden,
        feedback_state=args.feedback_state,
        token_feedback=args.token_feedback,
    )
    scores: dict[str, dict] = {}
    eval_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    names = {"none": "full", "output": "residual_only_prior_in_input", "all": "no_prior"}
    for mode, name in names.items():
        t0 = time.perf_counter()
        sink: list[tuple[torch.Tensor, torch.Tensor]] = []
        model.prior_ablation = mode
        model.state_sink = sink if mode != "output" else None  # "output" reads the same states as "none"
        try:
            nll, ppl, bpc, stats = evaluate(
                model, eval_ids, eval_segments, input_ids=eval_input_ids, streams=args.eval_streams,
                burn_in=args.eval_burn_in, **common, **eval_rows,
            )
        finally:
            model.prior_ablation = "none"
            model.state_sink = None
        first = mean_or_none(stats["first_token_loss"], stats["first_token_count"])
        later = mean_or_none(stats["later_token_loss"], stats["later_token_count"])
        scores[name] = {
            "heldout_ppl": ppl,
            "heldout_bpc": bpc,
            "first_token_ppl": math.exp(first) if first is not None else None,
            "later_token_ppl": math.exp(later) if later is not None else None,
            "eval_sec": time.perf_counter() - t0,
        }
        if sink:
            eval_states["prior_in_input" if mode == "none" else "no_prior"] = (
                torch.cat([s for s, _ in sink]),
                torch.cat([t for _, t in sink]),
            )
        print(f"prior_ablation {name}: heldout_ppl={ppl:.4f} bpc={bpc:.4f} sec={scores[name]['eval_sec']:.1f}", flush=True)

    targets = eval_states["no_prior"][1]
    counts = torch.bincount(train_ids.cpu(), minlength=model.gru_head.out_features).double() + 1.0
    unigram = float(-(counts / counts.sum()).log()[targets].mean())
    result = {
        "checkpoint": str(args.resume_checkpoint),
        "eval_chars": int(len(targets)),
        "scores": scores,
        "char_frequency_ppl": math.exp(unigram),
        "probe": {
            "train_chars": args.probe_train_chars,
            "streams": args.train_streams,
            "steps": args.probe_steps,
            "batch": args.probe_batch,
            "lr": args.probe_lr,
            "seed": args.probe_seed,
        },
        "probes": {},
    }
    for name, mode in (("prior_in_input", "none"), ("no_prior", "all")):
        t0 = time.perf_counter()
        model.prior_ablation = mode
        try:
            train_states, train_targets = collect_train_states(
                model, train_ids, train_segments, args.train_streams, 64, args.probe_train_chars, **common,
                **train_rows,
            )
        finally:
            model.prior_ablation = "none"
        print(f"probe {name}: {len(train_states)} training states in {time.perf_counter() - t0:.1f}s", flush=True)
        result["probes"][name] = train_probe(
            model.gru_head, train_states, train_targets.cpu(), eval_states[name][0], eval_states[name][1].cpu(),
            args.probe_steps, args.probe_batch, args.probe_lr, args.probe_seed,
        )
        del train_states, train_targets
    return result


def row_stats(stats: dict[str, float]) -> tuple[float | None, float | None, float | None]:
    return (
        mean_or_none(stats["gate_sum"], stats["gate_count"]),
        mean_or_none(stats["context_norm_sum"], stats["token_count"]),
        mean_or_none(stats["delta_norm_sum"], stats["token_count"]),
    )


def ppl_from_loss(total_loss: float, count: float) -> float | None:
    if count <= 0:
        return None
    nll = total_loss / count
    if nll > 700.0:
        return float("inf")
    return math.exp(nll)


def diagnostic_row_stats(
    stats: dict[str, float],
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    return (
        ppl_from_loss(stats["no_memory_loss"], stats["diagnostic_token_count"]),
        ppl_from_loss(stats["context_only_loss"], stats["diagnostic_token_count"]),
        mean_or_none(stats["attn_entropy_sum"], stats["attn_token_count"]),
        mean_or_none(stats["attn_top_weight_sum"], stats["attn_token_count"]),
        mean_or_none(stats["attn_char_distance_sum"], stats["attn_token_count"]),
        mean_or_none(stats["attn_prev_weight_sum"], stats["attn_token_count"]),
        mean_or_none(stats["attn_current_weight_sum"], stats["attn_token_count"]),
        mean_or_none(stats["attn_top_is_prev_sum"], stats["attn_token_count"]),
        ppl_from_loss(stats["first_token_loss"], stats["first_token_count"]),
        ppl_from_loss(stats["later_token_loss"], stats["later_token_count"]),
    )


def save_checkpoint(
    path: Path,
    model: SegmentMemoryLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    vocab: CharVocab,
    args: argparse.Namespace,
    best_val_ppl: float | None = None,
    best_epoch: int | None = None,
    best_step: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "vocab_chars": vocab.chars,
        "args": vars(args),
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "best_step": best_step,
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    args = parse_args()
    check_cross_fit_args(args)
    if args.eval_only:
        if args.resume_checkpoint is None:
            raise ValueError("--eval-only scores a trained model: give --resume-checkpoint")
        args.eval_initial = True
    if args.prior_ablation_output is not None and not args.eval_only:
        raise ValueError("--prior-ablation-output scores a trained model: give --eval-only and --resume-checkpoint")
    args.run_id = resolve_run_id(args.run_id, resume=args.resume_checkpoint is not None)
    args.output = prefixed_path(args.output, args.run_id)
    if args.checkpoint_output is None:
        args.checkpoint_output = args.output.with_suffix(".pt")
    else:
        args.checkpoint_output = prefixed_path(args.checkpoint_output, args.run_id)
    if args.best_checkpoint_output is None:
        args.best_checkpoint_output = args.output.with_suffix(".best.pt")
    else:
        args.best_checkpoint_output = prefixed_path(args.best_checkpoint_output, args.run_id)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    text = read_text(args.input)
    split = None if args.train_input is not None and args.eval_input is not None else split_text(text, train_fraction=args.train_fraction)
    train_source_text = read_text(args.train_input) if args.train_input is not None else split.train_text
    if args.max_train_chars is not None:
        train_text = train_source_text[: args.max_train_chars]
    else:
        train_text = train_source_text
    if args.eval_input is not None:
        eval_source_text = read_text(args.eval_input)
    else:
        if split is None:
            split = split_text(text, train_fraction=args.train_fraction)
        eval_source_text = split.val_text
    eval_text = eval_source_text[: args.eval_max_chars] if args.eval_max_chars is not None else eval_source_text
    vocab_text = read_text(args.vocab_input) if args.vocab_input is not None else text
    vocab = CharVocab.from_text(vocab_text)
    train_ids_list = vocab.encode(train_text)
    eval_ids_list = vocab.encode(eval_text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_storage_device = torch.device("cpu") if args.count_prior_storage in {"cpu", "mmap"} else device
    prior_log_prob_dtype = (
        torch.float16 if args.count_prior_log_prob_dtype == "float16" else torch.float32
    )
    train_ids = torch.tensor(train_ids_list, dtype=torch.long, device=device)
    eval_ids = torch.tensor(eval_ids_list, dtype=torch.long, device=device)
    # What held-out scoring predicts (eval_score_ids) and reads (eval_input_ids,
    # None for the targets themselves).
    eval_score_ids = eval_ids
    eval_input_ids: torch.Tensor | None = None
    if args.eval_clean_targets is not None and args.eval_residual_input is not None:
        raise ValueError("--eval-clean-targets and --eval-residual-input are two different tests: give one")

    def aligned_eval_ids(path: Path, flag: str) -> list[int]:
        source = read_text(path)
        other = vocab.encode(source[: args.eval_max_chars] if args.eval_max_chars is not None else source)
        if len(other) != len(eval_ids_list):
            raise ValueError(
                f"{flag} has {len(other)} characters, --eval-input {len(eval_ids_list)}: "
                "they must line up (substitutions only)"
            )
        changed = sum(a != b for a, b in zip(other, eval_ids_list))
        print(f"{flag.lstrip('-').replace('-', '_')}={path} eval_chars_changed={changed}", flush=True)
        return other

    if args.eval_clean_targets is not None:
        eval_score_ids = torch.tensor(aligned_eval_ids(args.eval_clean_targets, "--eval-clean-targets"), dtype=torch.long, device=device)
        eval_input_ids = eval_ids
    if args.eval_residual_input is not None:
        eval_input_ids = torch.tensor(aligned_eval_ids(args.eval_residual_input, "--eval-residual-input"), dtype=torch.long, device=device)
    train_unigram = unigram_probs(train_ids_list, vocab.size) if args.input_noise > 0.0 else None
    train_prior_log_probs: PriorRows | None = None
    eval_prior_log_probs: torch.Tensor | None = None
    train_prior_gate_features: PriorRows | None = None
    eval_prior_gate_features: torch.Tensor | None = None
    train_skip_prior_log_probs: PriorRows | None = None
    eval_skip_prior_log_probs: torch.Tensor | None = None
    train_skip_prior_strength: PriorRows | None = None
    eval_skip_prior_strength: torch.Tensor | None = None
    prior_gate_feature_count = 0
    prior_feature_names: list[str] | None = None
    if args.prior_rows is not None:
        if args.count_prior != "frozen":
            raise ValueError("--prior-rows replaces the frozen count prior: it needs --count-prior frozen")
        if args.skip_horizon_prior != "none" or args.skip_horizon_features != "none" or args.transition_category_features != "none":
            raise ValueError("--prior-rows doesn't support skip-horizon or transition-category features")
    if args.count_prior == "frozen":
        prior_started = time.perf_counter()
        count_history: list[dict[str, float]] = []
        # Only imm's own prior is cached; with --prior-rows its theta file
        # (a config default) needn't exist.
        cache_meta = None if args.prior_rows is not None else count_prior_cache_meta(args, train_ids_list, eval_ids_list, vocab)
        cache_loaded = False
        if args.prior_rows is not None:
            (
                train_prior_log_probs,
                eval_prior_log_probs,
                train_prior_gate_features,
                eval_prior_gate_features,
                prior_feature_names,
                prior_rows_meta,
            ) = load_prior_rows(args.prior_rows, train_ids_list, eval_ids_list, vocab, device)
            if train_prior_gate_features is not None:
                prior_gate_feature_count = int(train_prior_gate_features.shape[1])
        elif args.count_prior_cache is not None and args.count_prior_cache.exists():
            cache = torch.load(args.count_prior_cache, map_location="cpu", weights_only=False)
            if count_prior_cache_matches(cache, cache_meta):
                if cache.get("format") == "mmap":
                    train_prior_log_probs = load_prior_mmap(
                        Path(cache["train_log_probs_path"]),
                        cache["log_prob_dtype"],
                        tuple(cache["train_log_probs_shape"]),
                    )
                    eval_prior_log_probs_mmap = load_prior_mmap(
                        Path(cache["eval_log_probs_path"]),
                        cache["log_prob_dtype"],
                        tuple(cache["eval_log_probs_shape"]),
                    )
                    eval_prior_log_probs = prior_rows_to_device(
                        eval_prior_log_probs_mmap,
                        0,
                        eval_prior_log_probs_mmap.shape[0],
                        device,
                    )
                    if "train_gate_features_path" in cache and "eval_gate_features_path" in cache:
                        train_prior_gate_features = load_prior_mmap(
                            Path(cache["train_gate_features_path"]),
                            cache["feature_dtype"],
                            tuple(cache["train_gate_features_shape"]),
                        )
                        eval_prior_gate_features_mmap = load_prior_mmap(
                            Path(cache["eval_gate_features_path"]),
                            cache["feature_dtype"],
                            tuple(cache["eval_gate_features_shape"]),
                        )
                        eval_prior_gate_features = prior_rows_to_device(
                            eval_prior_gate_features_mmap,
                            0,
                            eval_prior_gate_features_mmap.shape[0],
                            device,
                        )
                        prior_gate_feature_count = int(train_prior_gate_features.shape[1])
                    count_history = cache.get("history", [])
                    cache_loaded = True
                else:
                    train_prior_log_probs = cache["train_log_probs"].to(prior_storage_device)
                    eval_prior_log_probs = cache["eval_log_probs"].to(device)
                    if "train_gate_features" in cache and "eval_gate_features" in cache:
                        train_prior_gate_features = cache["train_gate_features"].to(prior_storage_device)
                        eval_prior_gate_features = cache["eval_gate_features"].to(device)
                        prior_gate_feature_count = int(train_prior_gate_features.shape[1])
                    count_history = cache.get("history", [])
                    cache_loaded = True
            else:
                print(f"count_prior_cache_miss={args.count_prior_cache}", flush=True)
        if train_prior_log_probs is None or eval_prior_log_probs is None:
            count_model, count_theta, count_history = fit_count_gate_prior(
                train_ids_list,
                vocab.size,
                args.count_prior_depth,
                args.count_prior_valid_ratio,
                args.count_prior_epochs,
                args.count_prior_lr,
                args.count_prior_max_fit_positions,
                args.count_prior_extra_features,
                args.seed,
                args.count_prior_impl,
                args.count_prior_theta_json,
            )
            if args.count_prior_storage == "mmap":
                if args.count_prior_cache is None:
                    raise ValueError("--count-prior-storage mmap requires --count-prior-cache")
                train_log_path = prior_mmap_sidecar_path(args.count_prior_cache, "train_log_probs")
                eval_log_path = prior_mmap_sidecar_path(args.count_prior_cache, "eval_log_probs")
                train_feature_path = prior_mmap_sidecar_path(args.count_prior_cache, "train_gate_features")
                eval_feature_path = prior_mmap_sidecar_path(args.count_prior_cache, "eval_gate_features")
                if args.count_prior_precompute == "numpy":
                    if not isinstance(count_model, PackedCountModel):
                        raise ValueError("--count-prior-precompute numpy requires --count-prior-impl packed")
                    with tempfile.TemporaryDirectory(
                        dir=args.count_prior_cache.parent,
                        prefix=f"{args.count_prior_cache.stem}.depth_cache.",
                    ) as depth_cache_dir:
                        depth_prior_tables = build_depth_prior_tables(count_model, count_theta, depth_cache_dir)
                        if args.count_prior_train_folds > 0:
                            fold_started = time.perf_counter()
                            train_prior_log_probs, train_prior_gate_features = compile_cross_fit_rows_numpy_mmap(
                                train_ids_list,
                                vocab.size,
                                args.count_prior_depth,
                                expand_extra_features(args.count_prior_extra_features),
                                count_theta,
                                args.count_prior_train_folds,
                                args.count_prior_fold_chunk,
                                train_log_path,
                                train_feature_path,
                                log_dtype=numpy_dtype_for_torch(prior_log_prob_dtype),
                                cache_dir=Path(depth_cache_dir) / "folds",
                                progress=lambda fold, folds: print(
                                    f"count_prior_cross_fit fold={fold + 1}/{folds} "
                                    f"sec={time.perf_counter() - fold_started:.1f}",
                                    flush=True,
                                ),
                            )
                        else:
                            train_prior_log_probs, train_prior_gate_features = compile_prior_rows_numpy_mmap(
                                train_ids_list,
                                count_model,
                                depth_prior_tables,
                                train_log_path,
                                train_feature_path,
                                log_dtype=numpy_dtype_for_torch(prior_log_prob_dtype),
                            )
                        eval_prior_log_probs_mmap, eval_prior_gate_features_mmap = compile_prior_rows_numpy_mmap(
                            eval_ids_list,
                            count_model,
                            depth_prior_tables,
                            eval_log_path,
                            eval_feature_path,
                            log_dtype=numpy_dtype_for_torch(prior_log_prob_dtype),
                        )
                        del depth_prior_tables
                else:
                    train_prior_log_probs, train_prior_gate_features = precompute_count_prior_rows_mmap(
                        train_ids_list,
                        count_model,
                        count_theta,
                        args.count_prior_depth,
                        train_log_path,
                        train_feature_path,
                        prior_log_prob_dtype,
                        clear_cache_every=args.count_prior_precompute_cache_clear_every,
                    )
                    eval_prior_log_probs_mmap, eval_prior_gate_features_mmap = precompute_count_prior_rows_mmap(
                        eval_ids_list,
                        count_model,
                        count_theta,
                        args.count_prior_depth,
                        eval_log_path,
                        eval_feature_path,
                        prior_log_prob_dtype,
                        clear_cache_every=args.count_prior_precompute_cache_clear_every,
                    )
                eval_prior_log_probs = prior_rows_to_device(
                    eval_prior_log_probs_mmap,
                    0,
                    eval_prior_log_probs_mmap.shape[0],
                    device,
                )
                eval_prior_gate_features = prior_rows_to_device(
                    eval_prior_gate_features_mmap,
                    0,
                    eval_prior_gate_features_mmap.shape[0],
                    device,
                )
            else:
                train_prior_log_probs = precompute_count_prior_log_probs(
                    train_ids_list,
                    count_model,
                    count_theta,
                    args.count_prior_depth,
                    prior_storage_device,
                    prior_log_prob_dtype,
                )
                eval_prior_log_probs = precompute_count_prior_log_probs(
                    eval_ids_list,
                    count_model,
                    count_theta,
                    args.count_prior_depth,
                    device,
                    prior_log_prob_dtype,
                )
                train_prior_gate_features = precompute_count_prior_feature_rows(
                    train_ids_list,
                    count_model,
                    args.count_prior_depth,
                    prior_storage_device,
                )
                eval_prior_gate_features = precompute_count_prior_feature_rows(
                    eval_ids_list,
                    count_model,
                    args.count_prior_depth,
                    device,
                )
            prior_gate_feature_count = int(train_prior_gate_features.shape[1])
            if args.count_prior_cache is not None:
                args.count_prior_cache.parent.mkdir(parents=True, exist_ok=True)
                tmp_cache = args.count_prior_cache.with_suffix(f"{args.count_prior_cache.suffix}.tmp")
                if args.count_prior_storage == "mmap":
                    torch.save(
                        {
                            "meta": cache_meta,
                            "format": "mmap",
                            "history": count_history,
                            "log_prob_dtype": numpy_dtype_for_torch(prior_log_prob_dtype).name,
                            "feature_dtype": "float32",
                            "train_log_probs_path": str(train_log_path),
                            "eval_log_probs_path": str(eval_log_path),
                            "train_gate_features_path": str(train_feature_path),
                            "eval_gate_features_path": str(eval_feature_path),
                            "train_log_probs_shape": tuple(train_prior_log_probs.shape),
                            "eval_log_probs_shape": tuple(eval_prior_log_probs_mmap.shape),
                            "train_gate_features_shape": tuple(train_prior_gate_features.shape),
                            "eval_gate_features_shape": tuple(eval_prior_gate_features_mmap.shape),
                            "feature_count": prior_gate_feature_count,
                        },
                        tmp_cache,
                    )
                else:
                    torch.save(
                        {
                            "meta": cache_meta,
                            "history": count_history,
                            "train_log_probs": train_prior_log_probs.detach().cpu(),
                            "eval_log_probs": eval_prior_log_probs.detach().cpu(),
                            "train_gate_features": train_prior_gate_features.detach().cpu(),
                            "eval_gate_features": eval_prior_gate_features.detach().cpu(),
                        },
                        tmp_cache,
                    )
                tmp_cache.replace(args.count_prior_cache)
            del count_model, count_theta
            gc.collect()
        if args.prior_residual_gate == "context" and (
            train_prior_gate_features is None or eval_prior_gate_features is None
        ):
            raise ValueError("context residual gate requires count-prior feature rows; rebuild the count prior cache")
        if args.prior_rows is not None:
            print(
                f"prior_rows={args.prior_rows} builder={prior_rows_meta.get('builder')!r} "
                f"train_shape={tuple(train_prior_log_probs.shape)} eval_shape={tuple(eval_prior_log_probs.shape)} "
                f"gate_features={prior_feature_names} cross_fit={json.dumps(prior_rows_meta.get('cross_fit'))}",
                flush=True,
            )
        else:
            print(
                f"count_prior=frozen depth={args.count_prior_depth} "
                f"impl={args.count_prior_impl} "
                f"theta_json={args.count_prior_theta_json} "
                f"features={args.count_prior_extra_features} "
                f"fit_ppl={count_history[-1]['ppl'] if count_history else float('nan'):.3f} "
                f"cache_loaded={cache_loaded} cache={args.count_prior_cache} "
                f"build_precompute_sec={time.perf_counter() - prior_started:.2f}",
                flush=True,
            )
        if args.skip_horizon_prior != "none":
            if train_prior_log_probs is None or eval_prior_log_probs is None:
                raise ValueError("skip horizon prior requires count-prior log-prob rows")
            (
                train_prior_log_probs,
                eval_prior_log_probs,
                train_skip_prior_log_probs,
                eval_skip_prior_log_probs,
                train_skip_prior_strength,
                eval_skip_prior_strength,
            ) = add_skip_horizon_prior(
                train_ids_list,
                eval_ids_list,
                vocab,
                train_prior_log_probs,
                eval_prior_log_probs,
                args,
                device,
            )
        if args.transition_category_features != "none":
            if (
                train_prior_log_probs is None
                or eval_prior_log_probs is None
                or train_prior_gate_features is None
                or eval_prior_gate_features is None
            ):
                raise ValueError("transition category features require count-prior log-prob and feature rows")
            train_prior_gate_features, eval_prior_gate_features, prior_gate_feature_count = add_transition_category_features(
                train_prior_log_probs,
                eval_prior_log_probs,
                train_prior_gate_features,
                eval_prior_gate_features,
                args,
                device,
            )
        if args.skip_horizon_features != "none":
            if train_prior_gate_features is None or eval_prior_gate_features is None:
                raise ValueError("skip horizon features require count-prior feature rows")
            train_prior_gate_features, eval_prior_gate_features, prior_gate_feature_count = add_skip_horizon_features(
                train_ids_list,
                eval_ids_list,
                vocab,
                train_prior_gate_features,
                eval_prior_gate_features,
                args,
                device,
            )
    elif args.transition_category_features != "none":
        raise ValueError("--transition-category-features requires --count-prior frozen")
    elif args.skip_horizon_features != "none":
        raise ValueError("--skip-horizon-features requires --count-prior frozen")
    elif args.skip_horizon_prior != "none":
        raise ValueError("--skip-horizon-prior requires --count-prior frozen")
    if args.prior_residual_gate != "none" and args.count_prior != "frozen":
        raise ValueError("--prior-residual-gate requires --count-prior frozen")

    space_ids = [i for i, ch in enumerate(vocab.chars) if ch.isspace()]
    train_segments, eval_segments = rule_segments(
        args.segment_rule, train_ids_list, eval_ids_list, args.max_depth, args.unique_threshold, space_ids
    )
    mean_segment_len, max_segment_len = segment_stats(train_segments)

    prior_residual_gate_max = (
        args.prior_residual_gate_max if args.prior_residual_gate_max > 0.0 else 2.0 * args.prior_residual_scale
    )
    prior_residual_state_features = parse_feature_indices(args.prior_residual_state_features, prior_feature_names)
    if prior_residual_state_features and prior_gate_feature_count <= 0:
        raise ValueError("--prior-residual-state-features requires count-prior feature rows")
    if any(not 0 <= i < prior_gate_feature_count for i in prior_residual_state_features):
        raise ValueError(
            f"--prior-residual-state-features {prior_residual_state_features} outside the {prior_gate_feature_count} feature columns"
        )
    learned_skip_prior_modes = {"learned-distribution", "learned-horizon-distribution"}
    if args.skip_horizon_prior in learned_skip_prior_modes and prior_gate_feature_count <= 0:
        raise ValueError(f"--skip-horizon-prior {args.skip_horizon_prior} requires count-prior feature rows")
    if args.skip_horizon_prior not in learned_skip_prior_modes and (
        train_skip_prior_log_probs is not None
        or eval_skip_prior_log_probs is not None
        or train_skip_prior_strength is not None
        or eval_skip_prior_strength is not None
    ):
        raise ValueError("internal error: skip prior tensors should only be set for learned skip-prior modes")
    skip_prior_gate = (
        "horizon-context"
        if args.skip_horizon_prior == "learned-horizon-distribution"
        else "context"
        if args.skip_horizon_prior == "learned-distribution"
        else "none"
    )
    skip_prior_channels = args.skip_horizon_max if args.skip_horizon_prior == "learned-horizon-distribution" else 0
    invalid_state_features = [
        index for index in prior_residual_state_features if index < 0 or index >= prior_gate_feature_count
    ]
    if invalid_state_features:
        raise ValueError(
            f"invalid --prior-residual-state-features {invalid_state_features}; "
            f"feature count is {prior_gate_feature_count}"
        )
    if args.prior_input != "none" and args.count_prior != "frozen":
        raise ValueError("--prior-input requires --count-prior frozen")
    if args.prior_input != "none" and args.token_feedback != "none":
        raise ValueError("--prior-input is not implemented for --token-feedback")
    train_prior_evidence: PriorRows | None = None
    eval_prior_evidence: torch.Tensor | None = None
    prior_evidence_width = 0
    if args.prior_evidence_rows is not None and args.prior_input not in ("depth-evidence", "both"):
        raise ValueError("--prior-evidence-rows needs --prior-input depth-evidence or both")
    if args.prior_input in ("depth-evidence", "both"):
        if args.prior_evidence_rows is not None:
            train_prior_evidence, eval_prior_evidence = load_external_depth_evidence(
                args.prior_evidence_rows, train_ids_list, eval_ids_list, vocab, device
            )
        else:
            train_prior_evidence, eval_prior_evidence = load_or_build_depth_evidence(
                args, train_ids_list, eval_ids_list, vocab, device
            )
        prior_evidence_width = int(train_prior_evidence.shape[1] * train_prior_evidence.shape[2])
    model = SegmentMemoryLM(
        vocab.size,
        args.embedding_size,
        args.hidden_size,
        use_rope=not args.no_rope,
        rope_positions=args.rope_positions,
        memory_state=args.memory_state,
        memory_record=args.memory_record,
        rnn_core=args.rnn_core,
        attn_interface_norm=args.attn_interface_norm,
        mixing=args.mixing,
        attention_layers=args.attention_layers,
        attention_heads=args.attention_heads,
        input_rope=args.input_rope,
        feedback_gate_bias=args.feedback_gate_bias,
        feedback_delta_cap=args.feedback_delta_cap,
        carry_bound=args.carry_bound,
        carry_scale=args.carry_scale,
        prior_strength=args.prior_strength,
        prior_residual_scale=args.prior_residual_scale,
        prior_residual_l2=args.prior_residual_l2,
        prior_residual_gate=args.prior_residual_gate,
        prior_residual_gate_max=prior_residual_gate_max,
        prior_residual_gate_features=prior_gate_feature_count,
        prior_residual_state_features=prior_residual_state_features,
        skip_prior_gate=skip_prior_gate,
        skip_prior_gate_max=args.skip_horizon_prior_gate_max,
        skip_prior_gate_bias=args.skip_horizon_prior_gate_bias,
        skip_prior_channels=skip_prior_channels,
        record_aux_weight=args.record_aux_weight,
        terminal_record_aux_weight=args.terminal_record_aux_weight,
        retrieval_aux_weight=args.retrieval_aux_weight,
        utility_aux_weight=args.utility_aux_weight,
        utility_temperature=args.utility_temperature,
        prior_input=args.prior_input,
        prior_evidence_width=prior_evidence_width,
        dropout=args.dropout,
        prior_dropout=args.prior_dropout,
    ).to(device)
    if args.count_prior == "frozen":
        model.zero_prediction_heads()
    if args.train_streams < 1:
        raise ValueError("--train-streams must be at least 1")
    if args.prior_dropout > 0.0 and args.train_streams < 2:
        raise ValueError("--prior-dropout is built for the batched path: give --train-streams > 1")
    if args.train_streams > 1:
        model.check_streams_supported(args.token_feedback)
        stream_segments = split_streams(train_segments, args.train_streams)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    step = 0
    best_val_ppl: float | None = None
    best_epoch: int | None = None
    best_step: int | None = None
    checkpoint = load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint is not None else None
    if checkpoint is not None:
        if tuple(checkpoint["vocab_chars"]) != vocab.chars:
            raise ValueError("checkpoint vocab does not match input text vocab")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        completed_epoch = int(checkpoint["epoch"])
        step = int(checkpoint["step"])
        start_epoch = completed_epoch + 1
        best_val_ppl = checkpoint.get("best_val_ppl")
        best_epoch = checkpoint.get("best_epoch")
        best_step = checkpoint.get("best_step")
        print(
            f"resumed checkpoint={args.resume_checkpoint} completed_epoch={completed_epoch} "
            f"start_epoch={start_epoch} step={step}",
            flush=True,
        )

    if args.prior_ablation_output is not None:
        t0 = time.perf_counter()
        result = run_prior_ablation(
            model,
            args,
            eval_score_ids,
            eval_segments,
            {
                "prior_log_probs": eval_prior_log_probs,
                "prior_gate_features": eval_prior_gate_features,
                "prior_evidence": eval_prior_evidence,
            },
            eval_input_ids,
            train_ids,
            train_segments,
            {
                "prior_log_probs": train_prior_log_probs,
                "prior_gate_features": train_prior_gate_features,
                "prior_evidence": train_prior_evidence,
            },
        )
        result["run_id"] = args.run_id
        result["runtime_sec"] = time.perf_counter() - t0
        args.prior_ablation_output.parent.mkdir(parents=True, exist_ok=True)
        args.prior_ablation_output.write_text(json.dumps(result, indent=2))
        summary = {k: round(v["heldout_ppl"], 4) for k, v in result["scores"].items()}
        summary.update({f"probe_{k}": round(v["heldout_ppl"], 4) for k, v in result["probes"].items()})
        print(f"prior_ablation {summary} -> {args.prior_ablation_output}", flush=True)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(
        asdict(
            SegmentMemoryRow(
                epoch=0,
                step=0,
                train_nll=None,
                val_nll=None,
                val_ppl=None,
                val_bpc=None,
                segment_count=0,
                mean_segment_len=0.0,
                max_segment_len=0,
                mean_gate=None,
                mean_context_norm=None,
                mean_delta_norm=None,
                no_memory_ppl=None,
                context_only_ppl=None,
                mean_attn_entropy=None,
                mean_attn_top_weight=None,
                mean_attn_char_distance=None,
                mean_attn_prev_weight=None,
                mean_attn_current_weight=None,
                mean_attn_top_is_prev=None,
                first_token_ppl=None,
                later_token_ppl=None,
                runtime_sec=0.0,
                peak_rss_kb=0,
            )
        ).keys()
    )
    append_output = args.resume_checkpoint is not None and args.output.exists()
    with args.output.open("a" if append_output else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append_output:
            writer.writeheader()
        print(
            f"train_chars={len(train_ids_list)} train_segments={len(train_segments)} "
            f"eval_chars={len(eval_ids_list)} eval_segments={len(eval_segments)} "
            f"mean_segment_len={mean_segment_len:.2f} max_segment_len={max_segment_len} "
            f"memory_state={args.memory_state} rope_positions={args.rope_positions} "
            f"memory_record={args.memory_record} rnn_core={args.rnn_core} "
            f"attn_interface_norm={args.attn_interface_norm} "
            f"mixing={args.mixing} attention_layers={args.attention_layers} attention_heads={args.attention_heads} "
            f"carry_hidden={args.carry_hidden} train_streams={args.train_streams} "
            f"feedback_state={args.feedback_state} carry_bound={args.carry_bound} carry_scale={args.carry_scale} token_feedback={args.token_feedback} input_rope={args.input_rope} "
            f"feedback_gate_bias={args.feedback_gate_bias} feedback_delta_cap={args.feedback_delta_cap} "
            f"prior_strength={args.prior_strength} "
            f"prior_residual_scale={args.prior_residual_scale} prior_residual_l2={args.prior_residual_l2} "
            f"prior_residual_gate={args.prior_residual_gate} prior_residual_gate_max={prior_residual_gate_max} "
            f"prior_residual_state_features={prior_residual_state_features} "
            f"count_prior_storage={args.count_prior_storage} count_prior_log_prob_dtype={args.count_prior_log_prob_dtype} "
            f"count_prior_precompute={args.count_prior_precompute} "
            f"record_aux_weight={args.record_aux_weight} terminal_record_aux_weight={args.terminal_record_aux_weight} "
            f"retrieval_aux_weight={args.retrieval_aux_weight} "
            f"utility_aux_weight={args.utility_aux_weight} utility_temperature={args.utility_temperature} "
            f"count_prior={args.count_prior} count_prior_depth={args.count_prior_depth} "
            f"count_prior_impl={args.count_prior_impl} "
            f"count_prior_theta_json={args.count_prior_theta_json} "
            f"skip_horizon_features={args.skip_horizon_features} "
            f"skip_horizon_depth={args.skip_horizon_depth if args.skip_horizon_depth > 0 else args.count_prior_depth} "
            f"skip_horizon_max={args.skip_horizon_max} skip_horizon_db={args.skip_horizon_db} "
            f"skip_horizon_cache={args.skip_horizon_cache} "
            f"skip_horizon_prior={args.skip_horizon_prior} "
            f"skip_horizon_prior_weight={args.skip_horizon_prior_weight} "
            f"skip_horizon_prior_gate_max={args.skip_horizon_prior_gate_max} "
            f"skip_horizon_prior_gate_bias={args.skip_horizon_prior_gate_bias} "
            f"skip_horizon_count_db={args.skip_horizon_count_db} "
            f"checkpoint_output={args.checkpoint_output} best_checkpoint_output={args.best_checkpoint_output} "
            f"profile={args.profile} "
            f"device={device} output={args.output}",
            flush=True,
        )
        if args.eval_initial:
            t0 = time.perf_counter()
            val_nll, val_ppl, val_bpc, val_stats = evaluate(
                model,
                eval_score_ids,
                eval_segments,
                args.max_memory,
                carry_hidden=args.carry_hidden,
                feedback_state=args.feedback_state,
                token_feedback=args.token_feedback,
                prior_log_probs=eval_prior_log_probs,
                prior_gate_features=eval_prior_gate_features,
                skip_prior_log_probs=eval_skip_prior_log_probs,
                skip_prior_strength=eval_skip_prior_strength,
                prior_evidence=eval_prior_evidence,
                input_ids=eval_input_ids,
                streams=args.eval_streams,
                burn_in=args.eval_burn_in,
            )
            mean_gate, mean_context_norm, mean_delta_norm = row_stats(val_stats)
            mean_residual_alpha = mean_or_none(val_stats["residual_alpha_sum"], val_stats["residual_alpha_count"])
            mean_skip_prior_alpha = mean_or_none(
                val_stats["skip_prior_alpha_sum"],
                val_stats["skip_prior_alpha_count"],
            )
            (
                no_memory_ppl,
                context_only_ppl,
                mean_attn_entropy,
                mean_attn_top_weight,
                mean_attn_char_distance,
                mean_attn_prev_weight,
                mean_attn_current_weight,
                mean_attn_top_is_prev,
                first_token_ppl,
                later_token_ppl,
            ) = diagnostic_row_stats(val_stats)
            row = SegmentMemoryRow(
                epoch=start_epoch - 1 if args.eval_only else 0,
                step=step,
                train_nll=None,
                val_nll=val_nll,
                val_ppl=val_ppl,
                val_bpc=val_bpc,
                segment_count=len(train_segments),
                mean_segment_len=mean_segment_len,
                max_segment_len=max_segment_len,
                mean_gate=mean_gate,
                mean_context_norm=mean_context_norm,
                mean_delta_norm=mean_delta_norm,
                no_memory_ppl=no_memory_ppl,
                context_only_ppl=context_only_ppl,
                mean_attn_entropy=mean_attn_entropy,
                mean_attn_top_weight=mean_attn_top_weight,
                mean_attn_char_distance=mean_attn_char_distance,
                mean_attn_prev_weight=mean_attn_prev_weight,
                mean_attn_current_weight=mean_attn_current_weight,
                mean_attn_top_is_prev=mean_attn_top_is_prev,
                first_token_ppl=first_token_ppl,
                later_token_ppl=later_token_ppl,
                runtime_sec=time.perf_counter() - started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                mean_residual_alpha=mean_residual_alpha,
                mean_skip_prior_alpha=mean_skip_prior_alpha,
            )
            writer.writerow(asdict(row))
            handle.flush()
            print(
                f"epoch=0 initial step={step} val_ppl={val_ppl:.3f} "
                f"eval_sec={time.perf_counter() - t0:.2f} runtime_sec={row.runtime_sec:.2f} "
                f"peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
                flush=True,
            )
            if args.eval_only:
                print(f"eval_only checkpoint={args.resume_checkpoint} val_ppl={val_ppl:.4f}", flush=True)
                return
            if best_val_ppl is None or val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_epoch = 0
                best_step = step
                save_checkpoint(
                    args.best_checkpoint_output,
                    model,
                    optimizer,
                    0,
                    step,
                    vocab,
                    args,
                    best_val_ppl=best_val_ppl,
                    best_epoch=best_epoch,
                    best_step=best_step,
                )
                print(
                    f"best_checkpoint={args.best_checkpoint_output} "
                    f"best_epoch={best_epoch} best_step={best_step} best_val_ppl={best_val_ppl:.3f}",
                    flush=True,
                )
        if args.lr_schedule == "cosine" and not (0.0 <= args.min_lr <= args.lr and args.lr_warmup >= 0):
            raise ValueError("--lr-schedule cosine needs 0 <= --min-lr <= --lr and --lr-warmup >= 0")
        updates_per_epoch = math.ceil(
            (len(train_segments) if args.train_streams == 1 else len(stream_segments[0])) / args.segments_per_update
        )
        total_updates = args.epochs * updates_per_epoch
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            memory: list[MemoryEntry] | StreamMemory | None = [] if args.train_streams == 1 else None
            carried_hidden: torch.Tensor | None = None
            epoch_profile = EpochProfile()
            epoch_loss_sum = 0.0
            train_input_ids: torch.Tensor | None = None
            if train_unigram is not None:
                noisy, changed = substitute(
                    train_ids_list, args.input_noise, train_unigram, np.random.default_rng([args.seed, epoch])
                )
                train_input_ids = torch.from_numpy(noisy).to(device)
                print(f"epoch={epoch} input_noise={args.input_noise} chars_changed={changed}", flush=True)
            stream_length = len(train_segments) if args.train_streams == 1 else len(stream_segments[0])
            prefetcher = None
            pin_rows = args.pin_rows and args.row_gather == "vector" and device.type == "cuda"
            if args.train_streams > 1 and args.prefetch_rows:
                if args.prefetch_ahead < 1:
                    raise ValueError("--prefetch-ahead must be at least 1")
                prefetcher = RowPrefetcher(
                    {
                        "prior_log_probs": train_prior_log_probs,
                        "prior_gate_features": train_prior_gate_features,
                        "prior_evidence": train_prior_evidence,
                    },
                    drop_cache=("prior_evidence",) if args.drop_evidence_cache else (),
                    workers=args.prefetch_ahead,
                    gather=args.row_gather,
                    pin=pin_rows,
                )

                def update_spans(at: int) -> list[tuple[int, int]]:
                    last = min(at + args.segments_per_update, stream_length) - 1
                    return [(segs[at][0], segs[last][1]) for segs in stream_segments]

                # The next --prefetch-ahead updates' reads, oldest first.
                pending = collections.deque(
                    prefetcher.submit(update_spans(at))
                    for at in range(0, stream_length, args.segments_per_update)[: args.prefetch_ahead]
                )
                next_submit = min(stream_length, args.prefetch_ahead * args.segments_per_update)
            for offset in range(0, stream_length, args.segments_per_update):
                step += 1
                if args.lr_schedule == "cosine":
                    for group in optimizer.param_groups:
                        group["lr"] = scheduled_lr(step - 1, total_updates, args.lr, args.min_lr, args.lr_warmup)
                step_profile = EpochProfile()
                t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                step_profile.zero_sec += time.perf_counter() - t0
                t0 = time.perf_counter()
                if args.train_streams > 1:
                    batch_streams = [segs[offset : offset + args.segments_per_update] for segs in stream_segments]
                    spans = [(segs[0][0], segs[-1][1]) for segs in batch_streams]
                    if prefetcher is not None:
                        t_wait = time.perf_counter()
                        host_rows = pending.popleft().result()
                        step_profile.rows_wait_sec += time.perf_counter() - t_wait
                        if next_submit < stream_length:
                            pending.append(prefetcher.submit(update_spans(next_submit)))
                            next_submit += args.segments_per_update
                        batch_rows = {
                            name: rows_to_device(rows, device, non_blocking=pin_rows) for name, rows in host_rows.items()
                        }
                    else:
                        batch_rows = {
                            "prior_log_probs": prior_rows_for_spans(train_prior_log_probs, spans, device),
                            "prior_gate_features": prior_rows_for_spans(train_prior_gate_features, spans, device),
                            "prior_evidence": prior_rows_for_spans(train_prior_evidence, spans, device),
                        }
                    step_profile.rows_sec += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    loss, tokens, memory, carried_hidden, train_stats = model.segment_loss_streams(
                        train_ids,
                        batch_streams,
                        memory,
                        args.max_memory,
                        carried_hidden=carried_hidden,
                        carry_hidden=args.carry_hidden,
                        feedback_state=args.feedback_state,
                        token_feedback=args.token_feedback,
                        input_ids=train_input_ids,
                        **batch_rows,
                    )
                    batch_segments = batch_streams[0]
                else:
                    batch_segments = train_segments[offset : offset + args.segments_per_update]
                    prior_offset = batch_segments[0][0]
                    prior_end = batch_segments[-1][1]
                    batch_prior_log_probs = prior_rows_to_device(
                        train_prior_log_probs,
                        prior_offset,
                        prior_end,
                        device,
                    )
                    batch_prior_gate_features = prior_rows_to_device(
                        train_prior_gate_features,
                        prior_offset,
                        prior_end,
                        device,
                    )
                    batch_skip_prior_log_probs = prior_rows_to_device(
                        train_skip_prior_log_probs,
                        prior_offset,
                        prior_end,
                        device,
                    )
                    batch_skip_prior_strength = prior_rows_to_device(
                        train_skip_prior_strength,
                        prior_offset,
                        prior_end,
                        device,
                    )
                    batch_prior_evidence = prior_rows_to_device(train_prior_evidence, prior_offset, prior_end, device)
                    loss, tokens, memory, carried_hidden, train_stats = model.segment_loss(
                        train_ids,
                        batch_segments,
                        memory,
                        args.max_memory,
                        carried_hidden=carried_hidden,
                        carry_hidden=args.carry_hidden,
                        feedback_state=args.feedback_state,
                        token_feedback=args.token_feedback,
                        prior_log_probs=batch_prior_log_probs,
                        prior_gate_features=batch_prior_gate_features,
                        skip_prior_log_probs=batch_skip_prior_log_probs,
                        skip_prior_strength=batch_skip_prior_strength,
                        prior_offset=prior_offset,
                        prior_evidence=batch_prior_evidence,
                        input_ids=train_input_ids,
                    )
                step_profile.forward_sec += time.perf_counter() - t0
                if tokens == 0:
                    continue
                step_profile.updates += 1
                step_profile.tokens += tokens
                t0 = time.perf_counter()
                loss_value = float(loss.item())
                step_profile.wait_sec += time.perf_counter() - t0
                train_nll = loss_value / tokens
                epoch_loss_sum += loss_value
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch={epoch} step={step}")
                t0 = time.perf_counter()
                (loss / tokens).backward()
                step_profile.backward_sec += time.perf_counter() - t0
                t0 = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                step_profile.clip_sec += time.perf_counter() - t0
                t0 = time.perf_counter()
                optimizer.step()
                step_profile.optimizer_sec += time.perf_counter() - t0
                t0 = time.perf_counter()
                if args.train_streams > 1:
                    memory = (memory[0].detach(), memory[1])
                else:
                    memory = [(state.detach(), position) for state, position in memory[-args.max_memory:]]
                if carried_hidden is not None:
                    carried_hidden = carried_hidden.detach()
                step_profile.detach_sec += time.perf_counter() - t0
                epoch_profile.add(step_profile)
                if args.progress_every_steps > 0 and step % args.progress_every_steps == 0:
                    print(
                        f"progress epoch={epoch} step={step} offset={offset + len(batch_segments)}/{stream_length} "
                        f"train_nll={train_nll:.4f} {profile_summary(epoch_profile)}",
                        flush=True,
                    )
                if args.eval_every > 0 and step % args.eval_every == 0:
                    t0 = time.perf_counter()
                    val_nll, val_ppl, val_bpc, val_stats = evaluate(
                        model,
                        eval_score_ids,
                        eval_segments,
                        args.max_memory,
                        carry_hidden=args.carry_hidden,
                        feedback_state=args.feedback_state,
                        token_feedback=args.token_feedback,
                        prior_log_probs=eval_prior_log_probs,
                        prior_gate_features=eval_prior_gate_features,
                        skip_prior_log_probs=eval_skip_prior_log_probs,
                        skip_prior_strength=eval_skip_prior_strength,
                        prior_evidence=eval_prior_evidence,
                        input_ids=eval_input_ids,
                        streams=args.eval_streams,
                        burn_in=args.eval_burn_in,
                    )
                    epoch_profile.eval_sec += time.perf_counter() - t0
                    mean_gate, mean_context_norm, mean_delta_norm = row_stats(val_stats)
                    mean_residual_alpha = mean_or_none(
                        val_stats["residual_alpha_sum"],
                        val_stats["residual_alpha_count"],
                    )
                    mean_skip_prior_alpha = mean_or_none(
                        val_stats["skip_prior_alpha_sum"],
                        val_stats["skip_prior_alpha_count"],
                    )
                    (
                        no_memory_ppl,
                        context_only_ppl,
                        mean_attn_entropy,
                        mean_attn_top_weight,
                        mean_attn_char_distance,
                        mean_attn_prev_weight,
                        mean_attn_current_weight,
                        mean_attn_top_is_prev,
                        first_token_ppl,
                        later_token_ppl,
                    ) = diagnostic_row_stats(val_stats)
                    row = SegmentMemoryRow(
                        epoch=epoch,
                        step=step,
                        train_nll=train_nll,
                        val_nll=val_nll,
                        val_ppl=val_ppl,
                        val_bpc=val_bpc,
                        segment_count=len(train_segments),
                        mean_segment_len=mean_segment_len,
                        max_segment_len=max_segment_len,
                        mean_gate=mean_gate,
                        mean_context_norm=mean_context_norm,
                        mean_delta_norm=mean_delta_norm,
                        no_memory_ppl=no_memory_ppl,
                        context_only_ppl=context_only_ppl,
                        mean_attn_entropy=mean_attn_entropy,
                        mean_attn_top_weight=mean_attn_top_weight,
                        mean_attn_char_distance=mean_attn_char_distance,
                        mean_attn_prev_weight=mean_attn_prev_weight,
                        mean_attn_current_weight=mean_attn_current_weight,
                        mean_attn_top_is_prev=mean_attn_top_is_prev,
                        first_token_ppl=first_token_ppl,
                        later_token_ppl=later_token_ppl,
                        runtime_sec=time.perf_counter() - started,
                        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        mean_residual_alpha=mean_residual_alpha,
                        mean_skip_prior_alpha=mean_skip_prior_alpha,
                    )
                    writer.writerow(asdict(row))
                    handle.flush()
                    print(
                        f"epoch={epoch} step={step} train_nll={train_nll:.4f} "
                        f"val_ppl={val_ppl:.3f} "
                        f"mean_residual_alpha={mean_residual_alpha if mean_residual_alpha is not None else float('nan'):.4f} "
                        f"mean_skip_prior_alpha={mean_skip_prior_alpha if mean_skip_prior_alpha is not None else float('nan'):.4f} "
                        f"runtime_sec={row.runtime_sec:.2f} "
                        f"peak_rss_mb={row.peak_rss_kb / 1024:.1f}",
                        flush=True,
                    )
            if prefetcher is not None:
                prefetcher.close()
            t0 = time.perf_counter()
            val_nll, val_ppl, val_bpc, val_stats = evaluate(
                model,
                eval_score_ids,
                eval_segments,
                args.max_memory,
                carry_hidden=args.carry_hidden,
                feedback_state=args.feedback_state,
                token_feedback=args.token_feedback,
                prior_log_probs=eval_prior_log_probs,
                prior_gate_features=eval_prior_gate_features,
                skip_prior_log_probs=eval_skip_prior_log_probs,
                skip_prior_strength=eval_skip_prior_strength,
                prior_evidence=eval_prior_evidence,
                input_ids=eval_input_ids,
                streams=args.eval_streams,
                burn_in=args.eval_burn_in,
            )
            epoch_profile.eval_sec += time.perf_counter() - t0
            mean_gate, mean_context_norm, mean_delta_norm = row_stats(val_stats)
            mean_residual_alpha = mean_or_none(val_stats["residual_alpha_sum"], val_stats["residual_alpha_count"])
            mean_skip_prior_alpha = mean_or_none(
                val_stats["skip_prior_alpha_sum"],
                val_stats["skip_prior_alpha_count"],
            )
            (
                no_memory_ppl,
                context_only_ppl,
                mean_attn_entropy,
                mean_attn_top_weight,
                mean_attn_char_distance,
                mean_attn_prev_weight,
                mean_attn_current_weight,
                mean_attn_top_is_prev,
                first_token_ppl,
                later_token_ppl,
            ) = diagnostic_row_stats(val_stats)
            row = SegmentMemoryRow(
                epoch=epoch,
                step=step,
                train_nll=epoch_loss_sum / epoch_profile.tokens if epoch_profile.tokens else None,
                val_nll=val_nll,
                val_ppl=val_ppl,
                val_bpc=val_bpc,
                segment_count=len(train_segments),
                mean_segment_len=mean_segment_len,
                max_segment_len=max_segment_len,
                mean_gate=mean_gate,
                mean_context_norm=mean_context_norm,
                mean_delta_norm=mean_delta_norm,
                no_memory_ppl=no_memory_ppl,
                context_only_ppl=context_only_ppl,
                mean_attn_entropy=mean_attn_entropy,
                mean_attn_top_weight=mean_attn_top_weight,
                mean_attn_char_distance=mean_attn_char_distance,
                mean_attn_prev_weight=mean_attn_prev_weight,
                mean_attn_current_weight=mean_attn_current_weight,
                mean_attn_top_is_prev=mean_attn_top_is_prev,
                first_token_ppl=first_token_ppl,
                later_token_ppl=later_token_ppl,
                runtime_sec=time.perf_counter() - started,
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                mean_residual_alpha=mean_residual_alpha,
                mean_skip_prior_alpha=mean_skip_prior_alpha,
            )
            writer.writerow(asdict(row))
            handle.flush()
            print(
                f"epoch={epoch} done step={step} val_ppl={val_ppl:.3f} "
                f"mean_gate={mean_gate if mean_gate is not None else float('nan'):.4f} "
                f"mean_context_norm={mean_context_norm if mean_context_norm is not None else float('nan'):.4f} "
                f"mean_delta_norm={mean_delta_norm if mean_delta_norm is not None else float('nan'):.4f} "
                f"mean_residual_alpha={mean_residual_alpha if mean_residual_alpha is not None else float('nan'):.4f} "
                f"mean_skip_prior_alpha={mean_skip_prior_alpha if mean_skip_prior_alpha is not None else float('nan'):.4f} "
                f"no_memory_ppl={no_memory_ppl if no_memory_ppl is not None else float('nan'):.3f} "
                f"context_only_ppl={context_only_ppl if context_only_ppl is not None else float('nan'):.3f} "
                f"attn_top={mean_attn_top_weight if mean_attn_top_weight is not None else float('nan'):.3f} "
                f"prev_w={mean_attn_prev_weight if mean_attn_prev_weight is not None else float('nan'):.3f} "
                f"top_prev={mean_attn_top_is_prev if mean_attn_top_is_prev is not None else float('nan'):.3f} "
                f"runtime_sec={row.runtime_sec:.2f} peak_rss_mb={row.peak_rss_kb / 1024:.1f} "
                f"lr={optimizer.param_groups[0]['lr']:.4g}",
                flush=True,
            )
            t0 = time.perf_counter()
            improved_best = best_val_ppl is None or val_ppl < best_val_ppl
            if improved_best:
                best_val_ppl = val_ppl
                best_epoch = epoch
                best_step = step
                save_checkpoint(
                    args.best_checkpoint_output,
                    model,
                    optimizer,
                    epoch,
                    step,
                    vocab,
                    args,
                    best_val_ppl=best_val_ppl,
                    best_epoch=best_epoch,
                    best_step=best_step,
                )
            save_checkpoint(
                args.checkpoint_output,
                model,
                optimizer,
                epoch,
                step,
                vocab,
                args,
                best_val_ppl=best_val_ppl,
                best_epoch=best_epoch,
                best_step=best_step,
            )
            epoch_profile.checkpoint_sec += time.perf_counter() - t0
            if args.profile:
                print(f"epoch={epoch} {profile_summary(epoch_profile)}", flush=True)
            print(f"checkpoint={args.checkpoint_output}", flush=True)
            if improved_best:
                print(
                    f"best_checkpoint={args.best_checkpoint_output} "
                    f"best_epoch={best_epoch} best_step={best_step} best_val_ppl={best_val_ppl:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
