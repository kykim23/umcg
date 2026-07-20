"""Sequence-chunked LM head and FP32 causal cross entropy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TokenLossOutput:
    token_losses: torch.Tensor
    target_ids: torch.Tensor


def chunked_causal_token_loss(
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    input_ids: torch.Tensor,
    *,
    sequence_chunk_size: int,
) -> TokenLossOutput:
    if hidden_states.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("hidden_states and input_ids must have shapes [B,S,H] and [B,S]")
    if hidden_states.shape[:2] != input_ids.shape:
        raise ValueError("hidden_states and input_ids must share batch and sequence dimensions")
    if input_ids.dtype != torch.long or input_ids.shape[1] < 2:
        raise ValueError("causal targets require at least two torch.long input tokens")
    if sequence_chunk_size <= 0:
        raise ValueError("sequence_chunk_size must be positive")
    targets = input_ids[:, 1:]
    losses: list[torch.Tensor] = []
    prediction_count = targets.shape[1]

    def loss_chunk(hidden_chunk: torch.Tensor, target_chunk: torch.Tensor) -> torch.Tensor:
        logits = lm_head(hidden_chunk).float()
        result = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_chunk.reshape(-1),
            reduction="none",
        ).view_as(target_chunk)
        del logits
        return result

    for start in range(0, prediction_count, sequence_chunk_size):
        end = min(start + sequence_chunk_size, prediction_count)
        hidden_chunk = hidden_states[:, start:end]
        chunk_targets = targets[:, start:end]
        if torch.is_grad_enabled():
            chunk_losses = checkpoint(
                loss_chunk,
                hidden_chunk,
                chunk_targets,
                use_reentrant=False,
            )
        else:
            chunk_losses = loss_chunk(hidden_chunk, chunk_targets)
        losses.append(chunk_losses)
    token_losses = torch.cat(losses, dim=1)
    if not torch.isfinite(token_losses).all():
        raise FloatingPointError("non-finite causal token loss")
    return TokenLossOutput(token_losses=token_losses, target_ids=targets)


def per_token_causal_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> TokenLossOutput:
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("logits must have shape [B,S,V] matching input_ids")
    targets = input_ids[:, 1:]
    shifted = logits[:, :-1].float()
    losses = functional.cross_entropy(
        shifted.reshape(-1, shifted.shape[-1]), targets.reshape(-1), reduction="none"
    ).view_as(targets)
    return TokenLossOutput(token_losses=losses, target_ids=targets)
