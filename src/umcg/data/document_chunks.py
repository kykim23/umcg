"""Pure one-row-to-non-overlapping-parent transformation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch

from umcg.data.parent_dataset import ParentSample


def document_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_document_tokens(
    token_ids_without_special_tokens: Sequence[int],
    *,
    eos_token_id: int,
    pad_token_id: int,
    maximum_context: int,
    document_hash: str,
    url: str = "",
    timestamp: str = "",
) -> list[ParentSample]:
    if maximum_context < 2:
        raise ValueError("maximum_context must be at least 2")
    tokens = [int(token) for token in token_ids_without_special_tokens]
    original_token_count = len(tokens)
    tokens.append(int(eos_token_id))
    samples: list[ParentSample] = []
    for chunk_index, start in enumerate(range(0, len(tokens), maximum_context)):
        active = tokens[start : start + maximum_context]
        active_length = len(active)
        if active_length < 2:
            continue
        padded = active + [int(pad_token_id)] * (maximum_context - active_length)
        attention_mask = torch.zeros(maximum_context, dtype=torch.bool)
        attention_mask[:active_length] = True
        causal_target_mask = attention_mask[:-1] & attention_mask[1:]
        if not causal_target_mask.any():
            continue
        samples.append(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": attention_mask,
                "causal_target_mask": causal_target_mask,
                "position_ids": torch.arange(maximum_context, dtype=torch.long),
                "document_hash": str(document_hash),
                "chunk_index": chunk_index,
                "token_start": start,
                "token_end": min(start + active_length, original_token_count),
                "url": str(url),
                "timestamp": str(timestamp),
            }
        )
    return samples


def tokenize_c4_row(
    row: dict[str, object], tokenizer: object, maximum_context: int
) -> list[ParentSample]:
    text = row.get("text")
    if not isinstance(text, str):
        raise ValueError("C4 row text must be a string")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return split_document_tokens(
        token_ids,
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
        maximum_context=maximum_context,
        document_hash=document_sha256(text),
        url=str(row.get("url", "")),
        timestamp=str(row.get("timestamp", "")),
    )
