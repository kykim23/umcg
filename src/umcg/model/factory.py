"""Build the two model backends behind one training-only wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from umcg.config import load_json_object, validate_model_config
from umcg.model.attention import AttentionSelection, attention_kernel_context
from umcg.model.causal_loss import chunked_causal_token_loss
from umcg.model.reference_llama import ReferenceLlamaConfig, ReferenceLlamaForCausalLM


@dataclass(frozen=True)
class ModelBuildResult:
    model: PretrainingModel
    parameter_count: int
    trainable_parameter_count: int
    model_config: dict[str, Any]


class PretrainingModel(nn.Module):
    def __init__(
        self,
        causal_lm: nn.Module,
        *,
        model_backend: str,
        attention_selection: AttentionSelection,
        lm_head_chunk_size: int,
    ) -> None:
        super().__init__()
        self.causal_lm = causal_lm
        self.model_backend = model_backend
        self.attention_selection = attention_selection
        self.lm_head_chunk_size = lm_head_chunk_size

    @property
    def config(self):
        return self.causal_lm.config

    @property
    def decoder_layers(self) -> nn.ModuleList:
        return self.causal_lm.model.layers

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        with attention_kernel_context(self.attention_selection.resolved_backend):
            if self.model_backend == "huggingface":
                result = self.causal_lm.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                return result.last_hidden_state
            return self.causal_lm.forward_hidden(input_ids, attention_mask, position_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.forward_hidden(input_ids, attention_mask, position_ids)
        return chunked_causal_token_loss(
            hidden_states,
            self.causal_lm.lm_head,
            input_ids,
            sequence_chunk_size=self.lm_head_chunk_size,
        ).token_losses

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.causal_lm.lm_head(self.forward_hidden(input_ids, attention_mask, position_ids))

    def canonical_state_dict(self) -> dict[str, torch.Tensor]:
        return {name.removeprefix("causal_lm."): value for name, value in self.state_dict().items()}

    def load_canonical_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.causal_lm.state_dict())
        if set(state) != expected:
            missing = sorted(expected - set(state))
            unexpected = sorted(set(state) - expected)
            raise ValueError(
                f"initial weight keys differ from canonical model; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        self.causal_lm.load_state_dict(state, strict=True)


def _model_dtype(precision: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float32,
        "bfloat16": torch.float32,
        "float8": torch.bfloat16,
    }[precision]


def build_model(
    model_config_path: str | Path | Mapping[str, Any],
    *,
    model_backend: str,
    precision: str,
    attention_selection: AttentionSelection,
    activation_checkpointing: bool,
    device: torch.device,
    maximum_context: int,
    initial_weights: str | None = None,
) -> ModelBuildResult:
    model_config = (
        dict(model_config_path)
        if isinstance(model_config_path, Mapping)
        else load_json_object(model_config_path)
    )
    validate_model_config(model_config, maximum_context)
    local_chunk_size = int(model_config.get("lm_head_chunk_size", 128))
    if local_chunk_size <= 0:
        raise ValueError("lm_head_chunk_size must be positive")
    architecture_config = dict(model_config)
    architecture_config.pop("lm_head_chunk_size", None)
    if model_backend == "huggingface":
        config = LlamaConfig(**architecture_config)
        config.use_cache = False
        config._attn_implementation = attention_selection.huggingface_implementation
        causal_lm = LlamaForCausalLM(config)
        if activation_checkpointing:
            causal_lm.gradient_checkpointing_enable()
    elif model_backend == "reference":
        config = ReferenceLlamaConfig.from_mapping(architecture_config)
        causal_lm = ReferenceLlamaForCausalLM(
            config,
            attention_selection.resolved_backend,
            activation_checkpointing,
        )
    else:
        raise ValueError("model_backend must be huggingface or reference")
    model = PretrainingModel(
        causal_lm,
        model_backend=model_backend,
        attention_selection=attention_selection,
        lm_head_chunk_size=local_chunk_size,
    )
    model.to(device=device, dtype=_model_dtype(precision))
    if initial_weights is not None:
        weights = load_file(str(initial_weights), device=str(device))
        model.load_canonical_state_dict(weights)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ModelBuildResult(model, total, trainable, model_config)
