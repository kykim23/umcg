"""Independent LLaMA implementation.

The architecture follows the public LLaMA design and Hugging Face Transformers
LLaMA parameter naming. Transformers is Apache-2.0 licensed. No code or runtime
import is taken from the user's previous research repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.checkpoint import checkpoint

from umcg.model.attention import run_attention


@dataclass(frozen=True)
class ReferenceLlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    mlp_bias: bool
    attention_dropout: float
    hidden_act: str
    initializer_range: float
    pad_token_id: int
    eos_token_id: int
    tie_word_embeddings: bool
    use_cache: bool

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ReferenceLlamaConfig:
        defaults = {
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000.0,
            "attention_bias": False,
            "mlp_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "initializer_range": 0.02,
            "tie_word_embeddings": False,
        }
        fields = cls.__dataclass_fields__
        resolved = {name: value.get(name, defaults.get(name)) for name in fields}
        missing = [name for name, item in resolved.items() if item is None]
        if missing:
            raise ValueError(f"reference LLaMA config missing fields: {missing}")
        config = cls(**resolved)
        if config.hidden_act != "silu":
            raise ValueError("reference LLaMA supports hidden_act=silu only")
        if config.use_cache:
            raise ValueError("reference LLaMA requires use_cache=false")
        return config

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class ReferenceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


class ReferenceRotaryEmbedding(nn.Module):
    def __init__(self, config: ReferenceLlamaConfig) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inverse_frequency, persistent=False)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inverse = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        positions = position_ids[:, None, :].float()
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            frequencies = (inverse.to(hidden_states.device) @ positions).transpose(1, 2)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cosine = embedding.cos()
            sine = embedding.sin()
        return cosine.to(hidden_states.dtype), sine.to(hidden_states.dtype)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


class ReferenceAttention(nn.Module):
    def __init__(
        self, config: ReferenceLlamaConfig, layer_index: int, attention_backend: str
    ) -> None:
        super().__init__()
        self.layer_idx = layer_index
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.attention_backend = attention_backend
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = (
            self.q_proj(hidden_states)
            .view(batch_size, sequence_length, self.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(hidden_states)
            .view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(hidden_states)
            .view(batch_size, sequence_length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        cosine, sine = position_embeddings
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
        query = query * cosine + rotate_half(query) * sine
        key = key * cosine + rotate_half(key) * sine
        dropout = self.attention_dropout if self.training else 0.0
        output = run_attention(
            self.attention_backend,
            query,
            key,
            value,
            attention_mask,
            dropout_p=dropout,
        )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, -1).contiguous()
        return self.o_proj(output)


class ReferenceMLP(nn.Module):
    def __init__(self, config: ReferenceLlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class ReferenceDecoderLayer(nn.Module):
    def __init__(
        self, config: ReferenceLlamaConfig, layer_index: int, attention_backend: str
    ) -> None:
        super().__init__()
        self.self_attn = ReferenceAttention(config, layer_index, attention_backend)
        self.mlp = ReferenceMLP(config)
        self.input_layernorm = ReferenceRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = ReferenceRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, (cosine, sine)
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class ReferenceLlamaModel(nn.Module):
    def __init__(
        self,
        config: ReferenceLlamaConfig,
        attention_backend: str,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            ReferenceDecoderLayer(config, index, attention_backend)
            for index in range(config.num_hidden_layers)
        )
        self.norm = ReferenceRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = ReferenceRotaryEmbedding(config)
        self.activation_checkpointing = activation_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        cosine, sine = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    cosine,
                    sine,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask, cosine, sine)
        return self.norm(hidden_states)


class ReferenceLlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: ReferenceLlamaConfig,
        attention_backend: str,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = ReferenceLlamaModel(config, attention_backend, activation_checkpointing)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, attention_mask, position_ids)
