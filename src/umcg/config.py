"""Canonical production configuration and fail-fast validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DISTRIBUTED_BACKENDS = ("ddp", "fsdp2", "zero")
MODEL_BACKENDS = ("huggingface", "reference")
PRECISIONS = ("float32", "float16", "bfloat16", "float8")
ATTENTION_BACKENDS = (
    "automatic",
    "flash_attention_4",
    "flash_attention_3",
    "flash_attention_2",
    "pytorch_sdpa",
    "eager",
)
OPTIMIZERS = ("adamw", "adam", "sgd", "sgdm", "muon", "adamw_8bit")
SCHEDULERS = ("cosine", "linear")
GRADIENT_ESTIMATORS = ("full", "russian_roulette")
CONTEXT_PRESETS = (
    (128, 256, 512, 1024),
    (512, 1024, 2048, 4096),
)


@dataclass(frozen=True)
class EstimatorConfig:
    schema_version: int
    context_levels: tuple[int, ...]
    tail_probabilities: tuple[float, ...]
    sampling: str
    source: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> EstimatorConfig:
        value = load_json_object(path)
        required = {
            "schema_version",
            "context_levels",
            "tail_probabilities",
            "sampling",
            "source",
        }
        unknown = sorted(set(value) - required)
        missing = sorted(required - set(value))
        if missing or unknown:
            raise ValueError(
                "estimator config fields differ from the canonical schema; "
                f"missing={missing}, unknown={unknown}"
            )
        config = cls(
            schema_version=value["schema_version"],
            context_levels=tuple(value["context_levels"]),
            tail_probabilities=tuple(float(item) for item in value["tail_probabilities"]),
            sampling=value["sampling"],
            source=value["source"],
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("estimator schema_version must be 1")
        if self.context_levels not in CONTEXT_PRESETS:
            raise ValueError(
                f"context_levels must be exactly one supported preset: {list(CONTEXT_PRESETS)}"
            )
        if len(self.tail_probabilities) != len(self.context_levels):
            raise ValueError("tail_probabilities must match context_levels")
        if self.tail_probabilities[0] != 1.0:
            raise ValueError("the first tail probability must be 1.0")
        if any(not math.isfinite(item) or not 0 < item <= 1 for item in self.tail_probabilities):
            raise ValueError("tail probabilities must be finite and in (0, 1]")
        if any(
            left < right
            for left, right in zip(
                self.tail_probabilities, self.tail_probabilities[1:], strict=False
            )
        ):
            raise ValueError("tail probabilities must be non-increasing")
        if self.sampling != "shared_global_microbatch":
            raise ValueError("sampling must be shared_global_microbatch")
        if not isinstance(self.source, dict) or not self.source:
            raise ValueError("source must be a non-empty JSON object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context_levels": list(self.context_levels),
            "tail_probabilities": list(self.tail_probabilities),
            "sampling": self.sampling,
            "source": self.source,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    distributed_backend: str
    zero_stage: int | None
    model_backend: str
    model_config: str
    tokenizer: str
    tokenizer_revision: str
    precision: str
    attention_backend: str
    estimator_config: str
    gradient_estimator: str
    optimizer: str
    scheduler: str
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    momentum: float | None
    batch_size: str
    total_batch_size: int
    num_training_steps: int
    warmup_steps: int
    eval_every: int
    eval_parent_batches: int
    save_every: int
    save_at_steps: tuple[int, ...]
    save_dir: str
    c4_source: str
    c4_repo: str
    c4_revision: str
    c4_local_path: str | None
    workers: int
    seed: int
    use_torch_compile: bool
    compile_mode: str
    activation_checkpointing: bool
    gradient_clip_norm: float | None
    wandb_mode: str
    wandb_project: str
    wandb_entity: str | None
    name: str
    continue_from: str | None
    initial_weights: str | None

    def validate_before_model_creation(self, world_size: int) -> None:
        if self.distributed_backend not in DISTRIBUTED_BACKENDS:
            raise ValueError("invalid distributed_backend")
        if self.model_backend not in MODEL_BACKENDS:
            raise ValueError("invalid model_backend")
        if self.precision not in PRECISIONS:
            raise ValueError("invalid precision")
        if self.attention_backend not in ATTENTION_BACKENDS:
            raise ValueError("invalid attention_backend")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError("invalid optimizer")
        if self.scheduler not in SCHEDULERS:
            raise ValueError("invalid scheduler")
        if self.gradient_estimator not in GRADIENT_ESTIMATORS:
            raise ValueError("invalid gradient_estimator")
        if self.distributed_backend == "zero":
            if self.zero_stage not in {1, 2, 3}:
                raise ValueError("zero_stage must be 1, 2, or 3 when distributed_backend=zero")
        elif self.zero_stage is not None:
            raise ValueError("zero_stage is valid only when distributed_backend=zero")
        if self.distributed_backend == "fsdp2" and self.optimizer == "muon":
            raise ValueError("optimizer=muon is not supported with distributed_backend=fsdp2")
        if self.distributed_backend == "zero" and self.optimizer == "adamw_8bit":
            raise ValueError("optimizer=adamw_8bit is not supported with distributed_backend=zero")
        if self.distributed_backend == "zero" and self.precision == "float8":
            raise ValueError("precision=float8 is not supported with distributed_backend=zero")
        if self.optimizer in {"muon", "adamw_8bit"} and self.precision == "float8":
            raise ValueError(f"optimizer={self.optimizer} is not supported with precision=float8")
        if self.optimizer == "sgd" and self.momentum is not None:
            raise ValueError("optimizer=sgd does not accept momentum; use optimizer=sgdm")
        if self.optimizer not in {"sgd", "sgdm", "muon"} and self.momentum is not None:
            raise ValueError("momentum is valid only for optimizer=sgdm or optimizer=muon")
        if (
            self.optimizer in {"sgdm", "muon"}
            and self.momentum is not None
            and not 0 <= self.momentum < 1
        ):
            raise ValueError("momentum must be in [0, 1)")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if self.learning_rate <= 0 or self.epsilon <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate and epsilon must be positive; weight_decay non-negative"
            )
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.total_batch_size <= 0:
            raise ValueError("total_batch_size must be positive")
        if self.total_batch_size % world_size:
            raise ValueError("total_batch_size must be divisible by world_size")
        if self.batch_size != "auto":
            try:
                parsed_batch = int(self.batch_size)
            except ValueError as error:
                raise ValueError("batch_size must be a positive integer or auto") from error
            derive_accumulation_steps(parsed_batch, self.total_batch_size, world_size)
        positive_values = (
            self.num_training_steps,
            self.eval_every,
            self.eval_parent_batches,
            self.save_every,
            self.workers,
        )
        if min(positive_values) <= 0:
            raise ValueError("step, evaluation, save, and worker values must be positive")
        if any(step <= 0 or step > self.num_training_steps for step in self.save_at_steps):
            raise ValueError("save_at_steps must be within 1..num_training_steps")
        if len(set(self.save_at_steps)) != len(self.save_at_steps):
            raise ValueError("save_at_steps must not contain duplicates")
        if not 0 <= self.warmup_steps <= self.num_training_steps:
            raise ValueError("warmup_steps must be between 0 and num_training_steps")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.c4_source not in {"streaming", "local", "local_raw"}:
            raise ValueError("c4_source must be streaming, local, or local_raw")
        if self.c4_source in {"local", "local_raw"} and not self.c4_local_path:
            raise ValueError(
                "c4_local_path is required when c4_source is local or local_raw"
            )
        if self.c4_source == "streaming" and self.c4_local_path is not None:
            raise ValueError("c4_local_path is valid only when c4_source=local or local_raw")
        if self.continue_from is not None and self.initial_weights is not None:
            raise ValueError("continue_from and initial_weights are mutually exclusive")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if not self.name.strip():
            raise ValueError("name must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"JSON file does not exist: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"malformed JSON {source}: line {error.lineno}, column {error.colno}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def validate_model_config(model: dict[str, Any], maximum_context: int) -> None:
    allowed = {
        "architectures",
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "hidden_act",
        "initializer_range",
        "rms_norm_eps",
        "rope_theta",
        "attention_bias",
        "mlp_bias",
        "attention_dropout",
        "use_cache",
        "tie_word_embeddings",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "lm_head_chunk_size",
    }
    required = {
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "hidden_act",
        "initializer_range",
        "rms_norm_eps",
        "rope_theta",
        "attention_bias",
        "mlp_bias",
        "attention_dropout",
        "use_cache",
        "tie_word_embeddings",
        "pad_token_id",
        "eos_token_id",
        "lm_head_chunk_size",
    }
    missing = sorted(required - set(model))
    unknown = sorted(set(model) - allowed)
    if missing or unknown:
        raise ValueError(
            "model config fields differ from the canonical schema; "
            f"missing={missing}, unknown={unknown}"
        )
    if model["model_type"] != "llama":
        raise ValueError("only model_type=llama is supported")
    integer_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "lm_head_chunk_size",
    )
    for field in integer_fields:
        value = model[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"model field {field} must be a positive integer")
    if model["hidden_size"] % model["num_attention_heads"]:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if (model["hidden_size"] // model["num_attention_heads"]) % 2:
        raise ValueError("attention head dimension must be even for rotary embeddings")
    if model["num_attention_heads"] % model["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if model["max_position_embeddings"] < maximum_context:
        raise ValueError("model max_position_embeddings is shorter than the maximum context")
    if model["use_cache"] is not False:
        raise ValueError("model config must set use_cache=false")
    if model["hidden_act"] != "silu":
        raise ValueError("model config must set hidden_act=silu")
    if any(
        not isinstance(model[field], bool)
        for field in ("attention_bias", "mlp_bias", "tie_word_embeddings")
    ):
        raise ValueError("attention_bias, mlp_bias, and tie_word_embeddings must be booleans")
    for field in ("initializer_range", "rms_norm_eps", "rope_theta"):
        value = model[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"model field {field} must be a positive number")
    dropout = model["attention_dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("attention_dropout must be in [0, 1)")
    for field in ("pad_token_id", "eos_token_id"):
        value = model[field]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < model["vocab_size"]
        ):
            raise ValueError(f"model field {field} must be a valid vocabulary index")


def validate_tokenizer_contract(tokenizer: Any, model: dict[str, Any]) -> None:
    if len(tokenizer) != model["vocab_size"]:
        raise ValueError(
            f"tokenizer vocabulary ({len(tokenizer)}) does not equal model vocab_size "
            f"({model['vocab_size']}); automatic resize is forbidden"
        )
    if tokenizer.eos_token_id != model["eos_token_id"]:
        raise ValueError("tokenizer eos_token_id does not match model config")
    if tokenizer.pad_token_id != model["pad_token_id"]:
        raise ValueError("tokenizer pad_token_id does not match model config")


def derive_accumulation_steps(batch_size: int, total_batch_size: int, world_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    denominator = batch_size * world_size
    if total_batch_size % denominator:
        raise ValueError(
            "total_batch_size must be divisible by batch_size * world_size: "
            f"{total_batch_size} % ({batch_size} * {world_size}) != 0"
        )
    steps = total_batch_size // denominator
    if steps <= 0:
        raise ValueError("derived accumulation steps must be positive")
    return steps


def batch_divisors(total_batch_size: int, world_size: int) -> list[int]:
    if total_batch_size % world_size:
        raise ValueError("total_batch_size must be divisible by world_size")
    per_rank = total_batch_size // world_size
    return [candidate for candidate in range(per_rank, 0, -1) if per_rank % candidate == 0]


def canonical_json_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def source_tree_sha256(root: str | Path) -> str:
    base = Path(root).resolve()
    digest = hashlib.sha256()
    paths = [base / "pyproject.toml"]
    paths.extend(base.glob("*_main.py"))
    for directory in (base / "src", base / "configs"):
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    for path in sorted(set(paths)):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()
