"""Bounded CPU or one-GPU smoke test, separate from C4 training."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from umcg.config import (
    ATTENTION_BACKENDS,
    GRADIENT_ESTIMATORS,
    MODEL_BACKENDS,
    OPTIMIZERS,
    PRECISIONS,
    SCHEDULERS,
    EstimatorConfig,
    load_json_object,
    validate_model_config,
)
from umcg.data.collate import collate_parent_samples
from umcg.data.synthetic import SyntheticParentDataset
from umcg.estimators.global_objective import estimator_scalar, global_token_coefficients
from umcg.estimators.levels import LevelSpec
from umcg.estimators.russian_roulette import LevelSampler
from umcg.model.attention import resolve_attention_backend
from umcg.model.factory import build_model
from umcg.optim.factory import SchedulerController, build_optimizer
from umcg.precision import build_precision_runtime
from umcg.rng import seed_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smoke_main.py", allow_abbrev=False)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--model_backend", required=True, choices=MODEL_BACKENDS)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--precision", required=True, choices=PRECISIONS)
    parser.add_argument("--attention_backend", default="automatic", choices=ATTENTION_BACKENDS)
    parser.add_argument("--estimator_config", required=True)
    parser.add_argument("--gradient_estimator", required=True, choices=GRADIENT_ESTIMATORS)
    parser.add_argument("--optimizer", required=True, choices=OPTIMIZERS)
    parser.add_argument("--scheduler", required=True, choices=SCHEDULERS)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_training_steps", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--save_dir", required=True)
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.device == "cpu" and arguments.precision != "float32":
        raise ValueError("CPU smoke supports precision=float32 only")
    if arguments.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("CUDA smoke requires exactly one visible GPU")
    if arguments.optimizer == "sgd" and arguments.momentum is not None:
        raise ValueError("optimizer=sgd does not accept momentum; use optimizer=sgdm")
    if arguments.optimizer not in {"sgd", "sgdm", "muon"} and arguments.momentum is not None:
        raise ValueError("momentum is valid only for optimizer=sgdm or optimizer=muon")
    if arguments.precision == "float8" and arguments.optimizer in {"muon", "adamw_8bit"}:
        raise ValueError(f"optimizer={arguments.optimizer} is not supported with precision=float8")
    if arguments.batch_size <= 0 or arguments.num_training_steps <= 0:
        raise ValueError("batch_size and num_training_steps must be positive")
    device = torch.device("cuda", 0) if arguments.device == "cuda" else torch.device("cpu")
    estimator = EstimatorConfig.load(arguments.estimator_config)
    model_config_path = Path(arguments.model_config).resolve()
    model_config = load_json_object(model_config_path)
    validate_model_config(model_config, estimator.context_levels[-1])
    attention = resolve_attention_backend(
        arguments.attention_backend,
        device=device,
        precision=arguments.precision,
        context_levels=estimator.context_levels,
        num_attention_heads=model_config["num_attention_heads"],
        num_key_value_heads=model_config["num_key_value_heads"],
        head_dimension=model_config["hidden_size"] // model_config["num_attention_heads"],
    )
    seed_all(arguments.seed)
    build = build_model(
        model_config_path,
        model_backend=arguments.model_backend,
        precision=arguments.precision,
        attention_selection=attention,
        activation_checkpointing=arguments.activation_checkpointing,
        device=device,
        maximum_context=estimator.context_levels[-1],
    )
    precision = (
        build_precision_runtime(
            name=arguments.precision,
            device=device,
            distributed_backend="ddp",
            model=build.model,
        )
        if device.type == "cuda"
        else None
    )
    model = (
        torch.compile(build.model, mode=arguments.compile_mode)
        if arguments.use_torch_compile
        else build.model
    )
    optimizer = build_optimizer(
        build.model,
        name=arguments.optimizer,
        learning_rate=arguments.learning_rate,
        beta1=arguments.beta1,
        beta2=arguments.beta2,
        epsilon=arguments.epsilon,
        weight_decay=arguments.weight_decay,
        momentum=arguments.momentum,
    )
    scheduler = SchedulerController(
        optimizer,
        name=arguments.scheduler,
        total_updates=arguments.num_training_steps,
        warmup_updates=arguments.warmup_steps,
    )
    dataset = SyntheticParentDataset(
        num_parents=arguments.batch_size,
        parent_length=estimator.context_levels[-1],
        vocab_size=model_config["vocab_size"],
        seed=arguments.seed,
        pad_token_id=model_config["pad_token_id"],
    )
    parent = collate_parent_samples([dataset[index] for index in range(arguments.batch_size)])
    levels = LevelSpec(estimator.context_levels, estimator.tail_probabilities)
    sampler = LevelSampler(levels, arguments.seed + 17)
    metrics = []
    for update in range(1, arguments.num_training_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        level_index = (
            levels.num_levels - 1 if arguments.gradient_estimator == "full" else sampler.sample()
        )
        batch = parent.prefix(levels.lengths[level_index]).to(device)
        counts = torch.tensor(
            [int(parent.causal_target_mask[:, : length - 1].sum()) for length in levels.lengths],
            device=device,
            dtype=torch.long,
        )
        started = time.perf_counter()
        autocast = (
            precision.autocast() if precision is not None else torch.autocast("cpu", enabled=False)
        )
        with autocast:
            losses = model(batch.input_ids, batch.attention_mask, batch.position_ids)
            coefficients = global_token_coefficients(
                batch.causal_target_mask,
                levels=levels,
                global_target_counts=counts,
                sampled_level_index=level_index,
                gradient_estimator=arguments.gradient_estimator,
                gradient_scale=1.0,
            )
            scalar = estimator_scalar(losses, coefficients)
        if precision is None:
            scalar.backward()
            optimizer.step()
        else:
            precision.backward(scalar)
            for item in optimizer.optimizers:
                precision.step(item)
            precision.update()
        scheduler.step()
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in build.model.parameters()
        )
        if not torch.isfinite(scalar) or not finite_gradients:
            raise FloatingPointError("smoke test produced non-finite values")
        metrics.append(
            {
                "optimizer_update": update,
                "sampled_level_index": level_index,
                "active_length": levels.lengths[level_index],
                "signed_estimator_scalar": float(scalar.detach().cpu()),
                "wall_time_seconds": time.perf_counter() - started,
            }
        )
    report = {
        "schema_version": 1,
        "device": str(device),
        "model_backend": arguments.model_backend,
        "precision": arguments.precision,
        "attention": attention.to_dict(),
        "gradient_estimator": arguments.gradient_estimator,
        "optimizer": arguments.optimizer,
        "scheduler": arguments.scheduler,
        "parameter_count": build.parameter_count,
        "updates": metrics,
        "finite": True,
    }
    output = Path(arguments.save_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(arguments: Sequence[str] | None = None) -> None:
    report = run(build_parser().parse_args(arguments))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
