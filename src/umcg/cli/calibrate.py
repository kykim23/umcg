"""Optimizer-free calibration of fixed Russian-roulette tail probabilities."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from umcg.config import (
    ATTENTION_BACKENDS,
    CONTEXT_PRESETS,
    MODEL_BACKENDS,
    PRECISIONS,
    load_json_object,
    validate_model_config,
)
from umcg.data.c4_stream import build_c4_stream
from umcg.data.sources import load_tokenizer, resolve_c4_source
from umcg.model.attention import resolve_attention_backend
from umcg.model.factory import build_model
from umcg.precision import build_precision_runtime
from umcg.rng import seed_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calibrate_main.py", allow_abbrev=False)
    parser.add_argument("--model_backend", required=True, choices=MODEL_BACKENDS)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--tokenizer", default="t5-base")
    parser.add_argument("--tokenizer_revision", default="main")
    parser.add_argument("--precision", required=True, choices=PRECISIONS)
    parser.add_argument("--attention_backend", default="automatic", choices=ATTENTION_BACKENDS)
    parser.add_argument("--context_preset", required=True, type=int, choices=(1024, 4096))
    parser.add_argument("--c4_source", required=True, choices=("streaming", "local"))
    parser.add_argument("--c4_repo", default="allenai/c4")
    parser.add_argument("--c4_revision", default="main")
    parser.add_argument("--c4_local_path")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--parent_batches", type=int, default=64)
    parser.add_argument("--sketch_dimension", type=int, default=256)
    parser.add_argument("--monte_carlo_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output", required=True)
    return parser


def _levels(maximum: int) -> tuple[int, ...]:
    return next(levels for levels in CONTEXT_PRESETS if levels[-1] == maximum)


def _parameter_hash(name: str, seed: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    multiplier = int.from_bytes(digest[:8], "little") | 1
    offset = int.from_bytes(digest[8:16], "little")
    return multiplier, offset


def _gradient_sketch(
    gradients: tuple[torch.Tensor | None, ...],
    names: list[str],
    *,
    dimension: int,
    seed: int,
) -> torch.Tensor:
    sketch = torch.zeros(dimension, dtype=torch.float64)
    chunk_size = 1_000_000
    for name, gradient in zip(names, gradients, strict=True):
        if gradient is None:
            continue
        values = gradient.detach().float().reshape(-1).cpu()
        multiplier, offset = _parameter_hash(name, seed)
        for start in range(0, values.numel(), chunk_size):
            end = min(start + chunk_size, values.numel())
            indices = torch.arange(start, end, dtype=torch.int64)
            buckets = (indices * multiplier + offset) % dimension
            signs = (((indices * (multiplier + 2) + offset) & 1) * 2 - 1).double()
            sketch.scatter_add_(0, buckets, values[start:end].double() * signs)
    return sketch / math.sqrt(dimension)


def _candidate_schedules(level_count: int) -> list[tuple[float, ...]]:
    grid = (1.0, 0.75, 0.5, 0.25, 0.125)
    candidates = []
    for tail in itertools.product(grid, repeat=level_count - 1):
        schedule = (1.0, *tail)
        if all(left >= right for left, right in zip(schedule, schedule[1:], strict=False)):
            candidates.append(schedule)
    return candidates


def _score_schedule(
    schedule: tuple[float, ...],
    base_sketches: torch.Tensor,
    correction_sketches: torch.Tensor,
    level_costs: torch.Tensor,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    estimates = []
    for _ in range(samples):
        uniform = torch.rand(base_sketches.shape[0], generator=generator)
        estimate = base_sketches.clone()
        for level_index in range(1, len(schedule)):
            included = (uniform <= schedule[level_index]).double().unsqueeze(1)
            estimate += correction_sketches[:, level_index - 1] * included / schedule[level_index]
        estimates.append(estimate)
    stacked_estimates = torch.stack(estimates)
    mean_gradient = stacked_estimates.mean(dim=(0, 1), keepdim=True)
    variance = float((stacked_estimates - mean_gradient).square().sum(dim=2).mean())
    maximum_probabilities = [
        schedule[index] - schedule[index + 1] for index in range(len(schedule) - 1)
    ] + [schedule[-1]]
    expected_cost = float(
        sum(
            probability * float(level_costs[index])
            for index, probability in enumerate(maximum_probabilities)
        )
    )
    return {
        "tail_probabilities": list(schedule),
        "gradient_variance": variance,
        "expected_cuda_cost_ms": expected_cost,
        "objective": variance * expected_cost,
    }


def _pareto(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for candidate in scored:
        dominated = any(
            other["gradient_variance"] <= candidate["gradient_variance"]
            and other["expected_cuda_cost_ms"] <= candidate["expected_cuda_cost_ms"]
            and (
                other["gradient_variance"] < candidate["gradient_variance"]
                or other["expected_cuda_cost_ms"] < candidate["expected_cuda_cost_ms"]
            )
            for other in scored
        )
        if not dominated:
            result.append(candidate)
    return sorted(result, key=lambda item: item["expected_cuda_cost_ms"])


def calibrate(arguments: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("calibration requires CUDA")
    if (
        min(
            arguments.batch_size,
            arguments.parent_batches,
            arguments.sketch_dimension,
            arguments.monte_carlo_samples,
        )
        <= 0
    ):
        raise ValueError("calibration sizes must be positive")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    levels = _levels(arguments.context_preset)
    model_config_path = Path(arguments.model_config).resolve()
    model_config = load_json_object(model_config_path)
    validate_model_config(model_config, levels[-1])
    tokenizer, tokenizer_metadata = load_tokenizer(
        name=arguments.tokenizer,
        revision=arguments.tokenizer_revision,
        model_config=model_config,
    )
    source = resolve_c4_source(
        source=arguments.c4_source,
        repository=arguments.c4_repo,
        revision=arguments.c4_revision,
        local_path=arguments.c4_local_path,
    )
    data_revision = (
        str(source["resolved_commit"])
        if arguments.c4_source == "streaming"
        else arguments.c4_revision
    )
    attention = resolve_attention_backend(
        arguments.attention_backend,
        device=device,
        precision=arguments.precision,
        context_levels=levels,
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
        activation_checkpointing=False,
        device=device,
        maximum_context=levels[-1],
    )
    precision = build_precision_runtime(
        name=arguments.precision,
        device=device,
        distributed_backend="ddp",
        model=build.model,
    )
    stream = build_c4_stream(
        source=arguments.c4_source,
        repository=arguments.c4_repo,
        revision=data_revision,
        local_path=arguments.c4_local_path,
        split="train",
        tokenizer=tokenizer,
        tokenizer_metadata=tokenizer_metadata,
        maximum_context=levels[-1],
        seed=arguments.seed,
        rank=0,
        world_size=1,
        worker_count=1,
        train=True,
    )
    named_parameters = [
        (name, parameter)
        for name, parameter in build.model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    parents = [stream.next_batch(arguments.batch_size) for _ in range(arguments.parent_batches)]
    warmup_parent = parents[0]
    build.model.train()
    for level in levels:
        batch = warmup_parent.prefix(level).to(device)
        with precision.autocast():
            token_losses = build.model(batch.input_ids, batch.attention_mask, batch.position_ids)
            mask = batch.causal_target_mask
            loss = token_losses.float().masked_select(mask).mean()
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        torch.cuda.synchronize(device)
        del batch, gradients, token_losses, loss
    base_sketches = []
    correction_sketches = []
    level_times: list[list[float]] = []
    level_memories: list[list[int]] = []
    for batch_index, parent in enumerate(parents):
        sketches = []
        times = []
        memories = []
        for level in levels:
            batch = parent.prefix(level).to(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with precision.autocast():
                token_losses = build.model(
                    batch.input_ids, batch.attention_mask, batch.position_ids
                )
                mask = batch.causal_target_mask
                loss = token_losses.float().masked_select(mask).mean()
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            end_event.record()
            torch.cuda.synchronize(device)
            times.append(float(start_event.elapsed_time(end_event)))
            memories.append(int(torch.cuda.max_memory_allocated(device)))
            sketches.append(
                _gradient_sketch(
                    gradients,
                    names,
                    dimension=arguments.sketch_dimension,
                    seed=arguments.seed,
                )
            )
            del gradients, token_losses, loss
        stacked = torch.stack(sketches)
        base_sketches.append(stacked[0])
        correction_sketches.append(stacked[1:] - stacked[:-1])
        level_times.append(times)
        level_memories.append(memories)
        print(
            json.dumps(
                {"calibration_parent_batch": batch_index + 1, "cuda_time_ms": times},
                allow_nan=False,
            ),
            flush=True,
        )
    base_tensor = torch.stack(base_sketches)
    correction_tensor = torch.stack(correction_sketches)
    cost_tensor = torch.tensor(level_times, dtype=torch.float64).mean(dim=0)
    scored = [
        _score_schedule(
            schedule,
            base_tensor,
            correction_tensor,
            cost_tensor,
            samples=arguments.monte_carlo_samples,
            seed=arguments.seed + index,
        )
        for index, schedule in enumerate(_candidate_schedules(len(levels)))
    ]
    recommendation = min(scored, key=lambda item: item["objective"])
    estimator = {
        "schema_version": 1,
        "context_levels": list(levels),
        "tail_probabilities": recommendation["tail_probabilities"],
        "sampling": "shared_global_microbatch",
        "source": {
            "type": "calibration",
            "criterion": "gradient_variance_times_expected_cuda_cost",
            "parent_batches": arguments.parent_batches,
            "kernel_warmup_parent_batches": 1,
            "seed": arguments.seed,
        },
    }
    output = Path(arguments.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(estimator, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "estimator": estimator,
        "recommendation": recommendation,
        "pareto_candidates": _pareto(scored),
        "all_candidates": sorted(scored, key=lambda item: item["objective"]),
        "mean_level_cuda_time_ms": cost_tensor.tolist(),
        "maximum_level_peak_memory_bytes": torch.tensor(level_memories).max(dim=0).values.tolist(),
        "tokenizer": tokenizer_metadata,
        "data_source": source,
        "attention": attention.to_dict(),
        "precision": arguments.precision,
        "optimizer_updates": 0,
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"estimator_config": str(output), "report": str(report_path), **recommendation}


def main(arguments: Sequence[str] | None = None) -> None:
    result = calibrate(build_parser().parse_args(arguments))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
