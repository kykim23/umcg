"""Distributed, projection-free calibration of Russian-roulette tail probabilities."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as distributed
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel

from umcg.calibration.exact import (
    audit_schedule,
    candidate_schedules,
    correction_grams,
    cosine_matrix,
    gradient_gram,
    gradient_level_and_correction_grams,
    pareto_candidates,
    score_schedule,
    vector_cross_gram,
)
from umcg.config import (
    ATTENTION_BACKENDS,
    CONTEXT_PRESETS,
    MODEL_BACKENDS,
    PRECISIONS,
    EstimatorConfig,
    load_json_object,
    source_tree_sha256,
    validate_model_config,
)
from umcg.data.c4_stream import StatefulC4Stream, build_c4_stream
from umcg.data.collate import ParentBatch
from umcg.data.parent_dataset import ParentSample
from umcg.data.sources import load_tokenizer, resolve_c4_source
from umcg.distributed.runtime import DistributedContext
from umcg.model.attention import resolve_attention_backend
from umcg.model.factory import build_model
from umcg.precision import PrecisionRuntime, build_precision_runtime
from umcg.rng import seed_all

SPLIT_NAMES = ("measurement", "selection", "audit")
REPORT_SCHEMA_VERSION = 3
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SplitMeasurements:
    batch_level_grams: torch.Tensor
    batch_correction_grams: torch.Tensor
    mean_level_gram: torch.Tensor
    mean_correction_gram: torch.Tensor
    level_times_ms: torch.Tensor
    level_peak_memory_bytes: torch.Tensor
    full_gradient_cross_gram: torch.Tensor | None = None


@dataclass(frozen=True)
class CalibrationParentCache:
    manifest: dict[str, Any]
    splits: dict[str, dict[str, Any]]

    def rank_batch(
        self,
        split_name: str,
        batch_index: int,
        *,
        logical_batch_size: int,
        rank: int,
        world_size: int,
    ) -> ParentBatch:
        if logical_batch_size % world_size:
            raise ValueError("logical parent batch size must be divisible by world size")
        per_rank = logical_batch_size // world_size
        global_start = batch_index * logical_batch_size
        start = global_start + rank * per_rank
        end = start + per_rank
        split = self.splits[split_name]
        if end > split["input_ids"].shape[0]:
            raise IndexError("calibration parent cache batch is outside the split")
        input_ids = split["input_ids"][start:end].long()
        attention_mask = split["attention_mask"][start:end].bool()
        sequence_length = input_ids.shape[1]
        position_ids = torch.arange(sequence_length, dtype=torch.long).expand(per_rank, -1)
        metadata = split["metadata"]
        result = ParentBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            causal_target_mask=attention_mask[:, :-1] & attention_mask[:, 1:],
            position_ids=position_ids,
            document_hashes=list(metadata["document_hashes"][start:end]),
            chunk_indices=list(metadata["chunk_indices"][start:end]),
            token_starts=list(metadata["token_starts"][start:end]),
            token_ends=list(metadata["token_ends"][start:end]),
            urls=list(metadata["urls"][start:end]),
            timestamps=list(metadata["timestamps"][start:end]),
        )
        result.validate()
        return result


class GradientMeanAccumulator:
    """FP64 host running sums without retaining per-batch gradients."""

    def __init__(self, level_count: int, parameter_count: int) -> None:
        self.level_count = level_count
        self.parameter_count = parameter_count
        self.sums: list[list[torch.Tensor]] | None = None
        self.batch_count = 0

    def update(
        self,
        level_gradients: list[tuple[torch.Tensor, ...]],
        *,
        retain_full_gradient: bool,
    ) -> torch.Tensor | None:
        if len(level_gradients) != self.level_count:
            raise ValueError("gradient level count differs from the accumulator")
        if any(len(gradients) != self.parameter_count for gradients in level_gradients):
            raise ValueError("gradient parameter count differs from the accumulator")
        retained_parts: list[torch.Tensor] = []
        if self.sums is None:
            self.sums = [[] for _ in range(self.level_count)]
        for level_index, gradients in enumerate(level_gradients):
            for parameter_index, gradient in enumerate(gradients):
                cpu_gradient = gradient.detach().to(device="cpu", dtype=torch.float32)
                if self.batch_count == 0:
                    self.sums[level_index].append(cpu_gradient.double())
                else:
                    self.sums[level_index][parameter_index].add_(cpu_gradient)
                if retain_full_gradient and level_index == self.level_count - 1:
                    retained_parts.append(cpu_gradient.reshape(-1).clone())
        self.batch_count += 1
        if retain_full_gradient:
            return torch.cat(retained_parts)
        return None

    def mean_grams(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sums is None or self.batch_count <= 0:
            raise RuntimeError("gradient mean accumulator is empty")
        summed = [tuple(level) for level in self.sums]
        level_gram, correction_gram = gradient_level_and_correction_grams(summed)
        scale = self.batch_count * self.batch_count
        return level_gram / scale, correction_gram / scale


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def _removed_argument(name: str):
    def parse(_value: str) -> int:
        raise argparse.ArgumentTypeError(
            f"{name} was removed: calibration now uses full-coordinate analytic statistics"
        )

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calibrate_main.py",
        description="Exact two-rank calibration of Russian-roulette tail probabilities",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model = parser.add_argument_group("model, precision, and attention")
    model.add_argument(
        "--model_backend", required=True, choices=MODEL_BACKENDS, help="Model implementation."
    )
    model.add_argument(
        "--model_config", required=True, metavar="PATH", help="Canonical LLaMA model JSON."
    )
    model.add_argument(
        "--tokenizer", default="t5-base", metavar="NAME", help="Tokenizer repository or path."
    )
    model.add_argument(
        "--tokenizer_revision",
        default="main",
        metavar="REVISION",
        help="Tokenizer revision resolved to an immutable commit.",
    )
    model.add_argument(
        "--precision", required=True, choices=PRECISIONS, help="Calibration precision."
    )
    model.add_argument(
        "--attention_backend",
        default="automatic",
        choices=ATTENTION_BACKENDS,
        help="Attention kernel used for every measured context.",
    )
    model.add_argument(
        "--context_preset",
        required=True,
        type=int,
        choices=(1024, 4096),
        help="Maximum context preset to calibrate.",
    )
    model.add_argument(
        "--initial_weights",
        metavar="PATH",
        help="Optional canonical FP32 safetensors; absent means deterministic initialization.",
    )
    model.add_argument(
        "--activation_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recompute transformer activations during backward; enabled by the exact 350M "
            "protocol and matched by long training."
        ),
    )

    data = parser.add_argument_group("C4 data and immutable parent cache")
    data.add_argument(
        "--c4_source",
        required=True,
        choices=("streaming", "local", "local_raw"),
        help="C4 input used only when the parent cache must be created.",
    )
    data.add_argument("--c4_repo", default="allenai/c4", metavar="NAME", help="C4 repository.")
    data.add_argument(
        "--c4_revision", default="main", metavar="REVISION", help="Recorded C4 revision."
    )
    data.add_argument(
        "--c4_local_path", metavar="PATH", help="Required for local and local_raw sources."
    )
    data.add_argument(
        "--parent_cache",
        metavar="PATH",
        help="Immutable cache to create or reuse; defaults next to --output.",
    )

    sizes = parser.add_argument_group("logical batches, physical chunks, and split sizes")
    sizes.add_argument(
        "--logical_parent_batch_size",
        type=_positive_integer,
        default=128,
        metavar="INTEGER",
        help="Global parent samples represented by one gradient observation.",
    )
    physical = sizes.add_mutually_exclusive_group()
    physical.add_argument(
        "--max_parent_batch_size_per_gpu",
        type=_positive_integer,
        default=64,
        metavar="INTEGER",
        help="Largest physical parent chunk tested per rank.",
    )
    physical.add_argument(
        "--batch_size",
        dest="max_parent_batch_size_per_gpu",
        type=_positive_integer,
        default=argparse.SUPPRESS,
        metavar="INTEGER",
        help="Deprecated alias for --max_parent_batch_size_per_gpu.",
    )
    sizes.add_argument(
        "--memory_limit_fraction",
        type=_fraction,
        default=0.85,
        metavar="FLOAT",
        help="Maximum exact-path allocated VRAM fraction on either rank.",
    )
    measurement = sizes.add_mutually_exclusive_group()
    measurement.add_argument(
        "--measurement_parent_batches",
        type=_positive_integer,
        default=64,
        metavar="INTEGER",
        help="Logical batches used to measure gradient geometry and level costs.",
    )
    measurement.add_argument(
        "--parent_batches",
        dest="measurement_parent_batches",
        type=_positive_integer,
        default=argparse.SUPPRESS,
        metavar="INTEGER",
        help="Deprecated alias for --measurement_parent_batches.",
    )
    sizes.add_argument(
        "--selection_parent_batches",
        type=_positive_integer,
        default=32,
        metavar="INTEGER",
        help="Document-disjoint logical batches used to select one schedule.",
    )
    sizes.add_argument(
        "--audit_parent_batches",
        type=_positive_integer,
        default=32,
        metavar="INTEGER",
        help="Document-disjoint logical batches used only after schedule selection.",
    )
    sizes.add_argument(
        "--timing_parent_batches",
        type=_positive_integer,
        default=8,
        metavar="INTEGER",
        help="Leading measurement batches included in C_k timing.",
    )
    sizes.add_argument(
        "--timing_repeats",
        type=_positive_integer,
        default=1,
        metavar="INTEGER",
        help="Forward/backward timing repetitions; the scientific protocol uses one.",
    )
    sizes.add_argument(
        "--bootstrap_parent_resamples",
        type=_positive_integer,
        default=10_000,
        metavar="INTEGER",
        help="Audit logical-batch bootstrap resamples.",
    )
    sizes.add_argument(
        "--sketch_dimension",
        type=_removed_argument("--sketch_dimension"),
        help=argparse.SUPPRESS,
    )
    sizes.add_argument(
        "--monte_carlo_samples",
        type=_removed_argument("--monte_carlo_samples"),
        help=argparse.SUPPRESS,
    )

    output = parser.add_argument_group("reproducibility, diagnostics, and output")
    output.add_argument("--seed", type=int, default=777, help="Model, split, and bootstrap seed.")
    output.add_argument(
        "--locked_estimator",
        metavar="PATH",
        help="Original estimator evaluated without changing it during checkpoint diagnostics.",
    )
    output.add_argument(
        "--diagnostic_only",
        action="store_true",
        help="Write a report without creating or replacing an estimator configuration.",
    )
    output.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="New estimator path, or report stem in diagnostic-only mode.",
    )
    return parser


def _resolve_parent_batch_counts(arguments: argparse.Namespace) -> dict[str, int]:
    return {
        "measurement": int(arguments.measurement_parent_batches),
        "selection": int(arguments.selection_parent_batches),
        "audit": int(arguments.audit_parent_batches),
    }


def _levels(maximum: int) -> tuple[int, ...]:
    return next(levels for levels in CONTEXT_PRESETS if levels[-1] == maximum)


def _calibration_split_name(document_hash: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:calibration-split:{document_hash}".encode()).digest()
    bucket = digest[0] % 4
    return ("measurement", "measurement", "selection", "audit")[bucket]


def _document_manifest(document_hashes: set[str]) -> str:
    digest = hashlib.sha256()
    for document_hash in sorted(document_hashes):
        encoded = document_hash.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _collect_calibration_samples(
    stream: StatefulC4Stream,
    *,
    logical_batch_size: int,
    batch_counts: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[ParentSample]], dict[str, Any]]:
    required = {name: batch_counts[name] * logical_batch_size for name in SPLIT_NAMES}
    samples: dict[str, list[ParentSample]] = {name: [] for name in SPLIT_NAMES}
    scanned = 0
    while any(len(samples[name]) < required[name] for name in SPLIT_NAMES):
        sample = stream.next_sample()
        scanned += 1
        split_name = _calibration_split_name(str(sample["document_hash"]), seed)
        if len(samples[split_name]) < required[split_name]:
            samples[split_name].append(sample)
    hash_sets = {
        name: {str(sample["document_hash"]) for sample in samples[name]} for name in SPLIT_NAMES
    }
    overlaps = {
        "measurement_selection": len(hash_sets["measurement"] & hash_sets["selection"]),
        "measurement_audit": len(hash_sets["measurement"] & hash_sets["audit"]),
        "selection_audit": len(hash_sets["selection"] & hash_sets["audit"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"calibration document splits overlap: {overlaps}")
    metadata = {
        "assignment": "sha256(seed:calibration-split:document_hash) modulo 4",
        "ratio": {"measurement": 0.5, "selection": 0.25, "audit": 0.25},
        "seed": seed,
        "scanned_parent_samples": scanned,
        "overlap_document_counts": overlaps,
        "splits": {
            name: {
                "parent_batches": batch_counts[name],
                "parent_samples": len(samples[name]),
                "unique_documents": len(hash_sets[name]),
                "document_manifest_sha256": _document_manifest(hash_sets[name]),
            }
            for name in SPLIT_NAMES
        },
    }
    return samples, metadata


def _pack_split(samples: list[ParentSample]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([sample["input_ids"].to(torch.int32) for sample in samples]),
        "attention_mask": torch.stack([sample["attention_mask"].bool() for sample in samples]),
        "metadata": {
            "document_hashes": [str(sample["document_hash"]) for sample in samples],
            "chunk_indices": [int(sample["chunk_index"]) for sample in samples],
            "token_starts": [int(sample["token_start"]) for sample in samples],
            "token_ends": [int(sample["token_end"]) for sample in samples],
            "urls": [str(sample.get("url", "")) for sample in samples],
            "timestamps": [str(sample.get("timestamp", "")) for sample in samples],
        },
    }


def _write_cache_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"parent cache already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _build_or_load_parent_cache(
    path: Path,
    *,
    context: DistributedContext,
    stream: StatefulC4Stream | None,
    logical_batch_size: int,
    batch_counts: dict[str, int],
    maximum_context: int,
    seed: int,
    tokenizer_metadata: dict[str, Any],
    source: dict[str, Any],
) -> tuple[CalibrationParentCache, dict[str, Any]]:
    if context.is_primary and not path.exists():
        if stream is None:
            raise RuntimeError("rank zero requires a C4 stream to create the parent cache")
        samples, split_metadata = _collect_calibration_samples(
            stream,
            logical_batch_size=logical_batch_size,
            batch_counts=batch_counts,
            seed=seed,
        )
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "manifest": {
                "schema_version": CACHE_SCHEMA_VERSION,
                "logical_parent_batch_size": logical_batch_size,
                "batch_counts": batch_counts,
                "maximum_context": maximum_context,
                "seed": seed,
                "tokenizer": tokenizer_metadata,
                "data_source": source,
                "split_contract": split_metadata,
            },
            "splits": {name: _pack_split(samples[name]) for name in SPLIT_NAMES},
        }
        _write_cache_new(path, payload)
    context.barrier()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("calibration parent cache schema_version differs")
    manifest = payload["manifest"]
    expected = {
        "logical_parent_batch_size": logical_batch_size,
        "batch_counts": batch_counts,
        "maximum_context": maximum_context,
        "seed": seed,
        "tokenizer": tokenizer_metadata,
        "data_source": source,
    }
    differing = [key for key, value in expected.items() if manifest.get(key) != value]
    if differing:
        raise ValueError(f"calibration parent cache contract differs in fields: {differing}")
    return CalibrationParentCache(manifest, payload["splits"]), {
        "path": str(path),
        "sha256": _file_sha256(path),
        **manifest,
    }


def _slice_parent_batch(batch: ParentBatch, start: int, end: int) -> ParentBatch:
    result = ParentBatch(
        input_ids=batch.input_ids[start:end],
        attention_mask=batch.attention_mask[start:end],
        causal_target_mask=batch.causal_target_mask[start:end],
        position_ids=batch.position_ids[start:end],
        document_hashes=batch.document_hashes[start:end],
        chunk_indices=batch.chunk_indices[start:end],
        token_starts=batch.token_starts[start:end],
        token_ends=batch.token_ends[start:end],
        urls=batch.urls[start:end],
        timestamps=batch.timestamps[start:end],
    )
    result.validate()
    return result


def _global_level_gradient(
    model: DistributedDataParallel,
    parent: ParentBatch,
    *,
    active_length: int,
    physical_batch_size: int,
    precision: PrecisionRuntime,
    context: DistributedContext,
) -> tuple[tuple[torch.Tensor, ...], float, int]:
    local_count = parent.causal_target_mask[:, : active_length - 1].sum().to(context.device)
    distributed.all_reduce(local_count, op=distributed.ReduceOp.SUM)
    if int(local_count.item()) <= 0:
        raise RuntimeError(f"global target count is zero at context {active_length}")
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(context.device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    chunk_count = math.ceil(parent.input_ids.shape[0] / physical_batch_size)
    for chunk_index, start in enumerate(range(0, parent.input_ids.shape[0], physical_batch_size)):
        end = min(start + physical_batch_size, parent.input_ids.shape[0])
        chunk = _slice_parent_batch(parent, start, end).prefix(active_length).to(context.device)
        final_chunk = chunk_index == chunk_count - 1
        synchronization = contextlib.nullcontext() if final_chunk else model.no_sync()
        with synchronization:
            with precision.autocast():
                token_losses = model(chunk.input_ids, chunk.attention_mask, chunk.position_ids)
                numerator = token_losses.float().masked_select(chunk.causal_target_mask).sum()
                loss = numerator / local_count.to(dtype=torch.float32) * context.world_size
            loss.backward()
        del chunk, token_losses, numerator, loss
    end_event.record()
    torch.cuda.synchronize(context.device)
    elapsed = torch.tensor(
        float(start_event.elapsed_time(end_event)), device=context.device, dtype=torch.float64
    )
    distributed.all_reduce(elapsed, op=distributed.ReduceOp.MAX)
    gradients = []
    for parameter in model.module.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError("calibration model produced a missing parameter gradient")
        gradients.append(parameter.grad.detach().clone())
    model.zero_grad(set_to_none=True)
    peak = torch.tensor(
        int(torch.cuda.max_memory_allocated(context.device)),
        device=context.device,
        dtype=torch.long,
    )
    distributed.all_reduce(peak, op=distributed.ReduceOp.MAX)
    return tuple(gradients), float(elapsed.item()), int(peak.item())


def _probe_physical_batch_size(
    model: DistributedDataParallel,
    parent: ParentBatch,
    *,
    levels: tuple[int, ...],
    maximum_candidate: int,
    memory_limit_fraction: float,
    precision: PrecisionRuntime,
    context: DistributedContext,
) -> tuple[int, dict[str, Any]]:
    local_parent_count = parent.input_ids.shape[0]
    candidates = [
        candidate
        for candidate in (maximum_candidate, maximum_candidate // 2)
        if candidate > 0 and local_parent_count % candidate == 0
    ]
    candidates = list(dict.fromkeys(candidates))
    probes = []
    total_memory = torch.cuda.get_device_properties(context.device).total_memory
    for candidate in candidates:
        torch.cuda.empty_cache()
        level_gradients = []
        peak = 0
        try:
            for level in levels:
                gradients, _elapsed, level_peak = _global_level_gradient(
                    model,
                    parent,
                    active_length=level,
                    physical_batch_size=candidate,
                    precision=precision,
                    context=context,
                )
                level_gradients.append(gradients)
                peak = max(peak, level_peak)
            if context.is_primary:
                gradient_gram(level_gradients)
            fraction = peak / total_memory
            passed = fraction <= memory_limit_fraction
            error = None
        except torch.OutOfMemoryError as caught:
            passed = False
            fraction = 1.0
            error = f"{type(caught).__name__}: {caught}"
        probes.append(
            {
                "physical_parent_batch_size_per_gpu": candidate,
                "maximum_memory_fraction": fraction,
                "passed": passed,
                "error": error,
            }
        )
        del level_gradients
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        context.barrier()
        if passed:
            return candidate, {
                "memory_limit_fraction": memory_limit_fraction,
                "selected_physical_parent_batch_size_per_gpu": candidate,
                "probes": probes,
            }
    raise RuntimeError(f"exact calibration VRAM probe failed: {probes}")


def _measure_split(
    split_name: str,
    cache: CalibrationParentCache,
    *,
    logical_batch_size: int,
    logical_batch_count: int,
    physical_batch_size: int,
    timing_batch_count: int,
    levels: tuple[int, ...],
    model: DistributedDataParallel,
    precision: PrecisionRuntime,
    context: DistributedContext,
) -> SplitMeasurements | None:
    trainable_parameter_count = sum(
        1 for parameter in model.module.parameters() if parameter.requires_grad
    )
    accumulator = (
        GradientMeanAccumulator(len(levels), trainable_parameter_count)
        if context.is_primary
        else None
    )
    batch_grams = []
    batch_correction_grams = []
    times = []
    memories = []
    audit_vectors: list[torch.Tensor] = []
    for batch_index in range(logical_batch_count):
        parent = cache.rank_batch(
            split_name,
            batch_index,
            logical_batch_size=logical_batch_size,
            rank=context.rank,
            world_size=context.world_size,
        )
        level_gradients = []
        level_times = []
        level_memories = []
        for level in levels:
            gradients, elapsed_ms, peak = _global_level_gradient(
                model,
                parent,
                active_length=level,
                physical_batch_size=physical_batch_size,
                precision=precision,
                context=context,
            )
            level_gradients.append(gradients)
            level_times.append(elapsed_ms)
            level_memories.append(peak)
        if context.is_primary:
            level_gram, correction_gram = gradient_level_and_correction_grams(
                level_gradients
            )
            batch_grams.append(level_gram)
            batch_correction_grams.append(correction_gram)
            retained = accumulator.update(
                level_gradients, retain_full_gradient=split_name == "audit"
            )
            if retained is not None:
                audit_vectors.append(retained)
            if batch_index < timing_batch_count:
                times.append(level_times)
            memories.append(level_memories)
            print(
                json.dumps(
                    {
                        "calibration_split": split_name,
                        "logical_parent_batch": batch_index + 1,
                        "logical_parent_batches": logical_batch_count,
                        "logical_parent_batch_size": logical_batch_size,
                        "physical_parent_batch_size_per_gpu": physical_batch_size,
                        "cuda_time_ms": level_times,
                        "peak_memory_bytes": level_memories,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        del parent, level_gradients
    context.barrier()
    if not context.is_primary:
        return None
    if accumulator is None:
        raise RuntimeError("primary rank has no gradient mean accumulator")
    cross_gram = (
        vector_cross_gram(audit_vectors, device=context.device) if audit_vectors else None
    )
    mean_level_gram, mean_correction_gram = accumulator.mean_grams()
    result = SplitMeasurements(
        batch_level_grams=torch.stack(batch_grams),
        batch_correction_grams=torch.stack(batch_correction_grams),
        mean_level_gram=mean_level_gram,
        mean_correction_gram=mean_correction_gram,
        level_times_ms=torch.tensor(times, dtype=torch.float64),
        level_peak_memory_bytes=torch.tensor(memories, dtype=torch.long),
        full_gradient_cross_gram=cross_gram,
    )
    del audit_vectors, accumulator
    return result


def _relative_gram_residual(direct: torch.Tensor, derived: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(direct)), 1e-30)
    return float(torch.linalg.vector_norm(direct - derived)) / denominator


def _split_statistics(measurements: SplitMeasurements) -> dict[str, Any]:
    mean_batch_level = measurements.batch_level_grams.mean(dim=0)
    mean_batch_correction = measurements.batch_correction_grams.mean(dim=0)
    derived_batch_correction = correction_grams(mean_batch_level)
    mean_gradient_correction = measurements.mean_correction_gram
    derived_mean_gradient_correction = correction_grams(measurements.mean_level_gram)
    statistics: dict[str, Any] = {
        "logical_parent_batches": measurements.batch_level_grams.shape[0],
        "batch_level_grams": measurements.batch_level_grams.tolist(),
        "mean_per_batch_level_gram": mean_batch_level.tolist(),
        "mean_per_batch_level_cosine": cosine_matrix(mean_batch_level).tolist(),
        "mean_per_batch_correction_gram": mean_batch_correction.tolist(),
        "mean_per_batch_correction_cosine": cosine_matrix(mean_batch_correction).tolist(),
        "mean_gradient_level_gram": measurements.mean_level_gram.tolist(),
        "mean_gradient_level_cosine": cosine_matrix(measurements.mean_level_gram).tolist(),
        "mean_gradient_correction_gram": mean_gradient_correction.tolist(),
        "mean_gradient_correction_cosine": cosine_matrix(mean_gradient_correction).tolist(),
        "correction_second_moment_v_k": torch.diagonal(mean_batch_correction).tolist(),
        "per_batch_level_to_direct_correction_gram_relative_residual": (
            _relative_gram_residual(mean_batch_correction, derived_batch_correction)
        ),
        "mean_gradient_level_to_direct_correction_gram_relative_residual": (
            _relative_gram_residual(
                mean_gradient_correction, derived_mean_gradient_correction
            )
        ),
        "maximum_level_peak_memory_bytes": measurements.level_peak_memory_bytes.max(dim=0)
        .values.tolist(),
    }
    if measurements.level_times_ms.numel():
        level_costs = measurements.level_times_ms.mean(dim=0)
        statistics["timed_logical_parent_batches"] = measurements.level_times_ms.shape[0]
        statistics["level_time_samples_ms"] = measurements.level_times_ms.tolist()
        statistics["mean_level_cuda_time_ms_c_k"] = level_costs.tolist()
        statistics["incremental_cuda_time_ms"] = torch.cat(
            (level_costs[:1], level_costs[1:] - level_costs[:-1])
        ).tolist()
    return statistics


def _write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_outputs(
    output: Path,
    *,
    estimator: dict[str, Any],
    report: dict[str, Any],
    audit_passed: bool,
    diagnostic_only: bool,
) -> Path:
    report_path = output.with_suffix(output.suffix + ".report.json")
    if output.exists() or report_path.exists():
        existing = [str(path) for path in (output, report_path) if path.exists()]
        raise FileExistsError(f"calibration output already exists: {existing}")
    _write_json_new(report_path, report)
    if diagnostic_only:
        return report_path
    if not audit_passed:
        raise RuntimeError(f"calibration audit failed; diagnostic report: {report_path}")
    _write_json_new(output, estimator)
    return report_path


def _weight_metadata(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    return {"path": str(path), "sha256": _file_sha256(path), "metadata": metadata}


def _save_initialized_weights(path: Path, model: torch.nn.Module, seed: int) -> None:
    if path.exists():
        raise FileExistsError(f"initialized weight output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    weights = {
        name: tensor.detach().cpu().float().contiguous()
        for name, tensor in model.canonical_state_dict().items()
    }
    try:
        save_file(
            weights,
            str(temporary),
            metadata={
                "format": "umcg_canonical_fp32",
                "source": "calibration_init",
                "seed": str(seed),
            },
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_arguments(arguments: argparse.Namespace, world_size: int) -> None:
    if world_size <= 0:
        raise ValueError("world size must be positive")
    if arguments.logical_parent_batch_size % world_size:
        raise ValueError("logical parent batch size must be divisible by world size")
    local_size = arguments.logical_parent_batch_size // world_size
    if arguments.max_parent_batch_size_per_gpu > local_size:
        raise ValueError("maximum physical parent batch exceeds the rank-local logical batch")
    if local_size % arguments.max_parent_batch_size_per_gpu:
        raise ValueError("rank-local logical batch must be divisible by maximum physical batch")
    if arguments.timing_repeats != 1:
        raise ValueError("the exact scientific protocol requires timing_repeats=1")
    if arguments.diagnostic_only and not arguments.locked_estimator:
        raise ValueError("diagnostic_only requires --locked_estimator")
    if arguments.diagnostic_only and not arguments.initial_weights:
        raise ValueError("diagnostic_only requires checkpoint --initial_weights")


def calibrate(arguments: argparse.Namespace) -> dict[str, Any] | None:
    context = DistributedContext.initialize()
    started = time.time()
    try:
        _validate_arguments(arguments, context.world_size)
        batch_counts = _resolve_parent_batch_counts(arguments)
        output = Path(arguments.output).expanduser().resolve()
        report_path = output.with_suffix(output.suffix + ".report.json")
        cache_path = (
            Path(arguments.parent_cache).expanduser().resolve()
            if arguments.parent_cache
            else output.with_suffix(output.suffix + ".parents.pt")
        )
        existence_error: list[str | None] = [None]
        if context.is_primary and (output.exists() or report_path.exists()):
            existence_error[0] = "calibration output or report already exists"
        distributed.broadcast_object_list(existence_error, src=0)
        if existence_error[0] is not None:
            raise FileExistsError(existence_error[0])

        levels = _levels(arguments.context_preset)
        model_config_path = Path(arguments.model_config).expanduser().resolve()
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
        stream = None
        if context.is_primary and not cache_path.exists():
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
        cache, cache_metadata = _build_or_load_parent_cache(
            cache_path,
            context=context,
            stream=stream,
            logical_batch_size=arguments.logical_parent_batch_size,
            batch_counts=batch_counts,
            maximum_context=levels[-1],
            seed=arguments.seed,
            tokenizer_metadata=tokenizer_metadata,
            source=source,
        )

        attention = resolve_attention_backend(
            arguments.attention_backend,
            device=context.device,
            precision=arguments.precision,
            context_levels=levels,
            num_attention_heads=model_config["num_attention_heads"],
            num_key_value_heads=model_config["num_key_value_heads"],
            head_dimension=model_config["hidden_size"] // model_config["num_attention_heads"],
        )
        seed_all(arguments.seed)
        initial_weights = (
            None
            if arguments.initial_weights is None
            else str(Path(arguments.initial_weights).expanduser().resolve())
        )
        build = build_model(
            model_config_path,
            model_backend=arguments.model_backend,
            precision=arguments.precision,
            attention_selection=attention,
            activation_checkpointing=arguments.activation_checkpointing,
            device=context.device,
            maximum_context=levels[-1],
            initial_weights=initial_weights,
        )
        initialized_weights_path = output.with_suffix(output.suffix + ".initial.safetensors")
        if initial_weights is None and context.is_primary:
            _save_initialized_weights(initialized_weights_path, build.model, arguments.seed)
        context.barrier()
        weights_path = Path(initial_weights) if initial_weights else initialized_weights_path
        weight_metadata = _weight_metadata(weights_path) if context.is_primary else None
        weight_metadata_box = [weight_metadata]
        distributed.broadcast_object_list(weight_metadata_box, src=0)
        weight_metadata = weight_metadata_box[0]

        precision = build_precision_runtime(
            name=arguments.precision,
            device=context.device,
            distributed_backend="ddp",
            model=build.model,
        )
        model = DistributedDataParallel(
            build.model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
        )
        model.train()
        probe_parent = cache.rank_batch(
            "measurement",
            0,
            logical_batch_size=arguments.logical_parent_batch_size,
            rank=context.rank,
            world_size=context.world_size,
        )
        physical_batch_size, batch_probe = _probe_physical_batch_size(
            model,
            probe_parent,
            levels=levels,
            maximum_candidate=arguments.max_parent_batch_size_per_gpu,
            memory_limit_fraction=arguments.memory_limit_fraction,
            precision=precision,
            context=context,
        )
        del probe_parent

        measurements = {}
        for split_name in SPLIT_NAMES:
            value = _measure_split(
                split_name,
                cache,
                logical_batch_size=arguments.logical_parent_batch_size,
                logical_batch_count=batch_counts[split_name],
                physical_batch_size=physical_batch_size,
                timing_batch_count=(
                    min(arguments.timing_parent_batches, batch_counts[split_name])
                    if split_name == "measurement"
                    else 0
                ),
                levels=levels,
                model=model,
                precision=precision,
                context=context,
            )
            if context.is_primary:
                measurements[split_name] = value
        context.barrier()
        if not context.is_primary:
            return None

        measurement = measurements["measurement"]
        selection = measurements["selection"]
        audit = measurements["audit"]
        if measurement is None or selection is None or audit is None:
            raise RuntimeError("primary rank is missing calibration measurements")
        if audit.full_gradient_cross_gram is None:
            raise RuntimeError("audit split is missing the full-gradient cross Gram")
        level_costs = measurement.level_times_ms.mean(dim=0)
        schedules = candidate_schedules(len(levels))
        if len(levels) == 4 and len(schedules) != 35:
            raise RuntimeError(f"expected 35 monotone schedules, found {len(schedules)}")
        selection_evaluations = [
            score_schedule(
                schedule,
                selection.batch_level_grams,
                selection.mean_level_gram,
                level_costs,
            )
            for schedule in schedules
        ]
        recommendation_evaluation = min(
            selection_evaluations, key=lambda item: float(item.summary["objective"])
        )
        recommendation = recommendation_evaluation.summary
        locked_schedule = tuple(float(value) for value in recommendation["tail_probabilities"])
        audit_report, audit_passed = audit_schedule(
            locked_schedule,
            audit.batch_level_grams,
            audit.mean_level_gram,
            audit.full_gradient_cross_gram,
            level_costs,
            bootstrap_samples=arguments.bootstrap_parent_resamples,
            bootstrap_seed=arguments.seed + 30_000,
        )

        estimator = {
            "schema_version": 1,
            "context_levels": list(levels),
            "tail_probabilities": list(locked_schedule),
            "sampling": "shared_global_microbatch",
            "source": {
                "type": "full_coordinate_calibration",
                "criterion": "gradient_variance_times_expected_distributed_cuda_cost",
                "logical_parent_batch_size": arguments.logical_parent_batch_size,
                "measurement_parent_batches": batch_counts["measurement"],
                "selection_parent_batches": batch_counts["selection"],
                "audit_parent_batches": batch_counts["audit"],
                "timing_repeats": arguments.timing_repeats,
                "independent_audit_passed": audit_passed,
                "seed": arguments.seed,
            },
        }
        all_candidates = sorted(
            (evaluation.summary for evaluation in selection_evaluations),
            key=lambda item: float(item["objective"]),
        )
        locked_diagnostic = None
        if arguments.locked_estimator:
            original = EstimatorConfig.load(arguments.locked_estimator)
            if original.context_levels != levels:
                raise ValueError("locked estimator context levels differ from calibration")
            original_schedule = tuple(original.tail_probabilities)
            selection_locked = score_schedule(
                original_schedule,
                selection.batch_level_grams,
                selection.mean_level_gram,
                level_costs,
            )
            audit_locked, _locked_passed = audit_schedule(
                original_schedule,
                audit.batch_level_grams,
                audit.mean_level_gram,
                audit.full_gradient_cross_gram,
                level_costs,
                bootstrap_samples=arguments.bootstrap_parent_resamples,
                bootstrap_seed=arguments.seed + 40_000,
            )
            locked_diagnostic = {
                "estimator_path": str(Path(arguments.locked_estimator).resolve()),
                "selection": selection_locked.summary,
                "independent_audit": audit_locked,
            }

        split_statistics = {
            name: _split_statistics(value) for name, value in measurements.items()
        }
        residual_keys = (
            "per_batch_level_to_direct_correction_gram_relative_residual",
            "mean_gradient_level_to_direct_correction_gram_relative_residual",
        )
        maximum_residual = max(
            float(value[key])
            for value in split_statistics.values()
            for key in residual_keys
        )
        if maximum_residual > 1e-8:
            audit_passed = False
            audit_report["passed"] = False
            audit_report["gram_consistency_passed"] = False
        else:
            audit_report["gram_consistency_passed"] = True
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "estimator": estimator if audit_passed and not arguments.diagnostic_only else None,
            "proposed_estimator": estimator,
            "selection_recommendation": recommendation,
            "independent_audit": audit_report,
            "locked_estimator_diagnostic": locked_diagnostic,
            "split_statistics": split_statistics,
            "candidate_count": len(all_candidates),
            "pareto_candidates": pareto_candidates(all_candidates),
            "all_candidates": all_candidates,
            "gradient_statistics": {
                "method": "full_coordinate_streaming_gram",
                "projection": None,
                "gradient_storage": "current_logical_batch_only",
                "dot_product_accumulator_dtype": "float64",
                "model_parameter_count": build.parameter_count,
                "trainable_parameter_count": build.trainable_parameter_count,
                "audit_temporary_full_gradient_count": batch_counts["audit"],
            },
            "batch_contract": {
                "world_size": context.world_size,
                "logical_parent_batch_size": arguments.logical_parent_batch_size,
                "selected_physical_parent_batch_size_per_gpu": physical_batch_size,
                "parent_samples": {
                    name: batch_counts[name] * arguments.logical_parent_batch_size
                    for name in SPLIT_NAMES
                },
                "vram_probe": batch_probe,
            },
            "timing": {
                "method": "single distributed forward/backward including DDP gradient reduction",
                "timing_parent_batches": min(
                    arguments.timing_parent_batches, batch_counts["measurement"]
                ),
                "repeats_per_parent_batch_and_level": arguments.timing_repeats,
                "rank_aggregation": "maximum",
            },
            "parent_cache": cache_metadata,
            "weights": weight_metadata,
            "tokenizer": tokenizer_metadata,
            "data_source": source,
            "attention": attention.to_dict(),
            "precision": arguments.precision,
            "activation_checkpointing": arguments.activation_checkpointing,
            "optimizer_updates": 0,
            "diagnostic_only": arguments.diagnostic_only,
            "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[3]),
            "canonical_cli": [str(item) for item in sys.argv],
            "wall_time_seconds": time.time() - started,
        }
        written_report = _write_outputs(
            output,
            estimator=estimator,
            report=report,
            audit_passed=audit_passed,
            diagnostic_only=arguments.diagnostic_only,
        )
        return {
            "estimator_config": None if arguments.diagnostic_only else str(output),
            "report": str(written_report),
            "parent_cache": str(cache_path),
            "weights": weight_metadata,
            "audit_passed": audit_passed,
            **recommendation,
        }
    finally:
        context.close()


def main(arguments: Sequence[str] | None = None) -> None:
    result = calibrate(build_parser().parse_args(arguments))
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
