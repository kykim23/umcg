"""Local-only attention capability checks and distributed selection."""

from __future__ import annotations

import contextlib
import importlib.metadata
import math
import statistics
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as distributed
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class AttentionHardware:
    cuda_available: bool
    capability_major: int
    capability_minor: int
    device_name: str

    @classmethod
    def detect(cls, device: torch.device) -> AttentionHardware:
        if device.type != "cuda":
            return cls(False, 0, 0, "CPU")
        major, minor = torch.cuda.get_device_capability(device)
        return cls(True, major, minor, torch.cuda.get_device_name(device))


@dataclass(frozen=True)
class AttentionProbeResult:
    backend: str
    passed: bool
    worst_level_time_ms: float | None
    level_times_ms: dict[int, float]
    error: str | None


@dataclass(frozen=True)
class AttentionSelection:
    requested_backend: str
    resolved_backend: str
    huggingface_implementation: str
    hardware: dict[str, Any]
    package_versions: dict[str, str | None]
    rank_probe_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_attention_packages() -> dict[str, str | None]:
    return {
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "flash-attn": _package_version("flash-attn"),
        "flash-attn-3": _package_version("flash-attn-3"),
        "flash-attn-4": _package_version("flash-attn-4"),
    }


def candidate_backends(
    hardware: AttentionHardware,
    *,
    precision: str,
    flash_available: dict[int, bool] | None = None,
) -> list[str]:
    if flash_available is None:
        try:
            from transformers.utils import (
                is_flash_attn_2_available,
                is_flash_attn_3_available,
                is_flash_attn_4_available,
            )

            flash_available = {
                2: bool(is_flash_attn_2_available()),
                3: bool(is_flash_attn_3_available()),
                4: bool(is_flash_attn_4_available()),
            }
        except ImportError:
            flash_available = {2: False, 3: False, 4: False}
    compute_precision = "bfloat16" if precision == "float8" else precision
    candidates: list[str] = []
    if hardware.cuda_available and compute_precision in {"float16", "bfloat16"}:
        if hardware.capability_major >= 9 and flash_available.get(4, False):
            candidates.append("flash_attention_4")
        if hardware.capability_major >= 9 and flash_available.get(3, False):
            candidates.append("flash_attention_3")
        if hardware.capability_major >= 8 and flash_available.get(2, False):
            candidates.append("flash_attention_2")
    if hardware.cuda_available:
        candidates.extend(
            (
                "pytorch_sdpa_flash",
                "pytorch_sdpa_cudnn",
                "pytorch_sdpa_efficient",
                "pytorch_sdpa_math",
            )
        )
    else:
        candidates.append("pytorch_sdpa_math")
    candidates.append("eager")
    return candidates


def huggingface_attention_name(resolved_backend: str) -> str:
    if resolved_backend.startswith("pytorch_sdpa"):
        return "sdpa"
    return resolved_backend


@contextlib.contextmanager
def attention_kernel_context(resolved_backend: str):
    mapping = {
        "pytorch_sdpa_flash": SDPBackend.FLASH_ATTENTION,
        "pytorch_sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
        "pytorch_sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "pytorch_sdpa_math": SDPBackend.MATH,
    }
    selected = mapping.get(resolved_backend)
    if selected is None:
        yield
    else:
        with sdpa_kernel(backends=[selected]):
            yield


def _repeat_key_value(states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return states
    return states.repeat_interleave(groups, dim=1)


def run_attention(
    backend: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("attention dropout must be in [0, 1)")
    groups = query.shape[1] // key.shape[1]
    repeated_key = _repeat_key_value(key, groups)
    repeated_value = _repeat_key_value(value, groups)
    sequence_length = query.shape[2]
    causal = torch.ones(
        (sequence_length, sequence_length), device=query.device, dtype=torch.bool
    ).tril()
    allowed = causal.unsqueeze(0).unsqueeze(0) & attention_mask_2d[:, None, None, :]
    if backend == "eager":
        weights = torch.matmul(query, repeated_key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
        weights = weights.masked_fill(~allowed, torch.finfo(weights.dtype).min)
        probabilities = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        probabilities = functional.dropout(probabilities, p=dropout_p, training=dropout_p > 0)
        return torch.matmul(probabilities, repeated_value)
    if backend.startswith("pytorch_sdpa"):
        with attention_kernel_context(backend):
            return functional.scaled_dot_product_attention(
                query,
                repeated_key,
                repeated_value,
                attn_mask=allowed,
                dropout_p=dropout_p,
                is_causal=False,
            )
    from transformers import AttentionInterface

    class _AttentionConfig:
        _attn_implementation = backend

    class _AttentionMetadata:
        config = _AttentionConfig()
        num_key_value_groups = groups
        is_causal = True
        training = True
        layer_idx = 0

    function = AttentionInterface()[backend]
    output, _ = function(
        _AttentionMetadata(),
        query,
        key,
        value,
        attention_mask_2d,
        dropout=dropout_p,
        scaling=query.shape[-1] ** -0.5,
        is_causal=True,
    )
    if output.shape == (query.shape[0], query.shape[2], query.shape[1], query.shape[3]):
        output = output.transpose(1, 2)
    return output


def _run_unpadded_attention(
    backend: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask_2d: torch.Tensor,
) -> torch.Tensor:
    if backend.startswith("pytorch_sdpa"):
        groups = query.shape[1] // key.shape[1]
        with attention_kernel_context(backend):
            return functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=groups > 1,
            )
    return run_attention(backend, query, key, value, attention_mask_2d)


def _assert_attention_close(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    tolerance: float,
) -> None:
    if not torch.isfinite(candidate).all():
        raise FloatingPointError("non-finite forward output")
    torch.testing.assert_close(candidate.float(), reference.float(), rtol=tolerance, atol=tolerance)
    reference_gradients = torch.autograd.grad(reference.float().sum(), inputs, allow_unused=False)
    candidate_gradients = torch.autograd.grad(candidate.float().sum(), inputs, allow_unused=False)
    if any(
        not torch.isfinite(gradient).all()
        for gradient in (*reference_gradients, *candidate_gradients)
    ):
        raise FloatingPointError("non-finite backward output")
    for candidate_gradient, reference_gradient in zip(
        candidate_gradients, reference_gradients, strict=True
    ):
        torch.testing.assert_close(
            candidate_gradient.float(),
            reference_gradient.float(),
            rtol=tolerance,
            atol=tolerance,
        )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _probe_attention_backend(
    backend: str,
    *,
    device: torch.device,
    precision: str,
    context_levels: tuple[int, ...],
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dimension: int,
) -> AttentionProbeResult:
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8": torch.bfloat16,
    }[precision]
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32
    try:
        correctness_length = min(context_levels[0], 128)
        generator = torch.Generator(device=device)
        generator.manual_seed(1729)
        shapes = (
            (1, num_attention_heads, correctness_length, head_dimension),
            (1, num_key_value_heads, correctness_length, head_dimension),
        )
        query = torch.randn(
            shapes[0], generator=generator, device=device, dtype=dtype, requires_grad=True
        )
        key = torch.randn(
            shapes[1], generator=generator, device=device, dtype=dtype, requires_grad=True
        )
        value = torch.randn(
            shapes[1], generator=generator, device=device, dtype=dtype, requires_grad=True
        )
        tolerance = 2e-4 if dtype == torch.float32 else 5e-2
        full_mask = torch.ones((1, correctness_length), device=device, dtype=torch.bool)
        full_reference = run_attention("eager", query, key, value, full_mask)
        full_candidate = _run_unpadded_attention(backend, query, key, value, full_mask)
        _assert_attention_close(
            full_candidate,
            full_reference,
            (query, key, value),
            tolerance=tolerance,
        )
        padded_mask = full_mask.clone()
        if correctness_length > 8:
            padded_mask[:, -7:] = False
        padded_reference = run_attention("eager", query, key, value, padded_mask)
        padded_candidate = run_attention(backend, query, key, value, padded_mask)
        _assert_attention_close(
            padded_candidate,
            padded_reference,
            (query, key, value),
            tolerance=tolerance,
        )

        timings: dict[int, float] = {}
        benchmark_heads = num_attention_heads
        benchmark_key_heads = num_key_value_heads
        for level in context_levels:
            q = torch.randn(
                (1, benchmark_heads, level, head_dimension),
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            k = torch.randn(
                (1, benchmark_key_heads, level, head_dimension),
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            v = torch.randn_like(k, requires_grad=True)
            full_level_mask = torch.ones((1, level), device=device, dtype=torch.bool)
            padded_level_mask = full_level_mask.clone()
            if level > 8:
                padded_level_mask[:, -7:] = False
            scenario_times = []
            for level_mask, unpadded in (
                (full_level_mask, True),
                (padded_level_mask, False),
            ):
                repetitions = []
                for _ in range(3):
                    q.grad = None
                    k.grad = None
                    v.grad = None
                    _synchronize(device)
                    started = time.perf_counter()
                    output = (
                        _run_unpadded_attention(backend, q, k, v, level_mask)
                        if unpadded
                        else run_attention(backend, q, k, v, level_mask)
                    )
                    output.float().square().mean().backward()
                    _synchronize(device)
                    repetitions.append((time.perf_counter() - started) * 1000.0)
                    del output
                scenario_times.append(statistics.median(repetitions))
            timings[level] = max(scenario_times)
            del q, k, v
        return AttentionProbeResult(
            backend=backend,
            passed=True,
            worst_level_time_ms=max(timings.values()),
            level_times_ms=timings,
            error=None,
        )
    except Exception as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return AttentionProbeResult(
            backend=backend,
            passed=False,
            worst_level_time_ms=None,
            level_times_ms={},
            error=f"{type(error).__name__}: {error}",
        )


def probe_attention_backend(
    backend: str,
    *,
    device: torch.device,
    precision: str,
    context_levels: tuple[int, ...],
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dimension: int,
) -> AttentionProbeResult:
    # Unsupported kernels report their reason through the structured probe
    # result. PyTorch also emits the same reason as warnings, which would make
    # every automatic startup unnecessarily noisy.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return _probe_attention_backend(
            backend,
            device=device,
            precision=precision,
            context_levels=context_levels,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dimension=head_dimension,
        )


def resolve_attention_backend(
    requested_backend: str,
    *,
    device: torch.device,
    precision: str,
    context_levels: tuple[int, ...],
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dimension: int,
    probe: Callable[..., AttentionProbeResult] = probe_attention_backend,
) -> AttentionSelection:
    hardware = AttentionHardware.detect(device)
    candidates = candidate_backends(hardware, precision=precision)
    if requested_backend != "automatic":
        requested_candidates = (
            ["pytorch_sdpa"] if requested_backend == "pytorch_sdpa" else [requested_backend]
        )
    else:
        requested_candidates = candidates
    local_results: list[AttentionProbeResult] = []
    for candidate in requested_candidates:
        if candidate == "pytorch_sdpa":
            # Explicit pytorch_sdpa allows PyTorch to choose its own installed kernel.
            candidate = "pytorch_sdpa"
        elif candidate not in candidates:
            local_results.append(
                AttentionProbeResult(candidate, False, None, {}, "capability unavailable")
            )
            continue
        local_results.append(
            probe(
                candidate,
                device=device,
                precision=precision,
                context_levels=context_levels,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dimension=head_dimension,
            )
        )
    local_payload = [asdict(item) for item in local_results]
    if distributed.is_available() and distributed.is_initialized():
        rank_payloads: list[list[dict[str, Any]] | None] = [None] * distributed.get_world_size()
        distributed.all_gather_object(rank_payloads, local_payload)
        all_rank_payloads = [payload for payload in rank_payloads if payload is not None]
    else:
        all_rank_payloads = [local_payload]

    common: dict[str, float] = {}
    for candidate in requested_candidates:
        records = [
            next((item for item in payload if item["backend"] == candidate), None)
            for payload in all_rank_payloads
        ]
        if all(record is not None and record["passed"] for record in records):
            durations = [record["worst_level_time_ms"] for record in records]
            if any(duration is None for duration in durations):
                raise RuntimeError("passing attention probe omitted its benchmark duration")
            common[candidate] = max(float(duration) for duration in durations)
    if not common:
        errors = {f"rank_{rank}": payload for rank, payload in enumerate(all_rank_payloads)}
        raise RuntimeError(
            f"attention backend {requested_backend!r} passed on no common candidate: {errors}"
        )
    if requested_backend != "automatic":
        resolved = next(iter(common))
    else:
        fastest = min(common.values())
        close = {name for name, duration in common.items() if duration <= fastest * 1.03}
        pytorch_preference = (
            "pytorch_sdpa_flash",
            "pytorch_sdpa_cudnn",
            "pytorch_sdpa_efficient",
            "pytorch_sdpa_math",
        )
        resolved = next((name for name in pytorch_preference if name in close), None)
        if resolved is None:
            resolved = min(common, key=common.get)
    flattened = [
        {"rank": rank, "results": payload} for rank, payload in enumerate(all_rank_payloads)
    ]
    return AttentionSelection(
        requested_backend=requested_backend,
        resolved_backend=resolved,
        huggingface_implementation=huggingface_attention_name(resolved),
        hardware=asdict(hardware),
        package_versions=installed_attention_packages(),
        rank_probe_results=flattened,
    )


def verify_checkpoint_attention_selection(
    stored: dict[str, Any],
    *,
    device: torch.device,
    precision: str,
    context_levels: tuple[int, ...],
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dimension: int,
) -> AttentionSelection:
    current_versions = installed_attention_packages()
    if stored["package_versions"] != current_versions:
        raise RuntimeError(
            "checkpoint attention package versions are unavailable or changed: "
            f"checkpoint={stored['package_versions']}, current={current_versions}"
        )
    resolved = str(stored["resolved_backend"])
    result = probe_attention_backend(
        resolved,
        device=device,
        precision=precision,
        context_levels=context_levels,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dimension=head_dimension,
    )
    local = torch.tensor(1 if result.passed else 0, device=device, dtype=torch.int32)
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(local, op=distributed.ReduceOp.MIN)
    if not bool(local.item()):
        raise RuntimeError(
            f"checkpoint attention backend no longer passes startup verification: {asdict(result)}"
        )
    return AttentionSelection(
        requested_backend=str(stored["requested_backend"]),
        resolved_backend=resolved,
        huggingface_implementation=str(stored["huggingface_implementation"]),
        hardware=asdict(AttentionHardware.detect(device)),
        package_versions=current_versions,
        rank_probe_results=[{"resume_verification": asdict(result)}],
    )
