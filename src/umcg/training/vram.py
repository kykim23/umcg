"""Destructive-on-disposable-model microbatch search at maximum context."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as distributed

from umcg.config import batch_divisors
from umcg.distributed.runtime import BackendRuntime


@dataclass(frozen=True)
class BatchProbe:
    batch_size: int
    passed: bool
    maximum_memory_fraction: float
    wall_time_seconds: float
    error: str | None


@dataclass(frozen=True)
class BatchSelection:
    batch_size: int
    memory_limit_fraction: float
    probes: tuple[BatchProbe, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "memory_limit_fraction": self.memory_limit_fraction,
            "probes": [asdict(probe) for probe in self.probes],
        }


def find_automatic_batch_size(
    backend: BackendRuntime,
    *,
    total_batch_size: int,
    maximum_context: int,
    vocab_size: int,
    memory_limit_fraction: float = 0.9,
) -> BatchSelection:
    candidates = list(reversed(batch_divisors(total_batch_size, backend.context.world_size)))
    probes: list[BatchProbe] = []
    selected = None
    for candidate in candidates:
        backend.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(backend.context.device)
        started = time.perf_counter()
        local_passed = True
        error_text = None
        try:
            input_ids = torch.randint(
                3,
                vocab_size,
                (candidate, maximum_context),
                device=backend.context.device,
                dtype=torch.long,
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            position_ids = torch.arange(
                maximum_context, device=backend.context.device, dtype=torch.long
            ).expand(candidate, -1)
            with backend.precision.autocast():
                losses = backend.training_model(input_ids, attention_mask, position_ids)
                loss = losses.float().mean()
            backend.backward(loss, accumulation_steps=1)
            if backend.backend_name == "zero":
                if not backend.finish_microbatch(final_microbatch=True):
                    raise FloatingPointError("DeepSpeed skipped the VRAM probe update")
            else:
                for optimizer in backend.optimizer.optimizers:
                    backend.precision.step(optimizer)
                backend.precision.post_optimizer_step(backend.checkpoint_model)
                backend.precision.update()
            torch.cuda.synchronize(backend.context.device)
        except torch.OutOfMemoryError as error:
            local_passed = False
            error_text = f"{type(error).__name__}: {error}"
            backend.zero_grad()
            torch.cuda.empty_cache()
        local_peak = torch.cuda.max_memory_allocated(backend.context.device)
        total_memory = torch.cuda.get_device_properties(backend.context.device).total_memory
        local_fraction = local_peak / total_memory
        passed_tensor = torch.tensor(
            1 if local_passed and local_fraction <= memory_limit_fraction else 0,
            device=backend.context.device,
            dtype=torch.int32,
        )
        fraction_tensor = torch.tensor(
            local_fraction, device=backend.context.device, dtype=torch.float64
        )
        distributed.all_reduce(passed_tensor, op=distributed.ReduceOp.MIN)
        distributed.all_reduce(fraction_tensor, op=distributed.ReduceOp.MAX)
        passed = bool(passed_tensor.item())
        probes.append(
            BatchProbe(
                batch_size=candidate,
                passed=passed,
                maximum_memory_fraction=float(fraction_tensor.item()),
                wall_time_seconds=time.perf_counter() - started,
                error=error_text,
            )
        )
        if passed:
            selected = candidate
        else:
            break
    if selected is None:
        raise RuntimeError(
            "batch_size=auto found no divisor that stays within 90% GPU memory: "
            f"{[asdict(probe) for probe in probes]}"
        )
    return BatchSelection(selected, memory_limit_fraction, tuple(probes))
