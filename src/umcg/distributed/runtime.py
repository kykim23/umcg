"""Single-node distributed lifecycle and backend-specific mechanics."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from umcg.config import RuntimeConfig
from umcg.optim.factory import (
    OptimizerController,
    ParameterRoles,
    SchedulerController,
    build_optimizer,
    classify_muon_parameters,
)
from umcg.precision import PrecisionRuntime


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls) -> DistributedContext:
        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise RuntimeError(
                "torchrun_main.py must be launched with torchrun; missing environment "
                f"variables: {missing}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("production pretraining requires CUDA")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        if local_world_size != world_size:
            raise RuntimeError("production pretraining supports a single node only")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError("LOCAL_RANK is outside the visible CUDA device set")
        torch.cuda.set_device(local_rank)
        if not distributed.is_initialized():
            distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=torch.device("cuda", local_rank),
            )
        return cls(rank, local_rank, world_size, torch.device("cuda", local_rank))

    def barrier(self) -> None:
        distributed.barrier()

    def close(self) -> None:
        if distributed.is_initialized():
            distributed.destroy_process_group()


@dataclass
class BackendRuntime:
    context: DistributedContext
    backend_name: str
    training_model: nn.Module
    checkpoint_model: nn.Module
    optimizer: OptimizerController
    scheduler: SchedulerController
    precision: PrecisionRuntime
    fsdp_root: nn.Module | None = None
    deepspeed_engine: Any | None = None
    parameter_roles: ParameterRoles | None = None

    @property
    def loss_scale(self) -> float:
        return float(self.context.world_size)

    @contextlib.contextmanager
    def synchronization_context(self, *, final_microbatch: bool):
        if self.backend_name == "ddp" and not final_microbatch:
            with self.training_model.no_sync():
                yield
            return
        if self.backend_name == "fsdp2" and self.fsdp_root is not None:
            self.fsdp_root.set_requires_gradient_sync(final_microbatch)
            self.fsdp_root.set_reshard_after_backward(final_microbatch)
            try:
                yield
            finally:
                if final_microbatch:
                    self.fsdp_root.set_requires_gradient_sync(True)
                    self.fsdp_root.set_reshard_after_backward(True)
            return
        yield

    def backward(self, loss: torch.Tensor, *, accumulation_steps: int) -> None:
        if self.backend_name == "zero":
            # DeepSpeed divides each microbatch loss by its configured accumulation
            # count. The production objective is already normalized over the whole
            # update, so undo that division before DeepSpeed backward.
            self.deepspeed_engine.backward(loss * accumulation_steps)
        else:
            self.precision.backward(loss)

    def finish_microbatch(self, *, final_microbatch: bool, gradients_finite: bool = True) -> bool:
        if self.backend_name == "zero":
            self.deepspeed_engine.step()
            applied = bool(self.deepspeed_engine.was_step_applied())
            if applied and not final_microbatch:
                raise RuntimeError(
                    "DeepSpeed gradient-accumulation boundary differs from the derived contract"
                )
            if final_microbatch and applied:
                self.scheduler.step()
            if final_microbatch and not applied and self.precision.name != "float16":
                raise FloatingPointError("DeepSpeed skipped a non-FP16 optimizer update")
            return final_microbatch and applied
        if not final_microbatch:
            return False
        if not gradients_finite:
            self.precision.backoff_after_nonfinite_gradients()
            self.optimizer.zero_grad(set_to_none=True)
            return False
        for optimizer in self.optimizer.optimizers:
            self.precision.step(optimizer)
        self.precision.post_optimizer_step(self.checkpoint_model)
        self.precision.update()
        self.scheduler.step()
        return True

    def zero_grad(self) -> None:
        if self.backend_name == "zero":
            self.deepspeed_engine.zero_grad()
        else:
            self.optimizer.zero_grad(set_to_none=True)


def _mixed_precision_policy(precision: str):
    from torch.distributed.fsdp import MixedPrecisionPolicy

    dtype = {
        "float32": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8": torch.bfloat16,
    }[precision]
    return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype, output_dtype=dtype)


def _assign_deepspeed_muon_roles(model: nn.Module) -> ParameterRoles:
    _, _, roles = classify_muon_parameters(model)
    muon_names = set(roles.muon_names)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.use_muon = name in muon_names
    return roles


def _verify_deepspeed_muon_roles(model: nn.Module, roles: ParameterRoles) -> None:
    expected = set(roles.muon_names)
    actual = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and getattr(parameter, "use_muon", None) is True
    }
    missing_attributes = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not hasattr(parameter, "use_muon")
    ]
    if missing_attributes or actual != expected:
        raise RuntimeError(
            "DeepSpeed Muon parameter roles differ from the canonical LLaMA roles; "
            f"missing_attributes={missing_attributes[:10]}, "
            f"unexpected_muon={sorted(actual - expected)[:10]}, "
            f"missing_muon={sorted(expected - actual)[:10]}"
        )


def _build_deepspeed_config(
    config: RuntimeConfig,
    *,
    world_size: int,
    batch_size: int,
    accumulation_steps: int,
) -> dict[str, Any]:
    if config.total_batch_size != batch_size * world_size * accumulation_steps:
        raise ValueError("DeepSpeed batch values differ from the derived global batch contract")
    optimizer_parameters: dict[str, Any]
    if config.optimizer in {"adamw", "adam"}:
        optimizer_type = "Adam"
        optimizer_parameters = {
            "lr": config.learning_rate,
            "betas": [config.beta1, config.beta2],
            "eps": config.epsilon,
            "weight_decay": config.weight_decay,
            "torch_adam": True,
            "adam_w_mode": config.optimizer == "adamw",
        }
    elif config.optimizer in {"sgd", "sgdm"}:
        optimizer_type = "SGD"
        optimizer_parameters = {
            "lr": config.learning_rate,
            "momentum": (
                0.0
                if config.optimizer == "sgd"
                else (0.9 if config.momentum is None else config.momentum)
            ),
            "weight_decay": config.weight_decay,
        }
    elif config.optimizer == "muon":
        optimizer_type = "Muon"
        optimizer_parameters = {
            "lr": config.learning_rate,
            "betas": [config.beta1, config.beta2],
            "eps": config.epsilon,
            "momentum": 0.95 if config.momentum is None else config.momentum,
            "weight_decay": config.weight_decay,
            "muon_lr": config.learning_rate,
            "adam_lr": config.learning_rate,
            "ns_method": "gram",
        }
    else:
        raise ValueError("unsupported optimizer for DeepSpeed")
    return {
        "train_batch_size": config.total_batch_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "optimizer": {"type": optimizer_type, "params": optimizer_parameters},
        "zero_optimization": {
            "stage": config.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "fp16": {"enabled": config.precision == "float16"},
        "bf16": {"enabled": config.precision == "bfloat16"},
        "gradient_clipping": config.gradient_clip_norm or 0.0,
        "steps_per_print": 1_000_000_000,
        "wall_clock_breakdown": True,
        "zero_allow_untested_optimizer": False,
    }


def build_backend_runtime(
    model: nn.Module,
    *,
    config: RuntimeConfig,
    context: DistributedContext,
    precision: PrecisionRuntime,
    batch_size: int,
    accumulation_steps: int,
) -> BackendRuntime:
    parameter_roles = None
    if config.distributed_backend == "ddp":
        inner = (
            torch.compile(model, mode=config.compile_mode) if config.use_torch_compile else model
        )
        training_model = DistributedDataParallel(
            inner,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
        )
        optimizer = build_optimizer(
            model,
            name=config.optimizer,
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
        scheduler = SchedulerController(
            optimizer,
            name=config.scheduler,
            total_updates=config.num_training_steps,
            warmup_updates=config.warmup_steps,
        )
        return BackendRuntime(
            context,
            "ddp",
            training_model,
            model,
            optimizer,
            scheduler,
            precision,
            parameter_roles=optimizer.parameter_roles,
        )
    if config.distributed_backend == "fsdp2":
        from torch.distributed.fsdp import fully_shard

        policy = _mixed_precision_policy(config.precision)
        for layer in model.decoder_layers:
            fully_shard(layer, reshard_after_forward=True, mp_policy=policy)
        fully_shard(model, reshard_after_forward=False, mp_policy=policy)
        fsdp_root = model
        training_model = (
            torch.compile(model, mode=config.compile_mode) if config.use_torch_compile else model
        )
        optimizer = build_optimizer(
            model,
            name=config.optimizer,
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
        scheduler = SchedulerController(
            optimizer,
            name=config.scheduler,
            total_updates=config.num_training_steps,
            warmup_updates=config.warmup_steps,
        )
        return BackendRuntime(
            context,
            "fsdp2",
            training_model,
            model,
            optimizer,
            scheduler,
            precision,
            fsdp_root=fsdp_root,
        )
    if config.distributed_backend == "zero":
        try:
            import deepspeed
        except ImportError as error:
            raise RuntimeError("distributed_backend=zero requires DeepSpeed") from error
        if config.optimizer == "muon":
            parameter_roles = _assign_deepspeed_muon_roles(model)
        inner = (
            torch.compile(model, mode=config.compile_mode) if config.use_torch_compile else model
        )
        deepspeed_config = _build_deepspeed_config(
            config,
            world_size=context.world_size,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
        )
        engine, deepspeed_optimizer, _, _ = deepspeed.initialize(
            model=inner,
            model_parameters=inner.named_parameters(),
            config=deepspeed_config,
            dist_init_required=False,
        )
        if parameter_roles is not None:
            _verify_deepspeed_muon_roles(model, parameter_roles)
        optimizer = OptimizerController([deepspeed_optimizer], parameter_roles)
        scheduler = SchedulerController(
            optimizer,
            name=config.scheduler,
            total_updates=config.num_training_steps,
            warmup_updates=config.warmup_steps,
        )
        return BackendRuntime(
            context,
            "zero",
            engine,
            model,
            optimizer,
            scheduler,
            precision,
            deepspeed_engine=engine,
            parameter_roles=parameter_roles,
        )
    raise ValueError("unsupported distributed backend")
