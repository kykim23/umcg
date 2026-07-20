"""Strict FP32, FP16, BF16, and TorchAO mixed-FP8 policies."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.metadata
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class PrecisionRuntime:
    name: str
    device: torch.device
    distributed_backend: str
    scaler: torch.amp.GradScaler | None
    float8_metadata: dict[str, Any] | None

    def autocast(self):
        if self.name == "float32":
            return contextlib.nullcontext()
        dtype = torch.float16 if self.name == "float16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is None:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is None:
            optimizer.step()
        else:
            self.scaler.step(optimizer)

    def update(self) -> None:
        if self.scaler is not None:
            self.scaler.update()

    def backoff_after_nonfinite_gradients(self) -> None:
        if self.scaler is None:
            raise FloatingPointError("non-finite gradients without a dynamic GradScaler")
        next_scale = self.scaler.get_scale() * self.scaler.get_backoff_factor()
        self.scaler.update(new_scale=next_scale)

    @property
    def loss_scale(self) -> float | None:
        return None if self.scaler is None else float(self.scaler.get_scale())

    def post_optimizer_step(self, model: nn.Module) -> None:
        if self.name != "float8" or self.distributed_backend != "fsdp2":
            return
        try:
            from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
        except ImportError as error:
            raise RuntimeError("installed TorchAO lacks FSDP2 FP8 scale synchronization") from error
        precompute_float8_dynamic_scale_for_fsdp(model)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "float8_metadata": self.float8_metadata,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["name"] != self.name:
            raise ValueError("checkpoint precision differs from current precision")
        if (state["scaler"] is None) != (self.scaler is None):
            raise ValueError("checkpoint scaler presence differs from current precision")
        if self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])
        if state.get("float8_metadata") != self.float8_metadata:
            raise ValueError("checkpoint TorchAO FP8 metadata differs from current runtime")


def _torchao_version() -> str:
    try:
        return importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("precision=float8 requires TorchAO") from error


def apply_float8_training(
    model: nn.Module,
    device: torch.device,
    distributed_backend: str,
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("precision=float8 requires a CUDA device")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (8, 9):
        raise RuntimeError(
            "precision=float8 requires native FP8 Tensor Cores (Ada, Hopper, or Blackwell)"
        )
    version = _torchao_version()
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as error:
        raise RuntimeError("installed TorchAO does not provide float8 training APIs") from error

    def module_filter(module: nn.Module, fully_qualified_name: str) -> bool:
        eligible_role = ".self_attn." in fully_qualified_name or ".mlp." in fully_qualified_name
        if not isinstance(module, nn.Linear) or not eligible_role:
            return False
        return module.in_features % 16 == 0 and module.out_features % 16 == 0

    recipe = Float8LinearConfig.from_recipe_name("tensorwise")
    fsdp_float8_all_gather = distributed_backend == "fsdp2"
    if fsdp_float8_all_gather:
        recipe = dataclasses.replace(recipe, enable_fsdp_float8_all_gather=True)
    convert_to_float8_training(model, config=recipe, module_filter_fn=module_filter)
    converted = [
        name
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Float8Linear"
    ]
    if not converted:
        raise RuntimeError("TorchAO converted no attention or MLP Linear modules to FP8")
    return {
        "torchao_version": version,
        "recipe": "tensorwise",
        "fsdp_float8_all_gather": fsdp_float8_all_gather,
        "converted_module_count": len(converted),
        "attention_precision": "bfloat16",
        "normalization_precision": "bfloat16_or_float32",
        "loss_reduction_precision": "float32",
    }


def build_precision_runtime(
    *,
    name: str,
    device: torch.device,
    distributed_backend: str,
    model: nn.Module,
) -> PrecisionRuntime:
    if name not in {"float32", "float16", "bfloat16", "float8"}:
        raise ValueError("unsupported precision")
    if name == "float8" and device.type != "cuda":
        raise RuntimeError("precision=float8 requires a CUDA device")
    if name != "float32" and device.type != "cuda":
        raise RuntimeError(f"precision={name} requires CUDA in the production runner")
    float8_metadata = (
        apply_float8_training(model, device, distributed_backend) if name == "float8" else None
    )
    scaler = None
    if name == "float16" and distributed_backend != "zero":
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    return PrecisionRuntime(name, device, distributed_backend, scaler, float8_metadata)
