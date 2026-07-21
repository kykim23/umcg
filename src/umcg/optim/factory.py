"""Canonical optimizer families, Muon roles, and two schedulers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

MUON_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


@dataclass(frozen=True)
class ParameterRoles:
    muon_names: tuple[str, ...]
    adamw_names: tuple[str, ...]
    muon_parameter_count: int
    adamw_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "muon_names": list(self.muon_names),
            "adamw_names": list(self.adamw_names),
            "muon_parameter_count": self.muon_parameter_count,
            "adamw_parameter_count": self.adamw_parameter_count,
        }


def classify_muon_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], ParameterRoles]:
    muon_parameters: list[nn.Parameter] = []
    adamw_parameters: list[nn.Parameter] = []
    muon_names: list[str] = []
    adamw_names: list[str] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        is_hidden_matrix = parameter.ndim == 2 and name.endswith(MUON_WEIGHT_SUFFIXES)
        if is_hidden_matrix:
            muon_parameters.append(parameter)
            muon_names.append(name)
        else:
            adamw_parameters.append(parameter)
            adamw_names.append(name)
    if not muon_parameters:
        raise ValueError("Muon parameter classification found no hidden matrices")
    all_trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    classified = {id(parameter) for parameter in (*muon_parameters, *adamw_parameters)}
    if classified != all_trainable:
        raise RuntimeError("Muon parameter classification omitted or duplicated parameters")
    roles = ParameterRoles(
        muon_names=tuple(muon_names),
        adamw_names=tuple(adamw_names),
        muon_parameter_count=sum(parameter.numel() for parameter in muon_parameters),
        adamw_parameter_count=sum(parameter.numel() for parameter in adamw_parameters),
    )
    return muon_parameters, adamw_parameters, roles


class OptimizerController:
    def __init__(
        self,
        optimizers: list[torch.optim.Optimizer],
        parameter_roles: ParameterRoles | None,
    ) -> None:
        if not optimizers:
            raise ValueError("OptimizerController requires at least one optimizer")
        self.optimizers = optimizers
        self.parameter_roles = parameter_roles

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer_classes": [type(item).__name__ for item in self.optimizers],
            "optimizers": [item.state_dict() for item in self.optimizers],
            "parameter_roles": (
                None if self.parameter_roles is None else self.parameter_roles.to_dict()
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        classes = [type(item).__name__ for item in self.optimizers]
        if state["optimizer_classes"] != classes:
            raise ValueError("checkpoint optimizer classes differ from current runtime")
        if len(state["optimizers"]) != len(self.optimizers):
            raise ValueError("checkpoint optimizer count differs from current runtime")
        for optimizer, optimizer_state in zip(self.optimizers, state["optimizers"], strict=True):
            optimizer.load_state_dict(optimizer_state)


def build_optimizer(
    model: nn.Module,
    *,
    name: str,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    momentum: float | None,
) -> OptimizerController:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )
        return OptimizerController([optimizer], None)
    if name == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )
        return OptimizerController([optimizer], None)
    if name == "sgd":
        optimizer = torch.optim.SGD(
            parameters, lr=learning_rate, momentum=0.0, weight_decay=weight_decay
        )
        return OptimizerController([optimizer], None)
    if name == "sgdm":
        resolved_momentum = 0.9 if momentum is None else momentum
        optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=resolved_momentum,
            weight_decay=weight_decay,
        )
        return OptimizerController([optimizer], None)
    if name == "adamw_8bit":
        try:
            from torchao.optim import AdamW8bit
        except ImportError as error:
            raise RuntimeError("optimizer=adamw_8bit requires TorchAO") from error
        optimizer = AdamW8bit(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )
        return OptimizerController([optimizer], None)
    if name == "muon":
        muon_parameters, adamw_parameters, roles = classify_muon_parameters(model)
        muon_optimizer = torch.optim.Muon(
            muon_parameters,
            lr=learning_rate,
            momentum=0.95 if momentum is None else momentum,
            weight_decay=weight_decay,
        )
        adamw_optimizer = torch.optim.AdamW(
            adamw_parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )
        return OptimizerController([muon_optimizer, adamw_optimizer], roles)
    raise ValueError("unsupported optimizer")


def learning_rate_multiplier(
    scheduler: str,
    completed_updates: int,
    total_updates: int,
    warmup_updates: int,
) -> float:
    if scheduler not in {"cosine", "linear"}:
        raise ValueError("scheduler must be cosine or linear")
    if total_updates <= 0 or not 0 <= warmup_updates <= total_updates:
        raise ValueError("invalid scheduler update counts")
    if completed_updates < warmup_updates:
        return float(completed_updates + 1) / max(warmup_updates, 1)
    decay_updates = max(total_updates - warmup_updates, 1)
    progress = min(max((completed_updates - warmup_updates) / decay_updates, 0.0), 1.0)
    if scheduler == "linear":
        return 1.0 - progress
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class SchedulerController:
    def __init__(
        self,
        optimizer_controller: OptimizerController,
        *,
        name: str,
        total_updates: int,
        warmup_updates: int,
    ) -> None:
        def function(step: int) -> float:
            return learning_rate_multiplier(name, step, total_updates, warmup_updates)

        scheduler_optimizers: list[torch.optim.Optimizer] = []
        for optimizer in optimizer_controller.optimizers:
            if isinstance(optimizer, torch.optim.Optimizer):
                scheduler_optimizers.append(optimizer)
                continue
            inner_optimizer = getattr(optimizer, "optimizer", None)
            if not isinstance(inner_optimizer, torch.optim.Optimizer):
                raise TypeError(
                    f"{type(optimizer).__name__} does not expose a PyTorch optimizer"
                )
            scheduler_optimizers.append(inner_optimizer)
        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(optimizer, function)
            for optimizer in scheduler_optimizers
        ]

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        return {"schedulers": [scheduler.state_dict() for scheduler in self.schedulers]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if len(state["schedulers"]) != len(self.schedulers):
            raise ValueError("checkpoint scheduler count differs from current runtime")
        for scheduler, scheduler_state in zip(self.schedulers, state["schedulers"], strict=True):
            scheduler.load_state_dict(scheduler_state)
