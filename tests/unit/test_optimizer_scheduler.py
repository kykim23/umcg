import pytest
import torch
from torch import nn

from umcg.optim.factory import (
    OptimizerController,
    SchedulerController,
    build_optimizer,
    classify_muon_parameters,
    learning_rate_multiplier,
)


class ToyDecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.k_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.v_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.o_proj = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(8, 16, bias=False)
        self.mlp.up_proj = nn.Linear(8, 16, bias=False)
        self.mlp.down_proj = nn.Linear(16, 8, bias=False)
        self.input_layernorm = nn.LayerNorm(8)


class ToyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, 8)
        self.model.layers = nn.ModuleList([ToyDecoderBlock()])
        self.model.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


def optimizer_arguments(name, momentum=None):
    return {
        "name": name,
        "learning_rate": 1e-3,
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "momentum": momentum,
    }


def test_muon_parameter_roles_are_fixed_and_tied_weight_is_seen_once():
    model = ToyTiedModel()
    muon, adamw, roles = classify_muon_parameters(model)
    assert len(muon) == 7
    assert all(".self_attn." in name or ".mlp." in name for name in roles.muon_names)
    assert "model.embed_tokens.weight" in roles.adamw_names
    assert all(name != "lm_head.weight" for name in roles.adamw_names)
    classified_ids = [id(parameter) for parameter in (*muon, *adamw)]
    assert len(classified_ids) == len(set(classified_ids))
    assert id(model.model.embed_tokens.weight) in {id(parameter) for parameter in adamw}


def test_sgd_and_sgdm_have_distinct_momentum_contracts():
    model = nn.Linear(4, 4)
    sgd = build_optimizer(model, **optimizer_arguments("sgd"))
    sgdm = build_optimizer(model, **optimizer_arguments("sgdm"))
    explicit = build_optimizer(model, **optimizer_arguments("sgdm", momentum=0.8))
    assert sgd.optimizers[0].param_groups[0]["momentum"] == 0.0
    assert sgdm.optimizers[0].param_groups[0]["momentum"] == 0.9
    assert explicit.optimizers[0].param_groups[0]["momentum"] == 0.8


def test_cosine_and_linear_scheduler_formulas():
    assert learning_rate_multiplier("linear", 0, 10, 2) == pytest.approx(0.5)
    assert learning_rate_multiplier("linear", 1, 10, 2) == pytest.approx(1.0)
    assert learning_rate_multiplier("linear", 6, 10, 2) == pytest.approx(0.5)
    assert learning_rate_multiplier("linear", 10, 10, 2) == pytest.approx(0.0)
    assert learning_rate_multiplier("cosine", 2, 10, 2) == pytest.approx(1.0)
    assert learning_rate_multiplier("cosine", 6, 10, 2) == pytest.approx(0.5)
    assert learning_rate_multiplier("cosine", 10, 10, 2) == pytest.approx(0.0)


def test_scheduler_uses_pytorch_optimizer_inside_a_wrapper():
    parameter = nn.Parameter(torch.ones(1))
    inner = torch.optim.SGD([parameter], lr=1.0)

    class OptimizerWrapper:
        def __init__(self):
            self.optimizer = inner

        @property
        def param_groups(self):
            return self.optimizer.param_groups

    controller = OptimizerController([OptimizerWrapper()], None)
    scheduler = SchedulerController(
        controller,
        name="linear",
        total_updates=4,
        warmup_updates=0,
    )
    inner.step()
    scheduler.step()
    assert inner.param_groups[0]["lr"] == pytest.approx(0.75)
