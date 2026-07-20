import dataclasses

from torch import nn

from umcg.config import RuntimeConfig
from umcg.distributed.runtime import (
    _assign_deepspeed_muon_roles,
    _build_deepspeed_config,
)


class ToyDecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(8, 16, bias=False)
        self.norm = nn.LayerNorm(8)


class ToyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, 8)
        self.model.layers = nn.ModuleList([ToyDecoderBlock()])
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


def runtime_config(**changes):
    value = RuntimeConfig(
        distributed_backend="zero",
        zero_stage=2,
        model_backend="huggingface",
        model_config="model.json",
        tokenizer="t5-base",
        tokenizer_revision="main",
        precision="bfloat16",
        attention_backend="automatic",
        estimator_config="estimator.json",
        gradient_estimator="full",
        optimizer="adamw",
        scheduler="cosine",
        learning_rate=3e-4,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=0.1,
        momentum=None,
        batch_size="4",
        total_batch_size=32,
        num_training_steps=100,
        warmup_steps=10,
        eval_every=10,
        eval_parent_batches=2,
        save_every=10,
        save_dir="out",
        c4_source="streaming",
        c4_repo="allenai/c4",
        c4_revision="main",
        c4_local_path=None,
        workers=1,
        seed=777,
        use_torch_compile=False,
        compile_mode="default",
        activation_checkpointing=False,
        gradient_clip_norm=None,
        wandb_mode="disabled",
        wandb_project="umcg",
        wandb_entity=None,
        name="test",
        continue_from=None,
        initial_weights=None,
    )
    return dataclasses.replace(value, **changes)


def test_internal_zero_config_has_no_cpu_or_nvme_offload():
    value = _build_deepspeed_config(
        runtime_config(), world_size=2, batch_size=4, accumulation_steps=4
    )
    assert value["train_batch_size"] == 32
    assert value["train_micro_batch_size_per_gpu"] == 4
    assert value["gradient_accumulation_steps"] == 4
    assert value["zero_optimization"]["stage"] == 2
    assert "offload_param" not in value["zero_optimization"]
    assert "offload_optimizer" not in value["zero_optimization"]


def test_internal_zero_config_uses_deepspeed_muon():
    value = _build_deepspeed_config(
        runtime_config(optimizer="muon", zero_stage=3),
        world_size=2,
        batch_size=4,
        accumulation_steps=4,
    )
    assert value["optimizer"]["type"] == "Muon"
    assert value["zero_optimization"]["stage"] == 3
    assert value["optimizer"]["params"]["betas"] == [0.9, 0.95]
    assert value["optimizer"]["params"]["eps"] == 1e-8


def test_deepspeed_muon_flags_follow_the_same_canonical_parameter_roles():
    model: nn.Module = ToyTiedModel()
    roles = _assign_deepspeed_muon_roles(model)
    flags = {
        name: parameter.use_muon
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert {name for name, enabled in flags.items() if enabled} == set(roles.muon_names)
