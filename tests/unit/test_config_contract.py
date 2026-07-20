import dataclasses
import json

import pytest

from umcg.config import (
    EstimatorConfig,
    RuntimeConfig,
    derive_accumulation_steps,
    validate_model_config,
    validate_tokenizer_contract,
)


def runtime_config(**changes) -> RuntimeConfig:
    value = RuntimeConfig(
        distributed_backend="ddp",
        zero_stage=None,
        model_backend="huggingface",
        model_config="model.json",
        tokenizer="t5-base",
        tokenizer_revision="main",
        precision="bfloat16",
        attention_backend="automatic",
        estimator_config="estimator.json",
        gradient_estimator="russian_roulette",
        optimizer="adamw",
        scheduler="cosine",
        learning_rate=3e-4,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=0.1,
        momentum=None,
        batch_size="4",
        total_batch_size=16,
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


def test_safe_estimator_templates_are_canonical(project_root):
    one = EstimatorConfig.load(project_root / "configs/estimator/russian_roulette_safe_1024.json")
    four = EstimatorConfig.load(project_root / "configs/estimator/russian_roulette_safe_4096.json")
    assert one.context_levels == (128, 256, 512, 1024)
    assert four.context_levels == (512, 1024, 2048, 4096)
    assert one.tail_probabilities == (1.0, 1.0, 1.0, 1.0)


def test_batch_accumulation_is_derived_exactly():
    assert derive_accumulation_steps(8, 512, 4) == 16
    with pytest.raises(ValueError, match="must be divisible"):
        derive_accumulation_steps(7, 512, 4)


@pytest.mark.parametrize(
    "changes",
    [
        {"distributed_backend": "fsdp2", "optimizer": "muon"},
        {"distributed_backend": "zero", "zero_stage": 2, "optimizer": "adamw_8bit"},
        {"distributed_backend": "zero", "zero_stage": 3, "precision": "float8"},
        {"optimizer": "muon", "precision": "float8"},
        {"optimizer": "adamw_8bit", "precision": "float8"},
    ],
)
def test_unsupported_backend_optimizer_precision_combinations_fail(changes):
    with pytest.raises(ValueError):
        runtime_config(**changes).validate_before_model_creation(world_size=4)


@pytest.mark.parametrize(
    "changes",
    [
        {"distributed_backend": "ddp", "optimizer": "muon"},
        {"distributed_backend": "ddp", "optimizer": "adamw_8bit"},
        {"distributed_backend": "fsdp2", "optimizer": "adamw_8bit"},
        {"distributed_backend": "zero", "zero_stage": 1, "optimizer": "muon"},
        {"distributed_backend": "zero", "zero_stage": 2, "optimizer": "adam"},
        {"distributed_backend": "zero", "zero_stage": 3, "optimizer": "sgdm"},
    ],
)
def test_allowed_combination_matrix_passes_early_validation(changes):
    runtime_config(**changes).validate_before_model_creation(world_size=4)


def test_zero_stage_and_sgd_momentum_are_unambiguous():
    with pytest.raises(ValueError, match="zero_stage"):
        runtime_config(distributed_backend="zero").validate_before_model_creation(4)
    with pytest.raises(ValueError, match="valid only"):
        runtime_config(zero_stage=1).validate_before_model_creation(4)
    with pytest.raises(ValueError, match="does not accept momentum"):
        runtime_config(optimizer="sgd", momentum=0.0).validate_before_model_creation(4)
    with pytest.raises(ValueError, match="valid only"):
        runtime_config(optimizer="adamw", momentum=0.9).validate_before_model_creation(4)


def test_model_schema_rejects_unknown_fields(project_root):
    model = json.loads(
        (project_root / "configs/model/llama_tiny_smoke_1024.json").read_text(encoding="utf-8")
    )
    model["rope_scaling"] = {"type": "unsupported"}
    with pytest.raises(ValueError, match="unknown"):
        validate_model_config(model, maximum_context=1024)


def test_tokenizer_contract_never_resizes_embeddings(project_root):
    model = json.loads(
        (project_root / "configs/model/llama_tiny_smoke_1024.json").read_text(encoding="utf-8")
    )

    class Tokenizer:
        eos_token_id = model["eos_token_id"]
        pad_token_id = model["pad_token_id"]

        def __len__(self):
            return model["vocab_size"] + 1

    with pytest.raises(ValueError, match="automatic resize is forbidden"):
        validate_tokenizer_contract(Tokenizer(), model)
