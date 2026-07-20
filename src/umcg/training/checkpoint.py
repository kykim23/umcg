"""Atomic native checkpoints for DDP, FSDP2, and DeepSpeed ZeRO."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as distributed

from umcg.config import canonical_json_hash
from umcg.data.c4_stream import StatefulC4Stream
from umcg.distributed.runtime import BackendRuntime
from umcg.estimators.russian_roulette import LevelSampler
from umcg.rng import capture_rng_state, restore_rng_state
from umcg.training.state import TrainingState


def build_resume_contract(resolved_config: dict[str, Any]) -> dict[str, Any]:
    runtime = resolved_config["runtime"]
    estimator = resolved_config["estimator"]
    tokenizer = resolved_config["tokenizer"]
    attention = resolved_config["attention"]
    data_source = resolved_config["data_source"]
    return {
        "model_backend": runtime["model_backend"],
        "distributed_backend": runtime["distributed_backend"],
        "zero_stage": runtime["zero_stage"],
        "world_size": resolved_config["world_size"],
        "workers": runtime["workers"],
        "batch_size": resolved_config["batch_size"],
        "accumulation_steps": resolved_config["accumulation_steps"],
        "total_batch_size": runtime["total_batch_size"],
        "model_config_sha256": canonical_json_hash(resolved_config["model"]),
        "source_tree_sha256": resolved_config["source_tree_sha256"],
        "use_torch_compile": runtime["use_torch_compile"],
        "compile_mode": runtime["compile_mode"],
        "activation_checkpointing": runtime["activation_checkpointing"],
        "gradient_estimator": runtime["gradient_estimator"],
        "optimizer": runtime["optimizer"],
        "scheduler": runtime["scheduler"],
        "learning_rate": runtime["learning_rate"],
        "beta1": runtime["beta1"],
        "beta2": runtime["beta2"],
        "epsilon": runtime["epsilon"],
        "weight_decay": runtime["weight_decay"],
        "momentum": runtime["momentum"],
        "num_training_steps": runtime["num_training_steps"],
        "warmup_steps": runtime["warmup_steps"],
        "gradient_clip_norm": runtime["gradient_clip_norm"],
        "precision": runtime["precision"],
        "tokenizer": runtime["tokenizer"],
        "tokenizer_revision": runtime["tokenizer_revision"],
        "tokenizer_resolved_commit": tokenizer["resolved_commit"],
        "c4_source": runtime["c4_source"],
        "c4_repo": runtime["c4_repo"],
        "c4_revision": runtime["c4_revision"],
        "c4_resolved_source": data_source,
        "context_levels": estimator["context_levels"],
        "tail_probabilities": estimator["tail_probabilities"],
        "seed": runtime["seed"],
        "requested_attention_backend": attention["requested_backend"],
        "attention_backend": attention["resolved_backend"],
        "attention_package_versions": attention["package_versions"],
    }


def validate_resume_contract(
    checkpoint_contract: dict[str, Any], current_contract: dict[str, Any]
) -> None:
    differing = sorted(
        key
        for key in set(checkpoint_contract) | set(current_contract)
        if checkpoint_contract.get(key) != current_contract.get(key)
    )
    if differing:
        raise ValueError(f"native resume contract differs in fields: {differing}")


def _temporary_directory(target: Path, rank: int) -> Path:
    value = [f".{target.name}.tmp-{uuid.uuid4().hex}" if rank == 0 else None]
    distributed.broadcast_object_list(value, src=0)
    return target.parent / str(value[0])


def _rank_state_path(directory: Path, rank: int) -> Path:
    return directory / f"rank-{rank:05d}.pt"


def save_native_checkpoint(
    target_directory: str | Path,
    *,
    backend: BackendRuntime,
    training_state: TrainingState,
    train_stream: StatefulC4Stream,
    level_sampler: LevelSampler,
    resolved_config: dict[str, Any],
    wandb_run_id: str | None,
) -> Path:
    target = Path(target_directory).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"checkpoint target already exists: {target}")
    temporary = _temporary_directory(target, backend.context.rank)
    checkpoint_rng_state = capture_rng_state(backend.context.device)
    if backend.context.is_primary:
        temporary.mkdir(parents=False, exist_ok=False)
    backend.context.barrier()
    try:
        if backend.backend_name == "ddp":
            if backend.context.is_primary:
                torch.save(
                    {
                        "model": backend.checkpoint_model.canonical_state_dict(),
                        "optimizer": backend.optimizer.state_dict(),
                    },
                    temporary / "replicated-state.pt",
                )
        elif backend.backend_name == "fsdp2":
            import torch.distributed.checkpoint as checkpoint
            from torch.distributed.checkpoint.state_dict import get_state_dict

            model_state, optimizer_state = get_state_dict(
                backend.checkpoint_model, backend.optimizer.optimizers
            )
            checkpoint.save(
                {"model": model_state, "optimizer": optimizer_state},
                checkpoint_id=temporary / "distributed-state",
            )
        elif backend.backend_name == "zero":
            backend.deepspeed_engine.save_checkpoint(
                str(temporary / "deepspeed-state"),
                tag="native",
                client_state={"optimizer_update": training_state.optimizer_update},
            )
        else:
            raise ValueError("unknown distributed backend")
        rank_state = {
            "training_state": training_state.to_dict(),
            "train_stream": train_stream.state_dict(),
            "level_sampler": level_sampler.state_dict(),
            "scheduler": backend.scheduler.state_dict(),
            "precision": backend.precision.state_dict(),
            "rng": checkpoint_rng_state,
            "wandb_run_id": wandb_run_id,
        }
        torch.save(rank_state, _rank_state_path(temporary, backend.context.rank))
        backend.context.barrier()
        if backend.context.is_primary:
            manifest = {
                "schema_version": 1,
                "distributed_backend": backend.backend_name,
                "optimizer_update": training_state.optimizer_update,
                "resume_contract": build_resume_contract(resolved_config),
                "config_sha256": resolved_config["config_sha256"],
                "source_tree_sha256": resolved_config["source_tree_sha256"],
                "resolved_config": resolved_config,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
            os.replace(temporary, target)
        backend.context.barrier()
        restore_rng_state(checkpoint_rng_state, backend.context.device)
        return target
    except Exception:
        # Do not enter another collective after a rank-local save failure. Let
        # torchrun terminate peers. The hidden temporary directory remains
        # forensic evidence and can never be mistaken for a complete checkpoint.
        raise


def read_checkpoint_manifest(directory: str | Path) -> dict[str, Any]:
    source = Path(directory).resolve()
    if not (source / "COMPLETE").is_file():
        raise ValueError(f"checkpoint is incomplete: {source}")
    try:
        value = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"checkpoint manifest is missing: {source}") from error
    if value.get("schema_version") != 1:
        raise ValueError("checkpoint manifest schema_version must be 1")
    return value


def load_native_checkpoint(
    source_directory: str | Path,
    *,
    backend: BackendRuntime,
    train_stream: StatefulC4Stream,
    level_sampler: LevelSampler,
    resolved_config: dict[str, Any],
) -> tuple[TrainingState, str | None]:
    source = Path(source_directory).resolve()
    manifest = read_checkpoint_manifest(source)
    if manifest["distributed_backend"] != backend.backend_name:
        raise ValueError("native checkpoint cannot resume under another distributed backend")
    validate_resume_contract(manifest["resume_contract"], build_resume_contract(resolved_config))
    if backend.backend_name == "ddp":
        payload = torch.load(source / "replicated-state.pt", map_location="cpu", weights_only=False)
        backend.checkpoint_model.load_canonical_state_dict(payload["model"])
        backend.optimizer.load_state_dict(payload["optimizer"])
    elif backend.backend_name == "fsdp2":
        import torch.distributed.checkpoint as checkpoint
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        model_state, optimizer_state = get_state_dict(
            backend.checkpoint_model, backend.optimizer.optimizers
        )
        state = {"model": model_state, "optimizer": optimizer_state}
        checkpoint.load(state, checkpoint_id=source / "distributed-state")
        set_state_dict(
            backend.checkpoint_model,
            backend.optimizer.optimizers,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
    elif backend.backend_name == "zero":
        loaded, _ = backend.deepspeed_engine.load_checkpoint(
            str(source / "deepspeed-state"),
            tag="native",
            load_module_strict=True,
            load_optimizer_states=True,
            load_lr_scheduler_states=False,
        )
        if loaded is None:
            raise RuntimeError("DeepSpeed did not load the native checkpoint")
    rank_state = torch.load(
        _rank_state_path(source, backend.context.rank),
        map_location="cpu",
        weights_only=False,
    )
    backend.scheduler.load_state_dict(rank_state["scheduler"])
    backend.precision.load_state_dict(rank_state["precision"])
    train_stream.load_state_dict(rank_state["train_stream"])
    level_sampler.load_state_dict(rank_state["level_sampler"])
    restore_rng_state(rank_state["rng"], backend.context.device)
    return TrainingState.from_dict(rank_state["training_state"]), rank_state["wandb_run_id"]


def list_complete_checkpoints(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if not path.name.startswith(".")
        and path.is_dir()
        and (path / "COMPLETE").is_file()
        and (path / "manifest.json").is_file()
    )
