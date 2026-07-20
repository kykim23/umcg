"""Production torchrun training loop."""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as distributed

from umcg.cli.arguments import parse_runtime_config
from umcg.config import (
    EstimatorConfig,
    RuntimeConfig,
    canonical_json_hash,
    derive_accumulation_steps,
    load_json_object,
    source_tree_sha256,
    validate_model_config,
)
from umcg.data.c4_stream import StatefulC4Stream, build_c4_stream
from umcg.data.sources import load_tokenizer, resolve_c4_source
from umcg.distributed.runtime import (
    BackendRuntime,
    DistributedContext,
    build_backend_runtime,
)
from umcg.estimators.global_objective import (
    build_global_update_plan,
    estimator_scalar,
    global_token_coefficients,
)
from umcg.estimators.levels import LevelSpec
from umcg.estimators.russian_roulette import LevelSampler
from umcg.model.attention import (
    AttentionSelection,
    resolve_attention_backend,
    verify_checkpoint_attention_selection,
)
from umcg.model.factory import build_model
from umcg.precision import build_precision_runtime
from umcg.rng import capture_rng_state, restore_rng_state, seed_all
from umcg.training.checkpoint import (
    load_native_checkpoint,
    read_checkpoint_manifest,
    save_native_checkpoint,
)
from umcg.training.metrics import MetricsLogger, WandbLogger
from umcg.training.state import TrainingState
from umcg.training.vram import BatchSelection, find_automatic_batch_size


def _absolute_runtime_paths(config: RuntimeConfig) -> RuntimeConfig:
    def absolute(value: str | None) -> str | None:
        return None if value is None else str(Path(value).expanduser().resolve())

    return dataclasses.replace(
        config,
        model_config=str(Path(config.model_config).expanduser().resolve()),
        estimator_config=str(Path(config.estimator_config).expanduser().resolve()),
        save_dir=str(Path(config.save_dir).expanduser().resolve()),
        c4_local_path=absolute(config.c4_local_path),
        continue_from=absolute(config.continue_from),
        initial_weights=absolute(config.initial_weights),
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_save_directory(config: RuntimeConfig, context: DistributedContext) -> Path:
    save_directory = Path(config.save_dir)
    error_message: list[str | None] = [None]
    if context.is_primary:
        if save_directory.exists() and not save_directory.is_dir():
            error_message[0] = f"save_dir exists and is not a directory: {save_directory}"
        elif save_directory.is_dir() and any(save_directory.iterdir()):
            checkpoint_parent = (
                None if config.continue_from is None else Path(config.continue_from).parent
            )
            if checkpoint_parent is None or checkpoint_parent != save_directory:
                error_message[0] = (
                    "non-empty save_dir is allowed only when resuming a checkpoint "
                    "from that same run directory"
                )
        if error_message[0] is None:
            save_directory.mkdir(parents=True, exist_ok=True)
    distributed.broadcast_object_list(error_message, src=0)
    if error_message[0] is not None:
        raise FileExistsError(error_message[0])
    context.barrier()
    return save_directory


def _attention_selection(
    config: RuntimeConfig,
    *,
    context: DistributedContext,
    estimator: EstimatorConfig,
    model_config: dict[str, Any],
    checkpoint_manifest: dict[str, Any] | None,
) -> AttentionSelection:
    arguments = {
        "device": context.device,
        "precision": config.precision,
        "context_levels": estimator.context_levels,
        "num_attention_heads": int(model_config["num_attention_heads"]),
        "num_key_value_heads": int(model_config["num_key_value_heads"]),
        "head_dimension": int(model_config["hidden_size"])
        // int(model_config["num_attention_heads"]),
    }
    if checkpoint_manifest is not None:
        stored_attention = checkpoint_manifest["resolved_config"]["attention"]
        if config.attention_backend != stored_attention["requested_backend"]:
            raise ValueError("attention_backend differs from the native checkpoint request")
        return verify_checkpoint_attention_selection(stored_attention, **arguments)
    return resolve_attention_backend(config.attention_backend, **arguments)


def _probe_automatic_batch(
    config: RuntimeConfig,
    *,
    context: DistributedContext,
    estimator: EstimatorConfig,
    attention: AttentionSelection,
    model_config: dict[str, Any],
) -> BatchSelection:
    probe_config = dataclasses.replace(
        config,
        batch_size="1",
        total_batch_size=context.world_size,
        num_training_steps=1,
        warmup_steps=0,
        eval_every=1,
        save_every=1,
        continue_from=None,
        initial_weights=None,
        wandb_mode="disabled",
    )
    seed_all(config.seed)
    build = build_model(
        config.model_config,
        model_backend=config.model_backend,
        precision=config.precision,
        attention_selection=attention,
        activation_checkpointing=config.activation_checkpointing,
        device=context.device,
        maximum_context=estimator.context_levels[-1],
    )
    precision = build_precision_runtime(
        name=config.precision,
        device=context.device,
        distributed_backend=config.distributed_backend,
        model=build.model,
    )
    backend = build_backend_runtime(
        build.model,
        config=probe_config,
        context=context,
        precision=precision,
        batch_size=1,
        accumulation_steps=1,
    )
    selection = find_automatic_batch_size(
        backend,
        total_batch_size=config.total_batch_size,
        maximum_context=estimator.context_levels[-1],
        vocab_size=int(model_config["vocab_size"]),
    )
    del backend, precision, build
    gc.collect()
    torch.cuda.empty_cache()
    context.barrier()
    return selection


def _resolve_batch(
    config: RuntimeConfig,
    *,
    context: DistributedContext,
    estimator: EstimatorConfig,
    attention: AttentionSelection,
    model_config: dict[str, Any],
    checkpoint_manifest: dict[str, Any] | None,
) -> tuple[int, int, dict[str, Any]]:
    if checkpoint_manifest is not None and config.batch_size == "auto":
        batch_size = int(checkpoint_manifest["resolved_config"]["batch_size"])
        accumulation = int(checkpoint_manifest["resolved_config"]["accumulation_steps"])
        current_accumulation = derive_accumulation_steps(
            batch_size, config.total_batch_size, context.world_size
        )
        if current_accumulation != accumulation:
            raise ValueError("total_batch_size differs from the native checkpoint batch contract")
        return (
            batch_size,
            accumulation,
            {
                "source": "native_checkpoint",
                "batch_size": batch_size,
            },
        )
    if config.batch_size == "auto":
        selection = _probe_automatic_batch(
            config,
            context=context,
            estimator=estimator,
            attention=attention,
            model_config=model_config,
        )
        batch_size = selection.batch_size
        batch_report = selection.to_dict()
        batch_report["source"] = "maximum_context_probe"
    else:
        batch_size = int(config.batch_size)
        batch_report = {"source": "explicit", "batch_size": batch_size}
    accumulation = derive_accumulation_steps(
        batch_size, config.total_batch_size, context.world_size
    )
    return batch_size, accumulation, batch_report


def _build_train_stream(
    config: RuntimeConfig,
    *,
    tokenizer: object,
    tokenizer_metadata: dict[str, Any],
    maximum_context: int,
    context: DistributedContext,
) -> StatefulC4Stream:
    return build_c4_stream(
        source=config.c4_source,
        repository=config.c4_repo,
        revision=config.c4_revision,
        local_path=config.c4_local_path,
        split="train",
        tokenizer=tokenizer,
        tokenizer_metadata=tokenizer_metadata,
        maximum_context=maximum_context,
        seed=config.seed,
        rank=context.rank,
        world_size=context.world_size,
        worker_count=config.workers,
        train=True,
    )


def _evaluate(
    backend: BackendRuntime,
    config: RuntimeConfig,
    *,
    tokenizer: object,
    tokenizer_metadata: dict[str, Any],
    maximum_context: int,
    batch_size: int,
) -> tuple[float, float]:
    validation = build_c4_stream(
        source=config.c4_source,
        repository=config.c4_repo,
        revision=config.c4_revision,
        local_path=config.c4_local_path,
        split="validation",
        tokenizer=tokenizer,
        tokenizer_metadata=tokenizer_metadata,
        maximum_context=maximum_context,
        seed=config.seed,
        rank=backend.context.rank,
        world_size=backend.context.world_size,
        worker_count=config.workers,
        train=False,
    )
    backend.training_model.eval()
    numerator = torch.zeros((), device=backend.context.device, dtype=torch.float64)
    denominator = torch.zeros((), device=backend.context.device, dtype=torch.long)
    with torch.no_grad():
        for _ in range(config.eval_parent_batches):
            batch = validation.next_batch(batch_size).to(backend.context.device)
            with backend.precision.autocast():
                losses = backend.training_model(
                    batch.input_ids, batch.attention_mask, batch.position_ids
                )
            mask = batch.causal_target_mask
            numerator += losses.double().masked_select(mask).sum()
            denominator += mask.sum()
    distributed.all_reduce(numerator, op=distributed.ReduceOp.SUM)
    distributed.all_reduce(denominator, op=distributed.ReduceOp.SUM)
    if denominator.item() <= 0:
        raise RuntimeError("validation has no valid causal targets")
    loss = float((numerator / denominator).item())
    try:
        perplexity = math.exp(loss)
    except OverflowError as error:
        raise FloatingPointError("full-context validation perplexity overflowed") from error
    if not math.isfinite(perplexity):
        raise FloatingPointError("full-context validation perplexity is non-finite")
    backend.training_model.train()
    return loss, perplexity


def _gradients_are_finite(backend: BackendRuntime) -> bool:
    local_finite = True
    for parameter in backend.checkpoint_model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        local_gradient = gradient.to_local() if hasattr(gradient, "to_local") else gradient
        if not torch.isfinite(local_gradient).all():
            local_finite = False
            break
    status = torch.tensor(
        1 if local_finite else 0,
        device=backend.context.device,
        dtype=torch.int32,
    )
    distributed.all_reduce(status, op=distributed.ReduceOp.MIN)
    return bool(status.item())


def _gradient_norm_and_clip(
    backend: BackendRuntime, clip_norm: float | None
) -> tuple[float | None, bool]:
    if backend.backend_name == "zero":
        raise RuntimeError("ZeRO gradient norm is available only after engine.step()")
    for optimizer in backend.optimizer.optimizers:
        backend.precision.unscale_(optimizer)
    if not _gradients_are_finite(backend):
        if backend.precision.name != "float16":
            raise FloatingPointError("non-finite gradients without FP16 dynamic scaling")
        return None, False
    maximum = float("inf") if clip_norm is None else clip_norm
    value = torch.nn.utils.clip_grad_norm_(backend.checkpoint_model.parameters(), maximum)
    if isinstance(value, torch.Tensor):
        value = value.full_tensor() if hasattr(value, "full_tensor") else value
        return float(value.detach().float().cpu()), True
    return float(value), True


def _all_rank_peak_memory(context: DistributedContext) -> list[int]:
    local = int(torch.cuda.max_memory_allocated(context.device))
    values: list[int | None] = [None] * context.world_size
    distributed.all_gather_object(values, local)
    return [int(value) for value in values if value is not None]


def _train_update(
    backend: BackendRuntime,
    *,
    train_stream: StatefulC4Stream,
    batch_size: int,
    accumulation_steps: int,
    levels: LevelSpec,
    level_sampler: LevelSampler,
    gradient_estimator: str,
    gradient_clip_norm: float | None,
) -> dict[str, Any]:
    cpu_batches = [train_stream.next_batch(batch_size) for _ in range(accumulation_steps)]
    communication_started = time.perf_counter()
    plan = build_global_update_plan(
        cpu_batches,
        levels=levels,
        gradient_estimator=gradient_estimator,
        level_sampler=level_sampler,
        rank=backend.context.rank,
        device=backend.context.device,
    )
    communication_time = time.perf_counter() - communication_started
    backend.training_model.train()
    backend.zero_grad()
    torch.cuda.reset_peak_memory_stats(backend.context.device)
    forward_time = 0.0
    backward_time = 0.0
    local_scalar = torch.zeros((), device=backend.context.device, dtype=torch.float64)
    local_valid_tokens = torch.zeros((), device=backend.context.device, dtype=torch.long)
    gradient_norm: float | None = None
    optimizer_step_applied = False
    learning_rate = float(backend.optimizer.param_groups[0]["lr"])
    update_started = time.perf_counter()
    for microbatch_index, (cpu_batch, level_index) in enumerate(
        zip(cpu_batches, plan.sampled_level_indices, strict=True)
    ):
        active_length = levels.lengths[level_index]
        batch = cpu_batch.prefix(active_length).to(backend.context.device)
        final_microbatch = microbatch_index == accumulation_steps - 1
        forward_started = time.perf_counter()
        with backend.synchronization_context(final_microbatch=final_microbatch):
            with backend.precision.autocast():
                token_losses = backend.training_model(
                    batch.input_ids, batch.attention_mask, batch.position_ids
                )
                coefficients = global_token_coefficients(
                    batch.causal_target_mask,
                    levels=levels,
                    global_target_counts=plan.global_target_counts,
                    sampled_level_index=level_index,
                    gradient_estimator=gradient_estimator,
                    gradient_scale=backend.loss_scale,
                )
                scalar = estimator_scalar(token_losses, coefficients)
            torch.cuda.synchronize(backend.context.device)
            forward_time += time.perf_counter() - forward_started
            backward_started = time.perf_counter()
            backend.backward(scalar, accumulation_steps=accumulation_steps)
            torch.cuda.synchronize(backend.context.device)
            backward_time += time.perf_counter() - backward_started
        local_scalar += scalar.detach().double()
        local_valid_tokens += batch.causal_target_mask.sum()
        if final_microbatch:
            if backend.backend_name != "zero":
                gradient_norm, gradients_finite = _gradient_norm_and_clip(
                    backend, gradient_clip_norm
                )
            else:
                gradients_finite = True
        else:
            gradients_finite = True
        step_applied = backend.finish_microbatch(
            final_microbatch=final_microbatch,
            gradients_finite=gradients_finite,
        )
        optimizer_step_applied = optimizer_step_applied or step_applied
        if final_microbatch and backend.backend_name == "zero" and step_applied:
            zero_gradient_norm = backend.deepspeed_engine.get_global_grad_norm()
            if zero_gradient_norm is None:
                raise RuntimeError("DeepSpeed did not report a global gradient norm")
            gradient_norm = float(zero_gradient_norm)
    distributed.all_reduce(local_scalar, op=distributed.ReduceOp.SUM)
    local_scalar /= backend.context.world_size
    distributed.all_reduce(local_valid_tokens, op=distributed.ReduceOp.SUM)
    peak_memory = _all_rank_peak_memory(backend.context)
    if optimizer_step_applied and (gradient_norm is None or not math.isfinite(gradient_norm)):
        raise FloatingPointError("optimizer update produced no finite global gradient norm")
    return {
        "signed_estimator_scalar": float(local_scalar.item()),
        "valid_tokens": int(local_valid_tokens.item()),
        "full_context_equivalent_tokens": int(plan.global_target_counts[-1].item()),
        "selected_level_indices": list(plan.sampled_level_indices),
        "global_target_counts": [int(item) for item in plan.global_target_counts.tolist()],
        "gradient_norm": gradient_norm,
        "optimizer_step_applied": optimizer_step_applied,
        "learning_rate": learning_rate,
        "forward_time_seconds": forward_time,
        "backward_time_seconds": backward_time,
        "communication_time_seconds": communication_time,
        "optimizer_time_seconds": max(
            0.0,
            time.perf_counter() - update_started - forward_time - backward_time,
        ),
        "update_time_seconds": time.perf_counter() - update_started,
        "rank_peak_vram_bytes": peak_memory,
    }


def run(config: RuntimeConfig) -> None:
    context = DistributedContext.initialize()
    wandb = None
    try:
        config = _absolute_runtime_paths(config)
        config.validate_before_model_creation(context.world_size)
        estimator = EstimatorConfig.load(config.estimator_config)
        model_config = load_json_object(config.model_config)
        validate_model_config(model_config, estimator.context_levels[-1])
        tokenizer, tokenizer_metadata = load_tokenizer(
            name=config.tokenizer,
            revision=config.tokenizer_revision,
            model_config=model_config,
        )
        data_source = resolve_c4_source(
            source=config.c4_source,
            repository=config.c4_repo,
            revision=config.c4_revision,
            local_path=config.c4_local_path,
        )
        if config.c4_source == "streaming":
            config = dataclasses.replace(config, c4_revision=str(data_source["resolved_commit"]))
        checkpoint_manifest = (
            read_checkpoint_manifest(config.continue_from)
            if config.continue_from is not None
            else None
        )
        attention = _attention_selection(
            config,
            context=context,
            estimator=estimator,
            model_config=model_config,
            checkpoint_manifest=checkpoint_manifest,
        )
        batch_size, accumulation_steps, batch_report = _resolve_batch(
            config,
            context=context,
            estimator=estimator,
            attention=attention,
            model_config=model_config,
            checkpoint_manifest=checkpoint_manifest,
        )
        seed_all(config.seed)
        model_build = build_model(
            config.model_config,
            model_backend=config.model_backend,
            precision=config.precision,
            attention_selection=attention,
            activation_checkpointing=config.activation_checkpointing,
            device=context.device,
            maximum_context=estimator.context_levels[-1],
            initial_weights=config.initial_weights,
        )
        precision = build_precision_runtime(
            name=config.precision,
            device=context.device,
            distributed_backend=config.distributed_backend,
            model=model_build.model,
        )
        backend = build_backend_runtime(
            model_build.model,
            config=config,
            context=context,
            precision=precision,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
        )
        train_stream = _build_train_stream(
            config,
            tokenizer=tokenizer,
            tokenizer_metadata=tokenizer_metadata,
            maximum_context=estimator.context_levels[-1],
            context=context,
        )
        levels = LevelSpec(estimator.context_levels, estimator.tail_probabilities)
        levels.validate(max_position_embeddings=model_config["max_position_embeddings"])
        level_sampler = LevelSampler(levels, config.seed + 17)
        repository_root = Path(__file__).resolve().parents[3]
        resolved_config = {
            "schema_version": 1,
            "runtime": config.to_dict(),
            "model": model_config,
            "estimator": estimator.to_dict(),
            "tokenizer": tokenizer_metadata,
            "data_source": data_source,
            "attention": attention.to_dict(),
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "batch_resolution": batch_report,
            "world_size": context.world_size,
            "model_parameter_count": model_build.parameter_count,
            "trainable_parameter_count": model_build.trainable_parameter_count,
            "parameter_roles": (
                None if backend.parameter_roles is None else backend.parameter_roles.to_dict()
            ),
            "fp8": precision.float8_metadata,
            "source_tree_sha256": source_tree_sha256(repository_root),
            "canonical_cli": [str(item) for item in sys.argv],
        }
        resolved_config["config_sha256"] = canonical_json_hash(resolved_config)
        save_directory = _prepare_save_directory(config, context)
        state = TrainingState(selected_level_counts=[0] * levels.num_levels)
        resumed_wandb_id = None
        if config.continue_from is not None:
            state, resumed_wandb_id = load_native_checkpoint(
                config.continue_from,
                backend=backend,
                train_stream=train_stream,
                level_sampler=level_sampler,
                resolved_config=resolved_config,
            )
        if context.is_primary:
            _write_json(save_directory / "resolved_config.json", resolved_config)
            _write_json(
                save_directory / "run_manifest.json",
                {
                    "schema_version": 1,
                    "name": config.name,
                    "distributed_backend": config.distributed_backend,
                    "zero_stage": config.zero_stage,
                    "attention": attention.to_dict(),
                    "precision": config.precision,
                    "source_tree_sha256": resolved_config["source_tree_sha256"],
                    "external_distributed_validation": "pending",
                },
            )
        context.barrier()
        metrics = MetricsLogger(save_directory / "metrics.jsonl")
        logging_rng_state = capture_rng_state(context.device)
        wandb = WandbLogger(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            run_name=config.name,
            config=resolved_config,
            run_id=resumed_wandb_id,
            enabled=context.is_primary,
        )
        restore_rng_state(logging_rng_state, context.device)
        consecutive_scaler_retries = 0
        while state.optimizer_update < config.num_training_steps:
            retry_stream_state = (
                train_stream.state_dict() if config.precision == "float16" else None
            )
            retry_sampler_state = (
                level_sampler.state_dict() if config.precision == "float16" else None
            )
            retry_rng_state = (
                capture_rng_state(context.device) if config.precision == "float16" else None
            )
            update = _train_update(
                backend,
                train_stream=train_stream,
                batch_size=batch_size,
                accumulation_steps=accumulation_steps,
                levels=levels,
                level_sampler=level_sampler,
                gradient_estimator=config.gradient_estimator,
                gradient_clip_norm=config.gradient_clip_norm,
            )
            if not update["optimizer_step_applied"]:
                if (
                    retry_stream_state is None
                    or retry_sampler_state is None
                    or retry_rng_state is None
                ):
                    raise FloatingPointError("optimizer update was skipped without replay state")
                train_stream.load_state_dict(retry_stream_state)
                level_sampler.load_state_dict(retry_sampler_state)
                restore_rng_state(retry_rng_state, context.device)
                consecutive_scaler_retries += 1
                if context.is_primary:
                    print(
                        json.dumps(
                            {
                                "event": "fp16_gradient_scaler_retry",
                                "next_optimizer_update": state.optimizer_update + 1,
                                "loss_scale": precision.loss_scale,
                                "consecutive_retries": consecutive_scaler_retries,
                            },
                            sort_keys=True,
                            allow_nan=False,
                        ),
                        flush=True,
                    )
                if consecutive_scaler_retries >= 16:
                    raise FloatingPointError("FP16 update failed after 16 scaler retries")
                continue
            consecutive_scaler_retries = 0
            state.optimizer_update += 1
            state.valid_tokens += update["valid_tokens"]
            state.full_context_equivalent_tokens += update["full_context_equivalent_tokens"]
            for index in update["selected_level_indices"]:
                state.selected_level_counts[index] += 1
            selected_total = sum(state.selected_level_counts)
            selected_level_frequencies = [
                count / selected_total for count in state.selected_level_counts
            ]
            selected_level_tail_frequencies = [
                sum(state.selected_level_counts[index:]) / selected_total
                for index in range(len(state.selected_level_counts))
            ]
            elapsed = update["update_time_seconds"]
            metric = {
                "optimizer_update": state.optimizer_update,
                "valid_tokens": update["valid_tokens"],
                "total_valid_tokens": state.valid_tokens,
                "full_context_equivalent_tokens": update["full_context_equivalent_tokens"],
                "total_full_context_equivalent_tokens": state.full_context_equivalent_tokens,
                "selected_level_indices": update["selected_level_indices"],
                "selected_level_counts": list(state.selected_level_counts),
                "selected_level_frequencies": selected_level_frequencies,
                "selected_level_tail_frequencies": selected_level_tail_frequencies,
                "global_target_counts": update["global_target_counts"],
                "tail_probabilities": list(levels.tail_probabilities),
                "signed_estimator_scalar": update["signed_estimator_scalar"],
                "learning_rate": update["learning_rate"],
                "gradient_norm": update["gradient_norm"],
                "forward_time_seconds": update["forward_time_seconds"],
                "backward_time_seconds": update["backward_time_seconds"],
                "communication_time_seconds": update["communication_time_seconds"],
                "optimizer_time_seconds": update["optimizer_time_seconds"],
                "update_time_seconds": elapsed,
                "valid_tokens_per_second": update["valid_tokens"] / max(elapsed, 1e-12),
                "rank_peak_vram_bytes": update["rank_peak_vram_bytes"],
                "distributed_backend": config.distributed_backend,
                "zero_stage": config.zero_stage,
                "attention_backend": attention.resolved_backend,
                "precision": config.precision,
                "fp8": precision.float8_metadata,
                "source_tree_sha256": resolved_config["source_tree_sha256"],
            }
            if state.optimizer_update % config.eval_every == 0:
                validation_loss, validation_perplexity = _evaluate(
                    backend,
                    config,
                    tokenizer=tokenizer,
                    tokenizer_metadata=tokenizer_metadata,
                    maximum_context=levels.lengths[-1],
                    batch_size=batch_size,
                )
                metric["full_validation_loss"] = validation_loss
                metric["full_validation_perplexity"] = validation_perplexity
            if context.is_primary:
                metrics.log(metric)
                wandb.log(metric)
                print(json.dumps(metric, sort_keys=True, allow_nan=False), flush=True)
            if state.optimizer_update % config.save_every == 0:
                save_native_checkpoint(
                    save_directory / f"checkpoint-{state.optimizer_update:08d}",
                    backend=backend,
                    training_state=state,
                    train_stream=train_stream,
                    level_sampler=level_sampler,
                    resolved_config=resolved_config,
                    wandb_run_id=wandb.run_id,
                )
        if state.optimizer_update % config.save_every != 0:
            save_native_checkpoint(
                save_directory / f"checkpoint-{state.optimizer_update:08d}",
                backend=backend,
                training_state=state,
                train_stream=train_stream,
                level_sampler=level_sampler,
                resolved_config=resolved_config,
                wandb_run_id=wandb.run_id,
            )
    finally:
        if wandb is not None:
            wandb.finish()
        context.close()


def main(arguments: list[str] | None = None) -> None:
    run(parse_runtime_config(arguments))
