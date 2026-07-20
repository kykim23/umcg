"""Export any complete native checkpoint as canonical FP32 safetensors."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from umcg.model.attention import AttentionSelection
from umcg.model.factory import build_model
from umcg.training.checkpoint import read_checkpoint_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="export_weights_main.py", allow_abbrev=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _canonical_name(name: str) -> str:
    prefixes = ("module.", "_orig_mod.", "causal_lm.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def _load_fsdp_state(checkpoint: Path, manifest: dict[str, Any]) -> dict[str, torch.Tensor]:
    import torch.distributed.checkpoint as distributed_checkpoint

    resolved = manifest["resolved_config"]
    runtime = resolved["runtime"]
    attention_value = resolved["attention"]
    selection = AttentionSelection(
        requested_backend="eager",
        resolved_backend="eager",
        huggingface_implementation="eager",
        hardware=attention_value["hardware"],
        package_versions=attention_value["package_versions"],
        rank_probe_results=[],
    )
    result = build_model(
        resolved["model"],
        model_backend=runtime["model_backend"],
        precision="float32",
        attention_selection=selection,
        activation_checkpointing=False,
        device=torch.device("cpu"),
        maximum_context=max(resolved["estimator"]["context_levels"]),
    )
    state = {"model": result.model.state_dict()}
    distributed_checkpoint.load(state, checkpoint_id=checkpoint / "distributed-state")
    return state["model"]


def load_native_weights(checkpoint: Path, manifest: dict[str, Any]) -> dict[str, torch.Tensor]:
    backend = manifest["distributed_backend"]
    if backend == "ddp":
        payload = torch.load(
            checkpoint / "replicated-state.pt", map_location="cpu", weights_only=False
        )
        state = payload["model"]
    elif backend == "fsdp2":
        state = _load_fsdp_state(checkpoint, manifest)
    elif backend == "zero":
        try:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        except ImportError as error:
            raise RuntimeError("exporting a ZeRO checkpoint requires DeepSpeed") from error
        state = get_fp32_state_dict_from_zero_checkpoint(
            str(checkpoint / "deepspeed-state"), tag="native"
        )
    else:
        raise ValueError(f"unknown checkpoint distributed backend: {backend}")
    canonical: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        canonical_name = _canonical_name(name)
        if canonical_name in canonical:
            raise ValueError(f"duplicate canonical weight key: {canonical_name}")
        tensor = value.detach().cpu().contiguous().clone()
        canonical[canonical_name] = tensor.float() if tensor.is_floating_point() else tensor
    if not canonical:
        raise ValueError("native checkpoint contains no model tensors")
    return canonical


def export(arguments: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(arguments.checkpoint).resolve()
    output = Path(arguments.output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    manifest = read_checkpoint_manifest(checkpoint)
    weights = load_native_weights(checkpoint, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    try:
        save_file(
            weights,
            str(temporary),
            metadata={
                "format": "umcg_canonical_fp32",
                "source_backend": manifest["distributed_backend"],
                "optimizer_update": str(manifest["optimizer_update"]),
            },
        )
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "output": str(output),
        "tensor_count": len(weights),
        "source_backend": manifest["distributed_backend"],
        "optimizer_update": manifest["optimizer_update"],
    }


def main(arguments: Sequence[str] | None = None) -> None:
    result = export(build_parser().parse_args(arguments))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
