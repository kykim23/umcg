"""Resolve tokenizer and C4 revisions into immutable run metadata."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from umcg.config import validate_tokenizer_contract

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def local_raw_file_paths(directory: str | Path, split: str) -> list[Path]:
    if split not in {"train", "validation"}:
        raise ValueError("local_raw split must be train or validation")
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"local_raw C4 directory does not exist: {root}")
    paths = sorted(root.glob(f"c4-{split}.*.json.gz"))
    if not paths:
        raise ValueError(f"local_raw C4 has no {split} shards: {root}")
    return paths


def local_raw_snapshot(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).resolve()
    split_paths = {
        split: local_raw_file_paths(root, split) for split in ("train", "validation")
    }
    entries = [
        {"split": split, "path": path.name, "size": path.stat().st_size}
        for split, paths in split_paths.items()
        for path in paths
    ]
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "root": str(root),
        "train_shard_count": len(split_paths["train"]),
        "validation_shard_count": len(split_paths["validation"]),
        "file_manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def resolve_tokenizer_commit(name: str, revision: str, tokenizer: object | None = None) -> str:
    resolved_commit = (
        None if tokenizer is None else getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    )
    if COMMIT_PATTERN.fullmatch(revision):
        resolved_commit = revision
    if resolved_commit is None:
        try:
            resolved_commit = HfApi().model_info(name, revision=revision).sha
        except Exception as error:
            raise RuntimeError(
                "could not resolve tokenizer revision to an immutable Hub commit"
            ) from error
    if not resolved_commit:
        raise RuntimeError("tokenizer Hub metadata did not contain a resolved commit")
    return str(resolved_commit)


def load_tokenizer(
    *, name: str, revision: str, model_config: dict[str, Any]
) -> tuple[object, dict[str, Any]]:
    resolved_commit = resolve_tokenizer_commit(name, revision)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=resolved_commit, use_fast=True)
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define eos_token_id and pad_token_id")
    validate_tokenizer_contract(tokenizer, model_config)
    return tokenizer, {
        "name": name,
        "revision": revision,
        "resolved_commit": resolved_commit,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def resolve_c4_source(
    *, source: str, repository: str, revision: str, local_path: str | None
) -> dict[str, Any]:
    if source == "streaming":
        if COMMIT_PATTERN.fullmatch(revision):
            resolved_commit = revision
        else:
            try:
                resolved_commit = HfApi().dataset_info(repository, revision=revision).sha
            except Exception as error:
                raise RuntimeError(
                    "could not resolve C4 revision to an immutable Hub commit"
                ) from error
        return {
            "source": source,
            "repository": repository,
            "revision": revision,
            "resolved_commit": resolved_commit,
        }
    if source == "local":
        if local_path is None:
            raise ValueError("local_path is required for local C4")
        manifest_path = Path(local_path).resolve() / "manifest.json"
        content = manifest_path.read_bytes()
        manifest = json.loads(content)
        return {
            "source": source,
            "path": str(Path(local_path).resolve()),
            "manifest_sha256": hashlib.sha256(content).hexdigest(),
            "manifest": manifest,
        }
    if source == "local_raw":
        if local_path is None:
            raise ValueError("local_path is required for local_raw C4")
        snapshot = local_raw_snapshot(local_path)
        return {
            "source": source,
            "path": snapshot["root"],
            "upstream_revision": revision,
            "train_shard_count": snapshot["train_shard_count"],
            "validation_shard_count": snapshot["validation_shard_count"],
            "file_manifest_sha256": snapshot["file_manifest_sha256"],
        }
    raise ValueError("source must be streaming, local, or local_raw")
