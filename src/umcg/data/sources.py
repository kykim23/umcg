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
    raise ValueError("source must be streaming or local")
