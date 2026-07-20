"""Prepare local C4-like JSONL without crossing document boundaries."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

from transformers import AutoTokenizer

from umcg.config import EstimatorConfig
from umcg.data.document_chunks import tokenize_c4_row
from umcg.data.sources import resolve_tokenizer_commit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prepare_c4_main.py", allow_abbrev=False)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer", default="t5-base")
    parser.add_argument("--tokenizer_revision", default="main")
    parser.add_argument("--estimator_config", required=True)
    parser.add_argument("--parents_per_shard", type=int, default=10_000)
    return parser


def prepare(arguments: argparse.Namespace) -> dict[str, object]:
    input_path = Path(arguments.input_jsonl).resolve()
    output_path = Path(arguments.output_dir).resolve()
    if not input_path.is_file():
        raise ValueError(f"input_jsonl does not exist: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"output_dir already exists: {output_path}")
    if arguments.parents_per_shard <= 0:
        raise ValueError("parents_per_shard must be positive")
    estimator = EstimatorConfig.load(arguments.estimator_config)
    resolved_commit = resolve_tokenizer_commit(arguments.tokenizer, arguments.tokenizer_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        arguments.tokenizer,
        revision=resolved_commit,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define both eos_token_id and pad_token_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent))
    shard_names: list[str] = []
    parent_count = 0
    document_count = 0
    shard_handle = None
    try:
        with input_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at input line {line_number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"input line {line_number} is not a JSON object")
                chunks = tokenize_c4_row(row, tokenizer, estimator.context_levels[-1])
                document_count += 1
                for chunk in chunks:
                    if parent_count % arguments.parents_per_shard == 0:
                        if shard_handle is not None:
                            shard_handle.close()
                        shard_name = f"parents-{len(shard_names):05d}.jsonl"
                        shard_names.append(shard_name)
                        shard_handle = (temporary / shard_name).open("w", encoding="utf-8")
                    active_length = int(chunk["attention_mask"].sum().item())
                    record = {
                        "tokens": chunk["input_ids"][:active_length].tolist(),
                        "document_hash": chunk["document_hash"],
                        "chunk_index": chunk["chunk_index"],
                        "token_start": chunk["token_start"],
                        "token_end": chunk["token_end"],
                        "url": chunk["url"],
                        "timestamp": chunk["timestamp"],
                    }
                    assert shard_handle is not None
                    shard_handle.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    parent_count += 1
        if shard_handle is not None:
            shard_handle.close()
            shard_handle = None
        manifest = {
            "schema_version": 1,
            "source": str(input_path),
            "tokenizer": arguments.tokenizer,
            "tokenizer_revision": arguments.tokenizer_revision,
            "tokenizer_resolved_commit": resolved_commit,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "maximum_context": estimator.context_levels[-1],
            "document_count": document_count,
            "parent_count": parent_count,
            "shards": shard_names,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
        return manifest
    except Exception:
        if shard_handle is not None:
            shard_handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(arguments: Sequence[str] | None = None) -> None:
    result = prepare(build_parser().parse_args(arguments))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
