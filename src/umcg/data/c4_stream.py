"""Exactly resumable C4 streams with explicit row-to-chunk state."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Protocol

import torch

from umcg.data.collate import ParentBatch, collate_parent_samples
from umcg.data.document_chunks import tokenize_c4_row
from umcg.data.parent_dataset import ParentSample


class DocumentWorker(Protocol):
    def next_sample(self) -> ParentSample: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class HuggingFaceC4Worker:
    def __init__(
        self,
        *,
        repository: str,
        revision: str,
        split: str,
        tokenizer: object,
        maximum_context: int,
        seed: int,
        worker_index: int,
        total_workers: int,
        shuffle_shards: bool,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("c4_source=streaming requires the datasets package") from error
        dataset = load_dataset(
            repository,
            "en",
            split=split,
            streaming=True,
            revision=revision,
        )
        if shuffle_shards:
            # A one-row buffer leaves example order unchanged while the dataset
            # implementation still permutes its data-source shards by seed.
            dataset = dataset.shuffle(seed=seed, buffer_size=1)
        dataset = dataset.shard(num_shards=total_workers, index=worker_index)
        if not hasattr(dataset, "state_dict") or not hasattr(dataset, "load_state_dict"):
            raise RuntimeError("installed datasets version does not expose iterable state")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.maximum_context = maximum_context
        self.worker_index = worker_index
        self.total_workers = total_workers
        self.row_position = 0
        self.yielded_chunks = 0
        self.pending_chunks: list[ParentSample] = []
        self._iterator = iter(self.dataset)

    def next_sample(self) -> ParentSample:
        while not self.pending_chunks:
            try:
                row = next(self._iterator)
            except StopIteration as error:
                raise RuntimeError("C4 stream exhausted before training completed") from error
            self.row_position += 1
            self.pending_chunks = tokenize_c4_row(row, self.tokenizer, self.maximum_context)
        sample = self.pending_chunks.pop(0)
        self.yielded_chunks += 1
        return sample

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "huggingface_c4",
            "worker_index": self.worker_index,
            "total_workers": self.total_workers,
            "row_position": self.row_position,
            "yielded_chunks": self.yielded_chunks,
            "dataset_state": self.dataset.state_dict(),
            "pending_chunks": copy.deepcopy(self.pending_chunks),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (self.worker_index, self.total_workers)
        actual = (int(state["worker_index"]), int(state["total_workers"]))
        if actual != expected:
            raise ValueError(f"C4 worker topology changed: checkpoint={actual}, current={expected}")
        self.dataset.load_state_dict(state["dataset_state"])
        self.row_position = int(state["row_position"])
        self.yielded_chunks = int(state["yielded_chunks"])
        self.pending_chunks = copy.deepcopy(list(state["pending_chunks"]))
        self._iterator = iter(self.dataset)


class LocalC4Worker:
    def __init__(
        self,
        *,
        directory: str | Path,
        maximum_context: int,
        worker_index: int,
        total_workers: int,
        tokenizer_metadata: dict[str, Any],
        seed: int,
        shuffle_shards: bool,
    ) -> None:
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"local C4 manifest does not exist: {manifest_path}") from error
        if manifest.get("schema_version") != 1:
            raise ValueError("local C4 manifest schema_version must be 1")
        if manifest.get("maximum_context") != maximum_context:
            raise ValueError("local C4 maximum_context does not match estimator config")
        expected_tokenizer = {
            "tokenizer": tokenizer_metadata["name"],
            "tokenizer_resolved_commit": tokenizer_metadata["resolved_commit"],
            "vocab_size": tokenizer_metadata["vocab_size"],
            "eos_token_id": tokenizer_metadata["eos_token_id"],
            "pad_token_id": tokenizer_metadata["pad_token_id"],
        }
        differing = sorted(
            key for key, value in expected_tokenizer.items() if manifest.get(key) != value
        )
        if differing:
            raise ValueError(f"local C4 tokenizer contract differs in fields: {differing}")
        self._manifest = manifest
        self.shards = [self.directory / name for name in manifest["shards"]]
        if not self.shards or any(not path.is_file() for path in self.shards):
            raise ValueError("local C4 manifest references missing shards")
        if shuffle_shards:
            random.Random(seed).shuffle(self.shards)
        self.maximum_context = maximum_context
        self.pad_token_id = int(manifest["pad_token_id"])
        self.worker_index = worker_index
        self.total_workers = total_workers
        self.shard_index = 0
        self.line_index = 0
        self.global_record_index = 0
        self.yielded_chunks = 0
        self._handle = None
        self._open_current_shard()

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    def _open_current_shard(self) -> None:
        if self._handle is not None:
            self._handle.close()
        if self.shard_index >= len(self.shards):
            raise RuntimeError("local C4 stream exhausted before training completed")
        self._handle = self.shards[self.shard_index].open("r", encoding="utf-8")
        for _ in range(self.line_index):
            if not self._handle.readline():
                raise ValueError("saved local C4 line position exceeds shard length")

    def _next_record(self) -> dict[str, Any]:
        while True:
            assert self._handle is not None
            line = self._handle.readline()
            if line:
                current_index = self.global_record_index
                self.line_index += 1
                self.global_record_index += 1
                if current_index % self.total_workers == self.worker_index:
                    return json.loads(line)
                continue
            self.shard_index += 1
            self.line_index = 0
            self._open_current_shard()

    def next_sample(self) -> ParentSample:
        record = self._next_record()
        tokens = [int(item) for item in record["tokens"]]
        if not 2 <= len(tokens) <= self.maximum_context:
            raise ValueError("prepared C4 parent must contain 2..maximum_context active tokens")
        input_ids = torch.full((self.maximum_context,), self.pad_token_id, dtype=torch.long)
        input_ids[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.zeros(self.maximum_context, dtype=torch.bool)
        attention_mask[: len(tokens)] = True
        sample: ParentSample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "causal_target_mask": attention_mask[:-1] & attention_mask[1:],
            "position_ids": torch.arange(self.maximum_context, dtype=torch.long),
            "document_hash": str(record["document_hash"]),
            "chunk_index": int(record["chunk_index"]),
            "token_start": int(record["token_start"]),
            "token_end": int(record["token_end"]),
            "url": str(record.get("url", "")),
            "timestamp": str(record.get("timestamp", "")),
        }
        self.yielded_chunks += 1
        return sample

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "local_c4",
            "worker_index": self.worker_index,
            "total_workers": self.total_workers,
            "shards": [path.name for path in self.shards],
            "shard_index": self.shard_index,
            "line_index": self.line_index,
            "global_record_index": self.global_record_index,
            "yielded_chunks": self.yielded_chunks,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (self.worker_index, self.total_workers)
        actual = (int(state["worker_index"]), int(state["total_workers"]))
        if actual != expected:
            raise ValueError(
                f"local C4 worker topology changed: checkpoint={actual}, current={expected}"
            )
        if list(state["shards"]) != [path.name for path in self.shards]:
            raise ValueError("local C4 shard order changed")
        self.shard_index = int(state["shard_index"])
        self.line_index = int(state["line_index"])
        self.global_record_index = int(state["global_record_index"])
        self.yielded_chunks = int(state["yielded_chunks"])
        self._open_current_shard()


class StatefulC4Stream:
    def __init__(self, workers: list[DocumentWorker], *, rank: int, world_size: int) -> None:
        if not workers:
            raise ValueError("at least one logical data worker is required")
        self.workers = workers
        self.rank = rank
        self.world_size = world_size
        self.next_worker = 0
        self.yielded_chunks = 0

    def next_sample(self) -> ParentSample:
        worker = self.workers[self.next_worker]
        self.next_worker = (self.next_worker + 1) % len(self.workers)
        self.yielded_chunks += 1
        return worker.next_sample()

    def next_batch(self, batch_size: int) -> ParentBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return collate_parent_samples([self.next_sample() for _ in range(batch_size)])

    def state_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "worker_count": len(self.workers),
            "next_worker": self.next_worker,
            "yielded_chunks": self.yielded_chunks,
            "workers": [worker.state_dict() for worker in self.workers],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (self.rank, self.world_size, len(self.workers))
        actual = (
            int(state["rank"]),
            int(state["world_size"]),
            int(state["worker_count"]),
        )
        if actual != expected:
            raise ValueError(f"C4 stream topology changed: checkpoint={actual}, current={expected}")
        self.next_worker = int(state["next_worker"])
        self.yielded_chunks = int(state["yielded_chunks"])
        for worker, worker_state in zip(self.workers, state["workers"], strict=True):
            worker.load_state_dict(worker_state)


def build_c4_stream(
    *,
    source: str,
    repository: str,
    revision: str,
    local_path: str | None,
    split: str,
    tokenizer: object,
    tokenizer_metadata: dict[str, Any] | None,
    maximum_context: int,
    seed: int,
    rank: int,
    world_size: int,
    worker_count: int,
    train: bool,
) -> StatefulC4Stream:
    total_workers = world_size * worker_count
    workers: list[DocumentWorker] = []
    for local_worker in range(worker_count):
        worker_index = rank * worker_count + local_worker
        if source == "streaming":
            worker = HuggingFaceC4Worker(
                repository=repository,
                revision=revision,
                split=split,
                tokenizer=tokenizer,
                maximum_context=maximum_context,
                seed=seed,
                worker_index=worker_index,
                total_workers=total_workers,
                shuffle_shards=train,
            )
        elif source == "local":
            if local_path is None:
                raise ValueError("local_path is required for local C4")
            if tokenizer_metadata is None:
                raise ValueError("tokenizer_metadata is required for local C4")
            worker = LocalC4Worker(
                directory=local_path,
                maximum_context=maximum_context,
                worker_index=worker_index,
                total_workers=total_workers,
                tokenizer_metadata=tokenizer_metadata,
                seed=seed,
                shuffle_shards=train,
            )
        else:
            raise ValueError("source must be streaming or local")
        workers.append(worker)
    return StatefulC4Stream(workers, rank=rank, world_size=world_size)
