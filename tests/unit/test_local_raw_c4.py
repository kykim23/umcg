import gzip
import json

import torch

from umcg.data.c4_stream import build_c4_stream
from umcg.data.sources import local_raw_snapshot, resolve_c4_source


class TinyTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [index + 2 for index, _ in enumerate(text.split())]


def write_rows(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_raw_c4(root):
    root.mkdir()
    write_rows(
        root / "c4-train.00000-of-00002.json.gz",
        [
            {"text": "alpha beta gamma", "url": "train-0a", "timestamp": "t0"},
            {"text": "delta epsilon zeta", "url": "train-0b", "timestamp": "t1"},
        ],
    )
    write_rows(
        root / "c4-train.00001-of-00002.json.gz",
        [
            {"text": "eta theta iota", "url": "train-1a", "timestamp": "t2"},
            {"text": "kappa lambda mu", "url": "train-1b", "timestamp": "t3"},
        ],
    )
    write_rows(
        root / "c4-validation.00000-of-00002.json.gz",
        [{"text": "validation row", "url": "validation", "timestamp": "tv"}],
    )
    write_rows(
        root / "c4-validation.00001-of-00002.json.gz",
        [{"text": "second validation", "url": "validation-2", "timestamp": "tv2"}],
    )
    return root


def make_many_shard_raw_c4(root):
    root.mkdir()
    for shard in range(8):
        write_rows(
            root / f"c4-train.{shard:05d}-of-00008.json.gz",
            [
                {
                    "text": f"shard {shard} row {row} alpha beta",
                    "url": f"train-{shard}-{row}",
                    "timestamp": f"t-{shard}-{row}",
                }
                for row in range(4)
            ],
        )
    for shard in range(2):
        write_rows(
            root / f"c4-validation.{shard:05d}-of-00002.json.gz",
            [
                {
                    "text": f"validation {shard}",
                    "url": f"validation-{shard}",
                    "timestamp": f"tv-{shard}",
                }
            ],
        )
    return root


def build_stream(root, *, rank=0, world_size=1, train=True):
    return build_c4_stream(
        source="local_raw",
        repository="unused",
        revision="local-test-revision",
        local_path=str(root),
        split="train" if train else "validation",
        tokenizer=TinyTokenizer(),
        tokenizer_metadata=None,
        maximum_context=8,
        seed=777,
        rank=rank,
        world_size=world_size,
        worker_count=1,
        train=train,
    )


def assert_same_sample(left, right):
    for key in ("input_ids", "attention_mask", "causal_target_mask", "position_ids"):
        assert torch.equal(left[key], right[key])
    for key in (
        "document_hash",
        "chunk_index",
        "token_start",
        "token_end",
        "url",
        "timestamp",
    ):
        assert left[key] == right[key]


def test_local_raw_stream_restores_exact_dataset_and_pending_chunk_state(tmp_path):
    root = make_many_shard_raw_c4(tmp_path / "raw-c4")
    for rank in (0, 1):
        original = build_stream(root, rank=rank, world_size=2)
        for _ in range(3):
            original.next_sample()
        saved = original.state_dict()
        expected = [original.next_sample() for _ in range(4)]

        restored = build_stream(root, rank=rank, world_size=2)
        restored.load_state_dict(saved)
        actual = [restored.next_sample() for _ in range(4)]
        for expected_sample, actual_sample in zip(expected, actual, strict=True):
            assert_same_sample(expected_sample, actual_sample)


def test_local_raw_two_rank_sharding_has_distinct_documents(tmp_path):
    root = make_raw_c4(tmp_path / "raw-c4")
    left = build_stream(root, rank=0, world_size=2, train=False).next_sample()
    right = build_stream(root, rank=1, world_size=2, train=False).next_sample()
    assert left["document_hash"] != right["document_hash"]


def test_local_raw_snapshot_is_part_of_resolved_source(tmp_path):
    root = make_raw_c4(tmp_path / "raw-c4")
    snapshot = local_raw_snapshot(root)
    resolved = resolve_c4_source(
        source="local_raw",
        repository="unused",
        revision="upstream-commit",
        local_path=str(root),
    )
    assert resolved["file_manifest_sha256"] == snapshot["file_manifest_sha256"]
    assert resolved["train_shard_count"] == 2
    assert resolved["validation_shard_count"] == 2

    write_rows(
        root / "c4-train.00002-of-00003.json.gz",
        [{"text": "new row", "url": "new", "timestamp": "tn"}],
    )
    assert local_raw_snapshot(root)["file_manifest_sha256"] != snapshot[
        "file_manifest_sha256"
    ]
