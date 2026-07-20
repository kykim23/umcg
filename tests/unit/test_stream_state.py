import copy
import sys
import types

import torch

from umcg.data.c4_stream import HuggingFaceC4Worker


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [int(value) for value in text.split()]


class FakeIterable:
    def __init__(self, rows):
        self.rows = list(rows)
        self.position = 0
        self.shuffle_arguments = None
        self.shard_arguments = None

    def shuffle(self, *, seed, buffer_size):
        self.shuffle_arguments = (seed, buffer_size)
        return self

    def shard(self, *, num_shards, index):
        self.shard_arguments = (num_shards, index)
        self.rows = self.rows[index::num_shards]
        return self

    def __iter__(self):
        while self.position < len(self.rows):
            row = self.rows[self.position]
            self.position += 1
            yield row

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = int(state["position"])


def test_stream_resume_restores_dataset_position_and_pending_document_chunks(monkeypatch):
    created = []
    rows = [
        {"text": "10 11 12 13 14 15", "url": "a", "timestamp": "t1"},
        {"text": "20 21 22 23", "url": "b", "timestamp": "t2"},
    ]

    def load_dataset(*args, **kwargs):
        dataset = FakeIterable(copy.deepcopy(rows))
        created.append((args, kwargs, dataset))
        return dataset

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))

    def make_worker():
        return HuggingFaceC4Worker(
            repository="allenai/c4",
            revision="commit",
            split="train",
            tokenizer=FakeTokenizer(),
            maximum_context=4,
            seed=777,
            worker_index=0,
            total_workers=1,
            shuffle_shards=True,
        )

    original = make_worker()
    first = original.next_sample()
    saved = original.state_dict()
    expected = [original.next_sample(), original.next_sample()]

    restored = make_worker()
    restored.load_state_dict(saved)
    actual = [restored.next_sample(), restored.next_sample()]

    assert first["input_ids"].tolist() == [10, 11, 12, 13]
    assert saved["row_position"] == 1
    assert len(saved["pending_chunks"]) == 1
    assert created[0][2].shuffle_arguments == (777, 1)
    for expected_sample, actual_sample in zip(expected, actual, strict=True):
        assert expected_sample["document_hash"] == actual_sample["document_hash"]
        assert expected_sample["chunk_index"] == actual_sample["chunk_index"]
        assert torch.equal(expected_sample["input_ids"], actual_sample["input_ids"])


def test_stream_resume_rejects_worker_topology_change(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *args, **kwargs: FakeIterable([{"text": "1 2"}])),
    )
    worker = HuggingFaceC4Worker(
        repository="allenai/c4",
        revision="commit",
        split="train",
        tokenizer=FakeTokenizer(),
        maximum_context=4,
        seed=1,
        worker_index=0,
        total_workers=1,
        shuffle_shards=False,
    )
    state = worker.state_dict()
    state["total_workers"] = 2
    try:
        worker.load_state_dict(state)
    except ValueError as error:
        assert "topology changed" in str(error)
    else:
        raise AssertionError("topology change must be rejected")
