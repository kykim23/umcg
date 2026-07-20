import argparse
import json

from umcg.cli.prepare_c4 import prepare


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    init_kwargs = {"_commit_hash": "tokenizer-commit"}

    def __len__(self):
        return 2048

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return list(range(2, 2 + int(text)))


def test_local_preparation_preserves_document_and_chunk_boundaries(
    project_root, tmp_path, monkeypatch
):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "1100", "url": "first"}),
                json.dumps({"text": "3", "url": "second"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "umcg.cli.prepare_c4.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    output = tmp_path / "prepared"
    manifest = prepare(
        argparse.Namespace(
            input_jsonl=str(input_path),
            output_dir=str(output),
            tokenizer="fake",
            tokenizer_revision="a" * 40,
            estimator_config=str(
                project_root / "configs/estimator/russian_roulette_safe_1024.json"
            ),
            parents_per_shard=2,
        )
    )
    records = []
    for shard_name in manifest["shards"]:
        records.extend(
            json.loads(line)
            for line in (output / shard_name).read_text(encoding="utf-8").splitlines()
        )
    assert manifest["document_count"] == 2
    assert manifest["parent_count"] == 3
    assert [record["chunk_index"] for record in records] == [0, 1, 0]
    assert [record["token_start"] for record in records] == [0, 1024, 0]
    assert [record["token_end"] for record in records] == [1024, 1100, 3]
    assert records[0]["document_hash"] == records[1]["document_hash"]
    assert records[2]["document_hash"] != records[0]["document_hash"]
    assert len(records[0]["tokens"]) == 1024
    assert records[1]["tokens"][-1] == 1
    assert records[2]["tokens"][-1] == 1
