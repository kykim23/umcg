import json
from types import SimpleNamespace

import pytest

from umcg.training import runner
from umcg.training.checkpoint import (
    list_complete_checkpoints,
    read_checkpoint_manifest,
    validate_resume_contract,
)


def _save_directory_config(tmp_path, *, continue_from=None):
    return SimpleNamespace(
        save_dir=str(tmp_path / "run"),
        continue_from=None if continue_from is None else str(continue_from),
    )


class _PrimaryContext:
    is_primary = True

    def barrier(self):
        return None


@pytest.fixture
def local_broadcast(monkeypatch):
    monkeypatch.setattr(runner.distributed, "broadcast_object_list", lambda values, src: None)


def test_incomplete_checkpoint_is_never_loadable(tmp_path):
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        read_checkpoint_manifest(checkpoint)
    assert list_complete_checkpoints(tmp_path) == []


def test_only_complete_checkpoint_directories_are_listed(tmp_path):
    complete = tmp_path / "checkpoint-2"
    complete.mkdir()
    (complete / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "distributed_backend": "ddp"}),
        encoding="utf-8",
    )
    (complete / "COMPLETE").write_text("complete\n", encoding="utf-8")
    hidden = tmp_path / ".checkpoint-3.tmp-test"
    hidden.mkdir()
    (hidden / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    (hidden / "COMPLETE").write_text("complete\n", encoding="utf-8")
    assert read_checkpoint_manifest(complete)["distributed_backend"] == "ddp"
    assert list_complete_checkpoints(tmp_path) == [complete]


def test_resume_contract_reports_every_changed_field():
    checkpoint = {"world_size": 4, "workers": 2, "precision": "bfloat16"}
    current = {"world_size": 2, "workers": 1, "precision": "bfloat16"}
    with pytest.raises(ValueError, match="world_size") as raised:
        validate_resume_contract(checkpoint, current)
    assert "workers" in str(raised.value)


def test_new_run_rejects_nonempty_save_directory(tmp_path, local_broadcast):
    save_directory = tmp_path / "run"
    save_directory.mkdir()
    (save_directory / "existing.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty save_dir"):
        runner._prepare_save_directory(
            _save_directory_config(tmp_path),
            _PrimaryContext(),
        )


def test_resume_allows_only_its_existing_run_directory(tmp_path, local_broadcast):
    save_directory = tmp_path / "run"
    checkpoint = save_directory / "checkpoint-00000001"
    checkpoint.mkdir(parents=True)

    assert (
        runner._prepare_save_directory(
            _save_directory_config(tmp_path, continue_from=checkpoint),
            _PrimaryContext(),
        )
        == save_directory
    )

    other_checkpoint = tmp_path / "other" / "checkpoint-00000001"
    other_checkpoint.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="same run directory"):
        runner._prepare_save_directory(
            _save_directory_config(tmp_path, continue_from=other_checkpoint),
            _PrimaryContext(),
        )
