import argparse
import dataclasses
from pathlib import Path

import pytest

from umcg.cli.arguments import build_training_parser
from umcg.config import RuntimeConfig
from umcg.training.runner import SUPPORTED_RUNTIME_ARGUMENTS

CANONICAL_ENTRY_FILES = {
    "torchrun_main.py",
    "calibrate_main.py",
    "vram_check_main.py",
    "export_weights_main.py",
    "smoke_main.py",
    "prepare_c4_main.py",
}


def valid_arguments(project_root: Path) -> list[str]:
    return [
        "--distributed_backend",
        "ddp",
        "--model_backend",
        "huggingface",
        "--model_config",
        str(project_root / "configs/model/llama_tiny_smoke_1024.json"),
        "--precision",
        "float32",
        "--estimator_config",
        str(project_root / "configs/estimator/russian_roulette_safe_1024.json"),
        "--gradient_estimator",
        "full",
        "--optimizer",
        "adamw",
        "--scheduler",
        "cosine",
        "--learning_rate",
        "0.001",
        "--batch_size",
        "1",
        "--total_batch_size",
        "1",
        "--num_training_steps",
        "1",
        "--warmup_steps",
        "0",
        "--eval_every",
        "1",
        "--save_every",
        "1",
        "--save_at_step",
        "1",
        "--save_dir",
        "out",
        "--c4_source",
        "streaming",
        "--seed",
        "1",
        "--name",
        "test",
    ]


def test_only_six_canonical_root_entry_files(project_root):
    actual = {path.name for path in project_root.glob("*_main.py")}
    assert actual == CANONICAL_ENTRY_FILES
    assert not (project_root / "src/umcg/__main__.py").exists()
    assert "[project.scripts]" not in (project_root / "pyproject.toml").read_text()


def test_required_recovered_files_are_present_and_not_broadly_ignored(project_root):
    required = (
        project_root / "src/umcg/data/c4_stream.py",
        project_root / "src/umcg/data/document_chunks.py",
        project_root / "scripts/inspect_umcg_env.sh",
        project_root / "scripts/run_external_validation.sh",
    )
    assert all(path.is_file() for path in required)
    ignore_lines = {
        line.strip()
        for line in (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert "/data/" in ignore_lines
    assert "data/" not in ignore_lines
    assert "*.sh" not in ignore_lines


@pytest.mark.parametrize(
    "forbidden",
    [
        "--use_hf_model",
        "--dtype",
        "--max_length",
        "--gradient_accumulation",
        "--resume_from",
        "--distributed_back",
    ],
)
def test_legacy_and_abbreviated_flags_are_rejected(project_root, forbidden):
    parser = build_training_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*valid_arguments(project_root), forbidden, "value"])


def test_runner_argument_summary_matches_parser_and_runtime_config():
    parser = build_training_parser()
    parser_fields = {
        action.dest for action in parser._actions if action.dest not in {"help", argparse.SUPPRESS}
    }
    runtime_fields = {field.name for field in dataclasses.fields(RuntimeConfig)}
    runner_fields = {
        field
        for _group_name, group_fields in SUPPORTED_RUNTIME_ARGUMENTS
        for field in group_fields
    }
    runner_field_count = sum(len(fields) for _name, fields in SUPPORTED_RUNTIME_ARGUMENTS)

    assert runner_field_count == len(runner_fields)
    assert runner_fields == parser_fields == runtime_fields


def test_training_help_is_grouped_and_every_argument_is_explained():
    parser = build_training_parser()
    help_text = parser.format_help()
    normalized_help = " ".join(help_text.split())
    for group_name, _fields in SUPPORTED_RUNTIME_ARGUMENTS:
        assert f"{group_name}:" in help_text
    assert "INTEGER|auto" in help_text
    assert "not CPU cores, threads, processes" in normalized_help
    assert "Full gradients still require the context levels but ignore Q_k" in normalized_help
    for action in parser._actions:
        if action.dest != "help":
            assert action.help not in {None, argparse.SUPPRESS}


def test_package_initializers_do_not_reexport_public_objects(project_root):
    for path in (project_root / "src/umcg").rglob("__init__.py"):
        assert "from umcg" not in path.read_text(encoding="utf-8")


def test_source_has_no_previous_repository_or_bitsandbytes_dependency(project_root):
    sources = [*project_root.glob("*_main.py"), *(project_root / "src").rglob("*.py")]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "GaLore_DoRA_experiment" not in combined
    assert "import bitsandbytes" not in combined
    assert "from bitsandbytes" not in combined


def test_public_artifacts_have_no_legacy_entry_names(project_root):
    forbidden = (
        "--use_hf_model",
        "--dtype",
        "--max_length",
        "--gradient_accumulation",
        "--resume_from",
        "umcg train",
        "python -m umcg",
        "prepare-data",
        "train_smoke",
        "segment_ids",
    )
    paths = [
        project_root / "README.md",
        project_root / "pyproject.toml",
        project_root / "UMCG_IMPLEMENTATION_PROGRESS.md",
        *project_root.glob("*_main.py"),
        *(project_root / "src").rglob("*.py"),
        *(project_root / "configs").rglob("*.json"),
        *(project_root / "docs").rglob("*.md"),
        *(project_root / "scripts").rglob("*.sh"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for value in forbidden:
        assert value not in combined
    assert not any((project_root / "build").rglob("*.py"))
    ignore_lines = (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "*.egg-info/" in ignore_lines
