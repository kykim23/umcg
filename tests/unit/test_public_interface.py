from pathlib import Path

import pytest

from umcg.cli.arguments import build_training_parser

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
    assert not any((project_root / "src/umcg_pretraining.egg-info").glob("*"))
