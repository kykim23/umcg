import json
import subprocess
import sys


def test_cpu_smoke_entrypoint_runs_one_real_update(project_root, tmp_path):
    output = tmp_path / "smoke"
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "smoke_main.py"),
            "--device",
            "cpu",
            "--model_backend",
            "reference",
            "--model_config",
            str(project_root / "configs/model/llama_tiny_smoke_1024.json"),
            "--precision",
            "float32",
            "--attention_backend",
            "eager",
            "--estimator_config",
            str(project_root / "configs/estimator/russian_roulette_safe_1024.json"),
            "--gradient_estimator",
            "russian_roulette",
            "--optimizer",
            "adamw",
            "--scheduler",
            "cosine",
            "--batch_size",
            "1",
            "--num_training_steps",
            "1",
            "--warmup_steps",
            "0",
            "--seed",
            "777",
            "--save_dir",
            str(output),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "smoke_report.json").read_text(encoding="utf-8"))
    assert report["finite"] is True
    assert len(report["updates"]) == 1
    assert report["updates"][0]["active_length"] == 1024
