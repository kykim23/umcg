#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import importlib.metadata
import json
import shutil
import torch

packages = [
    "torch",
    "transformers",
    "datasets",
    "deepspeed",
    "torchao",
    "flash-attn",
    "wandb",
]
versions = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
report = {
    "packages": versions,
    "build_tools": {
        name: shutil.which(name) for name in ("gcc", "g++", "as", "ninja")
    },
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "devices": [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": torch.cuda.get_device_capability(index),
        }
        for index in range(torch.cuda.device_count())
    ],
}
print(json.dumps(report, indent=2, sort_keys=True))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi topo -m
fi
