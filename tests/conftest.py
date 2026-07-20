from pathlib import Path

import pytest
import torch

from umcg.model.attention import AttentionSelection

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def eager_selection() -> AttentionSelection:
    return AttentionSelection(
        requested_backend="eager",
        resolved_backend="eager",
        huggingface_implementation="eager",
        hardware={},
        package_versions={},
        rank_probe_results=[],
    )


@pytest.fixture
def flatten_gradients():
    def flatten(model: torch.nn.Module) -> torch.Tensor:
        values = []
        for parameter in model.parameters():
            if parameter.grad is None:
                values.append(torch.zeros_like(parameter).reshape(-1))
            else:
                values.append(parameter.grad.detach().reshape(-1).clone())
        return torch.cat(values)

    return flatten
