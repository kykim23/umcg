import pytest
import torch
from torch import nn

from umcg.model.factory import _model_dtype
from umcg.precision import build_precision_runtime


def test_float32_has_no_autocast_scaler_or_float8_metadata():
    runtime = build_precision_runtime(
        name="float32",
        device=torch.device("cpu"),
        distributed_backend="ddp",
        model=nn.Linear(4, 4),
    )
    assert runtime.scaler is None
    assert runtime.float8_metadata is None
    assert runtime.state_dict()["name"] == "float32"


def test_amp_precisions_keep_fp32_master_parameters():
    assert _model_dtype("float16") == torch.float32
    assert _model_dtype("bfloat16") == torch.float32
    assert _model_dtype("float8") == torch.bfloat16


def test_float8_on_unsupported_hardware_fails_without_fallback():
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        build_precision_runtime(
            name="float8",
            device=torch.device("cpu"),
            distributed_backend="ddp",
            model=nn.Linear(4, 4),
        )
