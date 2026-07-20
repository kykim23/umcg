import pytest
import torch
import transformers

from umcg.model.attention import (
    AttentionHardware,
    AttentionProbeResult,
    _run_unpadded_attention,
    candidate_backends,
    resolve_attention_backend,
    run_attention,
)


def test_unpadded_sdpa_probe_preserves_gqa_contract(monkeypatch):
    observed = {}

    def fake_sdpa(query, key, value, **kwargs):
        observed["query_heads"] = query.shape[1]
        observed["key_heads"] = key.shape[1]
        observed.update(kwargs)
        return query

    monkeypatch.setattr(
        "umcg.model.attention.functional.scaled_dot_product_attention",
        fake_sdpa,
    )
    query = torch.randn(1, 4, 8, 16)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    mask = torch.ones(1, 8, dtype=torch.bool)

    output = _run_unpadded_attention("pytorch_sdpa", query, key, value, mask)

    assert output.shape == query.shape
    assert observed["query_heads"] == 4
    assert observed["key_heads"] == 2
    assert observed["attn_mask"] is None
    assert observed["is_causal"] is True
    assert observed["enable_gqa"] is True


def test_turing_never_offers_flash_attention_packages():
    hardware = AttentionHardware(True, 7, 5, "Turing")
    candidates = candidate_backends(
        hardware,
        precision="float16",
        flash_available={2: True, 3: True, 4: True},
    )
    assert not any(name.startswith("flash_attention") for name in candidates)
    assert candidates[-1] == "eager"


def test_automatic_prefers_pytorch_when_it_is_within_three_percent(monkeypatch):
    monkeypatch.setattr(
        "umcg.model.attention.installed_attention_packages",
        lambda: {"torch": "test"},
    )

    def probe(backend, **kwargs):
        durations = {"pytorch_sdpa_math": 1.02, "eager": 1.0}
        return AttentionProbeResult(
            backend=backend,
            passed=True,
            worst_level_time_ms=durations[backend],
            level_times_ms={8: durations[backend]},
            error=None,
        )

    selection = resolve_attention_backend(
        "automatic",
        device=torch.device("cpu"),
        precision="float32",
        context_levels=(8,),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dimension=4,
        probe=probe,
    )
    assert selection.resolved_backend == "pytorch_sdpa_math"
    assert selection.huggingface_implementation == "sdpa"


def test_explicit_backend_failure_never_falls_back():
    with pytest.raises(RuntimeError, match="passed on no common candidate"):
        resolve_attention_backend(
            "flash_attention_2",
            device=torch.device("cpu"),
            precision="float32",
            context_levels=(8,),
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dimension=4,
            probe=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("unsupported explicit backend must not be probed")
            ),
        )


def test_external_attention_interface_receives_transformers_module_contract(monkeypatch):
    observed = {}

    def fake_attention(module, query, key, value, attention_mask, **kwargs):
        observed["implementation"] = module.config._attn_implementation
        observed["layer_idx"] = module.layer_idx
        observed["groups"] = module.num_key_value_groups
        output = value.repeat_interleave(module.num_key_value_groups, dim=1)
        return output.transpose(1, 2), None

    monkeypatch.setattr(
        transformers,
        "AttentionInterface",
        lambda: {"flash_attention_2": fake_attention},
    )
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    output = run_attention("flash_attention_2", query, key, value, mask)
    assert output.shape == query.shape
    assert observed == {
        "implementation": "flash_attention_2",
        "layer_idx": 0,
        "groups": 2,
    }
