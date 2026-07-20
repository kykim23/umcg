import torch

from umcg.model.attention import AttentionSelection
from umcg.model.causal_loss import per_token_causal_loss
from umcg.model.factory import build_model


def build_pair(project_root, eager_selection):
    arguments = {
        "model_config_path": project_root / "configs/model/llama_tiny_smoke_1024.json",
        "precision": "float32",
        "attention_selection": eager_selection,
        "activation_checkpointing": False,
        "device": torch.device("cpu"),
        "maximum_context": 16,
    }
    torch.manual_seed(11)
    huggingface = build_model(model_backend="huggingface", **arguments).model
    reference = build_model(model_backend="reference", **arguments).model
    reference.load_canonical_state_dict(huggingface.canonical_state_dict())
    return huggingface, reference


def test_huggingface_and_reference_logits_losses_and_gradients_match(
    project_root, eager_selection, flatten_gradients
):
    huggingface, reference = build_pair(project_root, eager_selection)
    input_ids = torch.tensor([[5, 7, 9, 11, 13, 15, 17, 19]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)

    hf_logits = huggingface.forward_logits(input_ids, attention_mask, position_ids)
    reference_logits = reference.forward_logits(input_ids, attention_mask, position_ids)
    torch.testing.assert_close(hf_logits, reference_logits, rtol=1e-5, atol=1e-6)

    hf_losses = huggingface(input_ids, attention_mask, position_ids)
    reference_losses = reference(input_ids, attention_mask, position_ids)
    torch.testing.assert_close(hf_losses, reference_losses, rtol=1e-5, atol=1e-6)
    hf_losses.mean().backward()
    reference_losses.mean().backward()
    torch.testing.assert_close(
        flatten_gradients(huggingface),
        flatten_gradients(reference),
        rtol=2e-5,
        atol=2e-6,
    )


def test_chunked_lm_head_matches_full_vocabulary_cross_entropy(project_root, eager_selection):
    model, _ = build_pair(project_root, eager_selection)
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    chunked = model(input_ids, attention_mask, position_ids)
    full = per_token_causal_loss(
        model.forward_logits(input_ids, attention_mask, position_ids), input_ids
    ).token_losses
    torch.testing.assert_close(chunked, full)


def test_causal_prefix_is_invariant_for_both_model_backends(project_root, eager_selection):
    for model in build_pair(project_root, eager_selection):
        input_ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
        full = model.forward_logits(input_ids, mask, positions)
        prefix = model.forward_logits(input_ids[:, :4], mask[:, :4], positions[:, :4])
        torch.testing.assert_close(full[:, :4], prefix, rtol=1e-5, atol=1e-6)


def test_right_padded_parent_matches_its_active_prefix(project_root, eager_selection):
    for model in build_pair(project_root, eager_selection):
        input_ids = torch.tensor([[4, 5, 6, 1, 0, 0, 0, 0]])
        mask = torch.tensor([[True, True, True, True, False, False, False, False]])
        positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
        padded = model.forward_logits(input_ids, mask, positions)
        prefix = model.forward_logits(input_ids[:, :4], mask[:, :4], positions[:, :4])
        torch.testing.assert_close(padded[:, :4], prefix, rtol=1e-5, atol=1e-6)


def test_huggingface_and_reference_match_with_pytorch_sdpa(project_root):
    selection = AttentionSelection(
        requested_backend="pytorch_sdpa",
        resolved_backend="pytorch_sdpa_math",
        huggingface_implementation="sdpa",
        hardware={},
        package_versions={},
        rank_probe_results=[],
    )
    huggingface, reference = build_pair(project_root, selection)
    input_ids = torch.tensor([[5, 7, 9, 1, 0, 0]])
    mask = torch.tensor([[True, True, True, True, False, False]])
    positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
    torch.testing.assert_close(
        huggingface.forward_logits(input_ids, mask, positions),
        reference.forward_logits(input_ids, mask, positions),
        rtol=1e-5,
        atol=1e-6,
    )
