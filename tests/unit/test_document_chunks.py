import torch

from umcg.data.collate import collate_parent_samples
from umcg.data.document_chunks import document_sha256, split_document_tokens


def test_nine_thousand_token_document_is_split_without_overlap_or_mid_chunk_eos():
    raw_tokens = list(range(10_000, 19_000))
    eos_token_id = 2
    pad_token_id = 0
    samples = split_document_tokens(
        raw_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        maximum_context=4096,
        document_hash="document-a",
    )

    assert len(samples) == 3
    assert [sample["chunk_index"] for sample in samples] == [0, 1, 2]
    assert [sample["token_start"] for sample in samples] == [0, 4096, 8192]
    assert [sample["token_end"] for sample in samples] == [4096, 8192, 9000]
    assert samples[0]["input_ids"].tolist() == raw_tokens[:4096]
    assert samples[1]["input_ids"].tolist() == raw_tokens[4096:8192]

    final_active_length = 9001 - 8192
    final = samples[2]
    assert final["input_ids"][final_active_length - 1].item() == eos_token_id
    assert torch.equal(
        final["input_ids"][final_active_length:],
        torch.full((4096 - final_active_length,), pad_token_id, dtype=torch.long),
    )
    assert final["attention_mask"].sum().item() == final_active_length
    assert final["causal_target_mask"].sum().item() == final_active_length - 1


def test_documents_never_mix_and_prefixes_keep_the_parent_origin():
    first = split_document_tokens(
        [10, 11, 12, 13, 14],
        eos_token_id=1,
        pad_token_id=0,
        maximum_context=4,
        document_hash=document_sha256("first"),
    )
    second = split_document_tokens(
        [20, 21, 22],
        eos_token_id=1,
        pad_token_id=0,
        maximum_context=4,
        document_hash=document_sha256("second"),
    )
    assert {sample["document_hash"] for sample in first}.isdisjoint(
        {sample["document_hash"] for sample in second}
    )

    parent = collate_parent_samples([first[0]])
    prefix = parent.prefix(2)
    assert prefix.document_hashes == parent.document_hashes
    assert prefix.chunk_indices == parent.chunk_indices
    assert prefix.token_starts == parent.token_starts
    assert prefix.input_ids.tolist() == [[10, 11]]
    assert prefix.causal_target_mask.tolist() == [[True]]


def test_a_document_without_a_causal_target_is_excluded():
    assert (
        split_document_tokens(
            [],
            eos_token_id=1,
            pad_token_id=0,
            maximum_context=4,
            document_hash="empty",
        )
        == []
    )


def test_collation_keeps_attention_and_target_masks_separate():
    samples = split_document_tokens(
        [7, 8],
        eos_token_id=1,
        pad_token_id=0,
        maximum_context=8,
        document_hash="short",
    )
    batch = collate_parent_samples(samples)
    assert batch.attention_mask.tolist() == [[True, True, True, False, False, False, False, False]]
    assert batch.causal_target_mask.tolist() == [[True, True, False, False, False, False, False]]
    batch.validate()
