"""Compare one-rank accumulation with a four-rank padded global batch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from umcg.data.collate import ParentBatch
from umcg.estimators.global_objective import (
    build_global_update_plan,
    estimator_scalar,
    global_token_coefficients,
)
from umcg.estimators.levels import LevelSpec
from umcg.estimators.russian_roulette import LevelSampler

GLOBAL_ATTENTION_MASKS = (
    (False, False, False, False, False, False),
    (True, True, False, False, False, False),
    (True, True, True, True, False, False),
    (True, True, True, True, True, True),
)

GLOBAL_FEATURES = (
    (1.0, 2.0, 3.0, 4.0, 5.0),
    (6.0, 7.0, 8.0, 9.0, 10.0),
    (11.0, 12.0, 13.0, 14.0, 15.0),
    (16.0, 17.0, 18.0, 19.0, 20.0),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    return parser


def make_batch(global_index: int) -> ParentBatch:
    attention_mask = torch.tensor([GLOBAL_ATTENTION_MASKS[global_index]], dtype=torch.bool)
    batch = ParentBatch(
        input_ids=torch.arange(6).unsqueeze(0),
        attention_mask=attention_mask,
        causal_target_mask=attention_mask[:, :-1] & attention_mask[:, 1:],
        position_ids=torch.arange(6).unsqueeze(0),
        document_hashes=[f"document-{global_index}"],
        chunk_indices=[0],
        token_starts=[0],
        token_ends=[int(attention_mask.sum())],
        urls=[""],
        timestamps=[""],
    )
    batch.validate()
    return batch


def main() -> None:
    arguments = build_parser().parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in {1, 2, 4}:
        raise RuntimeError("global gradient validation requires one, two, or four processes")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed.init_process_group("nccl", init_method="env://", device_id=device)
    model = nn.Linear(1, 1, bias=False, device=device)
    with torch.no_grad():
        model.weight.fill_(2.0)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank])
    levels = LevelSpec((2, 4, 6), (1.0, 1.0, 1.0))
    global_indices = list(range(rank, len(GLOBAL_FEATURES), world_size))
    batches = [make_batch(index) for index in global_indices]
    plan = build_global_update_plan(
        batches,
        levels=levels,
        gradient_estimator="full",
        level_sampler=LevelSampler(levels, seed=777),
        rank=rank,
        device=device,
    )
    wrapped.zero_grad(set_to_none=True)
    for microbatch_index, (global_index, batch) in enumerate(
        zip(global_indices, batches, strict=True)
    ):
        final_microbatch = microbatch_index == len(batches) - 1
        synchronization = wrapped.no_sync() if not final_microbatch else torch.enable_grad()
        with synchronization:
            features = torch.tensor(
                GLOBAL_FEATURES[global_index], device=device, dtype=torch.float32
            ).view(-1, 1)
            token_values = wrapped(features).view(1, -1)
            coefficients = global_token_coefficients(
                batch.causal_target_mask.to(device),
                levels=levels,
                global_target_counts=plan.global_target_counts,
                sampled_level_index=levels.num_levels - 1,
                gradient_estimator="full",
                gradient_scale=float(world_size),
            )
            estimator_scalar(token_values, coefficients).backward()
    gradient = float(model.weight.grad.detach().cpu())
    expected_values = []
    for mask, features in zip(GLOBAL_ATTENTION_MASKS, GLOBAL_FEATURES, strict=True):
        target_mask = [left and right for left, right in zip(mask[:-1], mask[1:], strict=True)]
        expected_values.extend(
            value for value, valid in zip(features, target_mask, strict=True) if valid
        )
    expected = sum(expected_values) / len(expected_values)
    if abs(gradient - expected) > 1e-6:
        raise AssertionError(f"distributed gradient {gradient} differs from expected {expected}")
    if rank == 0:
        report = {
            "world_size": world_size,
            "gradient": gradient,
            "expected_gradient": expected,
            "global_target_counts": plan.global_target_counts.cpu().tolist(),
        }
        if arguments.reference is not None:
            reference = json.loads(Path(arguments.reference).read_text(encoding="utf-8"))
            if abs(gradient - float(reference["gradient"])) > 1e-6:
                raise AssertionError("one-process and four-process gradients differ")
            report["reference"] = str(Path(arguments.reference).resolve())
        output = Path(arguments.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
