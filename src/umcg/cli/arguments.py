"""One canonical spelling for every public training argument."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from umcg.config import (
    ATTENTION_BACKENDS,
    DISTRIBUTED_BACKENDS,
    GRADIENT_ESTIMATORS,
    MODEL_BACKENDS,
    OPTIMIZERS,
    PRECISIONS,
    SCHEDULERS,
    RuntimeConfig,
)


def build_training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchrun_main.py",
        description="Independent C4 pretraining with full or Russian-roulette gradients",
        allow_abbrev=False,
    )
    parser.add_argument("--distributed_backend", required=True, choices=DISTRIBUTED_BACKENDS)
    parser.add_argument("--zero_stage", type=int, choices=(1, 2, 3))
    parser.add_argument("--model_backend", required=True, choices=MODEL_BACKENDS)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--tokenizer", default="t5-base")
    parser.add_argument("--tokenizer_revision", default="main")
    parser.add_argument("--precision", required=True, choices=PRECISIONS)
    parser.add_argument("--attention_backend", default="automatic", choices=ATTENTION_BACKENDS)
    parser.add_argument("--estimator_config", required=True)
    parser.add_argument("--gradient_estimator", required=True, choices=GRADIENT_ESTIMATORS)
    parser.add_argument("--optimizer", required=True, choices=OPTIMIZERS)
    parser.add_argument("--scheduler", required=True, choices=SCHEDULERS)
    parser.add_argument("--learning_rate", required=True, type=float)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--batch_size", required=True)
    parser.add_argument("--total_batch_size", required=True, type=int)
    parser.add_argument("--num_training_steps", required=True, type=int)
    parser.add_argument("--warmup_steps", required=True, type=int)
    parser.add_argument("--eval_every", required=True, type=int)
    parser.add_argument("--eval_parent_batches", type=int, default=32)
    parser.add_argument("--save_every", required=True, type=int)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument(
        "--c4_source", required=True, choices=("streaming", "local", "local_raw")
    )
    parser.add_argument("--c4_repo", default="allenai/c4")
    parser.add_argument("--c4_revision", default="main")
    parser.add_argument("--c4_local_path")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--gradient_clip_norm", type=float)
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb_project", default="umcg-pretraining")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--name", required=True)
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--continue_from")
    restart.add_argument("--initial_weights")
    return parser


def parse_runtime_config(arguments: Sequence[str] | None = None) -> RuntimeConfig:
    namespace = build_training_parser().parse_args(arguments)
    return RuntimeConfig(**vars(namespace))
