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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    distributed = parser.add_argument_group("distributed and model")
    distributed.add_argument(
        "--distributed_backend",
        required=True,
        choices=DISTRIBUTED_BACKENDS,
        help="Distributed runtime used by every rank.",
    )
    distributed.add_argument(
        "--zero_stage",
        type=int,
        choices=(1, 2, 3),
        help="Required only for distributed_backend=zero; rejected otherwise.",
    )
    distributed.add_argument(
        "--model_backend", required=True, choices=MODEL_BACKENDS, help="Model implementation."
    )
    distributed.add_argument(
        "--model_config", required=True, metavar="PATH", help="Canonical LLaMA model JSON."
    )
    distributed.add_argument(
        "--tokenizer", default="t5-base", metavar="NAME", help="Tokenizer repository or path."
    )
    distributed.add_argument(
        "--tokenizer_revision",
        default="main",
        metavar="REVISION",
        help="Tokenizer revision resolved to an immutable commit at startup.",
    )

    precision = parser.add_argument_group("precision and attention")
    precision.add_argument(
        "--precision", required=True, choices=PRECISIONS, help="Training numerical precision."
    )
    precision.add_argument(
        "--attention_backend",
        default="automatic",
        choices=ATTENTION_BACKENDS,
        help="Attention kernel; only automatic may fall back to another backend.",
    )

    estimator = parser.add_argument_group("gradient estimator")
    estimator.add_argument(
        "--estimator_config",
        required=True,
        metavar="PATH",
        help=(
            "Context levels and tail probabilities Q_k. Full gradients still require the "
            "context levels but ignore Q_k."
        ),
    )
    estimator.add_argument(
        "--gradient_estimator",
        required=True,
        choices=GRADIENT_ESTIMATORS,
        help="full always uses the maximum context; russian_roulette samples with Q_k.",
    )

    optimizer = parser.add_argument_group("optimizer and scheduler")
    optimizer.add_argument("--optimizer", required=True, choices=OPTIMIZERS, help="Optimizer.")
    optimizer.add_argument("--scheduler", required=True, choices=SCHEDULERS, help="LR scheduler.")
    optimizer.add_argument(
        "--learning_rate", required=True, type=float, metavar="FLOAT", help="Peak learning rate."
    )
    optimizer.add_argument("--beta1", type=float, default=0.9, help="First Adam moment decay.")
    optimizer.add_argument("--beta2", type=float, default=0.95, help="Second Adam moment decay.")
    optimizer.add_argument("--epsilon", type=float, default=1e-8, help="Optimizer epsilon.")
    optimizer.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    optimizer.add_argument(
        "--momentum",
        type=float,
        help="Optional only for sgdm or muon; sgd explicitly rejects it.",
    )
    optimizer.add_argument(
        "--gradient_clip_norm",
        type=float,
        metavar="FLOAT",
        help="Optional positive global gradient-norm limit.",
    )

    schedule = parser.add_argument_group("batch, training, evaluation, and saving")
    schedule.add_argument(
        "--batch_size",
        required=True,
        metavar="INTEGER|auto",
        help="Per-rank parent batch size, or auto for a VRAM probe.",
    )
    schedule.add_argument(
        "--total_batch_size",
        required=True,
        type=int,
        metavar="INTEGER",
        help="Global parent batch size across ranks and accumulation.",
    )
    schedule.add_argument(
        "--num_training_steps",
        required=True,
        type=int,
        metavar="INTEGER",
        help="Optimizer updates.",
    )
    schedule.add_argument(
        "--warmup_steps", required=True, type=int, metavar="INTEGER", help="LR warmup updates."
    )
    schedule.add_argument(
        "--eval_every",
        required=True,
        type=int,
        metavar="INTEGER",
        help="Updates between evaluations.",
    )
    schedule.add_argument(
        "--eval_parent_batches",
        type=int,
        default=32,
        metavar="INTEGER",
        help=(
            "Full-context validation batches per rank; total parents equal this value times "
            "batch_size times world_size."
        ),
    )
    schedule.add_argument(
        "--save_every",
        required=True,
        type=int,
        metavar="INTEGER",
        help="Updates between checkpoints.",
    )
    schedule.add_argument(
        "--save_at_step",
        dest="save_at_steps",
        action="append",
        type=int,
        default=[],
        metavar="INTEGER",
        help=(
            "Additional exact optimizer update to checkpoint; repeat for multiple milestones."
        ),
    )

    data = parser.add_argument_group("C4 data")
    data.add_argument(
        "--c4_source",
        required=True,
        choices=("streaming", "local", "local_raw"),
        help=(
            "streaming reads Hub C4; local reads prepared token parents; local_raw reads and "
            "tokenizes local .json.gz shards online."
        ),
    )
    data.add_argument("--c4_repo", default="allenai/c4", metavar="NAME", help="Hub C4 repository.")
    data.add_argument(
        "--c4_revision", default="main", metavar="REVISION", help="Recorded C4 source revision."
    )
    data.add_argument(
        "--c4_local_path",
        metavar="PATH",
        help="Required for local and local_raw; rejected for streaming.",
    )
    data.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="INTEGER",
        help=(
            "Logical C4 iterators per rank, not CPU cores, threads, processes, or parallel "
            "tokenizers."
        ),
    )

    runtime = parser.add_argument_group("runtime performance and output")
    runtime.add_argument(
        "--save_dir", required=True, metavar="PATH", help="New or resumable run output directory."
    )
    runtime.add_argument("--seed", required=True, type=int, help="Global deterministic seed.")
    runtime.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Compile the training model with torch.compile.",
    )
    runtime.add_argument(
        "--compile_mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
        help="torch.compile mode, used only with --use_torch_compile.",
    )
    runtime.add_argument(
        "--activation_checkpointing",
        action="store_true",
        help="Recompute transformer activations during backward to reduce VRAM.",
    )

    tracking = parser.add_argument_group("experiment tracking")
    tracking.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases logging mode.",
    )
    tracking.add_argument(
        "--wandb_project", default="umcg-pretraining", metavar="NAME", help="Tracking project."
    )
    tracking.add_argument("--wandb_entity", metavar="NAME", help="Optional tracking entity.")
    tracking.add_argument("--name", required=True, metavar="NAME", help="Unique run name.")

    restart_group = parser.add_argument_group("restart")
    restart = restart_group.add_mutually_exclusive_group()
    restart.add_argument(
        "--continue_from", metavar="PATH", help="Resume the complete native training state."
    )
    restart.add_argument(
        "--initial_weights",
        metavar="PATH",
        help="Start a new run from canonical model weights only.",
    )
    return parser


def parse_runtime_config(arguments: Sequence[str] | None = None) -> RuntimeConfig:
    namespace = build_training_parser().parse_args(arguments)
    namespace.save_at_steps = tuple(namespace.save_at_steps)
    return RuntimeConfig(**vars(namespace))
