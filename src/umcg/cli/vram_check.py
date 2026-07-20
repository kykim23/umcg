"""Run the production backend at maximum context and recommend a microbatch."""

from __future__ import annotations

import json
from collections.abc import Sequence

from umcg.cli.arguments import parse_runtime_config
from umcg.config import EstimatorConfig, load_json_object, validate_model_config
from umcg.distributed.runtime import DistributedContext
from umcg.training.runner import (
    _absolute_runtime_paths,
    _attention_selection,
    _probe_automatic_batch,
)


def main(arguments: Sequence[str] | None = None) -> None:
    config = _absolute_runtime_paths(parse_runtime_config(arguments))
    context = DistributedContext.initialize()
    try:
        config.validate_before_model_creation(context.world_size)
        estimator = EstimatorConfig.load(config.estimator_config)
        model_config = load_json_object(config.model_config)
        validate_model_config(model_config, estimator.context_levels[-1])
        attention = _attention_selection(
            config,
            context=context,
            estimator=estimator,
            model_config=model_config,
            checkpoint_manifest=None,
        )
        selection = _probe_automatic_batch(
            config,
            context=context,
            estimator=estimator,
            attention=attention,
            model_config=model_config,
        )
        report = {
            "distributed_backend": config.distributed_backend,
            "zero_stage": config.zero_stage,
            "precision": config.precision,
            "attention": attention.to_dict(),
            "maximum_context": estimator.context_levels[-1],
            "batch_selection": selection.to_dict(),
        }
        if context.is_primary:
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    finally:
        context.close()
