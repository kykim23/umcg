"""Always-on local JSONL metrics plus rank-zero W&B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metric: dict[str, Any]) -> None:
        required = {
            "optimizer_update",
            "valid_tokens",
            "full_context_equivalent_tokens",
            "signed_estimator_scalar",
            "learning_rate",
            "distributed_backend",
            "attention_backend",
            "precision",
        }
        missing = sorted(required - set(metric))
        if missing:
            raise ValueError(f"metric missing required fields: {missing}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()


class WandbLogger:
    def __init__(
        self,
        *,
        mode: str,
        project: str,
        entity: str | None,
        run_name: str,
        config: dict[str, Any],
        run_id: str | None,
        enabled: bool,
    ) -> None:
        self.run = None
        if not enabled or mode == "disabled":
            return
        import wandb

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=mode,
            config=config,
            id=run_id,
            resume="must" if run_id is not None else None,
        )

    @property
    def run_id(self) -> str | None:
        return None if self.run is None else str(self.run.id)

    def log(self, metric: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(metric, step=metric["optimizer_update"])

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
