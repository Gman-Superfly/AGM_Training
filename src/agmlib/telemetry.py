from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class MetricsLogger:
    """Append-only JSONL metrics logger for quick diagnostics."""

    def __init__(self, sink: Optional[Path] = None):
        self.sink = sink or (Path("logs") / "metrics.jsonl")
        self.sink.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        assert isinstance(record, dict), "record must be a dict"
        line = json.dumps(record)
        with open(self.sink, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def histogram(values: Any, *, bins: int = 10) -> Dict[str, Any]:
    """Compute simple histogram for telemetry (returns Python lists)."""
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy required for histogram") from e

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    elif hasattr(values, "numpy"):
        values = values.numpy()
    values = np.asarray(values).ravel()
    if values.size == 0:
        return {"bins": [], "counts": [], "min": 0.0, "max": 0.0}
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "min": float(values.min()),
        "max": float(values.max()),
    }

