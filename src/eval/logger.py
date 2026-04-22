"""Unified experiment logger — every run produces a reproducible JSON record."""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """
    Writes one JSON file per experiment run using a canonical schema.
    Compatible with pandas.read_json(lines=True) for downstream analysis.
    """

    def __init__(self, output_dir: str = "results/experiments"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def log(
        self,
        experiment_id: str,
        model: str,
        method: str,
        config: Dict[str, Any],
        eval_results: Dict[str, Any],
        seed: int = 42,
        notes: str = "",
    ) -> str:
        """
        Persist a complete experiment record.

        Parameters
        ----------
        experiment_id : str
            Human-readable identifier, e.g. "e1_main_table" or "ablation_layer_sweep".
        model : str
            HuggingFace model ID.
        method : str
            Algorithm name, e.g. "spectral_sharpening" or "repe_baseline".
        config : dict
            Hyperparameters — layer, alpha, n_pairs, quantization, etc.
        eval_results : dict
            Benchmark name → result dict produced by _eval_stats().
            Each value must contain at minimum: n, hits, rate, rate_pct,
            ci_95, ci_95_pct, ci_method.
        seed : int
            Global random seed used for dataset shuffling.
        notes : str
            Free-form annotation (e.g. "E3 Mistral anomaly re-run").

        Returns
        -------
        str  Path to the written JSON file.
        """
        record = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "method": method,
            "config": config,
            "seed": seed,
            "eval": eval_results,
            "notes": notes,
            "run_hash": self._content_hash(experiment_id, model, config, seed),
        }

        fname = f"{experiment_id}_{int(time.time())}.json"
        path = os.path.join(self.output_dir, fname)
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        return path

    def log_incremental(self, path: str, new_row: Dict[str, Any]) -> None:
        """Append a row to an existing JSONL checkpoint file (for sweep loops)."""
        with open(path, "a") as f:
            f.write(json.dumps(new_row) + "\n")

    @staticmethod
    def _content_hash(*args) -> str:
        payload = json.dumps(args, sort_keys=True, default=str).encode()
        return hashlib.md5(payload).hexdigest()[:8]