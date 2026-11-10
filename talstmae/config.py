"""Experiment configuration shared by every notebook and script.

All values mirror Sections 3-4 of the manuscript. Change them here (or on a
Config instance) rather than inside individual notebooks, so that every
experiment uses one documented configuration.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np


@dataclass
class Config:
    # ------------------------------------------------------------- run mode / paths
    data_dir: str = "data"
    out_dir: str = "outputs"
    smoke_test: bool = False          # tiny synthetic data ONLY to check that code runs; never report its numbers
    datasets: tuple = ("nsl_kdd", "unsw_nb15", "cicids2018")

    # ------------------------------------------------------------- seeds
    seed: int = 42
    seeds: tuple = (42, 43, 44)                            # repeated independent runs (uncertainty analysis)
    ts_seeds: dict = field(default_factory=lambda: {"nsl_kdd": 1001, "unsw_nb15": 1002})
    deterministic_ops: bool = False

    # ------------------------------------------------------------- data / splits (Sections 4.1-4.3)
    val_fraction: float = 0.10        # last contiguous block of the training partition
    cic_test_fraction: float = 0.30   # last 30% of the chronologically ordered CICIDS-2018 subset
    cic_subsample: int = 6            # systematic chronological subsampling (every sixth row)
    seq_len: int = 10
    mean_interarrival: float = 0.1    # synthetic exponential inter-arrival (NSL-KDD, UNSW-NB15)
    unsw_shuffle_seed: int = 2024     # UNSW-NB15 partitions are ordered by class; permute once (None = keep file order)
    ts_resolution_s: dict = field(default_factory=lambda: {"nsl_kdd": None, "unsw_nb15": 1.0, "cicids2018": 1.0})

    # ------------------------------------------------------------- TA-LSTMAE (Section 4.3)
    latent_dim: dict = field(default_factory=lambda: {"nsl_kdd": 16, "unsw_nb15": 23, "cicids2018": 16})
    tae_epochs: int = 10
    hblt_epochs: dict = field(default_factory=lambda: {"nsl_kdd": 12, "unsw_nb15": 9, "cicids2018": 5})
    ablation_epochs: int = 8
    batch_size: int = 512
    lr: float = 1e-3
    dropout: float = 0.2
    threshold: float = 0.5

    # ------------------------------------------------------------- baselines (head-to-head, common protocol)
    baseline_max_epochs: int = 50
    baseline_patience: int = 5

    # ------------------------------------------------------------- temporal metrics engine (Section 3.4)
    match_tolerance: int = 5
    t_response_s: float = 30.0        # configurable latency for projected MTTR (NOT a measured value)

    def smoke(self) -> "Config":
        """Configuration for the code-execution smoke test (synthetic data, few epochs)."""
        return replace(self, smoke_test=True, out_dir=self.out_dir + "_SMOKE_TEST_INVALID", seeds=(42, 43), batch_size=128,
                       tae_epochs=2, ablation_epochs=1, hblt_epochs={k: 3 for k in self.hblt_epochs},
                       baseline_max_epochs=3, baseline_patience=2)

    @property
    def out(self) -> Path:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, name: str = "run_config.json") -> Path:
        path = self.out / name
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
