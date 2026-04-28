"""TrajectoryDataset: yields (prefix_tokens, prefix_mask, target_tokens, target_mask)
samples from one or more parquet corpora.

Per the Phase 0c amendment, the prefix-length sampling strategy matters more
than it sounds. We support three strategies:
  - "uniform"  — sample uniformly from [1, T-1].
  - "long_first" — early in training, sample from the upper half [T/2, T-1];
                   anneal toward [1, T-1] as a curriculum.
  - "weighted" — sample uniformly but weight loss by 1/(T - prefix_length)
                 to up-weight harder (shorter prefix) cases. Implemented as a
                 sample weight returned alongside the sample.

A trajectory of length T frames yields up to T-1 (prefix, target) pairs:
  prefix = frames[:k] for k in [1, T-1], target = frames[-1].
We sample one (prefix_len, traj) pair per `__getitem__` call so the corpus
size scales linearly with #trajectories rather than #(traj, prefix) pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .trajectory import read_trajectories, unpack_frames


@dataclass
class DatasetConfig:
    paths: list[str]
    prefix_strategy: str = "uniform"   # "uniform" | "long_first" | "weighted"
    long_first_anneal_steps: int = 2000  # steps to fully anneal to uniform
    max_prefix_frames: int = 32  # truncate longer prefixes
    max_objects: int = 128
    seed: int = 0


class TrajectoryDataset(Dataset):
    def __init__(self, cfg: DatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._step = 0
        self.rng = np.random.default_rng(cfg.seed)

        # Load all rows (we keep them in memory; corpus is ~3 MB so trivial).
        self.rows: list[dict] = []
        for p in cfg.paths:
            self.rows.extend(read_trajectories(Path(p)))
        # Pre-unpack token arrays for speed (avoids reparsing each __getitem__).
        # We only keep trajectories with at least 2 frames (need a target distinct
        # from the prefix).
        self._tokens: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._meta: list[tuple[str, str]] = []  # (env_marker, run_id)
        for r in self.rows:
            tokens, mask = unpack_frames(r)
            T = tokens.shape[0]
            if T < 2:
                continue
            self._tokens.append(tokens.astype(np.float32))
            self._masks.append(mask.astype(np.float32))
            self._meta.append((r["env_marker"], r["run_id"]))

    def __len__(self) -> int:
        return len(self._tokens)

    def set_step(self, step: int) -> None:
        """Curriculum: training scripts call this once per step."""
        self._step = step

    def _sample_prefix_len(self, T: int) -> int:
        """Pick a prefix length in [1, T-1] per the strategy."""
        if T <= 2:
            return 1
        s = self.cfg.prefix_strategy
        if s == "uniform":
            return int(self.rng.integers(1, T))
        if s == "long_first":
            # Linear anneal: at step 0, sample from [T//2, T-1].
            # At step >= long_first_anneal_steps, sample from [1, T-1].
            frac = min(1.0, self._step / max(1, self.cfg.long_first_anneal_steps))
            lo = int(round((1 - frac) * (T // 2) + frac * 1))
            lo = max(1, lo)
            return int(self.rng.integers(lo, T))
        if s == "weighted":
            # Uniform sample; loss weight applied at training time.
            return int(self.rng.integers(1, T))
        raise ValueError(f"unknown prefix_strategy: {s}")

    def __getitem__(self, idx: int) -> dict:
        tokens = self._tokens[idx]
        mask = self._masks[idx]
        T = tokens.shape[0]

        prefix_len = self._sample_prefix_len(T)
        prefix_len = min(prefix_len, self.cfg.max_prefix_frames)
        prefix_tokens = tokens[:prefix_len]
        prefix_mask = mask[:prefix_len]

        target_tokens = tokens[T - 1]   # always the terminal frame
        target_mask = mask[T - 1]

        # Loss weight (only used for "weighted" strategy)
        if self.cfg.prefix_strategy == "weighted":
            distance = (T - 1) - (prefix_len - 1)  # frames between prefix end and terminal
            weight = 1.0 / max(distance, 1)
        else:
            weight = 1.0

        return {
            "prefix_tokens": torch.from_numpy(prefix_tokens),  # (prefix_len, max_objects, 13)
            "prefix_mask": torch.from_numpy(prefix_mask),       # (prefix_len, max_objects)
            "target_tokens": torch.from_numpy(target_tokens),   # (max_objects, 13)
            "target_mask": torch.from_numpy(target_mask),       # (max_objects,)
            "loss_weight": torch.tensor(weight, dtype=torch.float32),
            "env_marker": self._meta[idx][0],
            "T": T,
            "prefix_len": prefix_len,
        }


def collate_fn(batch: list[dict], max_prefix_frames: int = 32,
                fixed_pad: bool = True) -> dict:
    """Pad prefixes to a common length within the batch.

    `fixed_pad=True`: pad to `max_prefix_frames` for every batch. Enables MPS
    kernel reuse across batches (variable-shape recompiles every step). Wastes
    some compute per batch but is much faster overall on Apple Silicon.

    `fixed_pad=False`: pad only to the max length within the batch.
    """
    B = len(batch)
    if fixed_pad:
        max_prefix = max_prefix_frames
    else:
        max_prefix = max(b["prefix_tokens"].shape[0] for b in batch)
        max_prefix = min(max_prefix, max_prefix_frames)
    M = batch[0]["target_tokens"].shape[0]
    F = batch[0]["target_tokens"].shape[1]

    prefix_tokens = torch.zeros(B, max_prefix, M, F, dtype=torch.float32)
    prefix_mask = torch.zeros(B, max_prefix, M, dtype=torch.float32)
    target_tokens = torch.zeros(B, M, F, dtype=torch.float32)
    target_mask = torch.zeros(B, M, dtype=torch.float32)
    loss_weights = torch.zeros(B, dtype=torch.float32)
    env_markers: list[str] = []
    Ts: list[int] = []
    prefix_lens: list[int] = []

    for i, b in enumerate(batch):
        L = min(b["prefix_tokens"].shape[0], max_prefix)
        prefix_tokens[i, :L] = b["prefix_tokens"][:L]
        prefix_mask[i, :L] = b["prefix_mask"][:L]
        target_tokens[i] = b["target_tokens"]
        target_mask[i] = b["target_mask"]
        loss_weights[i] = b["loss_weight"]
        env_markers.append(b["env_marker"])
        Ts.append(b["T"])
        prefix_lens.append(b["prefix_len"])

    return {
        "prefix_tokens": prefix_tokens,
        "prefix_mask": prefix_mask,
        "target_tokens": target_tokens,
        "target_mask": target_mask,
        "loss_weights": loss_weights,
        "env_markers": env_markers,
        "T": Ts,
        "prefix_len": prefix_lens,
    }
