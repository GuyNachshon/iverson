"""Per-feature discretization for the terminal-state predictor.

Following AlphaFold 2's distogram head and Farebrother et al. 2024
("Stop Regressing"): regress-as-classification with soft (HL-Gauss) targets
for continuous features. Combined head outputs B logits per feature; we use
either argmax or expected value for decoding.

Feature buckets (chosen to match the corpus distribution we audited):

  Index  Feature              Strategy                Bins
  -----  -------------------  ----------------------  -----
   0     color_id             native categorical      256 (cross-env vocab)
   1     color_rank           native categorical       32
   2     log_size             HL-Gauss continuous      32
   3-6   bbox (x_min..y_max)  uniform [0,1]            32 each
   7-8   centroid (cx,cy)     uniform [0,1]            32 each
   9     aspect               HL-Gauss continuous      32
   10    is_singleton         binary (BCE)              -
   11    touches_edge         binary (BCE)              -
   12    log_touches_others   HL-Gauss continuous      16

Bins for log_size / aspect / log_touches_others use the train-set range
+ small margin. Edges stored in the discretizer for round-trip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# Feature indices in the 13-dim object token vector.
F_COLOR_ID = 0
F_COLOR_RANK = 1
F_LOG_SIZE = 2
F_BBOX_XMIN = 3
F_BBOX_YMIN = 4
F_BBOX_XMAX = 5
F_BBOX_YMAX = 6
F_CX = 7
F_CY = 8
F_ASPECT = 9
F_IS_SINGLETON = 10
F_TOUCHES_EDGE = 11
F_LOG_NEIGHBORS = 12

# Categorical sizes. color_id pulled from corpus max=160, round up.
N_COLOR_ID = 256
N_COLOR_RANK = 32

# Continuous-feature bin counts.
N_BINS_LOG_SIZE = 32
N_BINS_BBOX = 32       # for each of 4 bbox coords
N_BINS_CENTROID = 32   # for each of cx, cy
N_BINS_ASPECT = 32
N_BINS_NEIGHBORS = 16

# Per-feature bin specs. (start, end, n_bins, kind) where kind is:
#   "uniform"  — even bins over [start, end]
#   "log_size" — uniform over log_size's empirical range
#   "aspect"   — uniform over aspect range
#   "neighbor" — uniform over log_touches_others range
@dataclass
class BinSpec:
    feature_idx: int
    start: float
    end: float
    n_bins: int
    name: str

    def edges(self) -> np.ndarray:
        return np.linspace(self.start, self.end, self.n_bins + 1)

    def centers(self) -> np.ndarray:
        e = self.edges()
        return (e[:-1] + e[1:]) / 2

    def width(self) -> float:
        return (self.end - self.start) / self.n_bins


# Bin specs for the 6 continuous features we discretize.
# Ranges chosen from corpus stats (with light margin):
#   log_size: 0.69 - 5.07 (corpus min/max)
#   bbox / cx / cy: 0.0 - 1.0 (normalized)
#   aspect: 0.10 - 11.00 (corpus max)
#   log_touches_others: 0.69 - 3.69
CONTINUOUS_BIN_SPECS: list[BinSpec] = [
    BinSpec(F_LOG_SIZE, 0.0, 5.5, N_BINS_LOG_SIZE, "log_size"),
    BinSpec(F_BBOX_XMIN, 0.0, 1.0, N_BINS_BBOX, "bbox_xmin"),
    BinSpec(F_BBOX_YMIN, 0.0, 1.0, N_BINS_BBOX, "bbox_ymin"),
    BinSpec(F_BBOX_XMAX, 0.0, 1.0, N_BINS_BBOX, "bbox_xmax"),
    BinSpec(F_BBOX_YMAX, 0.0, 1.0, N_BINS_BBOX, "bbox_ymax"),
    BinSpec(F_CX, 0.0, 1.0, N_BINS_CENTROID, "cx"),
    BinSpec(F_CY, 0.0, 1.0, N_BINS_CENTROID, "cy"),
    BinSpec(F_ASPECT, 0.0, 12.0, N_BINS_ASPECT, "aspect"),
    BinSpec(F_LOG_NEIGHBORS, 0.0, 4.0, N_BINS_NEIGHBORS, "log_touches_others"),
]

CATEGORICAL_FEATURES = {
    F_COLOR_ID: N_COLOR_ID,
    F_COLOR_RANK: N_COLOR_RANK,
}

BINARY_FEATURES = [F_IS_SINGLETON, F_TOUCHES_EDGE]


def hl_gauss_target(values: torch.Tensor, n_bins: int, start: float, end: float,
                    sigma: float = 0.75) -> torch.Tensor:
    """HL-Gauss soft target.

    For each value, the target is a discretized Gaussian centered on the value
    with std `sigma * bin_width`. Provides a smooth gradient toward neighboring
    bins so MSE-like behavior is preserved while the head stays a classifier.

    values: (...,) tensor of continuous values.
    Returns: (..., n_bins) probability vector (sums to 1 over the last axis).
    """
    width = (end - start) / n_bins
    centers = torch.linspace(start + width / 2, end - width / 2, n_bins,
                              device=values.device, dtype=values.dtype)
    # centers shape: (n_bins,). values: (..., 1).
    dist = (values.unsqueeze(-1) - centers) / (sigma * width)
    # Use cumulative-Gaussian-difference for proper bin probabilities.
    # erf approximation:
    sqrt2 = math.sqrt(2.0)
    edges_l = (values.unsqueeze(-1) - (centers - width / 2)) / (sigma * width * sqrt2)
    edges_r = (values.unsqueeze(-1) - (centers + width / 2)) / (sigma * width * sqrt2)
    # Probability mass between edges_r and edges_l (cumulative Gaussian).
    target = 0.5 * (torch.erf(edges_l) - torch.erf(edges_r))
    target = target.clamp_min(0.0)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    return target


def hard_bin_target(values: torch.Tensor, n_bins: int, start: float, end: float
                     ) -> torch.Tensor:
    """Hard bin index for each value. Out-of-range values clamp to first/last bin."""
    width = (end - start) / n_bins
    idx = ((values - start) / width).long()
    idx = idx.clamp(0, n_bins - 1)
    return idx


def expected_value_decode(logits: torch.Tensor, start: float, end: float
                           ) -> torch.Tensor:
    """Decode bin-logits to a continuous value via expectation over softmax.

    logits: (..., n_bins). Returns (...,) tensor of decoded values.
    """
    n_bins = logits.shape[-1]
    width = (end - start) / n_bins
    centers = torch.linspace(start + width / 2, end - width / 2, n_bins,
                              device=logits.device, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=-1)
    return (probs * centers).sum(-1)
