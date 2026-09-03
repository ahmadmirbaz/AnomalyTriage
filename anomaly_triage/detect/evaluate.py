"""Scoring the forecaster, and checking it is honest enough to build on.

Three questions, in increasing order of how much they matter:

1. Is it accurate? Pinball loss at the median is comparable to the baselines'
   point forecasts, so the seasonal-naive opponent gets a fair fight.
2. Are its intervals the width they claim? A nominal 90% interval that covers
   97% of clean points is not conservative, it is wrong, and it will hide real
   incidents.
3. Is the residual uniform under the null? This is the one phase 3 depends on.
   Benjamini-Hochberg controls the false discovery rate only if p-values are
   uniform when nothing is broken. If the PIT is lumpy, every FDR guarantee
   downstream is decoration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pinball_loss(actual: pd.DataFrame, predicted: pd.DataFrame, quantile: float) -> float:
    """Mean pinball (quantile) loss, the proper score for a quantile forecast."""
    error = (actual - predicted).to_numpy(dtype=float)
    usable = np.isfinite(error)
    if not usable.any():
        return float("nan")
    error = error[usable]
    loss = np.where(error >= 0, quantile * error, (quantile - 1.0) * error)
    return float(loss.mean())


def coverage(actual: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame) -> float:
    """Fraction of points inside the interval, over points where all three exist."""
    a = actual.to_numpy(dtype=float)
    lo = lower.to_numpy(dtype=float)
    hi = upper.to_numpy(dtype=float)
    usable = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    if not usable.any():
        return float("nan")
    return float(((a >= lo) & (a <= hi))[usable].mean())


def pit_values(actual: pd.DataFrame, frames: dict[float, pd.DataFrame]) -> np.ndarray:
    """Where each point falls in its own predicted distribution.

    Interpolating the actual against the predicted quantile ladder gives the
    probability integral transform. Under a correct forecast these are uniform
    on [0, 1], which is exactly the input Benjamini-Hochberg wants.

    Points beyond the outermost predicted quantile clamp to 0 or 1. Those atoms
    are not a bug to hide - they mark the region the ladder cannot resolve, and
    sizing them is how we know how much work the extreme-value tail in phase 3
    has to do.
    """
    levels = np.array(sorted(frames), dtype=float)
    ladder = np.stack([frames[q].to_numpy(dtype=float) for q in sorted(frames)])
    ladder = np.sort(ladder, axis=0)
    values = actual.to_numpy(dtype=float)

    usable = np.isfinite(values) & np.isfinite(ladder).all(axis=0)
    flat_values = values[usable]
    flat_ladder = ladder[:, usable]

    out = np.empty(flat_values.shape, dtype=float)
    for i in range(flat_values.size):
        out[i] = np.interp(flat_values[i], flat_ladder[:, i], levels, left=0.0, right=1.0)
    return out


def ks_statistic(pit: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance from uniform. 0 is perfect."""
    if pit.size == 0:
        return float("nan")
    ordered = np.sort(pit)
    n = ordered.size
    upper = np.arange(1, n + 1) / n - ordered
    lower = ordered - np.arange(0, n) / n
    return float(max(upper.max(), lower.max()))


def tail_mass(pit: np.ndarray) -> tuple[float, float]:
    """Fraction of points pinned at each end of the ladder."""
    if pit.size == 0:
        return float("nan"), float("nan")
    return float((pit <= 0.0).mean()), float((pit >= 1.0).mean())


def pit_histogram(pit: np.ndarray, bins: int = 20) -> pd.DataFrame:
    """Observed vs expected mass per bin, for eyeballing non-uniformity."""
    counts, edges = np.histogram(pit, bins=bins, range=(0.0, 1.0))
    expected = pit.size / bins
    return pd.DataFrame(
        {
            "from": edges[:-1],
            "to": edges[1:],
            "observed": counts,
            "ratio": counts / expected if expected else np.nan,
        }
    )
