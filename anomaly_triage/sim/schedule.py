"""Randomised fault schedules for long unattended runs.

The schedule reserves a fault-free warm-up at the head of every run. The
detection layer has to estimate its null distribution from data it believes
is clean, and if the run opens mid-incident that estimate is poisoned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .faults import Fault, FaultKind
from .topology import Topology

# Rough incidence weights: latency and error problems dominate real pager
# traffic; a bad deploy is rarer but far more expensive to misread.
_KIND_WEIGHTS: dict[FaultKind, float] = {
    FaultKind.LATENCY_INJECTION: 0.26,
    FaultKind.ERROR_SPIKE: 0.24,
    FaultKind.CPU_SATURATION: 0.18,
    FaultKind.DEPENDENCY_FAILURE: 0.16,
    FaultKind.MEMORY_LEAK: 0.10,
    FaultKind.DEPLOY_REGRESSION: 0.06,
}

# Minutes; memory leaks smoulder, error spikes are short and loud.
_DURATION_MINUTES: dict[FaultKind, tuple[float, float]] = {
    FaultKind.LATENCY_INJECTION: (8.0, 45.0),
    FaultKind.ERROR_SPIKE: (4.0, 20.0),
    FaultKind.CPU_SATURATION: (10.0, 60.0),
    FaultKind.DEPENDENCY_FAILURE: (5.0, 30.0),
    FaultKind.MEMORY_LEAK: (45.0, 180.0),
    FaultKind.DEPLOY_REGRESSION: (10.0, 10.0),  # permanent; duration unused
}


@dataclass(frozen=True)
class ScheduleConfig:
    seed: int = 0
    faults_per_day: float = 6.0
    warmup_hours: float = 6.0
    min_gap_minutes: float = 20.0
    magnitude_range: tuple[float, float] = (0.35, 1.0)
    max_permanent: int = 2
    kind_weights: dict[FaultKind, float] = field(default_factory=lambda: dict(_KIND_WEIGHTS))


def schedule_faults(
    topology: Topology,
    index: pd.DatetimeIndex,
    config: ScheduleConfig | None = None,
) -> list[Fault]:
    """Draw a fault schedule spanning `index`, honouring warm-up and spacing."""
    config = config or ScheduleConfig()
    rng = np.random.default_rng(config.seed)

    step_seconds = int((index[1] - index[0]).total_seconds()) if len(index) > 1 else 1
    span_hours = (index[-1] - index[0]).total_seconds() / 3600
    usable_hours = span_hours - config.warmup_hours
    if usable_hours <= 0:
        return []

    target = int(round(config.faults_per_day * usable_hours / 24))
    if target <= 0:
        return []

    kinds = list(config.kind_weights)
    weights = np.array([config.kind_weights[k] for k in kinds], dtype=float)
    weights /= weights.sum()
    services = topology.names

    earliest = index[0] + pd.Timedelta(hours=config.warmup_hours)
    min_gap = pd.Timedelta(minutes=config.min_gap_minutes)

    starts: list[pd.Timestamp] = []
    faults: list[Fault] = []
    permanent_used = 0
    # Sample generously, then keep the draws that respect the spacing rule.
    for _ in range(target * 12):
        if len(faults) >= target:
            break
        kind = kinds[int(rng.choice(len(kinds), p=weights))]
        if kind.is_permanent and permanent_used >= config.max_permanent:
            continue

        offset_hours = float(rng.uniform(0.0, usable_hours))
        start = earliest + pd.Timedelta(hours=offset_hours)
        snapped = min(int(index.searchsorted(start, side="left")), len(index) - 1)
        start = index[snapped]
        if any(abs(start - existing) < min_gap for existing in starts):
            continue

        low, high = _DURATION_MINUTES[kind]
        # Snap to the sample grid; a fault that ends between two scrapes is
        # not something the data could ever express.
        step = pd.Timedelta(seconds=step_seconds)
        raw = pd.Timedelta(minutes=float(rng.uniform(low, high)))
        duration = max(step, raw.round(step))
        if not kind.is_permanent and start + duration > index[-1]:
            continue

        magnitude = float(rng.uniform(*config.magnitude_range))
        faults.append(Fault(kind, str(rng.choice(services)), start, duration, magnitude))
        starts.append(start)
        if kind.is_permanent:
            permanent_used += 1

    return sorted(faults, key=lambda f: f.start)
