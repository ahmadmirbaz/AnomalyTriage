"""Apply faults to healthy telemetry and emit ground-truth labels.

Two modelling choices here carry weight downstream:

  * Only latency and error rate propagate to callers. A caller does not leak
    memory because its callee did, so CPU and memory stay local to the
    origin. That asymmetry is the strongest localisation signal in the data.
  * Propagation is lagged by a hop. The cascade therefore has an ordering,
    which is what onset-time tie-breaking will later key off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .faults import Fault, origin_profile
from .topology import Topology

# Metrics that a caller inherits from a struggling callee.
PROPAGATING_METRICS = ("latency_p95_ms", "error_rate")

_BOUNDS = {"error_rate": (0.0, 1.0), "cpu_pct": (0.0, 100.0)}


@dataclass(frozen=True)
class InjectionConfig:
    attenuation: float = 0.55       # effect retained per hop upstream
    hop_lag_steps: int = 4          # propagation delay per hop
    effect_threshold: float = 0.05  # below this deviation, not called anomalous


def _shift_effect(multiplier: np.ndarray, steps: int) -> np.ndarray:
    """Delay an effect by `steps`, holding 1.0 (no effect) at the front."""
    if steps <= 0:
        return multiplier
    shifted = np.ones_like(multiplier)
    shifted[steps:] = multiplier[:-steps]
    return shifted


def _window_mask(fault: Fault, index: pd.DatetimeIndex) -> np.ndarray:
    if fault.kind.is_permanent:
        return np.asarray(index >= fault.start)
    return np.asarray((index >= fault.start) & (index < fault.end))


def _full_length_profile(
    fault: Fault, index: pd.DatetimeIndex
) -> dict[str, np.ndarray]:
    """Origin multipliers spanning the whole index, 1.0 outside the window."""
    mask = _window_mask(fault, index)
    n = int(mask.sum())
    profile = origin_profile(fault.kind, n, fault.magnitude)
    full: dict[str, np.ndarray] = {}
    for metric, values in profile.items():
        series = np.ones(len(index))
        series[mask] = values
        full[metric] = series
    return full


def spread(
    fault: Fault,
    topology: Topology,
    index: pd.DatetimeIndex,
    config: InjectionConfig,
) -> dict[tuple[str, str], np.ndarray]:
    """Multipliers for every (service, metric) this fault touches."""
    origin = _full_length_profile(fault, index)
    effects: dict[tuple[str, str], np.ndarray] = {
        (fault.service, metric): values for metric, values in origin.items()
    }

    for caller, hops in topology.upstream_of(fault.service).items():
        retained = config.attenuation**hops
        for metric in PROPAGATING_METRICS:
            if metric not in origin:
                continue
            delayed = _shift_effect(origin[metric], hops * config.hop_lag_steps)
            inherited = 1.0 + (delayed - 1.0) * retained
            if np.abs(inherited - 1.0).max() <= config.effect_threshold:
                continue  # too faint to count as affected
            key = (caller, metric)
            # Concurrent faults compound rather than overwrite.
            effects[key] = effects.get(key, np.ones(len(index))) * inherited
    return effects


def apply_faults(
    healthy: pd.DataFrame,
    topology: Topology,
    faults: Sequence[Fault],
    config: InjectionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (labelled telemetry, incident ground truth).

    The telemetry frame gains `is_anomalous` and `incident_id` columns.
    """
    config = config or InjectionConfig()
    wide = healthy.pivot_table(
        index="timestamp", columns=["service", "metric"], values="value"
    )
    index = wide.index
    values = wide.to_numpy(dtype=float, copy=True)
    columns = list(wide.columns)
    position = {col: i for i, col in enumerate(columns)}

    labels = np.zeros(values.shape, dtype=bool)
    incident_ids = np.full(values.shape, "", dtype=object)
    records = []

    for n, fault in enumerate(faults):
        incident_id = f"INC-{n:04d}"
        effects = spread(fault, topology, index, config)
        touched = set()

        for key, multiplier in effects.items():
            if key not in position:
                continue  # metric this simulator does not emit
            col = position[key]
            values[:, col] *= multiplier
            active = np.abs(multiplier - 1.0) > config.effect_threshold
            labels[:, col] |= active
            # First fault to claim a cell keeps it, so overlaps stay traceable.
            unclaimed = active & (incident_ids[:, col] == "")
            incident_ids[unclaimed, col] = incident_id
            if active.any():
                touched.add(key[0])

        end = index[-1] if fault.kind.is_permanent else fault.end
        records.append(
            {
                "incident_id": incident_id,
                "kind": fault.kind.value,
                "root_service": fault.service,
                "start": fault.start,
                "end": end,
                "magnitude": fault.magnitude,
                "permanent": fault.kind.is_permanent,
                "affected_services": sorted(touched),
            }
        )

    for (service, metric), col in position.items():
        low, high = _BOUNDS.get(metric, (0.0, None))
        values[:, col] = np.clip(values[:, col], low, high)

    faulted = pd.DataFrame(values, index=index, columns=wide.columns)
    long = faulted.stack(["service", "metric"], future_stack=True).rename("value").reset_index()
    flags = pd.DataFrame(labels, index=index, columns=wide.columns)
    ids = pd.DataFrame(incident_ids, index=index, columns=wide.columns)
    long["is_anomalous"] = flags.stack(["service", "metric"], future_stack=True).to_numpy()
    long["incident_id"] = ids.stack(["service", "metric"], future_stack=True).to_numpy()

    return long, pd.DataFrame(records)


def summarise(incidents: pd.DataFrame) -> str:
    if incidents.empty:
        return "no incidents"
    by_kind = incidents["kind"].value_counts().to_dict()
    fanout = incidents["affected_services"].apply(len).mean()
    return f"{len(incidents)} incidents, mean fan-out {fanout:.1f} services, {by_kind}"
