"""Fault taxonomy and the metric signatures each fault leaves behind.

Every fault has an origin service and a shape over time. The shapes are
deliberately different from one another: a latency injection is a step, a
memory leak is a slow ramp, a deploy regression never ends. Telling those
apart is the job the changepoint layer will inherit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class FaultKind(str, Enum):
    CPU_SATURATION = "cpu_saturation"
    MEMORY_LEAK = "memory_leak"
    LATENCY_INJECTION = "latency_injection"
    ERROR_SPIKE = "error_spike"
    DEPENDENCY_FAILURE = "dependency_failure"
    DEPLOY_REGRESSION = "deploy_regression"

    @property
    def is_permanent(self) -> bool:
        """A regression shipped in a deploy is a new normal, not a blip."""
        return self is FaultKind.DEPLOY_REGRESSION


@dataclass(frozen=True)
class Fault:
    kind: FaultKind
    service: str
    start: pd.Timestamp
    duration: pd.Timedelta
    magnitude: float = 0.6  # 0 = imperceptible, 1 = severe

    def __post_init__(self) -> None:
        if not 0.0 < self.magnitude <= 1.0:
            raise ValueError("magnitude must fall in (0, 1]")
        if self.duration <= pd.Timedelta(0):
            raise ValueError("duration must be positive")

    @property
    def end(self) -> pd.Timestamp:
        return self.start + self.duration


def _linear_ramp(n: int, fraction: float) -> np.ndarray:
    """Rise from 0 to 1 over the first `fraction` of the window, then hold."""
    if n == 0:
        return np.zeros(0)
    ramp_steps = max(1, int(round(n * fraction)))
    ramp = np.minimum(np.arange(n, dtype=float) / ramp_steps, 1.0)
    return ramp


def _decay(n: int, half_life_fraction: float) -> np.ndarray:
    if n == 0:
        return np.zeros(0)
    half_life = max(1.0, n * half_life_fraction)
    return 0.5 ** (np.arange(n, dtype=float) / half_life)


def origin_profile(kind: FaultKind, n: int, magnitude: float) -> dict[str, np.ndarray]:
    """Multiplicative effect on each metric of the *origin* service.

    Returns multipliers of length `n`; metrics not named are unaffected.
    """
    m = magnitude
    ones = np.ones(n)

    if kind is FaultKind.CPU_SATURATION:
        # Saturates quickly, then pins. Latency follows the queue, not the CPU.
        shape = _linear_ramp(n, 0.08)
        return {
            "cpu_pct": ones + 2.6 * m * shape,
            "latency_p95_ms": ones + 2.0 * m * shape,
            "error_rate": ones + 4.0 * m * np.clip(shape - 0.6, 0, None) / 0.4,
        }

    if kind is FaultKind.MEMORY_LEAK:
        # Grows across the whole window; latency only suffers late, under GC.
        shape = _linear_ramp(n, 1.0)
        return {
            "mem_mb": ones + 1.4 * m * shape,
            "latency_p95_ms": ones + 0.7 * m * shape**3,
            "cpu_pct": ones + 0.4 * m * shape**2,
        }

    if kind is FaultKind.LATENCY_INJECTION:
        shape = _linear_ramp(n, 0.02)
        return {
            "latency_p95_ms": ones + 4.5 * m * shape,
            "error_rate": ones + 1.5 * m * shape,
        }

    if kind is FaultKind.ERROR_SPIKE:
        # Sharp onset, partial self-recovery as retries succeed.
        shape = _linear_ramp(n, 0.02) * (0.45 + 0.55 * _decay(n, 0.35))
        return {
            "error_rate": ones + 70.0 * m * shape,
            "latency_p95_ms": ones + 0.5 * m * shape,
        }

    if kind is FaultKind.DEPENDENCY_FAILURE:
        shape = _linear_ramp(n, 0.03)
        return {
            "error_rate": ones + 90.0 * m * shape,
            "latency_p95_ms": ones + 2.5 * m * shape,
            # Callers give up, so the service itself sees less traffic.
            "request_rate_rps": ones - 0.55 * m * shape,
        }

    if kind is FaultKind.DEPLOY_REGRESSION:
        # A modest, permanent step - the class most likely to be mistaken
        # for a genuine incident.
        shape = _linear_ramp(n, 0.01)
        return {
            "latency_p95_ms": ones + 0.85 * m * shape,
            "cpu_pct": ones + 0.55 * m * shape,
        }

    raise ValueError(f"unhandled fault kind: {kind}")
