"""Healthy-state telemetry generation.

Three properties matter more than realism of the absolute levels, because
each one is something the detection layer will later have to cope with:

  * diurnal and weekly seasonality, so a seasonal-naive baseline is a
    genuinely competitive opponent;
  * autocorrelated noise, so residuals are not conveniently independent;
  * a heavy right tail on latency, so Gaussian tail probabilities are wrong
    and the extreme-value thresholds have something to earn their keep on.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .topology import Topology

METRICS = (
    "request_rate_rps",
    "latency_p95_ms",
    "error_rate",
    "cpu_pct",
    "mem_mb",
)

# Traffic peaks mid-afternoon and bottoms out in the small hours.
_PEAK_HOUR = 14.0
_DIURNAL_AMPLITUDE = 0.35
_WEEKEND_FACTOR = 0.72


@dataclass(frozen=True)
class GeneratorConfig:
    step_seconds: int = 15
    seed: int = 0
    ar_coefficient: float = 0.85
    noise_scale: float = 0.04
    latency_tail_sigma: float = 0.18
    mem_drift_scale: float = 0.002


def time_index(start: str | pd.Timestamp, hours: float, step_seconds: int) -> pd.DatetimeIndex:
    periods = int(round(hours * 3600 / step_seconds))
    return pd.date_range(start=start, periods=periods, freq=f"{step_seconds}s", tz="UTC")


def diurnal_factor(index: pd.DatetimeIndex) -> np.ndarray:
    """Traffic multiplier in roughly [0.45, 1.35], lower at weekends."""
    hours = index.hour + index.minute / 60 + index.second / 3600
    phase = 2 * np.pi * (hours - _PEAK_HOUR + 6.0) / 24.0
    daily = 1.0 + _DIURNAL_AMPLITUDE * np.sin(phase)
    weekly = np.where(index.dayofweek >= 5, _WEEKEND_FACTOR, 1.0)
    return np.asarray(daily * weekly, dtype=float)


def _rng_for(seed: int, *parts: str) -> np.random.Generator:
    """Stable per-series RNG.

    Derived from a checksum rather than hash() because hash() is salted per
    interpreter run, which would make runs irreproducible.
    """
    key = zlib.crc32("|".join(parts).encode("utf-8"))
    return np.random.default_rng([seed, key])


def ar1_noise(n: int, rho: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) series with the given marginal standard deviation."""
    innovation_scale = scale * np.sqrt(1.0 - rho**2)
    noise = np.empty(n, dtype=float)
    noise[0] = rng.normal(0.0, scale)
    shocks = rng.normal(0.0, innovation_scale, size=n)
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + shocks[i]
    return noise


def _random_walk(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    return np.cumsum(rng.normal(0.0, scale, size=n))


def generate_healthy(
    topology: Topology,
    index: pd.DatetimeIndex,
    config: GeneratorConfig | None = None,
) -> pd.DataFrame:
    """Long-format telemetry for every service and metric, fault-free.

    Columns: timestamp, service, metric, value.
    """
    config = config or GeneratorConfig()
    n = len(index)
    load = diurnal_factor(index)
    frames = []

    for service in topology:
        def noise(metric: str) -> np.ndarray:
            rng = _rng_for(config.seed, service.name, metric)
            return ar1_noise(n, config.ar_coefficient, config.noise_scale, rng)

        rps = service.base_rps * load * (1.0 + noise("request_rate_rps"))

        # Latency rises gently with load, then gets a lognormal kick so the
        # right tail is fat. Queueing does not produce Gaussian latency.
        tail_rng = _rng_for(config.seed, service.name, "latency_tail")
        tail = np.exp(tail_rng.normal(0.0, config.latency_tail_sigma, size=n))
        tail /= np.exp(config.latency_tail_sigma**2 / 2)  # keep the mean at 1
        latency = service.base_latency_ms * (1.0 + 0.25 * (load - 1.0)) * tail
        latency *= 1.0 + noise("latency_p95_ms")

        cpu = service.base_cpu_pct * (0.4 + 0.6 * load) * (1.0 + noise("cpu_pct"))

        mem_rng = _rng_for(config.seed, service.name, "mem_walk")
        mem = service.base_mem_mb * (
            1.0 + _random_walk(n, config.mem_drift_scale, mem_rng) + noise("mem_mb")
        )

        err_rng = _rng_for(config.seed, service.name, "error_tail")
        errors = service.base_error_rate * np.exp(err_rng.normal(0.0, 0.4, size=n))

        values = {
            "request_rate_rps": np.maximum(rps, 0.0),
            "latency_p95_ms": np.maximum(latency, 0.1),
            "error_rate": np.clip(errors, 0.0, 1.0),
            "cpu_pct": np.clip(cpu, 0.0, 100.0),
            "mem_mb": np.maximum(mem, 1.0),
        }
        for metric, series in values.items():
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": index,
                        "service": service.name,
                        "metric": metric,
                        "value": series,
                    }
                )
            )

    return pd.concat(frames, ignore_index=True)
