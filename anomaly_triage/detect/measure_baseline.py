"""Measure what per-metric threshold alerting actually costs.

    python -m anomaly_triage.detect.measure_baseline

Generates fault-free telemetry, runs the k-sigma rule over it, and reports
the alert volume. Every alert counted here is false by construction: there
is nothing wrong with the fleet.

The interesting part is the gap between the textbook rate and the measured
one. A two-sided three-sigma rule is supposed to fire on 0.27% of samples.
It does not, because that figure assumes independent Gaussian observations
and real telemetry is neither.
"""

from __future__ import annotations

import argparse

from ..sim.metrics import GeneratorConfig, generate_healthy, time_index
from ..sim.topology import default_topology
from .baseline import (
    RollingSigma,
    alerts_per_series_per_day,
    to_wide,
)

GAUSSIAN_TWO_SIDED = {2.0: 0.0455, 2.5: 0.0124, 3.0: 0.0027, 3.5: 0.000465}


def measure(
    days: float,
    step_seconds: int,
    window_hours: float,
    k: float,
    seed: int,
) -> dict:
    index = time_index("2026-08-24", hours=days * 24, step_seconds=step_seconds)
    healthy = generate_healthy(
        default_topology(), index, GeneratorConfig(step_seconds=step_seconds, seed=seed)
    )
    wide = to_wide(healthy)

    window_steps = int(window_hours * 3600 / step_seconds)
    alerts = RollingSigma(window_steps=window_steps, k=k).alerts(wide)

    per_series_day = alerts_per_series_per_day(alerts, step_seconds)
    samples_per_day = 86_400 / step_seconds
    observed_rate = per_series_day / samples_per_day

    return {
        "k": k,
        "step_seconds": step_seconds,
        "series": wide.shape[1],
        "days": days,
        "alerts_per_series_per_day": per_series_day,
        "observed_sample_rate": observed_rate,
        "gaussian_sample_rate": GAUSSIAN_TWO_SIDED.get(k),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=14.0)
    parser.add_argument("--step-seconds", type=int, default=60)
    parser.add_argument("--window-hours", type=float, default=4.0)
    parser.add_argument("--fleet", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5])
    args = parser.parse_args(argv)

    print(f"{args.days:.0f} fault-free days, {args.step_seconds}s scrapes, "
          f"{args.window_hours:.0f}h rolling window\n")
    print(f"{'k':>5}{'observed':>12}{'gaussian':>12}{'inflation':>11}"
          f"{'alerts/day, ' + str(args.fleet) + ' series':>30}")

    for k in args.k:
        result = measure(args.days, args.step_seconds, args.window_hours, k, args.seed)
        expected = result["gaussian_sample_rate"]
        inflation = result["observed_sample_rate"] / expected if expected else float("nan")
        fleet_alerts = result["alerts_per_series_per_day"] * args.fleet
        print(f"{k:>5.1f}{result['observed_sample_rate']:>11.3%}"
              f"{expected:>11.3%}{inflation:>10.1f}x{fleet_alerts:>30,.0f}")

    print("\nEvery one of these is a false alarm: the fleet is healthy throughout.")
    print("The inflation column is the cost of assuming independent Gaussian noise")
    print("when the data is autocorrelated and heavy-tailed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
