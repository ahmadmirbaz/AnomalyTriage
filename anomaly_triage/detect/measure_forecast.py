"""Score the quantile forecaster against the baselines it has to beat.

    python -m anomaly_triage.detect.measure_forecast --data data/train-01

Splits a labelled run chronologically - never at random, because shuffling
time series leaks the future into the training set and turns any forecaster
into a genius. Fits on the fault-free part of the training half, then reports
three things on the held-out half:

    accuracy     median pinball loss, against seasonal-naive and EWMA
    calibration  coverage of the nominal 90% and 98% intervals on clean points
    honesty      how uniform the residual is, which is what phase 3 needs

Separation is reported too: the same interval evaluated on clean points and
on labelled-anomalous points. If those two numbers are close, the forecaster
is not distinguishing anything, however good its loss looks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .baseline import EWMA, SeasonalNaive, step_seconds, to_wide
from .evaluate import (
    coverage,
    ks_statistic,
    pinball_loss,
    pit_histogram,
    pit_values,
    tail_mass,
)
from .features import steps_per_day
from .model import QuantileForecaster, enforce_monotone


def load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (values, clean) as aligned wide frames."""
    long = pd.read_csv(path / "metrics.csv.gz", parse_dates=["timestamp"])
    wide = to_wide(long)
    anomalous = long.pivot_table(
        index="timestamp",
        columns=["service", "metric"],
        values="is_anomalous",
        aggfunc="first",
    ).sort_index()
    clean = ~anomalous.reindex(index=wide.index, columns=wide.columns).astype(bool)
    return wide, clean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train-01"))
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--services", nargs="*", default=None,
                        help="restrict to these services (faster iteration)")
    args = parser.parse_args(argv)

    wide, clean = load(args.data)
    if args.services:
        keep = [c for c in wide.columns if c[0] in set(args.services)]
        wide, clean = wide[keep], clean[keep]

    step = step_seconds(wide)
    split = int(len(wide) * args.train_fraction)
    train, test = wide.iloc[:split], wide.iloc[split:]
    train_clean, test_clean = clean.iloc[:split], clean.iloc[split:]

    print(f"{wide.shape[1]} series, {step:.0f}s step")
    print(f"train {train.index[0]:%Y-%m-%d %H:%M} .. {train.index[-1]:%Y-%m-%d %H:%M}"
          f"  ({len(train):,} steps)")
    print(f" test {test.index[0]:%Y-%m-%d %H:%M} .. {test.index[-1]:%Y-%m-%d %H:%M}"
          f"  ({len(test):,} steps, "
          f"{1 - test_clean.to_numpy().mean():.2%} anomalous)\n")

    started = time.time()
    forecaster = QuantileForecaster().fit(train, step, train_clean)
    predicted = enforce_monotone(forecaster.predict(test, step))
    print(f"fit and predicted in {time.time() - started:.0f}s\n")

    day = steps_per_day(step)
    baselines = {
        "seasonal-naive": SeasonalNaive(period_steps=day).predict(wide).loc[test.index],
        "ewma": EWMA().predict(wide).loc[test.index],
    }

    print("accuracy - median pinball loss, lower is better")
    print(f"  {'quantile forecaster':<22}{pinball_loss(test, predicted[0.5], 0.5):>12.4f}")
    for name, prediction in baselines.items():
        print(f"  {name:<22}{pinball_loss(test, prediction, 0.5):>12.4f}")

    print("\ncalibration - interval coverage, clean points only")
    for lo, hi in ((0.05, 0.95), (0.01, 0.99)):
        nominal = hi - lo
        masked = test.where(test_clean)
        got = coverage(masked, predicted[lo], predicted[hi])
        print(f"  nominal {nominal:>5.0%}    covered {got:>7.2%}")

    print("\nseparation - the same interval on anomalous points")
    for lo, hi in ((0.05, 0.95), (0.01, 0.99)):
        inside_clean = coverage(test.where(test_clean), predicted[lo], predicted[hi])
        inside_bad = coverage(test.where(~test_clean), predicted[lo], predicted[hi])
        print(f"  nominal {hi - lo:>5.0%}    clean {inside_clean:>7.2%}"
              f"    anomalous {inside_bad:>7.2%}")

    pit = pit_values(test.where(test_clean), predicted)
    below, above = tail_mass(pit)
    print("\nhonesty - residual uniformity on clean points")
    print(f"  KS distance from uniform  {ks_statistic(pit):>8.4f}")
    print(f"  pinned at 0 / at 1        {below:>8.3%} / {above:.3%}")
    print("\n" + pit_histogram(pit, bins=10).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nA ratio far from 1.000 in the outer bins is the tail the quantile")
    print("ladder cannot resolve. Phase 3 fits an extreme-value tail there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
