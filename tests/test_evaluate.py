import math

import numpy as np
import pandas as pd
import pytest

from anomaly_triage.detect.evaluate import (
    coverage,
    ks_statistic,
    pinball_loss,
    pit_values,
    tail_mass,
)

LEVELS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def normal_ppf(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def frame(values) -> pd.DataFrame:
    return pd.DataFrame({"x": values})


def test_pinball_at_the_median_is_half_the_absolute_error():
    actual = frame([1.0, 2.0, 3.0])
    predicted = frame([0.0, 2.0, 5.0])
    assert pinball_loss(actual, predicted, 0.5) == pytest.approx((1 + 0 + 2) / 3 / 2)


def test_pinball_penalises_the_side_the_quantile_cares_about():
    actual = frame([10.0])
    under = frame([0.0])   # forecast too low
    over = frame([20.0])   # forecast too high, same distance
    # A high quantile should be punished harder for landing below the truth.
    assert pinball_loss(actual, under, 0.95) > pinball_loss(actual, over, 0.95)
    assert pinball_loss(actual, under, 0.05) < pinball_loss(actual, over, 0.05)


def test_pinball_ignores_missing_points():
    actual = frame([1.0, np.nan, 3.0])
    predicted = frame([1.0, 99.0, 3.0])
    assert pinball_loss(actual, predicted, 0.5) == 0.0


def test_coverage_counts_only_fully_observed_points():
    actual = frame([1.0, 5.0, np.nan])
    lower = frame([0.0, 0.0, 0.0])
    upper = frame([2.0, 2.0, 2.0])
    assert coverage(actual, lower, upper) == pytest.approx(0.5)


def test_pit_is_uniform_when_the_forecast_is_right():
    """The property phase 3 stands on: correct forecast -> uniform residual."""
    n = 20_000
    y = np.random.default_rng(1).normal(size=n)
    frames = {q: frame([normal_ppf(q)] * n) for q in LEVELS}

    pit = pit_values(frame(y), frames)
    # A seven-rung ladder interpolated linearly cannot be exactly uniform; it
    # should be close, and far closer than a wrong forecast (below).
    assert ks_statistic(pit) < 0.05
    below, above = tail_mass(pit)
    assert below == pytest.approx(0.01, abs=0.005)
    assert above == pytest.approx(0.01, abs=0.005)


def test_pit_is_visibly_wrong_when_the_spread_is_wrong():
    n = 20_000
    y = np.random.default_rng(2).normal(size=n)
    # Intervals three times too wide: everything piles into the middle.
    frames = {q: frame([3.0 * normal_ppf(q)] * n) for q in LEVELS}

    pit = pit_values(frame(y), frames)
    assert ks_statistic(pit) > 0.15
    below, above = tail_mass(pit)
    assert below < 0.001 and above < 0.001


def test_ks_statistic_is_zero_for_a_perfect_uniform():
    grid = (np.arange(1, 1001) - 0.5) / 1000
    assert ks_statistic(grid) == pytest.approx(0.0005, abs=1e-9)


def test_pit_clamps_outside_the_ladder():
    frames = {q: frame([normal_ppf(q)]) for q in LEVELS}
    assert pit_values(frame([1e9]), frames)[0] == 1.0
    assert pit_values(frame([-1e9]), frames)[0] == 0.0
