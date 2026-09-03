import numpy as np
import pandas as pd
import pytest

from anomaly_triage.detect.features import (
    calendar_features,
    feature_names,
    series_features,
    steps_per_day,
)

STEP = 15


def make_series(n: int, values=None) -> pd.Series:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq=f"{STEP}s")
    if values is None:
        values = np.arange(n, dtype=float)
    return pd.Series(values, index=index)


def test_steps_per_day_matches_the_simulator_cadence():
    assert steps_per_day(STEP) == 5760


def test_no_feature_can_see_its_own_target():
    """Perturb one point; only strictly later rows may change.

    This is the property the whole module exists to guarantee, and it is the
    one that silently breaks when a `.shift(1)` gets dropped in a refactor.
    """
    n = 8000
    clean = make_series(n, np.random.default_rng(0).normal(size=n))
    poisoned = clean.copy()
    target = 6000
    poisoned.iloc[target] += 1_000_000.0

    before = series_features(clean, STEP)
    after = series_features(poisoned, STEP)

    changed = (before != after) & ~(before.isna() & after.isna())
    rows_changed = changed.any(axis=1).to_numpy().nonzero()[0]
    assert rows_changed.min() > target


def test_lags_are_the_values_they_claim_to_be():
    features = series_features(make_series(500), STEP)
    assert features["lag_1"].iloc[100] == 99.0
    assert features["lag_16"].iloc[100] == 84.0
    # delta_1 is the last observed step, not the step into the target
    assert features["delta_1"].iloc[100] == 1.0


def test_daily_lag_reaches_exactly_one_day_back():
    day = steps_per_day(STEP)
    features = series_features(make_series(day + 100), STEP)
    at = day + 50
    assert features["lag_day_1"].iloc[at] == float(at - day)
    assert np.isnan(features["lag_day_2"].iloc[at])


def test_calendar_terms_wrap_smoothly_across_midnight():
    index = pd.DatetimeIndex(
        ["2026-01-01T23:59:45Z", "2026-01-02T00:00:00Z", "2026-01-02T12:00:00Z"]
    )
    calendar = calendar_features(index)
    across_midnight = np.hypot(
        calendar["day_sin"].iloc[1] - calendar["day_sin"].iloc[0],
        calendar["day_cos"].iloc[1] - calendar["day_cos"].iloc[0],
    )
    half_a_day = np.hypot(
        calendar["day_sin"].iloc[2] - calendar["day_sin"].iloc[1],
        calendar["day_cos"].iloc[2] - calendar["day_cos"].iloc[1],
    )
    # A 15s step is 2*pi*15/86400 radians, so the chord is that, to first
    # order. The point is that midnight is not a discontinuity.
    assert across_midnight == pytest.approx(2 * np.pi * STEP / 86_400, abs=1e-6)
    assert half_a_day == pytest.approx(2.0, abs=1e-6)


def test_feature_names_matches_a_real_build():
    assert feature_names(STEP) == list(series_features(make_series(50), STEP).columns)


def test_a_non_timestamp_index_is_rejected():
    with pytest.raises(TypeError):
        series_features(pd.Series([1.0, 2.0, 3.0]), STEP)
