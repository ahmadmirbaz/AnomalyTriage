import numpy as np
import pandas as pd
import pytest

from anomaly_triage.detect.baseline import (
    EWMA,
    RollingSigma,
    SeasonalNaive,
    alerts_per_series_per_day,
    projected_daily_alerts,
    step_seconds,
    to_wide,
)


def wide_series(values, step="15s"):
    index = pd.date_range("2026-09-01T00:00:00Z", periods=len(values), freq=step)
    columns = pd.MultiIndex.from_tuples([("cart", "cpu_pct")], names=["service", "metric"])
    return pd.DataFrame(np.asarray(values, dtype=float).reshape(-1, 1), index=index, columns=columns)


def test_to_wide_pivots_long_rows():
    long = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-09-01T00:00:00Z"] * 2 + ["2026-09-01T00:00:15Z"] * 2),
        "service": ["cart", "redis"] * 2,
        "metric": ["cpu_pct"] * 4,
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    wide = to_wide(long)
    assert wide.shape == (2, 2)
    assert wide.iloc[0][("cart", "cpu_pct")] == 1.0


def test_step_seconds_infers_the_cadence():
    assert step_seconds(wide_series(range(10), step="60s")) == 60.0


def test_step_seconds_needs_two_points():
    with pytest.raises(ValueError, match="two timestamps"):
        step_seconds(wide_series([1.0]))


def test_seasonal_naive_recovers_a_perfect_cycle():
    period = 24
    cycle = list(np.sin(np.linspace(0, 2 * np.pi, period, endpoint=False)) * 10 + 50)
    frame = wide_series(cycle * 5)
    predicted = SeasonalNaive(period_steps=period, lookback_periods=3).predict(frame)
    # after three full periods of warm-up the forecast should be exact
    tail = slice(period * 3, None)
    np.testing.assert_allclose(
        predicted.iloc[tail].to_numpy(), frame.iloc[tail].to_numpy(), atol=1e-9
    )


def test_seasonal_naive_ignores_a_single_contaminated_period():
    period = 24
    values = ([50.0] * period) * 5
    values[period * 2 + 5] = 5_000.0  # one bad day at that time of day
    frame = wide_series(values)
    predicted = SeasonalNaive(period_steps=period, lookback_periods=3).predict(frame)
    # the median across three lookbacks discards the outlier
    assert predicted.iloc[period * 4 + 5].item() == pytest.approx(50.0)


def test_forecasters_do_not_peek_at_the_present():
    frame = wide_series([1.0] * 40 + [900.0])
    for model in (EWMA(halflife_steps=5), SeasonalNaive(period_steps=10)):
        predicted = model.predict(frame)
        assert predicted.iloc[-1].item() < 100.0, model


def test_rolling_sigma_flags_an_obvious_spike():
    rng = np.random.default_rng(0)
    values = list(rng.normal(50, 1.0, 400)) + [200.0]
    alerts = RollingSigma(window_steps=100, k=3.0).alerts(wide_series(values))
    assert bool(alerts.iloc[-1].item())


def test_rolling_sigma_stays_quiet_on_a_flat_series():
    alerts = RollingSigma(window_steps=50, k=3.0).alerts(wide_series([7.0] * 300))
    assert not alerts.to_numpy().any()


def test_rolling_sigma_rate_is_in_the_right_ballpark_on_gaussian_noise():
    rng = np.random.default_rng(3)
    frame = wide_series(rng.normal(0, 1, 20_000), step="60s")
    alerts = RollingSigma(window_steps=240, k=3.0).alerts(frame)
    rate = alerts.fillna(False).to_numpy().mean()
    # independent Gaussian noise should sit near the textbook 0.27%
    assert 0.001 < rate < 0.008


def test_alerts_per_series_per_day_counts_only_evaluated_points():
    index = pd.date_range("2026-09-01T00:00:00Z", periods=2880, freq="60s")  # two days
    columns = pd.MultiIndex.from_tuples([("a", "m"), ("b", "m")], names=["service", "metric"])
    alerts = pd.DataFrame(False, index=index, columns=columns)
    alerts.iloc[:10, 0] = True  # 10 alerts across 2 series x 2 days = 4 series-days
    assert alerts_per_series_per_day(alerts, 60) == pytest.approx(2.5)


def test_projection_scales_linearly_with_fleet_size():
    index = pd.date_range("2026-09-01T00:00:00Z", periods=1440, freq="60s")
    columns = pd.MultiIndex.from_tuples([("a", "m")], names=["service", "metric"])
    alerts = pd.DataFrame(False, index=index, columns=columns)
    alerts.iloc[:5, 0] = True
    assert projected_daily_alerts(alerts, 60, 1000) == pytest.approx(5000.0)
