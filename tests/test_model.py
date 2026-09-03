import numpy as np
import pandas as pd
import pytest

from anomaly_triage.detect.model import QuantileForecaster, enforce_monotone

STEP = 60


def wide_frame(n: int, seed: int = 0) -> pd.DataFrame:
    """A seasonal series with noise, at a step coarse enough to fit quickly."""
    index = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq=f"{STEP}s")
    rng = np.random.default_rng(seed)
    clock = 2 * np.pi * np.arange(n) * STEP / 86_400
    values = 100 + 20 * np.sin(clock) + rng.normal(scale=3.0, size=n)
    columns = pd.MultiIndex.from_tuples([("api", "latency_p95_ms")],
                                        names=["service", "metric"])
    return pd.DataFrame(values.reshape(-1, 1), index=index, columns=columns)


def small_forecaster() -> QuantileForecaster:
    return QuantileForecaster(quantiles=(0.05, 0.5, 0.95), max_iter=30, max_depth=4)


def test_predicted_quantiles_land_in_order_after_repair():
    frame = wide_frame(3000)
    split = 2000
    fitted = small_forecaster().fit(frame.iloc[:split], STEP)
    predicted = enforce_monotone(fitted.predict(frame.iloc[split:], STEP))

    low = predicted[0.05].to_numpy()
    mid = predicted[0.5].to_numpy()
    high = predicted[0.95].to_numpy()
    assert (low <= mid).all()
    assert (mid <= high).all()


def test_enforce_monotone_fixes_crossed_quantiles():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="60s")
    crossed = {
        0.05: pd.DataFrame({"a": [5.0, 9.0, 1.0]}, index=index),
        0.5: pd.DataFrame({"a": [3.0, 8.0, 2.0]}, index=index),  # below the 5th
        0.95: pd.DataFrame({"a": [4.0, 7.0, 3.0]}, index=index),
    }
    fixed = enforce_monotone(crossed)
    assert list(fixed[0.05]["a"]) == [3.0, 7.0, 1.0]
    assert list(fixed[0.5]["a"]) == [4.0, 8.0, 2.0]
    assert list(fixed[0.95]["a"]) == [5.0, 9.0, 3.0]


def test_the_interval_brackets_most_of_a_clean_series():
    frame = wide_frame(3000, seed=3)
    split = 2000
    fitted = small_forecaster().fit(frame.iloc[:split], STEP)
    held_out = frame.iloc[split:]
    predicted = enforce_monotone(fitted.predict(held_out, STEP))

    inside = (held_out >= predicted[0.05]) & (held_out <= predicted[0.95])
    # Nominal 90%; a short fit on a small sample will not be exact.
    assert 0.75 < inside.to_numpy().mean() < 1.0


def test_training_through_an_incident_swallows_it():
    """Why `clean` exists: a model fit on its own faults expects them."""
    frame = wide_frame(4000, seed=5)
    column = frame.columns[0]
    # A sustained shift in the training half, at one time of day.
    faulty = frame.copy()
    fault = slice(1000, 1400)
    faulty.iloc[fault, 0] += 60.0

    clean = pd.DataFrame(True, index=faulty.index, columns=faulty.columns)
    clean.iloc[fault, 0] = False

    train = faulty.iloc[:2000]
    naive = small_forecaster().fit(train, STEP)
    careful = small_forecaster().fit(train, STEP, clean.iloc[:2000])

    at_fault = faulty.iloc[fault]
    lifted_naive = naive.predict(at_fault, STEP)[0.5][column].mean()
    lifted_careful = careful.predict(at_fault, STEP)[0.5][column].mean()
    # The model that saw the fault forecasts closer to it, so the residual it
    # would report is smaller - the incident partly disappears.
    assert lifted_naive > lifted_careful


def test_predict_before_fit_is_an_error():
    with pytest.raises(RuntimeError):
        QuantileForecaster().predict(wide_frame(10), STEP)


def test_too_little_clean_data_fails_loudly():
    frame = wide_frame(500)
    clean = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    clean.iloc[:50] = True
    with pytest.raises(ValueError, match="clean rows"):
        small_forecaster().fit(frame, STEP, clean)
