"""Features for the quantile forecaster.

Every feature here has to be computable at the moment of the forecast. The
easiest way to get a spectacular-looking detector is to build a feature from
the point being predicted, so the rule in this module is blunt: anything
derived from the series itself is shifted by at least one step before it is
used, and only calendar terms - which are known arbitrarily far ahead - are
read at the target timestamp.

Features are built one series at a time. Services differ by an order of
magnitude in traffic and latency, and a shared feature matrix would force the
model to learn that spread instead of the shape of each series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# In steps. At the simulator's 15s cadence these span 15s to 4 minutes, which
# is where most of the short-horizon autocorrelation lives.
SHORT_LAGS = (1, 2, 4, 8, 16)

# Rolling summaries: a quarter hour and an hour at 15s.
ROLL_WINDOWS = (60, 240)

SECONDS_PER_DAY = 86_400
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY


def steps_per_day(step_seconds: float) -> int:
    return int(round(SECONDS_PER_DAY / step_seconds))


def _cyclical(seconds: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Encode a clock position as a point on a circle.

    Midnight and 23:59 are one step apart in time but as far apart as possible
    on a raw hour-of-day feature; a tree would need a split at every wrap.
    """
    angle = 2.0 * np.pi * (seconds % period) / period
    return np.sin(angle), np.cos(angle)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    epoch_seconds = index.asi8 / 1e9
    day_sin, day_cos = _cyclical(epoch_seconds, SECONDS_PER_DAY)
    week_sin, week_cos = _cyclical(epoch_seconds, SECONDS_PER_WEEK)
    return pd.DataFrame(
        {
            "day_sin": day_sin,
            "day_cos": day_cos,
            "week_sin": week_sin,
            "week_cos": week_cos,
        },
        index=index,
    )


def series_features(series: pd.Series, step_seconds: float) -> pd.DataFrame:
    """Build the feature matrix for a single (service, metric) series."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must be indexed by timestamp")

    # One shift here, so no lag or window below can reach the target point.
    past = series.shift(1)
    day = steps_per_day(step_seconds)

    columns: dict[str, pd.Series] = {}
    for lag in SHORT_LAGS:
        columns[f"lag_{lag}"] = series.shift(lag)

    # Where the series was heading, not just where it was.
    columns["delta_1"] = series.shift(1) - series.shift(2)
    columns["delta_4"] = series.shift(1) - series.shift(5)

    for window in ROLL_WINDOWS:
        rolling = past.rolling(window, min_periods=window // 4)
        columns[f"mean_{window}"] = rolling.mean()
        columns[f"std_{window}"] = rolling.std()

    # Same clock time on previous days. This is the seasonal-naive signal,
    # handed to the model as a feature rather than used as a forecast.
    for back in (1, 2):
        columns[f"lag_day_{back}"] = series.shift(day * back)

    features = pd.DataFrame(columns, index=series.index)
    return features.join(calendar_features(series.index))


def feature_names(step_seconds: float) -> list[str]:
    """Column order produced by `series_features`, without building it."""
    index = pd.date_range("2026-01-01", periods=2, freq=f"{int(step_seconds)}s", tz="UTC")
    empty = pd.Series([0.0, 0.0], index=index)
    return list(series_features(empty, step_seconds).columns)
