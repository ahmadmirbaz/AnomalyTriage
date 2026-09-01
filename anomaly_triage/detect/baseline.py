"""Baseline detectors.

These exist to be beaten, but they have to be beaten honestly. A seasonal
naive forecast on strongly periodic metrics is a genuinely hard opponent,
and quietly using a weak baseline is the most common way a detection result
gets overstated.

`RollingSigma` is not really a forecaster at all - it is the per-metric
three-sigma rule the project argues against, kept here so its alert volume
can be measured rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SECONDS_PER_DAY = 86_400


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Long rows -> a timestamp-indexed frame with (service, metric) columns."""
    return long.pivot_table(
        index="timestamp", columns=["service", "metric"], values="value"
    ).sort_index()


def step_seconds(wide: pd.DataFrame) -> float:
    if len(wide.index) < 2:
        raise ValueError("need at least two timestamps to infer the step")
    return float(pd.Series(wide.index).diff().dropna().median().total_seconds())


@dataclass
class SeasonalNaive:
    """Predict a point from the same time of day, averaged over past days.

    The median across previous periods rather than the single most recent one,
    so yesterday's incident does not become today's forecast.
    """

    period_steps: int
    lookback_periods: int = 3

    def predict(self, wide: pd.DataFrame) -> pd.DataFrame:
        lags = [
            wide.shift(self.period_steps * k)
            for k in range(1, self.lookback_periods + 1)
        ]
        stacked = pd.concat(lags, keys=range(len(lags)))
        return stacked.groupby(level=1).median()


@dataclass
class EWMA:
    """Exponentially weighted mean of the recent past."""

    halflife_steps: float = 20.0

    def predict(self, wide: pd.DataFrame) -> pd.DataFrame:
        # shift first: a forecast may not see the point it is forecasting
        return wide.shift(1).ewm(halflife=self.halflife_steps, min_periods=5).mean()


@dataclass
class RollingSigma:
    """The per-metric k-sigma rule, as commonly deployed.

    Alerts when a point sits more than `k` rolling standard deviations from
    the rolling mean. No seasonality, no multiplicity correction - which is
    the point.
    """

    window_steps: int = 240
    k: float = 3.0

    def alerts(self, wide: pd.DataFrame) -> pd.DataFrame:
        rolling = wide.shift(1).rolling(self.window_steps, min_periods=self.window_steps // 4)
        mean = rolling.mean()
        sigma = rolling.std()
        deviation = (wide - mean).abs()
        # A flat series has zero sigma; without this every rounding wobble
        # becomes an alert.
        return (deviation > self.k * sigma) & (sigma > 0)


def residuals(wide: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    return wide - predicted


def alerts_per_series_per_day(alerts: pd.DataFrame, step: float) -> float:
    """Mean alerts raised per series per day."""
    usable = alerts.notna().to_numpy().sum()
    if usable == 0:
        return 0.0
    fired = int(alerts.fillna(False).to_numpy().sum())
    days_of_series = usable * step / SECONDS_PER_DAY
    return fired / days_of_series


def projected_daily_alerts(alerts: pd.DataFrame, step: float, series: int) -> float:
    """Extrapolate an observed alert rate to a fleet of `series` metrics."""
    return alerts_per_series_per_day(alerts, step) * series
