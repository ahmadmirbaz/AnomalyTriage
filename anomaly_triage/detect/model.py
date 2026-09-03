"""Per-series quantile forecasting.

The detector needs a *distribution* for the next point, not a number. A point
forecast plus a global sigma is the assumption that fails hardest on latency,
where the spread is wide, skewed, and changes with load. Predicting several
quantiles directly makes the interval the model's own output, and a residual
scored against it is on its way to being a p-value that phase 3 can trust.

`HistGradientBoostingRegressor` is used rather than the classic gradient
booster for two reasons: it takes NaN features natively, which matters because
every lag column has a ragged head, and it fits 60 series in a minute rather
than an afternoon.

The model is fit only on rows known to be fault-free. A forecaster trained
through its own incidents learns to expect them, and the anomaly it was built
to surface flattens into the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from anomaly_triage.detect.features import series_features

# Two central quantiles for the forecast itself and four in the tails, because
# the tails are where detection happens.
DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


@dataclass
class QuantileForecaster:
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    max_iter: int = 200
    learning_rate: float = 0.06
    max_depth: int | None = 6
    min_samples_leaf: int = 40
    random_state: int = 0

    models_: dict[tuple, dict[float, HistGradientBoostingRegressor]] = field(
        default_factory=dict, init=False, repr=False
    )
    columns_: list = field(default_factory=list, init=False, repr=False)

    def _estimator(self, quantile: float) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            early_stopping=False,
        )

    def fit(
        self,
        wide: pd.DataFrame,
        step_seconds: float,
        clean: pd.DataFrame | None = None,
    ) -> "QuantileForecaster":
        """Fit one model per (series, quantile).

        `clean` is a boolean frame shaped like `wide`, True where the point is
        known fault-free. Passing None trains on everything, which is what a
        real deployment has to do and what the unlabelled-data experiments in
        phase 3 will need.
        """
        self.columns_ = list(wide.columns)
        self.models_ = {}

        for column in self.columns_:
            series = wide[column]
            features = series_features(series, step_seconds)

            usable = series.notna() & features.notna().any(axis=1)
            if clean is not None:
                usable &= clean[column].fillna(False)

            X = features.loc[usable].to_numpy(dtype=float)
            y = series.loc[usable].to_numpy(dtype=float)
            if len(y) < 100:
                raise ValueError(f"{column}: only {len(y)} clean rows to fit on")

            self.models_[column] = {
                q: self._estimator(q).fit(X, y) for q in self.quantiles
            }
        return self

    def predict(
        self, wide: pd.DataFrame, step_seconds: float
    ) -> dict[float, pd.DataFrame]:
        """Predict every quantile for every series, as one frame per quantile."""
        if not self.models_:
            raise RuntimeError("fit before predict")

        per_quantile: dict[float, dict] = {q: {} for q in self.quantiles}
        for column in self.columns_:
            features = series_features(wide[column], step_seconds)
            X = features.to_numpy(dtype=float)
            for quantile, model in self.models_[column].items():
                per_quantile[quantile][column] = pd.Series(
                    model.predict(X), index=wide.index
                )

        frames = {}
        for quantile, columns in per_quantile.items():
            frame = pd.DataFrame(columns)
            frame.columns = pd.MultiIndex.from_tuples(
                frame.columns, names=wide.columns.names
            )
            frames[quantile] = frame.reindex(columns=wide.columns)
        return frames


def enforce_monotone(frames: dict[float, pd.DataFrame]) -> dict[float, pd.DataFrame]:
    """Sort the predicted quantiles so they cannot cross.

    Each quantile is fit independently, so nothing stops the 95th from landing
    below the 75th in a thin region. Left alone this produces negative interval
    widths and, downstream, p-values outside [0, 1]. Sorting is the standard
    repair and cannot make the pinball loss worse.
    """
    levels = sorted(frames)
    stacked = np.stack([frames[q].to_numpy(dtype=float) for q in levels])
    stacked.sort(axis=0)
    return {
        q: pd.DataFrame(stacked[i], index=frames[q].index, columns=frames[q].columns)
        for i, q in enumerate(levels)
    }
