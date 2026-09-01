import numpy as np
import pandas as pd
import pytest

from anomaly_triage.sim.metrics import (
    METRICS,
    GeneratorConfig,
    ar1_noise,
    diurnal_factor,
    generate_healthy,
    time_index,
)
from anomaly_triage.sim.topology import default_topology


def test_time_index_step_and_length():
    idx = time_index("2026-08-24", hours=2, step_seconds=15)
    assert len(idx) == 480
    assert (idx[1] - idx[0]).total_seconds() == 15


def test_diurnal_peaks_mid_afternoon_and_dips_overnight():
    idx = time_index("2026-08-24", hours=24, step_seconds=3600)  # a Monday
    factor = diurnal_factor(idx)
    assert idx[int(np.argmax(factor))].hour == 14
    assert idx[int(np.argmin(factor))].hour == 2


def test_weekends_carry_less_traffic():
    monday = diurnal_factor(time_index("2026-08-24T14:00", hours=1, step_seconds=3600))
    saturday = diurnal_factor(time_index("2026-08-29T14:00", hours=1, step_seconds=3600))
    assert saturday[0] < monday[0]


def test_ar1_noise_matches_requested_moments():
    noise = ar1_noise(200_000, rho=0.85, scale=0.04, rng=np.random.default_rng(7))
    assert noise.std() == pytest.approx(0.04, abs=0.002)
    lag1 = np.corrcoef(noise[:-1], noise[1:])[0, 1]
    assert lag1 == pytest.approx(0.85, abs=0.02)


def test_generate_healthy_covers_every_series():
    topo = default_topology()
    idx = time_index("2026-08-24", hours=1, step_seconds=15)
    df = generate_healthy(topo, idx)
    assert set(df["metric"].unique()) == set(METRICS)
    assert df["service"].nunique() == len(topo)
    assert len(df) == len(topo) * len(METRICS) * len(idx)


def test_generation_is_reproducible():
    topo = default_topology()
    idx = time_index("2026-08-24", hours=1, step_seconds=15)
    a = generate_healthy(topo, idx, GeneratorConfig(seed=3))
    b = generate_healthy(topo, idx, GeneratorConfig(seed=3))
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_diverge():
    topo = default_topology()
    idx = time_index("2026-08-24", hours=1, step_seconds=15)
    a = generate_healthy(topo, idx, GeneratorConfig(seed=1))
    b = generate_healthy(topo, idx, GeneratorConfig(seed=2))
    assert not np.allclose(a["value"], b["value"])


def test_values_stay_in_physical_bounds():
    idx = time_index("2026-08-24", hours=6, step_seconds=15)
    df = generate_healthy(default_topology(), idx)
    by_metric = df.groupby("metric")["value"]
    assert by_metric.min().min() >= 0.0
    assert by_metric.max()["cpu_pct"] <= 100.0
    assert by_metric.max()["error_rate"] <= 1.0


def test_latency_tail_is_heavier_than_gaussian():
    idx = time_index("2026-08-24", hours=72, step_seconds=15)
    df = generate_healthy(default_topology(), idx)
    latency = df[(df.service == "ad") & (df.metric == "latency_p95_ms")]["value"]
    assert latency.skew() > 0.3
