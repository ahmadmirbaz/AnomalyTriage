import pandas as pd
import pytest

from anomaly_triage.sim.faults import Fault, FaultKind
from anomaly_triage.sim.inject import (
    InjectionConfig,
    apply_faults,
    spread,
)
from anomaly_triage.sim.metrics import generate_healthy, time_index
from anomaly_triage.sim.topology import default_topology


@pytest.fixture
def index():
    return time_index("2026-08-24", hours=4, step_seconds=15)


@pytest.fixture
def healthy(index):
    return generate_healthy(default_topology(), index)


def _fault(kind, service, index, at=200, minutes=30, magnitude=0.8):
    return Fault(kind, service, index[at], pd.Timedelta(minutes=minutes), magnitude)


def test_magnitude_must_be_in_range(index):
    with pytest.raises(ValueError, match="magnitude"):
        Fault(FaultKind.ERROR_SPIKE, "cart", index[0], pd.Timedelta("5min"), 1.5)


def test_duration_must_be_positive(index):
    with pytest.raises(ValueError, match="duration"):
        Fault(FaultKind.ERROR_SPIKE, "cart", index[0], pd.Timedelta(0))


def test_cpu_and_memory_do_not_propagate(index):
    topo = default_topology()
    fault = _fault(FaultKind.MEMORY_LEAK, "postgres", index)
    effects = spread(fault, topo, index, InjectionConfig())
    touched = {(svc, metric) for svc, metric in effects}
    assert ("postgres", "mem_mb") in touched
    assert not any(metric == "mem_mb" for svc, metric in touched if svc != "postgres")


def test_latency_propagates_to_callers(index):
    topo = default_topology()
    fault = _fault(FaultKind.LATENCY_INJECTION, "postgres", index)
    effects = spread(fault, topo, index, InjectionConfig())
    assert ("product-catalog", "latency_p95_ms") in effects
    assert ("frontend", "latency_p95_ms") in effects


def test_effect_attenuates_with_distance(index):
    topo = default_topology()
    fault = _fault(FaultKind.LATENCY_INJECTION, "postgres", index)
    effects = spread(fault, topo, index, InjectionConfig())
    origin = effects[("postgres", "latency_p95_ms")].max()
    one_hop = effects[("product-catalog", "latency_p95_ms")].max()
    two_hop = effects[("frontend", "latency_p95_ms")].max()
    assert origin > one_hop > two_hop > 1.0


def test_propagation_is_lagged_so_onset_order_recovers_the_root(index, healthy):
    topo = default_topology()
    fault = _fault(FaultKind.LATENCY_INJECTION, "postgres", index)
    df, _ = apply_faults(healthy, topo, [fault])
    onsets = (
        df[df.is_anomalous & (df.metric == "latency_p95_ms")]
        .groupby("service")["timestamp"]
        .min()
        .sort_values()
    )
    assert onsets.index[0] == "postgres"
    assert onsets["product-catalog"] < onsets["frontend"]


def test_labels_are_confined_to_the_fault_window(index, healthy):
    topo = default_topology()
    fault = _fault(FaultKind.ERROR_SPIKE, "ad", index, at=200, minutes=20)
    df, _ = apply_faults(healthy, topo, [fault])
    flagged = df[df.is_anomalous]
    assert flagged["timestamp"].min() >= fault.start
    # allow the propagation lag to trail past the nominal end
    assert flagged["timestamp"].max() <= fault.end + pd.Timedelta(minutes=5)


def test_healthy_run_has_no_labels(healthy):
    df, incidents = apply_faults(healthy, default_topology(), [])
    assert not df["is_anomalous"].any()
    assert incidents.empty


def test_injection_leaves_values_within_bounds(index, healthy):
    topo = default_topology()
    faults = [
        _fault(FaultKind.ERROR_SPIKE, "payment", index, magnitude=1.0),
        _fault(FaultKind.CPU_SATURATION, "cart", index, at=300, magnitude=1.0),
    ]
    df, _ = apply_faults(healthy, topo, faults)
    assert df[df.metric == "error_rate"]["value"].max() <= 1.0
    assert df[df.metric == "cpu_pct"]["value"].max() <= 100.0
    assert df["value"].min() >= 0.0


def test_deploy_regression_never_ends(index, healthy):
    topo = default_topology()
    fault = _fault(FaultKind.DEPLOY_REGRESSION, "currency", index, at=100, minutes=10)
    df, incidents = apply_faults(healthy, topo, [fault])
    assert bool(incidents.loc[0, "permanent"])
    assert incidents.loc[0, "end"] == index[-1]
    tail = df[(df.service == "currency") & (df.metric == "latency_p95_ms")]
    assert bool(tail.iloc[-1]["is_anomalous"])


def test_incident_record_lists_the_root_and_its_blast_radius(index, healthy):
    topo = default_topology()
    fault = _fault(FaultKind.LATENCY_INJECTION, "postgres", index)
    _, incidents = apply_faults(healthy, topo, [fault])
    row = incidents.iloc[0]
    assert row["root_service"] == "postgres"
    assert "postgres" in row["affected_services"]
    assert "frontend" in row["affected_services"]
    assert "redis" not in row["affected_services"]  # unrelated branch


def test_faults_do_not_perturb_untouched_services(index, healthy):
    topo = default_topology()
    fault = _fault(FaultKind.MEMORY_LEAK, "redis", index)
    df, _ = apply_faults(healthy, topo, [fault])
    key = ["timestamp", "service", "metric"]
    before = healthy.set_index(key)["value"].sort_index()
    after = df.set_index(key)["value"].sort_index()
    pd.testing.assert_series_equal(
        after.xs("shipping", level="service"),
        before.xs("shipping", level="service"),
    )
