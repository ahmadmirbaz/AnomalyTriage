import pandas as pd
import pytest

from anomaly_triage.mesh import export


def telemetry(services=("postgres", "product-catalog", "payment")):
    index = pd.date_range("2026-09-01T00:00:00Z", periods=10, freq="15s")
    rows = [
        {"timestamp": ts, "service": svc, "metric": "latency_p95_ms", "value": 10.0}
        for ts in index
        for svc in services
    ]
    return pd.DataFrame(rows)


def incident(start_step, end_step, affected, incident_id="INC-0000"):
    index = pd.date_range("2026-09-01T00:00:00Z", periods=10, freq="15s")
    return {
        "incident_id": incident_id,
        "kind": "latency_injection",
        "root_service": affected[0],
        "start": index[start_step],
        "end": index[end_step],
        "affected_services": affected,
    }


def test_label_marks_only_affected_services_inside_the_window():
    data = telemetry()
    incidents = pd.DataFrame([incident(3, 6, ["postgres", "product-catalog"])])
    labelled = export.label(data, incidents)

    flagged = labelled[labelled.is_anomalous]
    assert set(flagged["service"]) == {"postgres", "product-catalog"}
    assert flagged["timestamp"].min() == data["timestamp"].unique()[3]
    assert flagged["timestamp"].max() == data["timestamp"].unique()[6]
    assert (flagged["incident_id"] == "INC-0000").all()


def test_label_accepts_a_comma_joined_string():
    data = telemetry()
    record = incident(2, 4, ["postgres"])
    record["affected_services"] = "postgres,product-catalog"
    labelled = export.label(data, pd.DataFrame([record]))
    assert set(labelled[labelled.is_anomalous]["service"]) == {"postgres", "product-catalog"}


def test_overlapping_incidents_keep_the_first_claim():
    data = telemetry()
    incidents = pd.DataFrame([
        incident(2, 6, ["postgres"], "INC-0000"),
        incident(4, 8, ["postgres"], "INC-0001"),
    ])
    labelled = export.label(data, incidents)
    postgres = labelled[(labelled.service == "postgres") & labelled.is_anomalous]
    # every cell the second incident also covers was already claimed
    assert set(postgres["incident_id"]) == {"INC-0000", "INC-0001"}
    overlap = postgres[postgres.timestamp.isin(data["timestamp"].unique()[4:7])]
    assert set(overlap["incident_id"]) == {"INC-0000"}


def test_no_incidents_leaves_everything_clean():
    labelled = export.label(telemetry(), pd.DataFrame())
    assert not labelled["is_anomalous"].any()


def test_fetch_shapes_prometheus_output(monkeypatch):
    def fake_query_range(query, start, end, step_seconds, base_url, timeout=30.0):
        return [
            {"metric": {"service": "cart"}, "values": [[1756684800, "12.5"], [1756684815, "13.0"]]},
            {"metric": {}, "values": [[1756684800, "99.0"]]},  # no service label; dropped
        ]

    monkeypatch.setattr(export, "query_range", fake_query_range)
    frame = export.fetch(
        pd.Timestamp("2026-09-01T00:00:00Z"),
        pd.Timestamp("2026-09-01T00:00:30Z"),
        metrics=["cpu_pct"],
    )
    assert list(frame.columns) == ["timestamp", "service", "metric", "value"]
    assert set(frame["service"]) == {"cart"}
    assert frame["value"].tolist() == [12.5, 13.0]


def test_fetch_returns_an_empty_frame_with_the_right_columns(monkeypatch):
    monkeypatch.setattr(export, "query_range", lambda *a, **k: [])
    frame = export.fetch(
        pd.Timestamp("2026-09-01T00:00:00Z"), pd.Timestamp("2026-09-01T00:01:00Z")
    )
    assert frame.empty
    assert list(frame.columns) == ["timestamp", "service", "metric", "value"]


def test_every_metric_has_a_query():
    from anomaly_triage.sim.metrics import METRICS

    assert set(export.QUERIES) == set(METRICS)
