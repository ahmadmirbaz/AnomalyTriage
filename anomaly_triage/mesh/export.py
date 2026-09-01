"""Pull a time window out of Prometheus in the simulator's schema.

Everything downstream of phase 1 consumes long-format rows of
(timestamp, service, metric, value). Emitting the mesh's telemetry in that
same shape means the detector never learns which source it is reading, and
a result can be reproduced against either one.
"""

from __future__ import annotations

import json
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

DEFAULT_PROMETHEUS = "http://localhost:9090"

# Metric name -> PromQL. Names and units deliberately match metrics.METRICS.
QUERIES: dict[str, str] = {
    "request_rate_rps": "sum by (service) (rate(service_requests_total[1m]))",
    "error_rate": (
        'sum by (service) (rate(service_requests_total{status=~"5.."}[1m]))'
        " / clamp_min(sum by (service) (rate(service_requests_total[1m])), 1e-9)"
    ),
    "latency_p95_ms": (
        "1000 * histogram_quantile(0.95, sum by (le, service) "
        "(rate(service_request_duration_seconds_bucket[1m])))"
    ),
    "cpu_pct": "service_cpu_percent",
    "mem_mb": "service_memory_mb",
}


class PrometheusError(RuntimeError):
    pass


def query_range(
    query: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    step_seconds: int,
    base_url: str = DEFAULT_PROMETHEUS,
    timeout: float = 30.0,
) -> list[dict]:
    params = urlencode(
        {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": f"{step_seconds}s",
        }
    )
    with urlopen(f"{base_url}/api/v1/query_range?{params}", timeout=timeout) as response:
        payload = json.load(response)
    if payload.get("status") != "success":
        raise PrometheusError(payload.get("error", "query failed"))
    return payload["data"]["result"]


def fetch(
    start: pd.Timestamp,
    end: pd.Timestamp,
    step_seconds: int = 15,
    base_url: str = DEFAULT_PROMETHEUS,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Long-format telemetry for the window, one row per service/metric/step."""
    wanted = list(metrics) if metrics else list(QUERIES)
    frames = []

    for metric in wanted:
        for series in query_range(QUERIES[metric], start, end, step_seconds, base_url):
            service = series["metric"].get("service")
            if not service:
                continue
            pairs = series["values"]
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            [float(ts) for ts, _ in pairs], unit="s", utc=True
                        ),
                        "service": service,
                        "metric": metric,
                        "value": [float(v) for _, v in pairs],
                    }
                )
            )

    if not frames:
        return pd.DataFrame(columns=["timestamp", "service", "metric", "value"])

    frame = pd.concat(frames, ignore_index=True)
    # Prometheus reports NaN as the string "NaN"; a gap is genuinely missing
    # rather than zero, so leave the hole for the detector to notice.
    return frame.sort_values(["timestamp", "service", "metric"], ignore_index=True)


def label(
    telemetry: pd.DataFrame,
    incidents: pd.DataFrame,
    affected_column: str = "affected_services",
) -> pd.DataFrame:
    """Attach is_anomalous / incident_id using the orchestrator's ground truth."""
    labelled = telemetry.copy()
    labelled["is_anomalous"] = False
    labelled["incident_id"] = ""

    for _, incident in incidents.iterrows():
        affected = incident[affected_column]
        if isinstance(affected, str):
            affected = [s for s in affected.split(",") if s]
        window = (
            labelled["timestamp"].between(incident["start"], incident["end"])
            & labelled["service"].isin(affected)
        )
        unclaimed = window & (labelled["incident_id"] == "")
        labelled.loc[window, "is_anomalous"] = True
        labelled.loc[unclaimed, "incident_id"] = incident["incident_id"]

    return labelled
