"""A single instrumented service.

One image, many containers: topology, latency and error levels all arrive
through the environment, so the compose file alone describes the fleet.

Faults here are real rather than modelled. Asking for cpu_saturation spins
worker threads; asking for a memory_leak actually allocates. The CPU and
memory gauges then report what the process is genuinely doing, which is the
whole reason for running containers instead of staying in the simulator.
"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass

import psutil
import requests
from flask import Flask, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

SERVICE = os.environ.get("SERVICE_NAME", "unnamed")
DEPENDS_ON = [d for d in os.environ.get("DEPENDS_ON", "").split(",") if d]
BASE_LATENCY_MS = float(os.environ.get("BASE_LATENCY_MS", "25"))
BASE_ERROR_RATE = float(os.environ.get("BASE_ERROR_RATE", "0.002"))
DEP_TIMEOUT_S = float(os.environ.get("DEP_TIMEOUT_S", "2.0"))
PORT = int(os.environ.get("PORT", "8000"))

app = Flask(__name__)
_process = psutil.Process()

REQUESTS = Counter(
    "service_requests_total", "Requests handled", ["service", "status"]
)
LATENCY = Histogram(
    "service_request_duration_seconds",
    "End-to-end handler duration",
    ["service"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
CPU = Gauge("service_cpu_percent", "Process CPU utilisation", ["service"])
MEMORY = Gauge("service_memory_mb", "Process resident set size", ["service"])
INFLIGHT = Gauge("service_inflight_requests", "Concurrent handlers", ["service"])
FAULT_ACTIVE = Gauge("service_fault_active", "1 while a fault runs", ["service", "kind"])


@dataclass
class Fault:
    kind: str
    magnitude: float
    expires_at: float  # monotonic; math.inf for a permanent regression

    @property
    def live(self) -> bool:
        return time.monotonic() < self.expires_at


class FaultRegistry:
    """Tracks active faults and owns the threads that make them real."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._faults: dict[str, Fault] = {}
        self._ballast: list[bytearray] = []
        threading.Thread(target=self._burn_cpu, daemon=True).start()
        threading.Thread(target=self._grow_heap, daemon=True).start()

    def add(self, kind: str, magnitude: float, seconds: float | None) -> Fault:
        expires = float("inf") if seconds is None else time.monotonic() + seconds
        fault = Fault(kind, magnitude, expires)
        with self._lock:
            self._faults[kind] = fault
        FAULT_ACTIVE.labels(SERVICE, kind).set(1)
        return fault

    def clear(self, kind: str | None = None) -> None:
        with self._lock:
            kinds = [kind] if kind else list(self._faults)
            for k in kinds:
                self._faults.pop(k, None)
                FAULT_ACTIVE.labels(SERVICE, k).set(0)
        if kind is None:
            self._ballast.clear()

    def get(self, kind: str) -> Fault | None:
        with self._lock:
            fault = self._faults.get(kind)
            if fault and not fault.live:
                del self._faults[kind]
                FAULT_ACTIVE.labels(SERVICE, kind).set(0)
                if kind == "memory_leak":
                    self._ballast.clear()
                return None
            return fault

    def active(self) -> dict[str, float]:
        return {k: f.magnitude for k in list(self._faults) if (f := self.get(k))}

    def _burn_cpu(self) -> None:
        """Duty-cycle a busy loop in proportion to the fault magnitude."""
        while True:
            fault = self.get("cpu_saturation")
            if not fault:
                time.sleep(0.2)
                continue
            duty = min(0.95, fault.magnitude)
            deadline = time.monotonic() + 0.05 * duty
            while time.monotonic() < deadline:
                pass
            time.sleep(0.05 * (1 - duty))

    def _grow_heap(self) -> None:
        while True:
            fault = self.get("memory_leak")
            if not fault:
                time.sleep(0.5)
                continue
            # Up to ~4 MB/s at full magnitude.
            self._ballast.append(bytearray(int(1_000_000 * fault.magnitude * 4)))
            time.sleep(1.0)


faults = FaultRegistry()


def _sample_resources() -> None:
    _process.cpu_percent()  # first call primes the interval
    while True:
        CPU.labels(SERVICE).set(_process.cpu_percent())
        MEMORY.labels(SERVICE).set(_process.memory_info().rss / 1_000_000)
        time.sleep(2.0)


def _added_latency_seconds() -> float:
    extra = 0.0
    if fault := faults.get("latency_injection"):
        extra += 0.4 * fault.magnitude
    if fault := faults.get("deploy_regression"):
        extra += 0.08 * fault.magnitude
    return extra


def _should_error() -> bool:
    rate = BASE_ERROR_RATE
    if fault := faults.get("error_spike"):
        rate += 0.6 * fault.magnitude
    return random.random() < rate


@app.get("/work")
def work():
    started = time.perf_counter()
    INFLIGHT.labels(SERVICE).inc()
    try:
        if faults.get("dependency_failure"):
            REQUESTS.labels(SERVICE, "503").inc()
            return jsonify(service=SERVICE, error="dependency unavailable"), 503

        if _should_error():
            REQUESTS.labels(SERVICE, "500").inc()
            return jsonify(service=SERVICE, error="internal"), 500

        downstream = []
        for dep in DEPENDS_ON:
            try:
                response = requests.get(
                    f"http://{dep}:8000/work", timeout=DEP_TIMEOUT_S
                )
                downstream.append({dep: response.status_code})
                if response.status_code >= 500:
                    REQUESTS.labels(SERVICE, "502").inc()
                    return jsonify(service=SERVICE, downstream=downstream), 502
            except requests.RequestException:
                downstream.append({dep: "timeout"})
                REQUESTS.labels(SERVICE, "504").inc()
                return jsonify(service=SERVICE, downstream=downstream), 504

        # Own work: a jittered base cost plus whatever a fault has added.
        time.sleep(BASE_LATENCY_MS / 1000 * random.uniform(0.7, 1.4))
        time.sleep(_added_latency_seconds())

        REQUESTS.labels(SERVICE, "200").inc()
        return jsonify(service=SERVICE, downstream=downstream)
    finally:
        LATENCY.labels(SERVICE).observe(time.perf_counter() - started)
        INFLIGHT.labels(SERVICE).dec()


@app.post("/fault")
def inject():
    payload = request.get_json(force=True, silent=True) or {}
    kind = payload.get("kind")
    if kind not in {
        "cpu_saturation",
        "memory_leak",
        "latency_injection",
        "error_spike",
        "dependency_failure",
        "deploy_regression",
    }:
        return jsonify(error=f"unknown fault kind: {kind!r}"), 400

    magnitude = float(payload.get("magnitude", 0.6))
    if not 0.0 < magnitude <= 1.0:
        return jsonify(error="magnitude must fall in (0, 1]"), 400

    seconds = payload.get("seconds")
    if kind == "deploy_regression":
        seconds = None  # permanent by definition
    faults.add(kind, magnitude, None if seconds is None else float(seconds))
    return jsonify(service=SERVICE, kind=kind, magnitude=magnitude, seconds=seconds)


@app.delete("/fault")
def heal():
    faults.clear(request.args.get("kind"))
    return jsonify(service=SERVICE, active=faults.active())


@app.get("/health")
def health():
    return jsonify(service=SERVICE, depends_on=DEPENDS_ON, faults=faults.active())


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


threading.Thread(target=_sample_resources, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
