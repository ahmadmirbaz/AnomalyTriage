"""Drive a labelled run against the containerised mesh.

    python -m anomaly_triage.mesh.orchestrate --minutes 45 --out data/mesh-01

Draws a schedule with the same code the simulator uses, injects each fault
over HTTP at its appointed wall-clock moment, then exports the window from
Prometheus in the simulator's schema and labels it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ..sim.schedule import ScheduleConfig, schedule_faults
from .client import MeshClient, MeshUnavailable, mesh_topology
from .export import fetch, label

LOADGEN = Path(__file__).resolve().parents[2] / "testbed" / "loadgen.py"


def affected_services(topology, root: str) -> list[str]:
    """Origin plus every transitive caller.

    Mesh calls are synchronous and blocking, so a slow callee really does
    delay everyone above it - unlike the simulator, there is no attenuation
    threshold below which a caller escapes.
    """
    return sorted({root, *topology.upstream_of(root)})


def run(
    minutes: float,
    out: Path,
    seed: int,
    faults_per_hour: float,
    warmup_minutes: float,
    duration_scale: float,
    rps: float,
    day_minutes: float,
    step_seconds: int,
    prometheus: str,
) -> int:
    client = MeshClient()
    if not client.ready():
        print("mesh is not up - run: docker compose -f testbed/docker-compose.yml up -d")
        return 1

    topology = mesh_topology()
    start = pd.Timestamp.now(tz="UTC").ceil("1s")
    end = start + pd.Timedelta(minutes=minutes)
    index = pd.date_range(start, end, freq=f"{step_seconds}s")

    schedule = ScheduleConfig(
        seed=seed,
        faults_per_day=faults_per_hour * 24,
        warmup_hours=warmup_minutes / 60,
        min_gap_minutes=max(1.0, minutes / 20),
        duration_scale=duration_scale,
    )
    faults = schedule_faults(topology, index, schedule)
    if not faults:
        print("schedule is empty - try a longer run or a higher fault rate")
        return 1

    print(f"{len(faults)} faults over {minutes:.0f} min; warm-up {warmup_minutes:.0f} min")
    client.heal()

    loadgen = subprocess.Popen(
        [sys.executable, str(LOADGEN), "--rps", str(rps),
         "--day-minutes", str(day_minutes), "--seconds", str(minutes * 60)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    records = []
    try:
        for n, fault in enumerate(faults):
            delay = (fault.start - pd.Timestamp.now(tz="UTC")).total_seconds()
            if delay > 0:
                time.sleep(delay)

            seconds = None if fault.kind.is_permanent else fault.duration.total_seconds()
            try:
                client.inject(fault.service, fault.kind.value, fault.magnitude, seconds)
            except MeshUnavailable as exc:
                print(f"  skipped {fault.kind.value} on {fault.service}: {exc}")
                continue

            fired_at = pd.Timestamp.now(tz="UTC")
            incident_end = end if fault.kind.is_permanent else fired_at + fault.duration
            records.append({
                "incident_id": f"INC-{n:04d}",
                "kind": fault.kind.value,
                "root_service": fault.service,
                "start": fired_at,
                "end": min(incident_end, end),
                "magnitude": round(fault.magnitude, 4),
                "permanent": fault.kind.is_permanent,
                "affected_services": affected_services(topology, fault.service),
            })
            print(f"  [{fired_at:%H:%M:%S}] {fault.kind.value} -> {fault.service} "
                  f"(m={fault.magnitude:.2f}, {seconds or 'permanent'})")

        remaining = (end - pd.Timestamp.now(tz="UTC")).total_seconds()
        if remaining > 0:
            time.sleep(remaining)
    except KeyboardInterrupt:
        print("interrupted; healing and exporting what we have")
        end = pd.Timestamp.now(tz="UTC")
    finally:
        client.heal()
        loadgen.terminate()

    # Let the last scrape land before asking for the window.
    time.sleep(step_seconds + 5)

    incidents = pd.DataFrame(records)
    telemetry = fetch(start, end, step_seconds, prometheus)
    if telemetry.empty:
        print("Prometheus returned nothing for the window")
        return 1
    telemetry = label(telemetry, incidents)

    out.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(out / "metrics.csv.gz", index=False, compression="gzip")
    flat = incidents.copy()
    if not flat.empty:
        flat["affected_services"] = flat["affected_services"].apply(",".join)
    flat.to_csv(out / "incidents.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "source": "mesh",
        "start": str(start), "end": str(end),
        "minutes": minutes, "step_seconds": step_seconds, "seed": seed,
        "services": len(topology), "rows": len(telemetry),
        "incidents": len(incidents),
        "anomalous_fraction": round(float(telemetry["is_anomalous"].mean()), 6),
        "rps": rps, "day_minutes": day_minutes,
    }, indent=2) + "\n")

    print(f"\nwrote {len(telemetry):,} rows -> {out}")
    print(f"{len(incidents)} incidents, {telemetry['is_anomalous'].mean():.2%} of cells anomalous")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a labelled experiment on the mesh.")
    parser.add_argument("--minutes", type=float, default=45.0)
    parser.add_argument("--out", type=Path, default=Path("data/mesh"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--faults-per-hour", type=float, default=6.0)
    parser.add_argument("--warmup-minutes", type=float, default=5.0)
    parser.add_argument("--duration-scale", type=float, default=0.06)
    parser.add_argument("--rps", type=float, default=25.0)
    parser.add_argument("--day-minutes", type=float, default=20.0)
    parser.add_argument("--step-seconds", type=int, default=15)
    parser.add_argument("--prometheus", default="http://localhost:9090")
    args = parser.parse_args(argv)
    return run(**vars(args))


if __name__ == "__main__":
    raise SystemExit(main())
