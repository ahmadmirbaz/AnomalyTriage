"""Generate a labelled telemetry run.

    python -m anomaly_triage.sim.run --hours 168 --out data/week-01

Writes long-format metrics, the incident ground truth, and a manifest
recording every parameter needed to reproduce the run.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .. import __version__
from .inject import InjectionConfig, apply_faults, summarise
from .metrics import GeneratorConfig, generate_healthy, time_index
from .schedule import ScheduleConfig, schedule_faults
from .topology import default_topology


def generate_run(
    hours: float,
    start: str,
    step_seconds: int,
    seed: int,
    faults_per_day: float,
    warmup_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    topology = default_topology()
    index = time_index(start, hours=hours, step_seconds=step_seconds)

    generator = GeneratorConfig(step_seconds=step_seconds, seed=seed)
    schedule = ScheduleConfig(
        seed=seed, faults_per_day=faults_per_day, warmup_hours=warmup_hours
    )
    injection = InjectionConfig()

    healthy = generate_healthy(topology, index, generator)
    faults = schedule_faults(topology, index, schedule)
    metrics, incidents = apply_faults(healthy, topology, faults, injection)

    manifest = {
        "package_version": __version__,
        "start": str(index[0]),
        "end": str(index[-1]),
        "hours": hours,
        "step_seconds": step_seconds,
        "seed": seed,
        "services": len(topology),
        "series": len(topology) * metrics["metric"].nunique(),
        "rows": len(metrics),
        "incidents": len(incidents),
        "anomalous_fraction": round(float(metrics["is_anomalous"].mean()), 6),
        "generator": asdict(generator),
        "schedule": {k: v for k, v in asdict(schedule).items() if k != "kind_weights"},
        "injection": asdict(injection),
    }
    return metrics, incidents, manifest


def write_run(out: Path, metrics: pd.DataFrame, incidents: pd.DataFrame, manifest: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv.gz", index=False, compression="gzip")

    flat = incidents.copy()
    if not flat.empty:
        flat["affected_services"] = flat["affected_services"].apply(",".join)
    flat.to_csv(out / "incidents.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate labelled telemetry.")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--start", default="2026-08-24T00:00:00Z")
    parser.add_argument("--step-seconds", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--faults-per-day", type=float, default=6.0)
    parser.add_argument("--warmup-hours", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=Path("data/run"))
    args = parser.parse_args(argv)

    metrics, incidents, manifest = generate_run(
        hours=args.hours,
        start=args.start,
        step_seconds=args.step_seconds,
        seed=args.seed,
        faults_per_day=args.faults_per_day,
        warmup_hours=args.warmup_hours,
    )
    write_run(args.out, metrics, incidents, manifest)

    print(f"wrote {manifest['rows']:,} rows across {manifest['series']} series -> {args.out}")
    print(summarise(incidents))
    print(f"anomalous cells: {manifest['anomalous_fraction']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
