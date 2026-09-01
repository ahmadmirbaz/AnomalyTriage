"""Smoke-test the testbed's core claim: faults are real and they propagate.

    python testbed/verify_propagation.py --service postgres --kind cpu_saturation

Takes a baseline, injects one fault, takes a second reading, and prints the
change per service ordered by hop distance from the origin. If the origin
does not move, the fault is not real; if distant callers move as much as
near ones, propagation is not being modelled the way the ranker assumes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly_triage.mesh.client import MeshClient, mesh_topology  # noqa: E402
from anomaly_triage.mesh.export import fetch  # noqa: E402

LOADGEN = Path(__file__).resolve().parent / "loadgen.py"


def snapshot(seconds: float, prometheus: str) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(seconds=seconds)
    frame = fetch(start, end, step_seconds=15, base_url=prometheus)
    return frame.groupby(["service", "metric"])["value"].median().unstack()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", default="postgres")
    parser.add_argument("--kind", default="cpu_saturation")
    parser.add_argument("--magnitude", type=float, default=0.9)
    parser.add_argument("--warmup", type=float, default=50.0)
    parser.add_argument("--hold", type=float, default=60.0)
    parser.add_argument("--rps", type=float, default=25.0)
    parser.add_argument("--prometheus", default="http://localhost:9090")
    args = parser.parse_args(argv)

    client = MeshClient()
    if not client.ready():
        print("mesh is not up - run: docker compose -f testbed/docker-compose.yml up -d")
        return 1

    topology = mesh_topology()
    hops = topology.upstream_of(args.service)
    client.heal()

    total = args.warmup + args.hold + 20
    load = subprocess.Popen(
        [sys.executable, str(LOADGEN), "--rps", str(args.rps),
         "--day-minutes", "600", "--seconds", str(total)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        print(f"warming up for {args.warmup:.0f}s")
        time.sleep(args.warmup)
        before = snapshot(30, args.prometheus)

        print(f"injecting {args.kind} (magnitude {args.magnitude}) into {args.service}")
        client.inject(args.service, args.kind, args.magnitude, args.hold)
        time.sleep(args.hold - 10)
        after = snapshot(30, args.prometheus)
    finally:
        client.heal()
        load.terminate()

    def distance(service: str) -> int:
        return 0 if service == args.service else hops.get(service, 99)

    print(f"\n{'service':18}{'hops':>6}{'cpu %':>16}{'p95 ms':>20}")
    for service in sorted(before.index, key=lambda s: (distance(s), s)):
        if service not in after.index:
            continue
        d = distance(service)
        tag = "origin" if d == 0 else (f"{d} hop" if d < 99 else "unrelated")
        cpu_b, cpu_a = before.loc[service, "cpu_pct"], after.loc[service, "cpu_pct"]
        p95_b, p95_a = before.loc[service, "latency_p95_ms"], after.loc[service, "latency_p95_ms"]
        ratio = p95_a / p95_b if p95_b else float("nan")
        print(f"{service:18}{tag:>6}{cpu_b:>7.1f} ->{cpu_a:>6.1f}"
              f"{p95_b:>9.1f} ->{p95_a:>7.1f}  ({ratio:.2f}x)")

    print("\nExpect: the origin moves most, effect decays with hop distance, "
          "and unrelated services stay flat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
