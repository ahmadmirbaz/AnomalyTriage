"""Drive traffic at the mesh's frontend.

Traffic follows a compressed diurnal cycle: a whole "day" passes in
`--day-minutes`, so a two-hour run still contains several peaks and troughs.
Without that, the seasonal baselines in phase 1 would have nothing to model
and would look artificially easy to beat.
"""

from __future__ import annotations

import argparse
import math
import random
import threading
import time
from collections import Counter
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

_PEAK_FRACTION = 14.0 / 24.0  # peak lands where 14:00 would fall
_AMPLITUDE = 0.35


def diurnal_multiplier(elapsed_s: float, day_seconds: float) -> float:
    phase = 2 * math.pi * (elapsed_s / day_seconds - _PEAK_FRACTION + 0.25)
    return 1.0 + _AMPLITUDE * math.sin(phase)


class LoadGenerator:
    def __init__(self, url: str, rps: float, day_minutes: float, workers: int) -> None:
        self.url = url
        self.rps = rps
        self.day_seconds = day_minutes * 60
        self.workers = workers
        self.counts: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _record(self, outcome: str) -> None:
        with self._lock:
            self.counts[outcome] += 1

    def _fire(self) -> None:
        try:
            with urlopen(self.url, timeout=10) as response:
                self._record(str(response.status))
        except HTTPError as exc:
            self._record(str(exc.code))
        except (URLError, OSError, TimeoutError):
            self._record("unreachable")

    def _worker(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            target = self.rps * diurnal_multiplier(started - self.origin, self.day_seconds)
            interval = self.workers / max(target, 0.1)
            self._fire()
            # Exponential gaps approximate Poisson arrivals better than a
            # fixed cadence, which would make the request rate suspiciously
            # smooth for a detector to model.
            delay = random.expovariate(1.0 / interval) - (time.monotonic() - started)
            if delay > 0:
                self._stop.wait(delay)

    def run(self, duration_s: float | None) -> Counter[str]:
        self.origin = time.monotonic()
        threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(self.workers)]
        for thread in threads:
            thread.start()
        try:
            deadline = None if duration_s is None else self.origin + duration_s
            while deadline is None or time.monotonic() < deadline:
                time.sleep(1.0)
                if int(time.monotonic() - self.origin) % 30 == 0:
                    with self._lock:
                        print(f"  {sum(self.counts.values()):>7,} requests  {dict(self.counts)}")
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
        return self.counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate load against the mesh.")
    parser.add_argument("--url", default="http://localhost:8080/work")
    parser.add_argument("--rps", type=float, default=20.0)
    parser.add_argument("--day-minutes", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=None)
    args = parser.parse_args(argv)

    generator = LoadGenerator(args.url, args.rps, args.day_minutes, args.workers)
    print(f"driving {args.rps} rps at {args.url} (a day every {args.day_minutes}min)")
    counts = generator.run(args.seconds)
    print(f"done: {sum(counts.values()):,} requests {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
