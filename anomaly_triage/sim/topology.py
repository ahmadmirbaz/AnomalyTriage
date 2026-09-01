"""Service dependency graph for the simulated fleet.

Names mirror the OpenTelemetry demo so that the simulator and the
containerised testbed stay swappable later on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Service:
    """One service and the steady-state level of its telemetry."""

    name: str
    depends_on: tuple[str, ...] = ()
    base_rps: float = 40.0
    base_latency_ms: float = 45.0
    base_cpu_pct: float = 25.0
    base_mem_mb: float = 512.0
    base_error_rate: float = 0.002


@dataclass
class Topology:
    """A directed call graph: an edge runs from caller to callee."""

    services: dict[str, Service] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for svc in self.services.values():
            for dep in svc.depends_on:
                if dep not in self.services:
                    raise ValueError(f"{svc.name} depends on unknown service {dep!r}")

    def __iter__(self) -> Iterator[Service]:
        return iter(self.services.values())

    def __len__(self) -> int:
        return len(self.services)

    @property
    def names(self) -> list[str]:
        return list(self.services)

    def edges(self) -> list[tuple[str, str]]:
        """(caller, callee) pairs."""
        return [(s.name, dep) for s in self for dep in s.depends_on]

    def callers_of(self, name: str) -> list[str]:
        """Direct callers — the services that would notice `name` degrading."""
        return [s.name for s in self if name in s.depends_on]

    def upstream_of(self, name: str) -> dict[str, int]:
        """Every transitive caller, mapped to its hop distance from `name`.

        Fault effects travel this direction: a slow callee makes its callers
        slow, attenuated by distance.
        """
        distances: dict[str, int] = {}
        frontier = [(name, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            for caller in self.callers_of(current):
                if caller in distances or caller == name:
                    continue
                distances[caller] = depth + 1
                frontier.append((caller, depth + 1))
        return distances


def _svc(name: str, deps: Iterable[str] = (), **kwargs: float) -> Service:
    return Service(name=name, depends_on=tuple(deps), **kwargs)


def default_topology() -> Topology:
    """A twelve-service storefront with a realistic fan-out."""
    services = [
        _svc("frontend", ["checkout", "product-catalog", "recommendation", "ad", "cart"],
             base_rps=220.0, base_latency_ms=140.0, base_cpu_pct=45.0),
        _svc("checkout", ["cart", "payment", "shipping", "email", "currency"],
             base_rps=35.0, base_latency_ms=210.0, base_cpu_pct=30.0),
        _svc("cart", ["redis"], base_rps=180.0, base_latency_ms=25.0),
        _svc("product-catalog", ["postgres"], base_rps=190.0, base_latency_ms=38.0),
        _svc("recommendation", ["product-catalog"], base_rps=90.0, base_latency_ms=85.0),
        _svc("ad", [], base_rps=95.0, base_latency_ms=30.0),
        _svc("payment", ["currency"], base_rps=32.0, base_latency_ms=120.0),
        _svc("shipping", [], base_rps=32.0, base_latency_ms=95.0),
        _svc("email", [], base_rps=30.0, base_latency_ms=60.0),
        _svc("currency", [], base_rps=140.0, base_latency_ms=12.0),
        _svc("redis", [], base_rps=400.0, base_latency_ms=3.0, base_mem_mb=1024.0),
        _svc("postgres", [], base_rps=260.0, base_latency_ms=9.0, base_mem_mb=2048.0),
    ]
    return Topology({s.name: s for s in services})
