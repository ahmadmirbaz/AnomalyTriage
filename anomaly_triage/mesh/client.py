"""Talk to the containerised mesh.

The port map and the topology here must agree with testbed/docker-compose.yml;
tests/test_mesh_client.py parses the compose file and fails if they drift.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..sim.topology import Service, Topology

SERVICE_PORTS: dict[str, int] = {
    "frontend": 8080,
    "checkout": 8001,
    "cart": 8002,
    "product-catalog": 8003,
    "recommendation": 8004,
    "payment": 8005,
    "redis": 8006,
    "postgres": 8007,
}

MESH_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "frontend": ("checkout", "product-catalog", "recommendation", "cart"),
    "checkout": ("cart", "payment"),
    "cart": ("redis",),
    "product-catalog": ("postgres",),
    "recommendation": ("product-catalog",),
    "payment": (),
    "redis": (),
    "postgres": (),
}


def mesh_topology() -> Topology:
    return Topology(
        {name: Service(name, deps) for name, deps in MESH_DEPENDENCIES.items()}
    )


class MeshUnavailable(RuntimeError):
    pass


class MeshClient:
    def __init__(self, host: str = "localhost", timeout: float = 5.0) -> None:
        self.host = host
        self.timeout = timeout

    def _url(self, service: str, path: str) -> str:
        try:
            port = SERVICE_PORTS[service]
        except KeyError:
            raise MeshUnavailable(f"{service!r} is not part of the mesh") from None
        return f"http://{self.host}:{port}{path}"

    def _call(self, service: str, path: str, method: str = "GET", body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        request = Request(
            self._url(service, path),
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            raise MeshUnavailable(f"{service} returned {exc.code}: {exc.read()!r}") from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise MeshUnavailable(f"{service} is unreachable: {exc}") from exc

    def health(self, service: str) -> dict:
        return self._call(service, "/health")

    def ready(self) -> bool:
        try:
            for service in SERVICE_PORTS:
                self.health(service)
        except MeshUnavailable:
            return False
        return True

    def inject(self, service: str, kind: str, magnitude: float, seconds: float | None) -> dict:
        return self._call(
            service,
            "/fault",
            method="POST",
            body={"kind": kind, "magnitude": magnitude, "seconds": seconds},
        )

    def heal(self, service: str | None = None) -> None:
        for target in [service] if service else list(SERVICE_PORTS):
            try:
                self._call(target, "/fault", method="DELETE")
            except MeshUnavailable:
                pass  # healing is best-effort; a dead container needs no healing
