"""The Python-side mesh description must not drift from the compose file."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from anomaly_triage.mesh.client import (
    MESH_DEPENDENCIES,
    SERVICE_PORTS,
    MeshClient,
    MeshUnavailable,
    mesh_topology,
)

COMPOSE = Path(__file__).resolve().parents[1] / "testbed" / "docker-compose.yml"


def parse_compose() -> dict[str, dict[str, str]]:
    """Pull service name, published port and DEPENDS_ON out of the compose file.

    Deliberately a regex rather than a YAML dependency: this test exists to
    catch drift, and it should not be able to fail because a parser is absent.
    """
    text = COMPOSE.read_text()
    blocks = re.split(r"\n  (?=[a-z][a-z-]*:\n)", text)
    parsed: dict[str, dict[str, str]] = {}
    for block in blocks:
        name = re.match(r"\s*([a-z][a-z-]*):\n", block)
        if not name or "SERVICE_NAME" not in block:
            continue
        service = re.search(r"SERVICE_NAME:\s*(\S+)", block).group(1)
        port = re.search(r'ports:\s*\["(\d+):8000"\]', block)
        depends = re.search(r'DEPENDS_ON:\s*(.*)', block)
        raw = depends.group(1).strip().strip('"') if depends else ""
        parsed[service] = {
            "port": port.group(1) if port else None,
            "depends_on": tuple(d for d in raw.split(",") if d),
        }
    return parsed


@pytest.fixture(scope="module")
def compose():
    return parse_compose()


def test_compose_defines_the_services_we_think_it_does(compose):
    assert set(compose) == set(SERVICE_PORTS)


def test_published_ports_match(compose):
    for service, details in compose.items():
        assert details["port"] is not None, f"{service} publishes no port"
        assert int(details["port"]) == SERVICE_PORTS[service]


def test_dependency_edges_match(compose):
    for service, details in compose.items():
        assert details["depends_on"] == MESH_DEPENDENCIES[service], service


def test_mesh_topology_builds_and_is_rooted_at_frontend():
    topo = mesh_topology()
    assert len(topo) == len(SERVICE_PORTS)
    for name in topo.names:
        if name != "frontend":
            assert "frontend" in topo.upstream_of(name), name


def test_unknown_service_is_rejected_before_any_request():
    with pytest.raises(MeshUnavailable, match="not part of the mesh"):
        MeshClient().health("nonexistent")


def test_client_reports_unreachable_rather_than_raising_urlerror():
    # nothing listens on this port
    client = MeshClient(host="127.0.0.1", timeout=0.5)
    import anomaly_triage.mesh.client as mod

    original = mod.SERVICE_PORTS.copy()
    mod.SERVICE_PORTS["frontend"] = 9
    try:
        with pytest.raises(MeshUnavailable, match="unreachable"):
            client.health("frontend")
    finally:
        mod.SERVICE_PORTS.clear()
        mod.SERVICE_PORTS.update(original)
