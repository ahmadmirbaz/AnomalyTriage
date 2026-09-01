import pytest

from anomaly_triage.sim.topology import Service, Topology, default_topology


def _chain() -> Topology:
    # a -> b -> c, plus d -> c
    return Topology({
        "a": Service("a", ("b",)),
        "b": Service("b", ("c",)),
        "c": Service("c", ()),
        "d": Service("d", ("c",)),
    })


def test_rejects_dangling_dependency():
    with pytest.raises(ValueError, match="unknown service"):
        Topology({"a": Service("a", ("nope",))})


def test_callers_are_direct_only():
    assert _chain().callers_of("c") == ["b", "d"]
    assert _chain().callers_of("a") == []


def test_upstream_records_hop_distance():
    assert _chain().upstream_of("c") == {"b": 1, "d": 1, "a": 2}


def test_upstream_of_root_is_empty():
    assert _chain().upstream_of("a") == {}


def test_upstream_terminates_on_a_cycle():
    cyclic = Topology({"a": Service("a", ("b",)), "b": Service("b", ("a",))})
    assert cyclic.upstream_of("a") == {"b": 1}


def test_default_topology_is_connected_to_frontend():
    topo = default_topology()
    # every service should be reachable from the frontend, or be the frontend
    for name in topo.names:
        if name == "frontend":
            continue
        assert "frontend" in topo.upstream_of(name), name


def test_default_topology_edge_count():
    topo = default_topology()
    assert len(topo) == 12
    assert len(topo.edges()) == 14
