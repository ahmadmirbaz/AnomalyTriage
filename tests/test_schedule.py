import pandas as pd

from anomaly_triage.sim.faults import FaultKind
from anomaly_triage.sim.metrics import time_index
from anomaly_triage.sim.schedule import ScheduleConfig, schedule_faults
from anomaly_triage.sim.topology import default_topology


def _week():
    return time_index("2026-08-24", hours=168, step_seconds=15)


def test_nothing_scheduled_during_warmup():
    idx = _week()
    config = ScheduleConfig(seed=3, warmup_hours=12.0)
    faults = schedule_faults(default_topology(), idx, config)
    assert faults
    assert min(f.start for f in faults) >= idx[0] + pd.Timedelta(hours=12)


def test_minimum_gap_is_respected():
    idx = _week()
    config = ScheduleConfig(seed=5, min_gap_minutes=30.0)
    faults = schedule_faults(default_topology(), idx, config)
    starts = sorted(f.start for f in faults)
    gaps = [(b - a) for a, b in zip(starts, starts[1:])]
    assert all(g >= pd.Timedelta(minutes=30) for g in gaps)


def test_rate_is_roughly_honoured():
    idx = _week()
    faults = schedule_faults(default_topology(), idx, ScheduleConfig(seed=1, faults_per_day=6.0))
    # 168h run minus a 6h warm-up leaves ~6.75 usable days
    assert 30 <= len(faults) <= 45


def test_permanent_faults_are_capped():
    idx = _week()
    config = ScheduleConfig(seed=9, faults_per_day=40.0, max_permanent=2)
    faults = schedule_faults(default_topology(), idx, config)
    permanent = [f for f in faults if f.kind.is_permanent]
    assert len(permanent) <= 2


def test_transient_faults_finish_inside_the_run():
    idx = _week()
    faults = schedule_faults(default_topology(), idx, ScheduleConfig(seed=4))
    for fault in faults:
        if not fault.kind.is_permanent:
            assert fault.end <= idx[-1]


def test_schedule_is_reproducible():
    idx = _week()
    a = schedule_faults(default_topology(), idx, ScheduleConfig(seed=42))
    b = schedule_faults(default_topology(), idx, ScheduleConfig(seed=42))
    assert a == b


def test_short_run_yields_nothing():
    idx = time_index("2026-08-24", hours=4, step_seconds=15)
    assert schedule_faults(default_topology(), idx, ScheduleConfig(warmup_hours=6.0)) == []


def test_all_kinds_appear_over_a_long_run():
    idx = time_index("2026-08-24", hours=24 * 60, step_seconds=60)
    faults = schedule_faults(default_topology(), idx, ScheduleConfig(seed=2))
    assert {f.kind for f in faults} == set(FaultKind)
