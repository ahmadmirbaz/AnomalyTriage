import json

import pandas as pd

from anomaly_triage.sim.run import generate_run, main, write_run


def _small_run(**overrides):
    params = dict(
        hours=18.0,
        start="2026-08-24T00:00:00Z",
        step_seconds=60,
        seed=0,
        faults_per_day=8.0,
        warmup_hours=4.0,
    )
    params.update(overrides)
    return generate_run(**params)


def test_run_produces_metrics_incidents_and_manifest():
    metrics, incidents, manifest = _small_run()
    assert manifest["rows"] == len(metrics)
    assert manifest["incidents"] == len(incidents)
    assert manifest["series"] == 60
    assert 0.0 < manifest["anomalous_fraction"] < 0.5


def test_labels_agree_with_the_incident_table():
    metrics, incidents, _ = _small_run()
    labelled = set(metrics.loc[metrics.is_anomalous, "incident_id"]) - {""}
    assert labelled == set(incidents["incident_id"])


def test_every_incident_marks_its_own_root_service():
    metrics, incidents, _ = _small_run()
    flagged = metrics[metrics.is_anomalous]
    for _, incident in incidents.iterrows():
        services = set(flagged.loc[flagged.incident_id == incident.incident_id, "service"])
        assert incident.root_service in services


def test_same_seed_reproduces_the_run():
    first, _, _ = _small_run(seed=17)
    second, _, _ = _small_run(seed=17)
    pd.testing.assert_frame_equal(first, second)


def test_write_run_round_trips(tmp_path):
    metrics, incidents, manifest = _small_run()
    write_run(tmp_path, metrics, incidents, manifest)

    reloaded = pd.read_csv(tmp_path / "metrics.csv.gz")
    assert len(reloaded) == len(metrics)

    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["seed"] == manifest["seed"]

    incident_csv = pd.read_csv(tmp_path / "incidents.csv")
    assert list(incident_csv["incident_id"]) == list(incidents["incident_id"])


def test_cli_writes_to_the_requested_directory(tmp_path, capsys):
    out = tmp_path / "nested" / "run"
    exit_code = main(["--hours", "10", "--step-seconds", "60", "--out", str(out)])
    assert exit_code == 0
    assert (out / "metrics.csv.gz").exists()
    assert (out / "manifest.json").exists()
    assert "wrote" in capsys.readouterr().out
