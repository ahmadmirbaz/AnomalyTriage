# AnomalyTriage

Watches thousands of service metrics, and when something breaks, tells you which one caused it.

## Why

A platform emitting 5,000 metrics on a one-minute cadence, alerting on a
two-sided 3σ band, fires roughly **19,000 false alerts a day** while
perfectly healthy. Nothing is misconfigured — the error rate was specified
per metric, but the on-call engineer receives the union of all of them.
Teams respond by widening thresholds until the noise stops, which is how
real incidents get missed.

This project treats detection as what it actually is: a very large
simultaneous hypothesis test. Alert volume then becomes a **false discovery
rate you choose** rather than a side effect of threshold tuning, and it
holds no matter how many series you watch.

## Where it is

Phase 0 of 6 — the labelled data generator.

| Phase | | Status |
|---|---|---|
| 0 | Testbed and labelled fault data | in progress |
| 1 | Ingestion, storage, seasonal baselines | |
| 2 | Quantile forecasting and eval harness | |
| 3 | FDR control, extreme-value thresholds, changepoints | |
| 4 | Trace topology and root-cause ranking | |
| 5 | Triage agent under a token budget | |
| 6 | Incident UI and write-up | |

## The data generator

Evaluating root-cause ranking needs labels that say *what broke*, not just
*when*. Public anomaly benchmarks almost never carry those, so the faults
are manufactured here and the ground truth falls out for free.

```bash
python -m anomaly_triage.sim.run --hours 168 --out data/week-01
```

Writes `metrics.csv.gz` (long format, one row per service/metric/timestamp,
carrying `is_anomalous` and `incident_id`), `incidents.csv` (root service,
fault kind, window, blast radius) and `manifest.json` (every parameter
needed to reproduce the run).

Three properties of the healthy baseline are deliberate, because each one is
something the detector will have to survive:

- **Diurnal and weekly seasonality**, so a seasonal-naive baseline is a
  genuinely competitive opponent rather than a strawman.
- **AR(1) noise**, so residuals are autocorrelated and independence
  assumptions do not come for free.
- **A lognormal tail on latency**, so Gaussian tail probabilities are wrong
  where it matters and the extreme-value work in phase 3 has a job to do.

Six fault kinds, each with a different shape over time — a step for latency
injection, a long ramp for a memory leak, a permanent step for a deploy
regression. That last one exists specifically to be the thing that looks
like an incident and is not.

Faults propagate from callee to caller, attenuated 0.55 per hop and delayed
by one hop-lag, so a cascade has a recoverable onset order. Latency and
error rate travel upstream; CPU and memory do not. That asymmetry is the
strongest localisation signal in the data, and the phase 4 ranker is meant
to find it.

## Layout

```
anomaly_triage/sim/
  topology.py   service call graph and upstream traversal
  metrics.py    healthy telemetry: seasonality, AR(1) noise, fat tails
  faults.py     fault taxonomy and per-kind metric signatures
  inject.py     propagation, attenuation, ground-truth labelling
  schedule.py   randomised schedules with a fault-free warm-up
  run.py        CLI producing a reproducible labelled run
```

## Development

```bash
pip install -r requirements.txt
pytest
```
