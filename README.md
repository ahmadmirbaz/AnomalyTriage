# AnomalyTriage

Watches thousands of service metrics, and when something breaks, tells you which one caused it.

## What this is, in plain terms

Big online systems are not one program. They are dozens of small services
that call each other: a checkout page asks a payment service, which asks a
database, which asks something else. Every one of those services constantly
reports numbers about itself. How fast is it replying? How much memory is it
using? How often is it failing?

When something goes wrong, two problems arrive together.

**The first is noise.** Put an alarm on every number, tell it to fire
whenever something looks unusual, and you get tens of thousands of alarms a
day on a system where nothing is actually broken. Engineers cope by turning
the alarms down until they go quiet, which is exactly how real outages end up
missed.

**The second is blame.** One slow database makes every service that depends
on it look slow too. The engineer woken at 3am gets fifty alerts and has to
work out which one is the cause and which forty-nine are just echoes of it.

This project attacks both. It decides what counts as unusual using statistics
that stay reliable no matter how many things you are watching, and it uses
the map of which service calls which to point at the one that actually
started the problem.

The sections below get progressively more technical. This one is the whole
idea.

## Why thresholds do not work

The usual approach is a threshold: alert whenever a number strays more than
three standard deviations from normal. The textbook says that should fire on
0.27% of readings. On realistic data it is worse:

```
14 fault-free days, 60s scrapes, 4h rolling window

    k    observed    gaussian  inflation       alerts/day, 5000 series
  2.0     9.243%     4.550%       2.0x                       665,518
  2.5     2.900%     1.240%       2.3x                       208,792
  3.0     0.881%     0.270%       3.3x                        63,446
  3.5     0.303%     0.046%       6.5x                        21,839
```

Every alert counted there is false by construction: nothing is wrong.
Reproduce with `python -m anomaly_triage.detect.measure_baseline`.

Two things are worth noticing. The three-sigma rule fires **3.3x** more
often than the Gaussian calculation promises, because real telemetry is
autocorrelated and heavy-tailed rather than independent and normal. And the
inflation *grows* as you push further out, 6.5x by 3.5 sigma, so tightening
the threshold buys less than it appears to. Teams widen thresholds until the
noise stops, which is how real incidents get missed.

Nothing is misconfigured here. The error rate was specified per metric, but
the engineer receives the union of all of them.

This project treats detection as what it actually is: a very large
simultaneous hypothesis test. Alert volume then becomes a **false discovery
rate you choose** rather than a side effect of threshold tuning, and it
holds no matter how many series you watch.

## Where it is

Phase 2 of 6: the forecaster.

| Phase | | Status |
|---|---|---|
| 0 | Testbed and labelled fault data | ipr |
| 0.5 | Containerised mesh with real faults | ipr |
| 1 | Ingestion, storage, seasonal baselines | ipr |
| 2 | Quantile forecasting and eval harness | ipr |
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

Six fault kinds, each with a different shape over time: a step for latency
injection, a long ramp for a memory leak, a permanent step for a deploy
regression. That last one exists specifically to be the thing that looks
like an incident and is not.

Faults propagate from callee to caller, attenuated 0.55 per hop and delayed
by one hop-lag, so a cascade has a recoverable onset order. Latency and
error rate travel upstream; CPU and memory do not. That asymmetry is the
strongest localisation signal in the data, and the phase 4 ranker is meant
to find it.

## The forecaster

Detection needs a *distribution* for the next point, not a number. A point
forecast plus one global sigma is the assumption that fails hardest on
latency, where the spread is wide, skewed, and moves with load. So the model
predicts seven quantiles per series directly, and the interval becomes its
own output rather than an afterthought.

One gradient-boosted model per `(service, metric)` per quantile. Services
differ by an order of magnitude in traffic and latency, and a shared model
would spend its capacity learning that spread instead of either series'
shape. Models are fit **only on rows labelled fault-free**: a forecaster
trained through its own incidents learns to expect them, and the anomaly it
was built to surface flattens into the baseline.

```bash
python -m anomaly_triage.detect.measure_forecast --data data/train-01
```

### Predict the change, not the level

The first version predicted the value directly and lost. Across all 60
series it scored *worse* than a five-line exponential moving average.

The cause is specific. A regression tree can only ever output a value it saw
while training. Memory climbs steadily under a leak, so a level-predicting
model walks off the end of its own training range at exactly the moment an
incident begins. An exponential average follows a drift for free.

Anchoring on the previous observation and predicting the *step* keeps the
target stationary and hands extrapolation back to the anchor. On `mem_mb`
and `latency_p95_ms` across three services, held out on the last 30% of a
14-day run:

```
              pinball  cover90  cover98      KS   tail atoms
  level        31.97    67.4%    74.3%    0.194   6.3% / 19.4%
  change       20.15    85.7%    96.1%    0.032   1.9% /  2.0%
```

The KS column is the load-bearing one. Benjamini-Hochberg controls the false
discovery rate only if p-values are uniform under the null, so a residual at
KS 0.194 is not a p-value at all and every guarantee in phase 3 built on it
would be decoration.

### Against the baselines

Median pinball loss on `postgres`, each metric scaled by its own spread so
that `mem_mb` does not drown out `error_rate`. Pooling them unscaled meant
two of the five metrics contributed under 3% of the headline number:

```
  metric              forecaster    seasonal        ewma
  cpu_pct                 0.0328      0.2262      0.0663
  error_rate              0.1725      0.2155      0.1829
  latency_p95_ms          0.3070      0.3914      0.3081
  mem_mb                  0.0952      0.4656      0.1614
  request_rate_rps        0.0287      0.2897      0.0522
```

Nominal 90% and 98% intervals covered 88.7% and 97.0% of clean points.

Two honest caveats. These are one service; the full-fleet numbers for the
corrected model have not been measured yet. And EWMA is a *strong* opponent
here by construction. The healthy baseline is AR(1), and for an AR(1)
process the optimal one-step forecast is an exponential smoother, so there
may be little headroom on point accuracy by design. Feeding EWMA to the
model as an input feature was tried and did not help.

The claim worth making is not "more accurate". It is that EWMA emits a bare
number carrying no notion of how surprised to be, and this emits a calibrated
distribution, which is the only thing the FDR machinery in phase 3 can
consume.

## The containerised mesh

The simulator is fast and its labels are exact, but its metrics are drawn
from a model rather than measured. The mesh closes that gap: eight
instrumented Flask services behind Prometheus, wired into the same topology
and scraped every five seconds.

```bash
docker compose -f testbed/docker-compose.yml up -d --build
python -m anomaly_triage.mesh.orchestrate --minutes 45 --out data/mesh-01
```

Faults here are **real**, not modelled. `cpu_saturation` duty-cycles a busy
loop, `memory_leak` genuinely allocates, `latency_injection` sleeps in the
handler, `dependency_failure` refuses calls. The CPU and memory gauges then
report what the process is actually doing.

Injecting `cpu_saturation` at magnitude 0.9 into `postgres`:

| service | hops | CPU | p95 latency |
|---|---|---|---|
| postgres | origin | 4.5% → 86.9% | 9.8 → 41.6 ms (4.3x) |
| product-catalog | 1 | n/a | 48.4 → 96.6 ms (2.0x) |
| recommendation | 2 | n/a | 97.5 → 227.6 ms (2.3x) |
| frontend | 2 | n/a | 487 → 494 ms (1.01x) |
| payment | unrelated | n/a | unchanged |

The frontend barely moves, and that is the interesting part. Its p95 is
dominated by the slower `checkout -> payment` branch, so a 48 ms bump on the
catalog branch disappears into it. An incident that is plainly visible three
hops down is invisible at the edge, which is the argument for per-service
detection and graph localisation rather than watching the front door.

Reproduce with:

```bash
python testbed/verify_propagation.py --service postgres --kind cpu_saturation
```

Both sources emit the identical schema, `(timestamp, service, metric,
value)` over the same five metric names, so nothing downstream knows or
cares which one it is reading.

## Layout

```
anomaly_triage/sim/
  topology.py   service call graph and upstream traversal
  metrics.py    healthy telemetry: seasonality, AR(1) noise, fat tails
  faults.py     fault taxonomy and per-kind metric signatures
  inject.py     propagation, attenuation, ground-truth labelling
  schedule.py   randomised schedules with a fault-free warm-up
  run.py        CLI producing a reproducible labelled run

anomaly_triage/detect/
  baseline.py   seasonal-naive, EWMA and the k-sigma rule it argues against
  features.py   lag, rolling and calendar features, shifted so none can peek
  model.py      per-series quantile forecaster on change-from-last
  evaluate.py   pinball loss, interval coverage, PIT uniformity
  measure_baseline.py  what threshold alerting costs on a healthy fleet
  measure_forecast.py  the forecaster against the baselines, held out in time

anomaly_triage/mesh/
  client.py     fault injection against the running containers
  export.py     Prometheus -> the simulator's schema
  orchestrate.py  drives a labelled run end to end

testbed/
  service/      one instrumented service, configured by environment
  docker-compose.yml  the eight-service mesh plus Prometheus
  loadgen.py    Poisson traffic on a compressed diurnal cycle
  verify_propagation.py  smoke test: are faults real, do they propagate
```

## How this was built

Built with the assistance of Claude Code, Anthropic's command-line tool,
used for implementation and for working through design trade-offs. The
measurements in this README were produced by running the code, not by
asking a model what it expected the numbers to be.

## Development

```bash
pip install -r requirements.txt
pytest
```
