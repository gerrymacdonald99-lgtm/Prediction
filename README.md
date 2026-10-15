# j1939pf: J1939 per-engine exposure and early-warning analysis

> **What this program can and cannot tell you.**
> Without failure labels this program can only report **exposure** (time × severity of
> deviation from this engine's own learned baseline) and **trend**. It cannot state that a
> condition shortens engine life. Life-shortening claims come only from label-based layers
> (supervised / survival), and those always show sample size and confidence.
>
> It never converts low-oil-pressure exposure into "percent of life lost", and it never
> invents engine spec numbers. Every limit comes from the per-engine config, written from
> that engine's manual.

The headline question is: *do certain oil pressures at certain RPMs (and related operating
conditions) shorten engine life?* The honest answer this tool can give without failure
history is "here is how much time this engine spent at each deviation from its own normal,
and which way it is trending". Nothing stronger.

## Build status

The program is built in five phases. **Phases 1–4 are implemented.** Phase 5 modules
(anomaly detection, supervised/survival models, health score, HTML reports, CLI) do not
exist yet. There are no stubs for them.

| Phase | Scope | Status |
|---|---|---|
| 1 | Parser, TP reassembly, SPN decode, canonical series | done |
| 2 | Synthetic generator, baseline layers (a)–(d), core alerts | done |
| 3 | Derived values, exposure reports, oil-change tracking | done |
| 4 | Correlations, change-points, feedback loop | done |
| 5 | Anomaly detection, supervised/survival, reporting, CLI | not started |

## Install and test

```bash
pip install -e '.[test]'      # Python 3.11+; pandas, numpy, scipy, scikit-learn, PyYAML
pytest                        # lifelines is optional, needed only for phase 5 survival models
```

The program makes no network calls.

## Using Phase 1 on real logs

```python
from j1939pf.pipeline import canonical_from_candump, canonical_from_csv

r = canonical_from_candump("logs/truck12.log")      # .log or .log.gz, streamed line by line
r.canonical       # one row per engine per resample interval
r.dtc_episodes    # DM1/DM2 SPN/FMI episodes (first seen, last seen, max occurrence count)
r.report          # parse quality: malformed lines, duplicates, TP stats, per-signal status counts,
                  # which source addresses sent which PGNs, late or dropped samples, unverified SPNs

r = canonical_from_csv("logs/telematics.csv", "config/my_logger_mapping.yaml")
```

### Inputs

* **candump text**: `candump -l` log format `(ts) can0 18FEEE00#...` or `candump -ta`
  screen format `(ts) can0 18FEEE00 [8] 7D 7D ...`. Timestamps are required. Standard
  (11-bit), remote and CAN-FD frames are counted and skipped.
* **CSV from telematics loggers**: requires a column mapping. See
  `config/csv_mapping.example.yaml`. Each mapped column names an SPN (or a canonical name)
  and **must** state its unit (psi, degF, gal/h, …). Units are converted, never guessed.
  Timestamp format and timezone are configured. Blank cells mean "no sample". Configured
  `na_values` mean "not available" and `error_values` mean a sensor error.
* Both paths emit the same internal sample stream and go through the same resampler, so
  the **same bus data produces the same canonical series** (tested).

### The SPN table (`config/spn_table.yaml`): verify it before trusting anything

The decode table is data, not code. Each entry has PGN, SPN, name, start byte/bit, length,
scale, offset, unit, valid range and not-available/error handling. **Every entry ships with
`verified: false`.** The positions were transcribed from memory of J1939-71 and have not
been checked against a licensed J1939DA. Check each signal you rely on, set
`verified: true`, and note the source. Every parse report lists the unverified SPNs that
were used.

Places where this table departs from, or answers, the draft table:

* **SPN 99 (oil filter Δp)**: the draft placed it in PGN 65263. My reading puts it in
  PGN 65276 (Dash Display) byte 4. Verify.
* **SPN 97 (water in fuel)**: placed in PGN 65279 byte 1, bits 1–2 (2-bit state). Verify.
* **SPN 107 (air filter Δp)**: believed to be in IC1 (PGN 65270) byte 5. Verify.
* **TSC1 (PGN 0)** is a command *to* the engine, so its values are attributed to the
  destination address, not the sender.
* OEM-specific signals (turbo speed, fan, DPF Δp, ash load, DEF level) are decoded only
  where they are actually broadcast. Their positions are low-confidence. SCR temperatures,
  NOx, EGR position, PTO and derate status are not in the table until a verified position
  is available.

### Decoding rules

* 29-bit ID → priority, PGN, source address, destination address. PDU1 (PF < 240) carries
  the DA in PS, which is not part of the PGN. PDU2 is broadcast.
* **Not-available (0xFF…), error (0xFE…), reserved (0xFB–0xFD…) and out-of-range values
  decode to NULL, never to a reading.** Bad readings (error, reserved, out of range, short
  payload) are counted per interval in `<signal>__err` columns. That count feeds sensor-fault
  versus engine-wear discrimination in later phases.
* Oil pressure (SPN 100) is 4 kPa/bit, about 0.58 psi/bit. Later phases floor baseline
  spread at this quantization.
* **Transport protocol (TP.CM/TP.DT)**: BAM and RTS/CTS, with J1939-21 timeouts evaluated
  on frame timestamps. A message is emitted only when every packet arrived. A lost BAM packet
  drops the message. RTS/CTS retransmissions after a CTS are accepted. Exact duplicate packets
  are ignored. A conflicting duplicate in a BAM aborts the message. Every outcome is counted
  in `report["transport"]`.
* **DM1** (single- or multi-frame): lamp states plus an SPN/FMI/OC list, using conversion
  method 4. Messages with CM = 1 are counted, because their SPNs may be mis-decoded. DM2 goes
  to the DTC episode list only.
* **Multi-engine buses**: each source address is a separate engine (`SA00`, `SA01`, …)
  unless `sources.engine_map` in the config maps it, for example mapping an aftertreatment
  ECU's SA onto its engine. `report["decode"]["source_addresses"]` shows which SA sent which
  PGNs, so you can write that mapping.

### Canonical series

One row per engine per `resample_interval_s` (default 1 s):

* Fast signals (EEC1 at about 20 ms) are averaged within the interval. Counters (hours,
  total fuel) take the last value.
* Slow signals (ET1 at about 1 s) are held across empty intervals for at most
  max(3 s, 3 × typical rate), then become NULL.
* An interval whose only samples were error or not-available is NULL and never filled from
  an earlier value.
* Duplicate captures (same interface, ID and data within 0.5 ms) are dropped.
* Silence longer than `gap_s` starts a new `segment`. No rows are produced in the gap and
  nothing is held across it.
* Samples more than `lateness_s` out of order (for example a logger clock stepping
  backwards) are dropped and counted, not silently merged.
* The whole pipeline streams: neither the file nor one object per frame is held in memory.
  The test parses a 6 MB+ log under a quarter of its size in peak traced memory.

`config/defaults.yaml` holds **tunable program defaults, not engine specifications**:
interval, gap, hold, TP timeouts and DTC episode gap.

## Phase 2: per-engine baselines and alerts

```python
from j1939pf.analysis import analyze
from j1939pf.engine_config import load_engine_configs

r = analyze(parsed.canonical, load_engine_configs("config/my_engines/"),
            dtc_episodes=parsed.dtc_episodes, events=maintenance_log_df)
print(r.summary_text())        # header always states the exposure-not-life-loss framing
r.alerts_frame()               # one row per alert episode
r.status["SA00"]               # learning period, layer availability, notes
r.alert_data(r.alerts[0])      # the canonical rows (with expected value and z) behind an alert
```

### Writing the engine config from the manual

Copy `config/engines/TEMPLATE.yaml`, one file per engine. Every limit takes
`{value, unit, source}`, where `source` is the manual section or page. Units are converted
(psi/kPa/bar, degF/degC). **No limit has a default.** If a limit is missing, layer (a) is
disabled for that signal, and the status says so. The learned layers (b)–(d) still run.

The `example_*_PLACEHOLDER.yaml` files hold the on-highway placeholder minimums only:
Cummins ~15/35 psi, Caterpillar ~10–20/30–45 psi (lower bound used), Cummins ISB ~10/30 psi.
The ~248 °F maximum oil temperature is marked UNVERIFIED. All of these are flagged
`placeholder: true`, and every alert that uses one says so. There are no industrial
(QSK/C32/MTU) values; take those from the manual. Rated RPM and the idle band are left empty
in the examples on purpose.

### The four layers

| Layer | What | Learned from |
|---|---|---|
| (a) | Manual thresholds: oil pressure minimum (at all running speeds), minimum at rated speed (within ±100 rpm of the configured rated RPM), max oil and coolant temperature | the engine config only |
| (b) | Per-bin baseline: RPM (100 rpm) × oil temperature (5 °C) × load (25 %); median and robust spread | this engine's first 40 eligible engine hours |
| (c) | z = (actual − expected) / spread, alert when \|z\| ≥ 4 for ≥ 120 s | (b) plus a capped online update |
| (d) | Engine-level drift vs the fixed early-hours reference, with a Theil–Sen slope and 95 % CI | (b) |

* **Eligible (learnable) rows**: engine running, past the 60 s startup grace, no active DM1
  code, reading decoded OK (no error/not-available encoding). If DM1 is not broadcast, the
  status says that learning is unguarded by fault codes.
* **Trusted bin**: at least 30 samples from at least 3 separate sessions. The spread
  includes session-to-session variation and is floored at the bus resolution (4 kPa for oil
  pressure) and at 2.5 % of the bin median, so quantization cannot inflate z-scores.
* **Online learning guardrails**: after the learning period a bin's expected value follows
  new data slowly, and only when that block is not itself anomalous. It never moves more
  than 3 % from the reference (the drift cap). Bins that hit the cap are counted, and the
  drift alert says "flagged, not absorbed". New bins are learned after the learning period
  only while the engine is near its reference.
* **"Drift flagged before threshold"**, as tested: a layer (d) alert fires when the
  engine-level drift against the fixed reference window passes −4 % and the slope's 95 % CI
  excludes zero. The test asserts that this happens before any layer (a) alert, while the
  engine's oil pressure (1st percentile, idle and rated) is still above the configured
  minimums.
* **New engines** announce `LEARNING PERIOD: x of 40 engine hours observed`. Until then only
  layer (a) judges them. Baselines are per engine: nothing is copied between engines, and an
  engine's model is identical whether it is analysed alone or with its fleet (tested).
* Runs less than 10 engine hours apart merge into one alert episode, so a persistent fault
  produces one alert rather than one per day.
* All of these numbers are **tunable program defaults** in `config/defaults.yaml`. They are
  not engine specs.

### Reading an alert

Each alert records the following:

* the signal and layer;
* the operating point (RPM, oil temperature, load, dominant bin);
* expected vs actual values (kPa and psi), z, and duration;
* a severity: watch, warning, or critical (critical is for manual-limit breaches);
* a **falsifiable explanation**;
* confounders: maintenance events in the window, marked as before or after the onset, and
  ambient temperature shifts;
* **what to check next**, cheapest first: the reading itself (mechanical gauge,
  connector/wiring, DTCs), then oil level and grade, then the filter, then mechanical
  causes.

### Sensor fault or engine wear?

Before an alert blames the engine, `sensor_check` weighs the evidence:

* DM1 codes for the SPN with measurement-circuit FMIs (2, 3, 4, 5, 6, 8, 10, 13);
* error encodings in the samples;
* an **engine-off zero check**: a stopped engine reads ~0 kPa, so an offset there is the
  sensor;
* the deviation pattern: high-RPM-only and proportional patterns point to the oil system,
  while a uniform-kPa pattern looks like a sensor offset;
* corroborating signals: filter dp rising, oil level falling.

ECU low-pressure codes (FMI 1/17/18) come from the same sensor and are reported as *not
independent*. The verdict is `sensor_suspected`, `engine_suspected` or `undetermined`.
Readings clip at 0 kPa, so a sensor drifting *low* cannot be told apart from uniform wear by
the zero check. Such cases are reported as `undetermined`, not guessed.

### Synthetic generator (`j1939pf.generator`)

The generator is seeded and works at canonical-series level. `render_candump` turns its
output into J1939 frames, so the whole pipeline can run end to end.

* Scenarios: healthy, bearing wear, clogged filter, stuck relief valve, failing pump, sensor
  drift, high-idle duty cycle, chronic cold operation.
* Events: oil change, oil-grade change, sensor replacement.
* Each engine draws its own profile, with gaussian, Student-t or AR(1) noise, jittered duty
  cycles and missing samples.
* Every generator coefficient is a toy-model parameter, **not an engine specification**.
* Tests require zero alerts across healthy, high-idle and cold-operation engines under all
  three noise models and multiple seeds, plus detection of every fault scenario.

Phase 2 lists maintenance events as confounders but does not yet re-baseline after them.
Re-baselining and the feedback loop come in Phase 4.

## Phase 3: derived values, exposure, oil-change tracking

```python
from j1939pf.usage import usage_report

rep = usage_report(parsed.canonical, configs, dtc_episodes=parsed.dtc_episodes, events=maintenance_log_df)
print(rep.summary_text())     # header: exposure framing; exposure is NOT life consumed
rep.flags()                   # {engine: ["CHRONIC COLD OPERATION ...", "HIGH IDLE ...", "UNLOGGED OIL CHANGE ..."]}
rep.exposure_frame()          # one row per engine x condition
rep.oil_history_frame()       # oil service history: logged and/or detected
rep.engines["SA00"].metrics   # derived values (each says "unavailable: why" when it cannot be computed)
```

### Derived values

Each value below is either computed, or reported as `unavailable` with the reason: the
signal isn't broadcast, the limit is missing from the engine config, or the row interval is
too coarse. All times are counted in canonical rows × row interval.

| Value | Definition |
|---|---|
| Idle % | running AND `idle_rpm_band.low ≤ rpm ≤ idle_rpm_band.high` (inclusive) AND load `<` `idle_load_cutoff` (strict). Load is SPN 92, falling back to SPN 513. A row with unknown load is not idle; the unknown-load share is reported. Flagged only above the config's `idle_pct_flag_threshold`. |
| Idle / working / PTO split | idle vs running-not-idle. PTO and driving are reported as unavailable: there is no verified PTO SPN and vehicle speed is not decoded. |
| Cold starts, time to temperature | a start whose coolant (or oil) temperature is below the config's low-temp cutoff; minutes until the cutoff is reached (median, p90, never reached). |
| Low-temp running % | running below the cutoff, both overall and after the first 30 min. **Chronic cold** is flagged when the after-warm-up share exceeds 25 % (a program definition). |
| Load histogram | % of running time per 10 % load band, plus an RPM × load table. |
| Fuel-per-power drift | fuel rate ÷ (torque % × rpm) at ≥ 50 % load, compared per RPM × load bin against the first 40 h, with a Theil–Sen trend. A **proxy**, not calibrated BSFC. |
| Cranking | crank duration and battery-voltage sag per start (SPN 168). Resolution is limited to the row interval, and the report says so. |
| Torque spikes | a rise of ≥ 50 %-points between rows ≤ 2 s apart. Unavailable on coarser data. |
| Oil consumption | oil level (SPN 98) slope per oil interval, in %-points/100 h. The sender's % is not converted to litres. |
| Faults | DM1 episodes per 100 running h, by SPN/FMI, and clusters within 2 engine h. |
| Hours since service | hours since the last oil change, logged or detected; the source is stated. |
| Fan on, derate | fan-on uses SPN 975 when it is broadcast. Derate is unavailable: there is no verified derate SPN. |

### Exposure report (time × severity)

Each condition reports hours, severity-weighted exposure in a physical unit, episodes, hours
per 100 running hours, the rate per 50 h period, and its definition:

* oil pressure below this engine's own baseline (z-hours, broken down by RPM band; this is
  the headline question, answered as exposure);
* oil pressure below the configured minimums (kPa-hours);
* oil and coolant over-temperature (°C-hours);
* lugging: RPM ≤ the config's `lugging_rpm_max` (from the manual: below peak-torque speed)
  at ≥ 80 % load;
* cold oil at ≥ 75 % of rated RPM;
* extended idling (> 10 min);
* high load while cold;
* hot shutdowns (stopped within 3 min of ≥ 75 % load);
* low-temperature running after warm-up.

Conditions that need a manual value are unavailable until the config has it. **Exposure is
not life consumed.** Nothing converts it into life lost.

### Oil-change detection from oil pressure / temperature vs RPM

An oil change needs the engine stopped, so every engine-off boundary is a candidate. That
means a gap between sessions, or a stop of ≥ 20 min. For each boundary the program compares
this engine's data from 8 engine hours before with 8 hours after:

* **Primary signal: oil pressure at the same RPM × oil temperature × load bins.** Used oil
  thins, so pressure at a fixed operating point sags over the interval, and fresh oil
  restores it: a step up.
* **Supporting signals:**
  * an oil level step up (SPN 98);
  * a filter dp step down (new filter, SPN 99);
  * a shift in oil temperature vs RPM/load;
  * the change in the ln(pressure) vs oil-temperature slope at fixed RPM (viscosity
    behaviour), which is reported.
* **Self-calibrating:** every step is judged against this engine's own boundary-to-boundary
  noise. A detection needs a robust z ≥ 3 and a step ≥ 1.5 %. Noise floors match bus
  resolution: 4 kPa pressure, 0.4 % oil level.
* **Exclusions:**
  * A shift in the engine-off (zero) reading at the boundary means the **pressure sensor**
    changed.
  * A pressure step **down** is an oil-system step (a thinner grade? dilution?). It is not
    counted as an oil change.
  * A level rise without a pressure step is recorded as an `oil_level_rise` (oil added or
    changed), not a confirmed change.
* **Reconciliation with the maintenance log:**
  * `logged; signature confirms`;
  * `logged; no clear signature`;
  * `detected; NOT in the maintenance log`, which raises the **UNLOGGED OIL CHANGE** flag.
    Detected changes are passed to the Phase 2 analysis as confounders.
* **Tracking:**
  * the oil service history, with interval lengths;
  * hours since the last change;
  * per oil interval, the oil-pressure trend at fixed operating points (%/100 h) and oil
    consumption.

On synthetic data, all oil changes are found with no false detections across seeds and
noise models, under normal, high-idle and abusive duty. **Limitation:** when oil runs
chronically cold, pressure sits against the relief valve and the viscosity step can be
invisible. Those changes appear only as oil-level rises, if SPN 98 is broadcast.

The golden fixture `tests/fixtures/idle_golden.log` is a 70 s log covering every idle-rule
edge case: inclusive band edge, strict load cutoff, not-available load and a duplicate
frame. It is computed by hand in `tests/fixtures/idle_golden.md` (34 / 60 s = 56.67 %), and
the test must match it exactly.

The synthetic generator now also models oil ageing (viscosity loss, oil consumption, filter
loading), maintenance at session starts, cranking with voltage sag, and an `abusive`
scenario: lugging, high load from a cold start, and no cool-down before shutdown.

## Phase 4: change-points, feedback loop, correlations

```python
from j1939pf.feedback import FeedbackStore
from j1939pf.correlation import correlation_report

store = FeedbackStore("feedback.json")             # local, append-only decision log
rep = usage_report(parsed.canonical, configs, events=log_df)  # passes logged + detected events on
r = analyze(parsed.canonical, configs, events=rep.events_used, feedback=store)
store.dismiss(r.alerts[0], "gauge reads the same; known weak pump, replacement booked")
store.confirm(r.alerts[1], "bearing wear confirmed at teardown")
r.transitions                                       # re-baselining record per change-point
r.suppressed                                        # alerts silenced by a dismissal (with the reason)
corr = correlation_report(r, configs, events=rep.events_used)
print(corr.summary_text()); corr.flags()            # {"SA00": ["RELATIONSHIP BROKE: boost_vs_fuel_rate ..."]}
```

### Change-points: re-baselining, not alert floods

Change-points come from the maintenance log, plus the oil and sensor changes detected in
Phase 3.

| Event | Handling |
|---|---|
| Oil-grade change | **Multiplicative re-baseline** of oil pressure and filter dp, since viscosity scales both. |
| Sensor replacement (logged, or detected by the engine-off zero shift) | **Additive re-baseline** of that sensor's signal (default: oil pressure). |
| Oil change | No re-baseline: fresh oil restores the reference state. It is recorded as a transition. A low drift that **resets** at an oil change is annotated "consistent with oil ageing rather than wear". |
| Season | Reported as a transition when the ambient median moves more than 10 °C. Bins already hold oil temperature constant. |

How a re-baseline works:

* For `relearn_h` (10 engine h) after the event, that signal is **not judged**. This is a
  flagged transition.
* The step is then measured per operating bin: 10 h before vs 10 h after, weighted median.
* It is applied causally, at the exact end of the transition.
* The step is **measured locally**, not against the original reference. Wear that had
  already accumulated is therefore kept, not absorbed (tested: bearing wear still alerts
  across a grade change).
* A step larger than the event can explain (> 15 % for a grade change, > 30 kPa for a
  sensor) is **refused**. The change stays visible to the alerts.
* A change-point inside the learning window restarts learning.
* Tested: a ±12 kPa sensor replacement and a −10 %/+8 % grade change, each on top of 100 h
  oil changes, give **zero alerts**. The measured steps come within 2.5 kPa / 1.5 % of the
  truth.

### Feedback loop

`FeedbackStore` saves confirm/dismiss decisions to a local JSON file. Records are
append-only, and a reason is required.

* Each alert has a stable `pattern_key`: engine | layer | signal | direction | where (the
  RPM band, the limit name, or "engine").
* **Dismissed:** the pattern is suppressed on later runs and listed in `suppressed` with the
  reason. It comes back only on **new evidence**:
  * the deviation grows to at least 1.5× its size when dismissed (for layer c, the larger of
    the episode median |z| and the most recent 20 %), or
  * a DM1 code for the SPN appears after the dismissal.
  The re-raised alert says why it returned.
* **Confirmed:** the pattern is escalated one severity level, cites the confirmation, and is
  detected at 0.75× the normal z / drift threshold, so it is followed earlier.

### Correlation discovery

Each relationship reports:

* Spearman rho with a 95 % CI (bootstrap over sessions, seeded);
* n rows **and** n independent units (sessions, blocks, periods or engines);
* the valid operating range and condition;
* the confounders;
* a status.

| Kind | Relationships |
|---|---|
| Curve (learned on the reference window, re-checked every 25 h) | oil pressure vs RPM (oil temperature held ±5 °C), coolant temperature vs load, boost vs fuel rate, fuel pressure vs fuel rate, turbo speed vs boost, coolant pressure vs coolant temperature, DPF dp vs fuel rate |
| Trend vs engine hours | oil-pressure drift, crankcase pressure, oil level, coolant level, DPF ash |
| Per period / across the fleet | idle % vs oil-pressure degradation, idle % vs regen frequency, low-temp % vs wear trend |

* **Break detection:** a curve is **BROKEN** when two consecutive periods sit at least 3
  reference spreads off it, and the result is flagged (`RELATIONSHIP BROKE`). A break that
  coincides with a known change-point is reported as "broken (explained)".
  * Tested: a synthetic charge-air leak breaks boost vs fuel rate at its onset, while turbo
    speed vs boost correctly stays stable.
  * A grade change shows as an explained break.
* Period and fleet relationships with fewer than 8 points are labelled **inconclusive**
  rather than over-claimed.
* **Regenerations are INFERRED** from DPF dp drops, since there is no verified regen-status
  SPN. On synthetic data they match on normal duty, but idle-heavy duty is over-counted by
  about 35 %.

The generator now also produces:

* boost and turbo speed;
* intake manifold and exhaust temperature;
* fuel delivery pressure;
* crankcase pressure (slow blow-by growth);
* coolant pressure and level;
* DPF dp from a soot model with active regenerations;
* ash load;
* a `boost_leak` fault.

## Honest limitations

* The SPN table is unverified (see above). A wrong byte position gives plausible-looking but
  wrong numbers, so verify the signals you depend on.
* **Synthetic-data caveat:** the tests run on self-built bus data and self-scripted faults.
  They prove the plumbing (decoding, reassembly, resampling, equivalence, baselines, alerts)
  works. They say nothing about real-world accuracy. Thresholds were checked against several
  noise models and seeds so they are not tuned to one scenario, but real engines will differ.
* Normal oil ageing makes healthy engines drift a few % within each oil interval. The drift
  threshold allows for it, and a low drift that resets at an oil change is annotated as oil
  ageing. With very long oil intervals that margin is thinner.
* Correlations are associations in one engine's own data, not causes. Fleet-level
  relationships need many engines before they mean anything.
* Without failure labels, alerts report deviation, exposure and trend. They never say a
  condition shortens life, and there is no life-loss formula anywhere.
* CSV inputs carry no raw frames, so they have no TP/DM1 frame decoding. DM1 lamp and count
  columns can be mapped if the logger exports them.
* A single corrupted far-future timestamp would make later samples look "late". Such drops
  are counted in `report["canonical"]["late_samples_dropped"]`, so check it.
* Live socketCAN input (bonus) is not implemented.
* Off-bus data (oil analysis, vibration, weather, inspections) is out of scope by design.
