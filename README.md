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

The program is built in five phases. **Phases 1 and 2 are implemented.** Later-phase
modules (exposure reports, correlations, change-point re-baselining, feedback loop,
anomaly/survival models, HTML reports, CLI) do not exist yet. There are no stubs for them.

| Phase | Scope | Status |
|---|---|---|
| 1 | Parser, TP reassembly, SPN decode, canonical series | done |
| 2 | Synthetic generator, baseline layers (a)–(d), core alerts | done |
| 3 | Derived values, exposure reports | not started |
| 4 | Correlations, change-points, feedback loop | not started |
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

## Honest limitations

* The SPN table is unverified (see above). A wrong byte position gives plausible-looking but
  wrong numbers, so verify the signals you depend on.
* **Synthetic-data caveat:** the tests run on self-built bus data and self-scripted faults.
  They prove the plumbing (decoding, reassembly, resampling, equivalence, baselines, alerts)
  works. They say nothing about real-world accuracy. Thresholds were checked against several
  noise models and seeds so they are not tuned to one scenario, but real engines will differ.
* Without failure labels, alerts report deviation, exposure and trend. They never say a
  condition shortens life, and there is no life-loss formula anywhere.
* CSV inputs carry no raw frames, so they have no TP/DM1 frame decoding. DM1 lamp and count
  columns can be mapped if the logger exports them.
* A single corrupted far-future timestamp would make later samples look "late". Such drops
  are counted in `report["canonical"]["late_samples_dropped"]`, so check it.
* Live socketCAN input (bonus) is not implemented.
* Off-bus data (oil analysis, vibration, weather, inspections) is out of scope by design.
