# j1939pf: per-engine J1939 exposure, trend and early-warning analysis

> **Read this first.**
> Without failure labels this program can only report EXPOSURE (time x severity of deviation from this engine's own learned baseline) and TREND. It cannot state that a condition shortens engine life. Life-shortening claims are made only by label-based layers (supervised / survival) and always show sample size and confidence.
>
> The program never turns low-oil-pressure exposure into "percent of life lost", and never
> invents engine specifications. Every limit comes from the per-engine config written from
> that engine's manual. If a limit is missing, the program says so and does not judge that
> signal against it.

The headline question is *"do certain oil pressures at certain RPMs, and related operating
conditions, shorten engine life?"* Without failure history, the honest answer this program
gives is: how much time this engine spent at each deviation from its **own** normal, and
which way it is trending. Only when your event log contains enough failures do the
label-based layers estimate effects on time to failure, and then always with n and
confidence intervals.

**Synthetic-data caveat.** Every test runs on self-built bus data with self-scripted faults.
That proves the plumbing works: decoding, baselines, alerts, exposure, re-baselining,
correlations and reports. It says **nothing** about real-world accuracy. Run the program on
your own engines, check its alerts against inspections, and use the feedback loop.

---

## 1. Install and test

```bash
pip install -e '.[test]'        # Python 3.11+: pandas, numpy, scipy, scikit-learn, PyYAML (+ pytest)
pytest                          # 189 tests, about 11 minutes (synthetic engines are generated on the fly)
```

* **No network calls** at runtime.
* All randomness is seeded through `seed` in `config/defaults.yaml` (default 0): the
  generator, the isolation forest, and the bootstraps.
* `lifelines` is **not required**. The survival layer uses its own scipy implementation (a
  Weibull AFT model and Kaplan–Meier). `lifelines` could not be built in the development
  environment (Python 3.13), so there is no lifelines code path to go untested.

## 2. Quick start with synthetic data

```bash
j1939pf generate --scenario stuck_relief_valve --hours 300 --seed 1 --out demo.csv --events-out demo_events.csv
j1939pf analyze demo.csv --engines config/engines/TEMPLATE.yaml --events demo_events.csv --synthetic --out out/
open out/report.html            # or read out/summary.txt
```

`TEMPLATE.yaml` has no limits, so layer (a) is disabled and the learned layers do all the
work. Available scenarios: `healthy`, `bearing_wear`, `clogged_filter`,
`stuck_relief_valve`, `failing_pump`, `sensor_drift`, `high_idle`, `cold_operation`,
`abusive`, `boost_leak`. Outputs from `--synthetic` runs are labelled SYNTHETIC.

## 3. Running on real logs

```bash
j1939pf analyze truck12.log.gz --engines my_engines/ --events maintenance.csv --feedback feedback.json --out out/
j1939pf analyze telematics.csv --csv-mapping my_logger.yaml --engines my_engines/ --out out/
j1939pf parse truck12.log --out truck12_canonical.csv       # decode once, analyse the CSV many times
```

* **candump:**
  * Formats: `candump -l` (`(ts) can0 18FEEE00#...`) or `candump -ta` screen format; `.gz`
    files are read directly.
  * Timestamps are required.
  * Logs are streamed line by line, never loaded whole.
* **Telematics CSV:** copy `config/csv_mapping.example.yaml`.
  * Every column names an SPN (or canonical name) **and its unit**. psi, °F, gal/h and so on
    are converted, never guessed.
  * Timestamp format and timezone are configurable.
  * Blank cell = no sample; `na_values` = "not available"; `error_values` = a sensor error.
  * The same bus data gives the same canonical series from candump and from CSV (tested).
* **Several engines on one bus:**
  * Each source address is its own engine (`SA00`, `SA01`, …).
  * Map addresses in an override file passed with `--defaults`, for example
    `sources: {engine_map: {0: port_main, 61: port_main}}`, which puts an aftertreatment ECU
    on its engine.
  * `parse_report.json` lists which SA sent which PGNs.
* **Overrides:** `--defaults my_overrides.yaml` overrides any value in
  `config/defaults.yaml`. Every value there is a tunable program default, **not** an engine
  spec.

### The SPN decode table: verify it before trusting decoded values

`config/spn_table.yaml` is data, not code. Every entry has PGN, SPN, start byte/bit, length,
scale, offset, unit, valid range and not-available handling, and **every entry ships with
`verified: false`**. The positions were transcribed from memory of J1939-71. Check each
signal you rely on against your licensed J1939DA or the OEM's broadcast list, set
`verified: true`, and record the source. Every report lists the unverified SPNs it used.

Places where the table departs from, or answers, the original draft (verify each):

* SPN 99 (oil filter Δp): PGN 65276 (DD) byte 4, not 65263.
* SPN 97 (water in fuel): PGN 65279, 2-bit state.
* SPN 107 (air filter Δp): IC1 byte 5.
* TSC1 is a command *to* the engine, so it is attributed to the destination address.
* OEM-specific signals (turbo speed, fan, DPF Δp, ash, DEF) are low-confidence and decoded
  only where broadcast.
* SCR temperatures, NOx, EGR, PTO and derate status are not in the table yet.

## 4. Writing the engine config from the manual

Copy `config/engines/TEMPLATE.yaml`, one file per engine (`engine_id` = `SA00`… or your mapped
name). Every limit is `{value, unit, source}`, where `source` is the manual section or page.
**No limit has a default.**

| Limit | Used by |
|---|---|
| `oil_pressure_min_idle`, `oil_pressure_min_rated`, `rated_rpm` | layer (a) oil-pressure thresholds (the rated minimum needs `rated_rpm`) |
| `oil_temp_max`, `coolant_temp_max` | layer (a), over-temperature exposure |
| `idle_rpm_band`, `idle_load_cutoff`, `load_signal` (SPN 92, falling back to 513) | idle %, extended idling |
| `coolant_low_temp_cutoff`, `oil_low_temp_cutoff` | cold starts, time to temperature, low-temp running, cold-oil-high-RPM |
| `idle_pct_flag_threshold` | the HIGH IDLE flag (fleet policy is a valid source) |
| `lugging_rpm_max` | lugging exposure (manual: below peak-torque speed) |

* A missing limit disables only what needs it. Each output says "unavailable: …missing from
  engine config", and the learned layers still run.
* The `example_*_PLACEHOLDER.yaml` files hold **on-highway placeholder** oil-pressure
  minimums only:
  * Cummins ~15 / 35 psi;
  * Caterpillar ~10–20 / 30–45 psi (lower bound used);
  * Cummins ISB ~10 / 30 psi;
  * ~248 °F max oil temperature, marked **UNVERIFIED**.
* All placeholders carry `placeholder: true`, and every alert that uses one says so.
* There are **no industrial (QSK, C32, MTU) values**: take them from the manual.

## 5. Maintenance / failure event log (optional; everything works with none)

CSV with `engine_id, engine_hours, event_type` and optional `t, detail, signal`:

| event_type | effect |
|---|---|
| `oil_change` | change-point; confirmed or not by its oil-pressure signature; resets "hours since oil change" |
| `oil_grade_change` | multiplicative re-baseline of oil pressure and filter dp |
| `sensor_replacement` | additive re-baseline of that sensor's signal (`signal` column; default oil pressure) |
| `failure` / `failure_<component>` | failure label for the supervised and survival layers |

## 6. Outputs and how to read them

`j1939pf analyze … --out out/` writes:

| File | What it is |
|---|---|
| `report.html` | self-contained report (no network): health, alerts with the data behind them, trends, what it learned, exposure, oil service, relationships, correlation matrix, anomalies, label layers |
| `summary.txt` | the same in text form |
| `alerts.json` / `alerts.csv` | one record per alert episode, plus alerts suppressed by feedback |
| `alert_data/<alert_id>.csv` | **the data behind each alert**: canonical rows with the learned expected value and z |
| `trend.csv` | per engine and signal: drift vs the early-hours reference, every 5 engine h |
| `learned_baselines.csv` | **what it learned**: per operating bin (RPM × oil temp × load), reference median, spread, n, sessions, trusted, current online value |
| `exposure.csv` | time × severity per condition |
| `metrics.json` | derived values and flags |
| `oil_service.csv`, `oil_intervals.csv` | oil changes (logged and/or detected), per-interval pressure trend and oil consumption |
| `transitions.csv` | change-points and re-baselining decisions |
| `correlations.csv`, `correlation_matrix_<engine>.csv` | learned relationships, breaks, Spearman matrix |
| `anomalies.csv`, `anomaly_rates.csv` | joint-signal anomaly episodes and rates |
| `health.json` | health score with its decomposition |
| `labels.json`, `kaplan_meier.csv` | supervised and survival layers (or why they are inactive) |
| `parse_report.json` | decode quality: malformed lines, duplicates, TP stats, per-signal status counts, unverified SPNs |

Every JSON export and report carries the framing statement. `FRAMING.txt` repeats it next to
the CSVs.

### Health score (0–100), always with its decomposition

The score is a weighted mean of component sub-scores:

* learned oil pressure and filter dp (drift and recent alerts);
* limit-only oil and coolant temperature;
* joint anomaly rate;
* broken learned relationships;
* DM1 faults;
* data quality.

Each component shows its sub-score, weight, **points lost**, and reason. The points lost add
up exactly to 100 − score. Unavailable components are listed with the reason.

**It is distance from this engine's own normal and its limits.** It is not remaining life and
not a failure probability. New engines say which parts are still in their learning period.

### Alerts

Each alert records:

* the signal and layer;
* the operating point (RPM, oil temperature, load, dominant bin);
* expected vs actual (kPa and psi), z, duration and number of runs;
* severity;
* a **falsifiable explanation**;
* the **sensor-vs-engine verdict** with its evidence;
* confounders: maintenance events, marked before or after the onset, and ambient shifts;
* **what to check next**, cheapest first: the reading (gauge, connector, DTCs), then oil
  level and grade, then the filter, then mechanical causes.

| Layer | Meaning |
|---|---|
| (a) | breach of a limit from your engine config (critical) |
| (c) | sustained deviation (\|z\| ≥ 4 for ≥ 120 s) from this engine's own learned value at that RPM × oil temp × load |
| (d) | drift across all learned bins vs the fixed early-hours reference (±4 % with a significant Theil–Sen slope): a trend, not a threshold breach |

**Sensor-vs-engine verdicts:**

* **Sensor evidence:** measurement-circuit FMIs, error encodings, an engine-off reading that
  is not ~0 kPa, or a uniform-kPa offset.
* **Engine evidence:** the deviation is confined to high RPM, pressure no longer rises with
  RPM, the deviation is proportional, or filter dp or oil level corroborate.
* **`undetermined`** whenever the evidence is missing or conflicting. A sensor drifting low
  cannot be told apart from uniform wear with one sensor.

### Exposure (time × severity; not life consumed)

Each condition reports hours, exposure in physical units, episodes, hours per 100 running
hours, the rate per 50 h period, and its definition:

* **oil pressure below this engine's own baseline**, in z-hours by RPM band. This is the
  headline question, answered as exposure;
* oil pressure below the configured minimums (kPa-hours);
* over-temperature (°C-hours);
* lugging;
* cold oil at high RPM;
* extended idling;
* high load while cold;
* hot shutdowns;
* low-temperature running after warm-up.

### Derived values (`metrics.json`)

Each value below is computed, or reported as "unavailable: why":

* idle % and the idle / working split;
* cold starts and time to temperature;
* low-temp running %, flagged **CHRONIC COLD** above 25 % after warm-up;
* load histogram and RPM × load table;
* fuel-per-power proxy drift;
* crank duration and battery-voltage sag (resolution-limited);
* torque spikes (need rows ≤ 2 s apart);
* oil consumption per oil interval;
* DM1 fault frequency and clusters;
* hours since the last oil change;
* fan-on %;
* inferred DPF regenerations.

PTO, driving and derate are reported as unavailable: there is no verified SPN for them.

Idle % = running AND `low ≤ rpm ≤ high` (inclusive) AND load `<` cutoff (strict). A row with
unknown load is not idle, and the unknown share is reported. The golden fixture
`tests/fixtures/idle_golden.log` is computed by hand in `idle_golden.md` (34 / 60 s) and must
match exactly.

### Oil-change detection and tracking

At every engine-off boundary (between sessions, or a stop ≥ 20 min), the program compares
this engine's **oil pressure at the same RPM × oil temperature × load** in the 8 engine hours
before and after. Used oil thins, so pressure sags over an interval; fresh oil steps it back
up.

* **Supporting evidence:**
  * oil level up;
  * filter dp down;
  * a shift in oil temperature vs RPM/load;
  * the ln(pressure) vs oil-temperature slope.
* **Thresholds calibrate themselves:** a step must stand out from this engine's own
  boundary-to-boundary noise (robust z ≥ 3, ≥ 1.5 %).
* **Not counted as an oil change:**
  * an engine-off zero shift → *sensor change*;
  * a step down → *oil-system step*;
  * a level-only rise → *oil level rise*.
* **Reconciliation with the log:**
  * *logged; signature confirms*;
  * *logged; no clear signature*;
  * *detected; NOT in the log*, which raises the **UNLOGGED OIL CHANGE** flag.
* Detected changes feed the alert analysis as confounders and change-points.

### Change-points and re-baselining

Grade changes and sensor replacements (logged, or detected through the zero shift) trigger a
**flagged transition**:

* The signal is not judged for 10 engine hours.
* The step is then measured per operating bin, 10 h before vs 10 h after, and applied at
  exactly the end of the transition.
* Because the step is measured locally, accumulated wear is **not** absorbed.
* Steps too large to blame on the event (> 15 % or > 30 kPa) are **refused** and stay
  visible to the alerts.
* Oil changes are recorded but not re-baselined: fresh oil restores the reference state. A
  drift that resets at an oil change is annotated "consistent with oil ageing".

### Feedback loop

```bash
j1939pf feedback dismiss --store feedback.json --alerts out/alerts.json --alert-id 4583d750aa30 --reason "gauge reads the same"
j1939pf feedback confirm --store feedback.json --alerts out/alerts.json --alert-id 29d2102d3727 --reason "pump wear at teardown"
j1939pf explain --out out/ --alert-id 4583d750aa30
```

* Decisions go to an append-only JSON file, and a reason is required.
* **Dismissed** patterns (engine | layer | signal | direction | RPM band) stay quiet. They
  are listed under "suppressed" with the reason until there is **new evidence**: the
  deviation grows ≥ 1.5×, or a new DM1 code appears for that SPN.
* **Confirmed** patterns are escalated one severity level and detected at 0.75× the normal
  threshold.

### Learned relationships (`correlations.csv`)

Each relationship reports:

* Spearman rho with a session-bootstrap 95 % CI;
* n rows and n independent sessions, blocks, periods or engines;
* the valid operating range and condition;
* the confounders;
* a status.

| Kind | Relationships |
|---|---|
| Curves (learned on the reference window, re-checked every 25 h) | oil pressure vs RPM at fixed oil temperature, coolant vs load, boost vs fuel rate, fuel pressure vs fuel rate, turbo speed vs boost, coolant pressure vs temperature, DPF dp vs fuel rate |
| Trends vs engine hours | oil-pressure drift, crankcase pressure, oil level, coolant level, DPF ash |
| Per period and across the fleet | idle % vs oil-pressure degradation, idle % vs regen frequency, low-temp % vs wear trend |

* **BROKEN** = two consecutive periods ≥ 3 reference spreads off the learned curve (flag
  **RELATIONSHIP BROKE**). *Broken (explained)* = the break coincides with a known
  change-point.
* Fewer than 8 points = **inconclusive**.
* Correlation is association in your data, not cause.

### Joint-signal anomalies

* Each signal is first turned into a **residual from its own reference median at the same
  RPM × load**, with its learned dependence on ambient temperature removed.
* Robust Mahalanobis (MCD) and an isolation forest (seeded) score the residual vector,
  learned on warmed-up reference rows.
* A row is anomalous only when **both** exceed their 99.5 % reference quantile. Episodes
  need 120 s.
* Each episode lists its top deviating signals.
* This catches combinations the oil layers cannot, such as low boost for the fuel being
  burned.
* If a signal is missing it is dropped; if scikit-learn is missing, classical Mahalanobis
  runs alone; if there is too little data, the reason is stated.

### Label-based layers (`labels.json`)

These are the **only** layers allowed to say anything about failure risk or life.

* **Supervised:**
  * Rows are engine × 50 h periods of exposure rates and drift. The label is a failure
    within 200 engine h.
  * Logistic regression with `class_weight="balanced"` (class imbalance) and **engine-grouped
    cross-validation**.
  * Reports out-of-fold ROC AUC with an engine-bootstrap CI, PR AUC vs the base rate, and the
    counts.
  * Skill is claimed only when the AUC CI lower bound is above 0.5. Probabilities are not
    calibrated and are not reported as such.
  * Needs ≥ 10 pre-failure periods from ≥ 5 failed engines.
* **Survival:**
  * Kaplan–Meier with left truncation, plus a Weibull accelerated-failure-time model by
    maximum likelihood.
  * Covariate effects are reported as **time ratios with 95 % CI**: < 1 means shorter time
    to failure per unit of exposure.
  * Remaining life is a conditional median with a parameter-uncertainty CI, and only with
    ≥ 10 failures.
* With fewer labels: **"INSUFFICIENT DATA: … No prediction made."** With none: "no failure
  labels … exposure and trend only."

## 7. How it works (by layer)

1. **Parser.**
   * 29-bit ID → priority, PGN (PDU1/PDU2), source and destination address.
   * TP.CM/TP.DT reassembly: BAM and RTS/CTS, J1939-21 timeouts, lost, duplicate and
     conflicting packets. A message is emitted only when complete.
   * Data-driven SPN decode. Not-available, error, reserved and out-of-range values →
     **NULL**, counted in `<signal>__err`.
   * DM1/DM2 decode into lamps, SPN/FMI/OC and DTC episodes.
2. **Canonical series.**
   * One row per engine per interval (default 1 s).
   * Fast signals are averaged; slow signals are held only briefly; errors are never held.
   * Clock gaps start a new segment; late samples are dropped and counted.
3. **Baselines.**
   * Per RPM (100) × oil temp (5 °C) × load (25 %) bin.
   * Learned from the first 40 eligible engine hours: running, past the startup grace, no
     active DM1, decoded OK.
   * A bin is trusted at ≥ 30 rows from ≥ 3 sessions.
   * The spread includes session-to-session variation and is floored at the bus resolution
     (4 kPa for oil pressure) and at 2.5 %.
   * The online baseline follows slowly but is capped at ±3 %, so slow faults are flagged,
     not absorbed.
4. **Layers (a), (c), (d), sensor check, exposure, oil tracking, change-points, feedback,
   correlations, anomalies, labels, health:** described above.

All tunable numbers live in `config/defaults.yaml`, grouped by section and labelled as
program defaults.

## 8. Honest limitations

* **Synthetic validation only.** Detection rates, zero-false-alert results and thresholds
  were checked on a toy engine model with several noise models and seeds, so they are not
  tuned to one scenario. They are not evidence of real-world performance.
* **The SPN table is unverified.** A wrong byte position gives plausible-looking but wrong
  numbers.
* **Exposure is not life consumed.** Nothing in the program converts exposure or deviation
  into life lost. Life statements come only from the label layers, with n and CIs, and only
  with enough failures. A handful of failures is not enough, and the program says so.
* **Sensor vs wear is sometimes undecidable.** A pressure sensor drifting *low* looks like
  uniform wear (readings clip at 0 kPa, so the zero check cannot see it). The verdict is
  then `undetermined`, and the first check is always the mechanical gauge.
* **Oil ageing uses part of the trend margin.** Normal oil ageing makes healthy engines
  drift a few % within each oil interval, and the 4 % trend threshold allows for it. Very
  long intervals narrow the margin.
* **Cold oil hides oil changes.** With chronically cold oil, pressure sits on the relief
  valve and a fresh-oil step can be invisible; such changes show only as oil-level rises.
* **Inferred regens.** DPF regenerations are inferred from DPF dp drops (no verified
  regen-status SPN). Idle-heavy duty is over-counted by ~35 % on synthetic data.
* **The joint anomaly layer is supplementary.** It finds faults the oil layers miss (charge-
  air leak), but can miss some the oil layers catch strongly (one stuck-valve seed in
  testing). Unusual ambient conditions outside the reference range can raise its rate.
* **Coarse row intervals limit some metrics.** At 10 s rows, crank duration, voltage sag and
  torque spikes are resolution-limited or unavailable, and the outputs say so.
* **CSV inputs have no DM1 frames** unless the logger exports DM1 columns.
* **Not built:** live socketCAN input (bonus).
* **Out of scope by design:** off-bus data (oil analysis, vibration, weather, inspections).
* **Correlations are associations.** Fleet-level relationships need many engines before they
  mean anything.

## 9. Project layout

```
src/j1939pf/   ids, spn_table, transport, dm, candump, csv_source, decoder, canonical, pipeline   (parse)
               engine_config, baseline, detection, alerts, sensor_check, analysis                 (layers a-d)
               derived, exposure, oil_change, usage                                               (exposure)
               feedback, correlation                                                              (phase 4)
               anomaly, labels, health, report, html_report, io, cli                              (phase 5)
               generator, encode                                                                  (synthetic data)
config/        spn_table.yaml, defaults.yaml, csv_mapping.example.yaml, engines/
tests/         one file per phase + test_acceptance.py (maps every acceptance item to its test)
```
