"""Phase 2 gate: per-engine baselines, layers (a)-(d), learning period, sensor-vs-wear."""

import re

import numpy as np
import pandas as pd
import pytest

from j1939pf.analysis import analyze
from j1939pf.framing import HONEST_FRAMING
from j1939pf.generator import generate_engine, generate_fleet
from synth import analyzed, engine, synthetic_config


# ---------------------------------------------------------------- healthy: zero alerts
@pytest.mark.parametrize("scenario", ["healthy", "high_idle", "cold_operation"])
@pytest.mark.parametrize("noise_model", ["gaussian", "student_t", "ar1"])
@pytest.mark.parametrize("seed", [0, 1])
def test_zero_alerts_on_healthy_engines(scenario, noise_model, seed):
    e, r = analyzed(scenario, seed, noise_model)
    assert r.alerts == [], [a.headline() for a in r.alerts]
    assert r.status["SA00"]["learned"]["oil_pressure_kpa"]["status"] == "active"


# ---------------------------------------------------------------- low pressure at high RPM
@pytest.mark.parametrize("seed", [0, 1, 3])
def test_stuck_relief_valve_low_pressure_at_high_rpm(seed):
    e, r = analyzed("stuck_relief_valve", seed)
    c = [a for a in r.alerts if a.layer == "c" and a.signal == "oil_pressure_kpa" and a.direction == "low"]
    assert c, "layer (c) must flag low oil pressure"
    a = max(c, key=lambda x: x.duration_s)
    op = a.operating_point
    # operating point reported: upper RPM range, reaching rated speed (the stuck valve caps pressure there)
    assert op["rpm"] > (e.profile.idle_rpm + e.profile.rated_rpm) / 2
    assert op["rpm_range"][1] >= e.profile.rated_rpm - 150
    assert "rpm" in op["label"] and "oil" in op["label"]
    assert a.actual < a.expected and a.z <= -8
    assert abs(a.start_h - e.truth["onset_engine_hours"]) < 5
    assert "Falsifiable" in a.explanation and "learned" in a.explanation
    assert a.sensor_verdict == "engine_suspected"
    assert any(ev.startswith("ENGINE: deviation") for ev in a.sensor_evidence)
    assert "gauge" in a.check_next[0] and "level" in a.check_next[1]
    assert any("relief" in s for s in a.check_next)
    # layer (a): placeholder rated-speed minimum breached too, and labelled as a placeholder
    la = [x for x in r.alerts if x.layer == "a"]
    assert la and all("PLACEHOLDER" in x.limit for x in la)


# ---------------------------------------------------------------- slow trend, flagged before threshold
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_bearing_wear_drift_flagged_before_threshold(seed):
    e, r = analyzed("bearing_wear", seed)
    cfg = synthetic_config(e)
    d = [a for a in r.alerts if a.layer == "d" and a.signal == "oil_pressure_kpa" and a.direction == "low"]
    assert d, "slow downward trend must raise a layer (d) drift alert"
    drift = d[0]
    sb = r.baselines[("SA00", "oil_pressure_kpa")]
    # drift is measured against a FIXED reference window of early healthy hours
    assert sb.learn_end_h - sb.learn_start_h == pytest.approx(40.0)
    assert sb.learn_end_h <= e.truth["onset_engine_hours"]
    assert drift.start_h > e.truth["onset_engine_hours"]
    assert drift.operating_point["drift_pct_at_first_detection"] <= -4.0
    # "before threshold": no manual-threshold alert precedes it ...
    assert all(a.start_h > drift.start_h for a in r.alerts if a.layer == "a")
    # ... and when it fires, the engine's oil pressure is still above the configured minimums
    df = r.prepared["SA00"].df
    win = df["_judgeable"] & (df["_hours"] > drift.start_h - 10) & (df["_hours"] <= drift.start_h)
    p = df.loc[win, "oil_pressure_kpa"].dropna()
    assert p.quantile(0.01) > cfg.limit("oil_pressure_min_idle").value
    rated = win & (df["engine_speed_rpm"] >= cfg.limit("rated_rpm").value - 100)
    assert df.loc[rated, "oil_pressure_kpa"].dropna().quantile(0.01) > cfg.limit("oil_pressure_min_rated").value
    assert "not a threshold breach" in drift.explanation


def test_slow_fault_is_not_absorbed_drift_cap():
    e, r = analyzed("bearing_wear", 0)
    sb = r.baselines[("SA00", "oil_pressure_kpa")]
    ref = sb.reference.loc[sb.online.index, "median"]
    assert ((sb.online - ref).abs() <= 0.03 * ref + 1e-9).all()
    assert len(sb.capped) > 0
    assert r.status["SA00"]["learned"]["oil_pressure_kpa"]["capped_bins"] == len(sb.capped)
    d = [a for a in r.alerts if a.layer == "d"][0]
    assert "drift cap" in d.explanation and "not absorbed" in d.explanation
    # the late-hours z-scores keep seeing the fault instead of a baseline that followed it down
    df = r.prepared["SA00"].df
    late = df["_hours"] > df["_hours"].max() - 30
    assert np.nanmedian(sb.z[late.to_numpy()]) < -2


@pytest.mark.parametrize("scenario", ["failing_pump", "clogged_filter"])
def test_other_oil_faults_detected(scenario):
    e, r = analyzed(scenario, 0)
    lows = [a for a in r.alerts if a.signal == "oil_pressure_kpa" and a.direction == "low"]
    assert lows and min(a.start_h for a in lows) > e.truth["onset_engine_hours"]
    if scenario == "clogged_filter":
        assert any(a.signal == "oil_filter_dp_kpa" and a.direction == "high" for a in r.alerts)


# ---------------------------------------------------------------- sensor vs wear
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sensor_drift_is_attributed_to_the_sensor(seed):
    e, r = analyzed("sensor_drift", seed)
    d = [a for a in r.alerts if a.layer == "d" and a.signal == "oil_pressure_kpa"]
    assert d and d[0].direction == "high"
    assert d[0].sensor_verdict == "sensor_suspected"
    assert any("engine-off zero check" in ev and ev.startswith("SENSOR") for ev in d[0].sensor_evidence)
    assert d[0].check_next[0].startswith("FIRST (sensor evidence present)")


@pytest.mark.parametrize("scenario", ["bearing_wear", "stuck_relief_valve", "failing_pump"])
def test_engine_faults_are_not_blamed_on_the_sensor(scenario):
    _, r = analyzed(scenario, 0)
    assert r.alerts and all(a.sensor_verdict != "sensor_suspected" for a in r.alerts)


def test_dm1_sensor_fmi_is_sensor_evidence_and_low_fmi_is_not_independent():
    e = engine("stuck_relief_valve", 0)
    t0 = e.canonical["t"].iloc[0]
    t1 = e.canonical["t"].iloc[-1]
    base = {"engine_id": "SA00", "source_address": 0, "kind": "DM1", "spn": 100, "first_t": t0, "last_t": t1,
            "max_oc": 5, "messages": 100}
    dtcs = pd.DataFrame([dict(base, fmi=3), dict(base, fmi=1)])
    r = analyze(e.canonical, {"SA00": synthetic_config(e)}, dtc_episodes=dtcs)
    a = [x for x in r.alerts if x.layer == "c"][0]
    assert any("FMI 3" in ev and ev.startswith("SENSOR") and "measurement-circuit" in ev for ev in a.sensor_evidence)
    assert any("FMI 1" in ev and "not independent" in ev for ev in a.sensor_evidence)
    assert a.sensor_verdict != "engine_suspected"  # conflicting evidence is not resolved by guessing


def test_error_encodings_are_sensor_evidence():
    e = engine("stuck_relief_valve", 0)
    df = e.canonical.copy()
    rng = np.random.default_rng(0)
    bad = (df["engine_speed_rpm"] > 0).to_numpy() & (rng.random(len(df)) < 0.05)
    df["oil_pressure_kpa__err"] = bad.astype(int)
    df.loc[bad, "oil_pressure_kpa"] = np.nan
    r = analyze(df, {"SA00": synthetic_config(e)})
    a = [x for x in r.alerts if x.layer == "c"][0]
    assert any("error/out-of-range encodings" in ev for ev in a.sensor_evidence)


# ---------------------------------------------------------------- learning guardrails
def test_learning_excludes_active_dm1_rows():
    e = engine("healthy", 0)
    df = e.canonical.copy()
    h = df["engine_hours_h"] - df["engine_hours_h"].iloc[0]
    hit = (h > 1) & (h < 39) & (df["engine_speed_rpm"] > 0) & (np.arange(len(df)) % 5 != 0)
    df.loc[hit, "oil_pressure_kpa"] = df.loc[hit, "oil_pressure_kpa"] * 0.5   # nonsense while a fault is active
    df.loc[hit, "dm1_active_count"] = 1.0
    clean = analyze(e.canonical).baselines[("SA00", "oil_pressure_kpa")].reference
    guarded = analyze(df).baselines[("SA00", "oil_pressure_kpa")].reference
    unguarded = analyze(df.drop(columns=["dm1_active_count"])).baselines[("SA00", "oil_pressure_kpa")].reference
    common = clean.index[clean.trusted].intersection(guarded.index[guarded.trusted]).intersection(unguarded.index)
    d_guard = (guarded.loc[common, "median"] - clean.loc[common, "median"]).abs().median()
    d_unguard = (unguarded.loc[common, "median"] - clean.loc[common, "median"]).abs().median()
    assert d_guard <= 4.0 < d_unguard
    assert "DM1 not broadcast" in " ".join(analyze(df.drop(columns=["dm1_active_count"])).status["SA00"]["notes"])


def test_new_engine_announces_learning_period_and_judges_only_layer_a():
    e = generate_engine("SA07", "stuck_relief_valve", hours=25, seed=9, fault={"onset_h": 5.0})
    r = analyze(e.canonical, {"SA07": synthetic_config(e)})
    st = r.status["SA07"]
    assert st["learned"]["oil_pressure_kpa"]["status"] == "learning"
    assert "LEARNING PERIOD" in st["learning_announcement"] and "of 40 engine hours" in st["learning_announcement"]
    assert "Layer (a)" in st["learning_announcement"]
    assert {a.layer for a in r.alerts} == {"a"}   # the fault is still caught by the manual threshold
    assert "LEARNING PERIOD" in r.summary_text()


def test_baselines_are_per_engine_and_never_copied():
    df, ev, eng = generate_fleet([{"engine_id": "SA00", "scenario": "healthy", "hours": 120},
                                  {"engine_id": "SA01", "scenario": "healthy", "hours": 120},
                                  {"engine_id": "SA02", "scenario": "healthy", "hours": 10}], seed=3)
    r = analyze(df)
    a, b = r.baselines[("SA00", "oil_pressure_kpa")], r.baselines[("SA01", "oil_pressure_kpa")]
    common = a.reference.index.intersection(b.reference.index)
    assert len(common) > 5
    assert (a.reference.loc[common, "median"] - b.reference.loc[common, "median"]).abs().median() > 8
    # a short-history engine is still learning even though its neighbours have full baselines
    assert r.status["SA02"]["learned"]["oil_pressure_kpa"]["status"] == "learning"
    # each engine's model is identical whether analysed alone or with the fleet
    alone = analyze(df[df.engine_id == "SA00"]).baselines[("SA00", "oil_pressure_kpa")]
    pd.testing.assert_frame_equal(alone.reference, a.reference)
    assert a.reference is not b.reference


# ---------------------------------------------------------------- config handling / honesty
def test_missing_limits_disable_layer_a_only():
    e, _ = analyzed("stuck_relief_valve", 0)
    r = analyze(e.canonical, {"SA00": synthetic_config(e, rated_rpm=None, coolant_temp_max=None)})
    la = r.status["SA00"]["layer_a"]
    assert la["oil_pressure_min_rated"].startswith("disabled: rated_rpm missing")
    assert la["coolant_temp_max"].startswith("disabled")
    assert la["oil_pressure_min_idle"].startswith("active")
    assert any(a.layer == "c" for a in r.alerts)
    r2 = analyze(e.canonical)                       # no config at all
    assert all(v.startswith("disabled: no engine config") for v in r2.status["SA00"]["layer_a"].values())
    assert any(a.layer == "c" for a in r2.alerts) and not any(a.layer == "a" for a in r2.alerts)


def test_alert_records_and_report_are_honest():
    _, r = analyzed("clogged_filter", 0)
    text = r.summary_text()
    assert text.startswith(HONEST_FRAMING)
    for a in r.alerts:
        d = a.to_dict()
        for k in ("signal", "operating_point", "expected", "actual", "duration_s", "severity", "explanation",
                  "check_next", "sensor_verdict"):
            assert k in d
        assert a.check_next and a.explanation
        assert not re.search(r"life|remaining useful|% of life", a.explanation + " ".join(a.check_next), re.I)
    a = r.alerts[0]
    rows = r.alert_data(a)
    assert len(rows) > 0 and "oil_pressure_kpa" in rows


def test_end_to_end_candump_to_alert(tmp_path):
    """Generator -> candump text -> parser -> canonical -> analysis (shortened learning period)."""
    from j1939pf.config import load_defaults
    from j1939pf.generator import render_candump
    from j1939pf.pipeline import canonical_from_candump
    e = generate_engine("SA00", "stuck_relief_valve", hours=45, seed=1, fault={"onset_h": 25.0})
    log = tmp_path / "bus.log"
    render_candump(e.canonical, log)
    params = load_defaults(overrides={"canonical": {"resample_interval_s": 10, "gap_s": 60},
                                      "baseline": {"learning_period_h": 15}})
    parsed = canonical_from_candump(log, config=params)
    r = analyze(parsed.canonical, {"SA00": synthetic_config(e)}, dtc_episodes=parsed.dtc_episodes, params=params)
    c = [a for a in r.alerts if a.layer == "c" and a.direction == "low"]
    assert c and c[0].start_h >= e.truth["onset_engine_hours"] - 0.1
    assert not [a for a in r.alerts if a.start_h < e.truth["onset_engine_hours"] - 0.1]
