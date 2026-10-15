"""Phase 4 gate: change-points / re-baselining, feedback loop, correlation discovery."""

import json

import numpy as np
import pandas as pd
import pytest

from j1939pf.analysis import analyze
from j1939pf.correlation import correlation_report
from j1939pf.feedback import FeedbackStore
from j1939pf.generator import generate_engine, generate_fleet
from j1939pf.usage import usage_report
from synth import synthetic_config, usage_config


def run(scenario="healthy", seed=0, **kw):
    e = generate_engine("SA00", scenario, hours=300, seed=seed, **kw)
    return e, usage_report(e.canonical, {"SA00": usage_config(e)}, events=e.events)


# ---------------------------------------------------------------- change-points: no alert flood
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kw,mode,truth", [
    (dict(sensor_replacement_at_h=120, sensor_offset_step_kpa=12), "add", 12.0),
    (dict(sensor_replacement_at_h=120, sensor_offset_step_kpa=-12), "add", -12.0),
    (dict(grade_change_at_h=120, grade_factor=0.9), "mul", -0.10),
    (dict(grade_change_at_h=120, grade_factor=1.08), "mul", 0.08),
])
def test_no_alert_flood_after_sensor_replacement_or_grade_change(seed, kw, mode, truth):
    e, rep = run(seed=seed, oil_change_interval_h=100.0, **kw)
    assert rep.analysis.alerts == [], [a.headline() for a in rep.analysis.alerts]
    tr = rep.analysis.transitions
    adj = tr[(tr["signal"] == "oil_pressure_kpa") & (tr["mode"] == mode)]
    assert len(adj) == 1 and bool(adj["accepted"].iloc[0])
    step = float(adj["step"].iloc[0])
    assert step == pytest.approx(truth, abs=2.5 if mode == "add" else 0.015)
    # oil changes are recorded as transitions too (no re-baseline: fresh oil restores the reference)
    assert (tr["kind"] == "oil_change").sum() >= 2
    assert any("re-baselined" in t for t in rep.analysis.status["SA00"]["transitions"])


def test_oil_changes_alone_cause_no_alerts():
    for seed in (0, 1):
        _, rep = run(seed=seed, oil_change_interval_h=60.0)
        assert rep.analysis.alerts == []


def test_step_too_large_for_the_event_is_not_absorbed():
    e, rep = run(seed=0, grade_change_at_h=120, grade_factor=0.7)
    tr = rep.analysis.transitions
    adj = tr[(tr["signal"] == "oil_pressure_kpa") & (tr["kind"] == "oil_grade_change")].iloc[0]
    assert adj["accepted"] is False or adj["accepted"] == False  # noqa: E712
    assert "NOT re-baselined" in adj["note"] and "larger than" in adj["note"]
    assert any(a.signal == "oil_pressure_kpa" and a.direction == "low" for a in rep.analysis.alerts)


def test_wear_is_not_absorbed_by_a_change_point():
    e = generate_engine("SA00", "bearing_wear", hours=300, seed=0, grade_change_at_h=150, grade_factor=0.93)
    rep = usage_report(e.canonical, {"SA00": usage_config(e)}, events=e.events)
    d = [a for a in rep.analysis.alerts if a.layer == "d" and a.signal == "oil_pressure_kpa"]
    assert d and d[0].direction == "low"
    adj = rep.analysis.transitions
    assert bool(adj[(adj["signal"] == "oil_pressure_kpa") & (adj["kind"] == "oil_grade_change")]["accepted"].iloc[0])
    # the step is measured locally (just before vs just after), so wear accumulated so far survives:
    # drift keeps falling after the grade change
    tr = rep.analysis.baselines[("SA00", "oil_pressure_kpa")].trend
    assert tr["drift_rel"].iloc[-3:].median() < -0.10


def test_change_point_inside_learning_window_restarts_learning():
    e = generate_engine("SA00", "healthy", hours=200, seed=1, grade_change_at_h=15, grade_factor=0.9)
    r = analyze(e.canonical, {"SA00": synthetic_config(e)}, events=e.events)
    sb = r.baselines[("SA00", "oil_pressure_kpa")]
    gc = float(e.events.loc[e.events.event_type == "oil_grade_change", "engine_hours"].iloc[0])
    assert sb.learn_start_h >= gc and "learning restarted after oil grade change" in sb.message
    assert r.alerts == []


def test_unlogged_sensor_change_is_detected_and_rebaselined():
    e = generate_engine("SA00", "healthy", hours=300, seed=2, sensor_replacement_at_h=120, sensor_offset_step_kpa=14)
    rep = usage_report(e.canonical, {"SA00": usage_config(e)}, events=None)   # nothing logged
    tr = rep.analysis.transitions
    adj = tr[(tr["kind"] == "sensor_change") & (tr["signal"] == "oil_pressure_kpa")]
    assert len(adj) == 1 and bool(adj["accepted"].iloc[0]) and "detected, unlogged" in adj["label"].iloc[0]
    assert not [a for a in rep.analysis.alerts if a.signal == "oil_pressure_kpa"]


# ---------------------------------------------------------------- feedback loop
@pytest.fixture
def pump():
    e = generate_engine("SA00", "failing_pump", hours=300, seed=0)
    return e, {"SA00": synthetic_config(e)}


def _c_low(r):
    return [a for a in r.alerts if a.layer == "c" and a.direction == "low" and a.signal == "oil_pressure_kpa"]


def test_dismissed_alert_stays_quiet_and_is_persisted(tmp_path, pump):
    e, cfg = pump
    store = FeedbackStore(tmp_path / "fb.json")
    r1 = analyze(e.canonical, cfg, feedback=store)
    target = _c_low(r1)[0]
    store.dismiss(target, "mechanical gauge reads the same; pump known weak, replacement booked", at="2024-03-01T00:00:00Z")
    # persisted, append-only, with reason
    doc = json.loads((tmp_path / "fb.json").read_text())
    assert doc["records"][0]["action"] == "dismiss" and "gauge" in doc["records"][0]["reason"]
    # a fresh store loaded from disk silences the same pattern on the next run
    r2 = analyze(e.canonical, cfg, feedback=FeedbackStore(tmp_path / "fb.json"))
    assert target.alert_id not in {a.alert_id for a in r2.alerts}
    assert any(a.pattern_key == target.pattern_key and "no new evidence" in why for a, why in r2.suppressed)
    assert "suppressed by feedback" in r2.summary_text()
    # other patterns are unaffected
    assert {a.pattern_key for a in r1.alerts} - {target.pattern_key} == {a.pattern_key for a in r2.alerts}


def test_dismissed_pattern_refires_only_on_new_evidence(tmp_path, pump):
    e, cfg = pump
    df = e.canonical
    early = df[df["engine_hours_h"] < df["engine_hours_h"].iloc[0] + 225]
    store = FeedbackStore(tmp_path / "fb.json")
    first = _c_low(analyze(early, cfg))[0]
    store.dismiss(first, "nuisance at the time")
    # same data again: quiet
    assert not [a for a in _c_low(analyze(early, cfg, feedback=store)) if a.pattern_key == first.pattern_key]
    # the pump keeps failing: the deviation grows past 1.5x the dismissed size -> re-raised, with the reason
    later = [a for a in _c_low(analyze(df, cfg, feedback=store)) if a.pattern_key == first.pattern_key]
    from j1939pf.feedback import magnitude
    assert later and magnitude(later[0]) >= 1.5 * magnitude(first)
    assert "previously DISMISSED" in later[0].feedback_note and "new evidence" in later[0].feedback_note
    # a new DM1 code for SPN 100 after the dismissal is also new evidence
    dtc = pd.DataFrame([{"engine_id": "SA00", "source_address": 0, "kind": "DM1", "spn": 100, "fmi": 1,
                         "first_t": early["t"].iloc[-1], "last_t": early["t"].iloc[-1], "max_oc": 1, "messages": 1}])
    store2 = FeedbackStore(tmp_path / "fb2.json")
    store2.dismiss(first, "nuisance")
    again = [a for a in _c_low(analyze(early, cfg, feedback=store2, dtc_episodes=dtc)) if a.pattern_key == first.pattern_key]
    assert again and "new DM1 code for SPN 100" in again[0].feedback_note


def test_confirmed_fault_strengthens_the_pattern(tmp_path, pump):
    e, cfg = pump
    base = analyze(e.canonical, cfg)
    a = _c_low(base)[0]
    store = FeedbackStore(tmp_path / "fb.json")
    store.confirm(a, "pump wear confirmed at teardown")
    r = analyze(e.canonical, cfg, feedback=store)
    same = [x for x in _c_low(r) if x.pattern_key == a.pattern_key]
    assert same and "CONFIRMED" in same[0].feedback_note
    order = ["watch", "warning", "critical"]
    assert order.index(same[0].severity) >= min(order.index(a.severity) + 1, 2)
    # detection threshold lowered for the confirmed pattern: the episode starts no later, and covers more time
    assert same[0].start_h <= a.start_h and same[0].duration_s >= a.duration_s


def test_feedback_requires_reason_and_valid_action(tmp_path, pump):
    e, cfg = pump
    a = _c_low(analyze(e.canonical, cfg))[0]
    store = FeedbackStore(tmp_path / "fb.json")
    with pytest.raises(ValueError, match="reason"):
        store.dismiss(a, "  ")
    with pytest.raises(ValueError, match="action"):
        store.record(a, "ignore", "x")


# ---------------------------------------------------------------- correlations
NEWER = {"boost_vs_fuel_rate", "fuel_pressure_vs_fuel_rate", "turbo_speed_vs_boost", "coolant_pressure_vs_coolant_temp",
         "dpf_dp_vs_fuel_rate", "crankcase_pressure_vs_hours", "oil_level_vs_hours", "coolant_level_vs_hours",
         "dpf_ash_vs_hours"}


def _corr(scenario, seed=0, **kw):
    e, rep = run(scenario, seed, **kw)
    return e, rep, correlation_report(rep.analysis, {"SA00": usage_config(e)}, events=rep.events_used)


def test_correlation_report_complete_with_newer_signals():
    e, rep, c = _corr("healthy")
    f = c.frame()
    names = set(f["name"])
    assert NEWER <= names and {"oil_pressure_vs_rpm", "coolant_temp_vs_load", "oil_pressure_drift_vs_hours",
                               "idle_pct_vs_oil_pressure_degradation", "idle_pct_vs_regen_frequency",
                               "low_temp_pct_vs_wear_trend"} <= names
    eng = f[f["engine_id"] == "SA00"]
    for _, r in eng[eng["kind"].isin(["curve", "trend"])].iterrows():
        assert r["available"], (r["name"], r["reason"])
        assert r["rho"] is not None and r["n_rows"] > 0 and r["n_independent"] >= 3
        assert r["valid_range"] and r["confounders"] and r["status"]
    curves = eng[eng["kind"] == "curve"].set_index("name")
    assert curves.loc["oil_pressure_vs_rpm", "rho"] > 0.8 and "oil temperature" in curves.loc["oil_pressure_vs_rpm", "condition"]
    assert curves.loc["oil_pressure_vs_rpm", "rho_ci"][0] > 0.5
    assert not c.broken() and c.flags() == {}
    # small samples are labelled, never over-claimed
    per = eng[eng["kind"] == "period"]
    assert all(("inconclusive" in s) or (n >= 8) for s, n in zip(per["status"], per["n_independent"]))
    assert c.summary_text().startswith(rep.analysis.summary_text().split("\n")[0])


def test_learned_relationship_break_is_flagged():
    e, rep, c = _corr("boost_leak")
    broken = {r.name: r for r in c.broken()}
    assert set(broken) == {"boost_vs_fuel_rate"}
    r = broken["boost_vs_fuel_rate"]
    start = float(r.breaks[0].split("from ")[1].split(" h")[0])
    assert abs(start - e.truth["onset_engine_hours"]) <= 30
    assert "-1" in r.breaks[0]                                  # boost below the learned curve (~ -20 %)
    assert "RELATIONSHIP BROKE: boost_vs_fuel_rate" in c.flags()["SA00"][0]


def test_break_at_known_change_point_is_explained():
    e, rep, c = _corr("healthy", 1, grade_change_at_h=120, grade_factor=0.85)
    r = next(x for x in c.relationships if x.name == "oil_pressure_vs_rpm" and x.engine_id == "SA00")
    assert r.status == "broken (explained)" and "oil_grade_change" in r.breaks[0]
    assert not c.broken()


def test_fleet_relationships_and_regen_inference():
    specs = [{"engine_id": f"SA{i:02X}", "scenario": sc, "hours": 300}
             for i, sc in enumerate(["healthy", "high_idle", "abusive", "healthy", "high_idle", "cold_operation"])]
    df, ev, eng = generate_fleet(specs, seed=5)
    cfgs = {k: usage_config(v) for k, v in eng.items()}
    rep = usage_report(df, cfgs, events=ev)
    c = correlation_report(rep.analysis, cfgs, events=rep.events_used)
    fleet = {r.name: r for r in c.relationships if r.engine_id == "FLEET"}
    r = fleet["idle_pct_vs_regen_frequency"]
    assert r.available and r.n_independent == 6 and r.rho > 0.5
    assert "inconclusive" in r.status                           # 6 engines < 8: reported, not concluded
    # inferred regens track the generator's truth on normal duty
    for k in ("SA00", "SA03"):
        m = rep.engines[k].metrics["dpf_regens"]
        assert abs(m["inferred_regens"] - len(eng[k].truth["regen_engine_hours"])) <= 1 and "INFERRED" in m["note"]
