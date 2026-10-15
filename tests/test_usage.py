"""Phase 3 gate: derived values, exposure, flags, oil-change tracking."""

import re
from pathlib import Path

import numpy as np
import pytest

from j1939pf.engine_config import load_engine_config
from j1939pf.framing import HONEST_FRAMING
from j1939pf.generator import generate_engine
from j1939pf.pipeline import canonical_from_candump
from j1939pf.usage import usage_report
from synth import engine, usage, usage_config

FIX = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------- golden idle fixture
def test_idle_pct_matches_hand_computed_golden_fixture():
    parsed = canonical_from_candump(FIX / "idle_golden.log")
    assert parsed.report["decode"]["duplicate_frames"] == 1
    cfg = load_engine_config(FIX / "idle_golden_engine.yaml")
    rep = usage_report(parsed.canonical, {"SA00": cfg})
    idle = rep.engines["SA00"].metrics["idle"]
    assert idle["running_s"] == 60.0
    assert idle["idle_s"] == 34.0
    assert idle["idle_pct"] == 100.0 * 34 / 60          # 56.666...%, see fixtures/idle_golden.md
    assert idle["load_unknown_running_pct"] == 100.0 * 1 / 60
    assert idle["flagged"] is False and idle["flag_threshold_pct"] is None


def test_idle_unavailable_without_manual_values():
    e = engine("healthy", 0)
    rep = usage_report(e.canonical, {"SA00": usage_config(e, idle_load_cutoff=None)}, events=e.events)
    m = rep.engines["SA00"].metrics
    assert m["idle"] == {"available": False, "reason": "idle_load_cutoff missing from engine config"}
    rep2 = usage_report(e.canonical, events=e.events)
    assert "no engine config" in rep2.engines["SA00"].metrics["idle"]["reason"]
    ex = rep2.engines["SA00"].exposure.set_index("condition")
    assert not ex.loc["lugging", "available"] and "lugging_rpm_max" in ex.loc["lugging", "reason"]


# ---------------------------------------------------------------- flags
@pytest.mark.parametrize("seed,noise", [(0, "gaussian"), (1, "student_t"), (2, "ar1")])
def test_cold_operation_is_flagged(seed, noise):
    e, rep = usage("cold_operation", seed, noise)
    flags = rep.engines["SA00"].flags
    assert any(f.startswith("CHRONIC COLD OPERATION (coolant)") for f in flags)
    lt = rep.engines["SA00"].metrics["low_temp_running_coolant"]
    assert lt["low_temp_running_pct_after_warmup"] > 90
    ex = rep.engines["SA00"].exposure.set_index("condition")
    assert ex.loc["low_temp_running_after_warmup", "hours"] > 100


@pytest.mark.parametrize("scenario", ["healthy", "high_idle", "abusive"])
@pytest.mark.parametrize("seed,noise", [(0, "gaussian"), (1, "student_t"), (2, "ar1")])
def test_no_cold_flags_on_warm_engines(scenario, seed, noise):
    e, rep = usage(scenario, seed, noise)
    u = rep.engines["SA00"]
    assert not [f for f in u.flags if "COLD" in f], u.flags
    assert u.metrics["low_temp_running_coolant"]["low_temp_running_pct_after_warmup"] < 5
    if scenario == "healthy":
        assert u.flags == [], u.flags      # warm healthy engine with a complete maintenance log: no flags
        assert rep.analysis.alerts == []


def test_high_idle_flag_uses_configured_threshold():
    _, hi = usage("high_idle", 0, "gaussian")
    _, ok = usage("healthy", 0, "gaussian")
    assert any(f.startswith("HIGH IDLE") for f in hi.engines["SA00"].flags)
    assert hi.engines["SA00"].metrics["idle"]["idle_pct"] > 45 > ok.engines["SA00"].metrics["idle"]["idle_pct"]


# ---------------------------------------------------------------- exposure
def test_abusive_use_shows_up_as_exposure_not_life_loss():
    _, ab = usage("abusive", 0, "gaussian")
    _, ok = usage("healthy", 0, "gaussian")
    a = ab.engines["SA00"].exposure.set_index("condition")
    h = ok.engines["SA00"].exposure.set_index("condition")
    assert a.loc["lugging", "hours"] > 10 and h.loc["lugging", "hours"] == 0
    assert a.loc["hot_shutdown", "exposure"] > 20 and h.loc["hot_shutdown", "exposure"] == 0
    assert a.loc["high_load_while_cold", "exposure"] > 3 * h.loc["high_load_while_cold", "exposure"]
    for cond in ("lugging", "cold_oil_high_rpm", "high_load_while_cold"):
        assert a.loc[cond, "unit"] and a.loc[cond, "definition"] and isinstance(a.loc[cond, "by_period"], dict)
    text = ab.summary_text()
    assert text.startswith(HONEST_FRAMING) and "NOT life consumed" in text
    body = text[len(HONEST_FRAMING):].replace("NOT life consumed; no formula here converts exposure into life lost", "")
    assert not re.search(r"life (lost|consumed|remaining)|% of life|remaining useful", body, re.I)


def test_oil_pressure_exposure_by_rpm_band_grows_with_a_fault():
    e = generate_engine("SA00", "stuck_relief_valve", hours=300, seed=0)
    rep = usage_report(e.canonical, {"SA00": usage_config(e)}, events=e.events)
    ex = rep.engines["SA00"].exposure.set_index("condition")
    _, ok = usage("healthy", 0)
    assert ex.loc["oil_pressure_below_own_baseline", "exposure"] > 20 * ok.engines["SA00"].exposure.set_index(
        "condition").loc["oil_pressure_below_own_baseline", "exposure"]
    bands = ex.loc["oil_pressure_below_own_baseline", "detail"]["by_rpm_band"]
    top = max(bands, key=lambda b: bands[b]["z_hours"])
    assert int(top.split("-")[0]) >= 1000            # concentrated at higher RPM
    assert ex.loc["oil_pressure_below_min_rated", "hours"] > 0


def test_derived_metrics_healthy_values():
    _, rep = usage("healthy", 0, "gaussian")
    m = rep.engines["SA00"].metrics
    assert 10 < m["idle"]["idle_pct"] < 40
    assert m["cold_starts_coolant"]["cold_starts"] > 0 and m["cold_starts_coolant"]["time_to_temp_min_median"] < 30
    assert abs(sum(m["load_histogram"]["pct_running_time"].values()) - 100) < 1e-6
    assert abs(m["fuel_per_power"]["drift_pct_latest"]) < 3
    assert m["cranking"]["voltage_sag_v_median"] > 1.0 and "resolved only" in m["cranking"]["resolution_note"]
    assert not m["torque_spikes"]["available"] and "too coarse" in m["torque_spikes"]["reason"]
    assert not m["derate"]["available"] and not m["fan_on"]["available"]
    cons = m["oil_consumption"]["per_interval"]
    assert cons and all(-2.0 < r["oil_level_trend_pct_per_100h"] < -0.5 for r in cons)  # generator: -1.2 %/100 h


# ---------------------------------------------------------------- oil change detection / tracking
@pytest.mark.parametrize("seed,noise", [(0, "gaussian"), (1, "student_t"), (2, "ar1"), (3, "gaussian")])
def test_oil_changes_detected_from_pressure_vs_rpm_and_temp(seed, noise):
    e, rep = usage("healthy", seed, noise, with_log=False, oil_interval=100.0)
    oil = rep.engines["SA00"].oil
    h = oil.history
    det = h[h["event"] == "oil_change"]
    truth = e.truth["oil_change_engine_hours"]
    assert len(truth) == 2
    # detected hours come from SPN 247 (0.05 h/bit); the generator's truth is unquantized
    assert np.allclose(sorted(det["engine_hours"]), sorted(truth), atol=0.06)
    assert (det["source"] == "detected").all() and (det["pressure_step_pct"] > 2).all()
    assert len(h) == len(det)                         # nothing else detected
    assert oil.hours_since_change == pytest.approx(oil.hours_now - max(truth), abs=0.06)
    assert any(f.startswith("UNLOGGED OIL CHANGE") for f in rep.engines["SA00"].flags)
    # detected-but-unlogged changes are handed to the analysis as confounders
    ev = rep.events_used
    assert (ev["detail"].str.contains("detected from J1939 signature")).sum() == 2


def test_logged_oil_changes_are_confirmed_and_tracked():
    e, rep = usage("healthy", 1, "student_t", with_log=True, oil_interval=100.0)
    h = rep.engines["SA00"].oil.history
    assert list(h["source"]) == ["logged+detected", "logged+detected"]
    assert all(s == "logged; signature confirms" for s in h["status"])
    assert h["interval_h"].iloc[1] == pytest.approx(e.truth["oil_change_engine_hours"][1]
                                                    - e.truth["oil_change_engine_hours"][0], abs=0.1)
    assert not [f for f in rep.engines["SA00"].flags if "UNLOGGED" in f]
    iv = rep.engines["SA00"].oil.intervals
    assert len(iv) == 3 and "pressure_trend_pct_per_100h" in iv


def test_logged_change_without_signature_is_reported_honestly():
    e = engine("healthy", 0)
    ev = e.events.copy()
    fake = ev.iloc[[0]].copy()
    fake["engine_hours"] = e.canonical["engine_hours_h"].iloc[0] + 120.0
    fake["t"] = e.canonical["t"].iloc[len(e.canonical) // 3]
    rep = usage_report(e.canonical, {"SA00": usage_config(e)}, events=__import__("pandas").concat([ev, fake]))
    h = rep.engines["SA00"].oil.history
    assert (h["status"] == "logged; no clear signature (e.g. same oil after a short interval, or too little data)").sum() == 1


def test_sensor_replacement_and_thinner_grade_are_not_counted_as_oil_changes():
    e = generate_engine("SA00", "healthy", hours=300, seed=4, oil_change_interval_h=None,
                        sensor_replacement_at_h=100, sensor_offset_step_kpa=12, grade_change_at_h=200, grade_factor=0.9)
    rep = usage_report(e.canonical, {"SA00": usage_config(e)})
    h = rep.engines["SA00"].oil.history
    assert not (h["event"] == "oil_change").any()
    kinds = dict(zip(h["event"], h["engine_hours"]))
    truth = dict(zip(e.events["event_type"], e.events["engine_hours"]))
    assert kinds["sensor_change"] == pytest.approx(truth["sensor_replacement"], abs=0.06)
    assert kinds["oil_system_step"] == pytest.approx(truth["oil_grade_change"], abs=0.06)
    assert rep.engines["SA00"].oil.hours_since_change is None


@pytest.mark.parametrize("scenario", ["cold_operation", "high_idle", "abusive"])
def test_no_false_oil_change_detections_under_other_duty_cycles(scenario):
    e, rep = usage(scenario, 0, "gaussian", with_log=False, oil_interval=100.0)
    h = rep.engines["SA00"].oil.history
    truth = e.truth["oil_change_engine_hours"]
    # precision: every detection (pressure signature or oil-level-only) is a real change
    assert all(any(abs(d - t) < 0.5 for t in truth) for d in h["engine_hours"]), h[["event", "engine_hours"]]
    # recall: every change is seen at least as an oil level rise. With chronically cold oil the pressure sits on
    # the relief valve, so the viscosity step can be invisible; it is then reported as level-only, not confirmed.
    assert len(h) == len(truth)
    if scenario != "cold_operation":
        assert (h["event"] == "oil_change").all()
