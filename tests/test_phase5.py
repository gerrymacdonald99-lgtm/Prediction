"""Phase 5: joint anomalies, label-based layers, health score, exports, HTML, CLI."""

import json
import subprocess
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from j1939pf import anomaly as anomaly_mod
from j1939pf.anomaly import anomaly_report
from j1939pf.config import load_defaults
from j1939pf.framing import HONEST_FRAMING
from j1939pf.generator import generate_engine, generate_fleet
from j1939pf.labels import kaplan_meier, remaining_life, supervised, survival
from j1939pf.report import run_full
from j1939pf.spn_table import load_spn_table
from j1939pf.usage import usage_report
from synth import usage_config

P = load_defaults()
TABLE = load_spn_table()


@lru_cache(maxsize=16)
def prepared(scenario, seed):
    e = generate_engine("SA00", scenario, hours=300, seed=seed)
    rep = usage_report(e.canonical, {"SA00": usage_config(e)}, events=e.events)
    sb = rep.analysis.baselines[("SA00", "oil_pressure_kpa")]
    return e, rep.analysis.prepared["SA00"], sb.learn_end_h


# ---------------------------------------------------------------- joint anomalies
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_anomaly_layer_finds_boost_leak_that_oil_layers_cannot_see(seed):
    e, prep, ref_end = prepared("boost_leak", seed)
    a = anomaly_report(prep, P, ref_end, TABLE)
    assert a.available and "boost_kpa" in a.features
    assert a.methods["iforest"]["random_state"] == P["seed"]
    assert a.recent_rate >= 8 * a.reference_rate
    after = [ep for ep in a.episodes if ep["start_h"] >= e.truth["onset_engine_hours"]]
    assert after and sum(ep["top_signals"][0].startswith("boost_kpa") for ep in after) >= len(after) / 2


@pytest.mark.parametrize("seed", [0, 2])
def test_anomaly_rate_stays_near_reference_on_healthy_engines(seed):
    _, prep, ref_end = prepared("healthy", seed)
    a = anomaly_report(prep, P, ref_end, TABLE)
    assert a.reference_rate < 0.01 and a.recent_rate < 0.015 and len(a.episodes) <= 1


def test_anomaly_is_deterministic_and_degrades_gracefully(monkeypatch):
    _, prep, ref_end = prepared("boost_leak", 0)
    a, b = anomaly_report(prep, P, ref_end, TABLE), anomaly_report(prep, P, ref_end, TABLE)
    assert np.array_equal(a.flag, b.flag)
    # signals missing -> dropped, with the reason
    p2 = load_defaults(overrides={"anomaly": {"features": ["engine_speed_rpm", "load", "oil_pressure_kpa", "nope"]}})
    c = anomaly_report(prep, p2, ref_end, TABLE)
    assert not c.available and "at least 2 signals" in c.reason and c.dropped_features["nope"] == "not broadcast"
    # too little reference data -> unavailable, not a crash
    d = anomaly_report(prep, P, prep.df["_hours"].min() + 0.5, TABLE)
    assert not d.available and "reference window" in d.reason
    # scikit-learn missing -> classical Mahalanobis only
    monkeypatch.setattr(anomaly_mod, "HAVE_SKLEARN", False)
    f = anomaly_report(prep, P, ref_end, TABLE)
    assert f.available and "unavailable" in f.methods["iforest"]["status"] and "classical" in f.methods["mahalanobis"]["method"]


# ---------------------------------------------------------------- label-based layers
def _survival_table(n, seed=0, beta=-0.06):
    rng = np.random.default_rng(seed)
    lug = rng.uniform(0, 10, n)
    W = np.log(-np.log(rng.uniform(size=n)))
    T = np.exp(np.log(9000) + beta * lug + 0.35 * W)
    entry = rng.uniform(500, 3000, n)
    end = entry + rng.uniform(2000, 8000, n)
    keep = T > entry
    failed = (T <= end) & keep
    return pd.DataFrame({"engine_id": [f"E{i}" for i in range(n)], "entry_h": entry,
                         "exit_h": np.where(failed, T, end), "failed": failed.astype(int), "lugging": lug})[keep]


def test_survival_recovers_a_known_effect_with_ci_and_sample_size():
    r = survival(_survival_table(80), ["lugging"], P)
    eff = r.details["covariate_effects"]["lugging"]
    lo, hi = eff["ci95"]
    assert r.available and lo <= np.exp(-0.06) <= hi and eff["significant"]
    assert f"{r.details['failures']} failures" in r.message and "engines" in r.message
    rl_hi = remaining_life(r, 4000, {"lugging": 9}, P)
    rl_lo = remaining_life(r, 4000, {"lugging": 1}, P)
    assert rl_hi["median_remaining_h"] < rl_lo["median_remaining_h"]
    assert rl_hi["ci95"][0] < rl_hi["median_remaining_h"] < rl_hi["ci95"][1] and rl_hi["n_failures"] > 10


def test_survival_says_plainly_when_data_is_insufficient():
    t = _survival_table(80)
    small = pd.concat([t[t.failed == 1].head(5), t[t.failed == 0].head(10)])   # 5 failures < 10 needed
    r = survival(small, ["lugging"], P)
    assert not r.available and r.message.startswith("INSUFFICIENT DATA") and "no remaining-life" in r.message
    assert remaining_life(r, 4000, {"lugging": 1}, P)["available"] is False
    km = r.details["kaplan_meier"]
    assert (np.diff(km["survival"]) <= 1e-12).all() and (km["ci_low"] <= km["survival"]).all()
    none = survival(small.assign(failed=0), ["lugging"], P)
    assert "no failure labels" in none.message


def test_supervised_handles_imbalance_and_refuses_without_enough_failures():
    rng = np.random.default_rng(1)
    rows, fails = [], []
    for i in range(40):
        risky = i % 2 == 0
        fh = rng.uniform(300, 1200) if risky else None
        if fh:
            fails.append({"engine_id": f"E{i}", "engine_hours": fh, "event_type": "failure_bearing"})
        for p0 in range(0, 1200, 50):
            near = fh is not None and p0 + 50 < fh <= p0 + 250
            rows.append({"engine_id": f"E{i}", "from_h": p0, "to_h": p0 + 50,
                         "x1": rng.normal(2.0 if near else 0.0, 0.7), "x2": rng.normal()})
    d, f = pd.DataFrame(rows), pd.DataFrame(fails)
    r = supervised(d, f, ["x1", "x2"], P)
    assert r.available and r.details["skill"] and r.details["base_rate"] < 0.2
    assert "GroupKFold by engine" in r.details["cv"] and "NOT calibrated" in r.details["class_weight"]
    assert r.details["roc_auc_ci95"][0] > 0.5 and "n = " in r.message
    r2 = supervised(d, f.head(3), ["x1", "x2"], P)
    assert not r2.available and r2.message.startswith("INSUFFICIENT DATA") and "No prediction made" in r2.message
    r3 = supervised(d.assign(x1=rng.normal(size=len(d))), f, ["x1", "x2"], P)
    assert r3.available and (not r3.details["skill"] or r3.details["roc_auc_ci95"][0] < 0.6)


# ---------------------------------------------------------------- full pipeline, health, exports
@lru_cache(maxsize=1)
def fleet_result():
    specs = [{"engine_id": "SA00", "scenario": "stuck_relief_valve", "hours": 300},
             {"engine_id": "SA01", "scenario": "healthy", "hours": 300},
             {"engine_id": "SA02", "scenario": "healthy", "hours": 25}]          # new engine, still learning
    df, ev, eng = generate_fleet(specs, seed=2)
    cfgs = {k: usage_config(v) for k, v in eng.items() if k != "SA01"}           # SA01: no config at all
    ev = pd.concat([ev, pd.DataFrame([{"engine_id": "SA00", "engine_hours": float(df[df.engine_id == "SA00"]
                                       .engine_hours_h.max()) - 1, "event_type": "failure_oil_pump", "detail": "",
                                       "t": pd.NaT}])], ignore_index=True)
    return run_full(df, cfgs, events=ev, synthetic=True), eng


def test_health_score_always_has_its_decomposition():
    res, _ = fleet_result()
    assert set(res.health) == {"SA00", "SA01", "SA02"}
    for h in res.health.values():
        assert h.components and {c["component"] for c in h.components} >= {"oil_pressure_kpa", "joint_anomaly", "faults"}
        for c in h.components:
            assert c["reason"] and ("subscore" in c) and ("points_lost" in c)
        if h.score is not None:
            assert 100 - sum(c["points_lost"] for c in h.components) == pytest.approx(h.score)
        assert "not remaining life" in h.note
    assert res.health["SA01"].score >= 95                              # healthy, no config: layer (a) parts unavailable
    assert not next(c for c in res.health["SA01"].components if c["component"] == "oil_temp_c")["available"]
    sa00 = res.health["SA00"]
    assert sa00.score < res.health["SA01"].score - 20
    assert max(sa00.components, key=lambda c: c["points_lost"])["component"] == "oil_pressure_kpa"
    assert "LEARNING PERIOD" in res.health["SA02"].note             # new engine: learned parts not yet included


def test_one_failure_is_not_enough_for_label_layers():
    res, _ = fleet_result()
    assert not res.supervised.available and res.supervised.message.startswith("INSUFFICIENT DATA")
    assert not res.survival.available and "1 failures among 3 engines" in res.survival.message
    assert res.remaining_life == {}


def test_exports_html_and_show_what_it_learned(tmp_path):
    res, _ = fleet_result()
    paths = res.export(tmp_path)
    for f in ("alerts.json", "alerts.csv", "trend.csv", "exposure.csv", "metrics.json", "oil_service.csv",
              "transitions.csv", "correlations.csv", "anomalies.csv", "health.json", "labels.json", "summary.txt",
              "learned_baselines.csv", "report.html", "correlation_matrix_SA00.csv", "FRAMING.txt"):
        assert f in paths, f
    for f in ("alerts.json", "health.json", "labels.json", "metrics.json"):
        assert json.loads((tmp_path / f).read_text())["framing"] == HONEST_FRAMING
    alerts = json.loads((tmp_path / "alerts.json").read_text())["alerts"]
    assert alerts and all((tmp_path / "alert_data" / f"{a['alert_id']}.csv").exists() for a in alerts)
    behind = pd.read_csv(tmp_path / "alert_data" / f"{alerts[0]['alert_id']}.csv")
    assert {"oil_pressure_kpa", "engine_speed_rpm"} <= set(behind.columns)
    lb = pd.read_csv(tmp_path / "learned_baselines.csv")
    assert set(lb.engine_id) == {"SA00", "SA01", "SA02"} and lb.trusted.any() and "operating_point" in lb
    cm = pd.read_csv(tmp_path / "correlation_matrix_SA01.csv")
    assert "oil_pressure_kpa" in set(cm["signal"])
    html = (tmp_path / "report.html").read_text()
    import html as _html
    assert _html.escape(HONEST_FRAMING) in html and "SYNTHETIC DATA" in html
    assert "Decomposition" in html and html.count("<svg") >= 4 and "What it learned" in html
    assert all(a["alert_id"] in html for a in alerts) and "Correlation matrix" in html
    assert "http://" not in html and "https://" not in html and "<script" not in html   # self-contained, offline
    txt = (tmp_path / "summary.txt").read_text()
    assert txt.startswith(HONEST_FRAMING) and "TREND REPORT" in txt and "LABEL-BASED LAYERS" in txt


def test_cli_end_to_end(tmp_path):
    run = lambda *a: subprocess.run([sys.executable, "-m", "j1939pf", *a], capture_output=True, text=True,  # noqa: E731
                                    check=True, cwd=tmp_path)
    run("generate", "--scenario", "stuck_relief_valve", "--hours", "120", "--seed", "3", "--out", "g.csv",
        "--events-out", "ev.csv")
    (tmp_path / "eng.yaml").write_text("engine_id: SA00\nlimits:\n  oil_pressure_min_idle: {value: 15, unit: psi, "
                                       "placeholder: true, source: 'PLACEHOLDER on-highway'}\n")
    out = run("analyze", "g.csv", "--engines", "eng.yaml", "--events", "ev.csv", "--feedback", "fb.json",
              "--synthetic", "--out", "out").stdout
    assert out.startswith(HONEST_FRAMING) and "health" in out
    alerts = json.loads((tmp_path / "out" / "alerts.json").read_text())["alerts"]
    aid = alerts[0]["alert_id"]
    assert aid in run("explain", "--out", "out", "--alert-id", aid).stdout
    run("feedback", "dismiss", "--store", "fb.json", "--alerts", "out/alerts.json", "--alert-id", aid,
        "--reason", "gauge reads the same")
    run("analyze", "g.csv", "--engines", "eng.yaml", "--events", "ev.csv", "--feedback", "fb.json", "--out", "out2")
    doc = json.loads((tmp_path / "out2" / "alerts.json").read_text())
    assert aid not in {a["alert_id"] for a in doc["alerts"]} and doc["suppressed"][0]["alert"]["alert_id"] == aid
    # parse sub-command on a candump log
    run("generate", "--scenario", "healthy", "--hours", "2", "--format", "candump", "--out", "bus.log")
    (tmp_path / "d.yaml").write_text("canonical: {resample_interval_s: 10, gap_s: 60}\n")
    run("parse", "bus.log", "--defaults", "d.yaml", "--out", "canon.csv")
    assert pd.read_csv(tmp_path / "canon.csv").shape[0] > 100
