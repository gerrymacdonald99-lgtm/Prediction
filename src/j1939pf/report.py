"""Full pipeline (phase 5): every layer, the summary, and the exports.

    result = run_full(canonical, configs, dtc_episodes=..., events=..., feedback=store)
    print(result.summary_text())
    result.export("out/")      # CSV / JSON / report.html / data behind every alert
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .anomaly import AnomalyResult, anomaly_report
from .config import load_defaults
from .correlation import CorrelationReport, correlation_matrix, correlation_report
from .framing import HONEST_FRAMING
from .health import Health, health_score
from .labels import LayerResult, failures_from_events, remaining_life, supervised, survival
from .spn_table import load_spn_table
from .usage import UsageReport, usage_report

EXPOSURE_FEATURES = ["oil_pressure_below_own_baseline", "lugging", "cold_oil_high_rpm", "extended_idling",
                     "high_load_while_cold", "low_temp_running_after_warmup"]


@dataclass
class FullResult:
    usage: UsageReport
    correlations: CorrelationReport
    anomalies: dict[str, AnomalyResult]
    health: dict[str, Health]
    supervised: LayerResult
    survival: LayerResult
    remaining_life: dict
    params: dict
    parse_report: dict | None = None
    synthetic: bool = False
    matrices: dict = field(default_factory=dict)

    @property
    def analysis(self):
        return self.usage.analysis

    # ------------------------------------------------------------------ tables
    def trend_frame(self) -> pd.DataFrame:
        rows = []
        for (eid, sig), sb in self.analysis.baselines.items():
            if sb.trend is None:
                continue
            for _, r in sb.trend.iterrows():
                rows.append({"engine_id": eid, "signal": sig, "engine_hours": r["hours"], "t": r["t"],
                             "drift_pct": r["drift_rel"] * 100, "drift_abs": r["drift_abs"], "n_bins": r["n_bins"],
                             "capped_frac": r["capped_frac"]})
        return pd.DataFrame(rows)

    def baselines_frame(self) -> pd.DataFrame:
        from .baseline import bin_label
        rows = []
        for (eid, sig), sb in self.analysis.baselines.items():
            if sb.reference is None:
                continue
            prep = self.analysis.prepared[eid]
            for k, r in sb.reference.iterrows():
                rows.append({"engine_id": eid, "signal": sig, "bin": int(k),
                             "operating_point": bin_label(int(k), self.params, prep.load_col),
                             "reference_median": r["median"], "spread": r["spread"], "n": int(r["n"]),
                             "sessions": r.get("sessions"), "trusted": bool(r["trusted"]),
                             "late_learned": bool(r["late_learned"]),
                             "online_expected": float(sb.online[k]) if sb.online is not None and k in sb.online.index
                             else None})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ text
    def summary_text(self) -> str:
        L = [HONEST_FRAMING]
        if self.synthetic:
            L.append("SYNTHETIC DATA: results show the plumbing works, not real-world accuracy.")
        if self.parse_report and self.parse_report.get("unverified_spns"):
            L.append(f"UNVERIFIED SPN TABLE: decoded SPNs {self.parse_report['unverified_spns']} are not yet checked "
                     f"against J1939DA.")
        L.append("=" * 78)
        L.append("HEALTH")
        for h in self.health.values():
            L.append(h.table())
        L += ["=" * 78, "ALERTS / LEARNING / TRANSITIONS", self.analysis.summary_text().split("\n", 2)[-1]]
        L += ["=" * 78, "TREND REPORT"]
        tf = self.trend_frame()
        for (eid, sig), g in tf.groupby(["engine_id", "signal"]) if len(tf) else []:
            last = g.tail(3)["drift_pct"].median()
            L.append(f"  {eid} {sig}: {len(g)} trend points, latest drift {last:+.2f} % vs reference "
                     f"(min {g['drift_pct'].min():+.2f}, max {g['drift_pct'].max():+.2f})")
        for r in self.correlations.relationships:
            if r.kind == "trend" and r.available:
                L.append(f"  {r.engine_id} {r.name}: {r.status}, slope {r.slope:+.3g} {r.slope_unit}, n={r.n_independent}")
        L += ["=" * 78, "USAGE / EXPOSURE / OIL SERVICE", self.usage.summary_text().split("\n", 3)[-1]]
        L += ["=" * 78, "CORRELATIONS", self.correlations.summary_text().split("\n", 3)[-1]]
        L += ["=" * 78, "JOINT ANOMALIES"]
        for a in self.anomalies.values():
            if not a.available:
                L.append(f"  {a.engine_id}: unavailable - {a.reason}")
                continue
            L.append(f"  {a.engine_id}: features {a.features}; reference rate {a.reference_rate * 100:.2f} %, recent "
                     f"{(a.recent_rate or 0) * 100:.2f} %, {len(a.episodes)} sustained episode(s)")
            for ep in a.episodes[:10]:
                L.append(f"    {ep['start_h']:.1f}-{ep['end_h']:.1f} h, {ep['duration_s'] / 60:.0f} min: "
                         f"{', '.join(ep['top_signals'])}")
        L += ["=" * 78, "LABEL-BASED LAYERS (the only layers that may speak about failure risk)",
              f"  supervised: {self.supervised.message}", f"  survival: {self.survival.message}"]
        for eid, r in self.remaining_life.items():
            if r.get("available"):
                L.append(f"  remaining life {eid}: median {r['median_remaining_h']:.0f} engine h (95 % CI "
                         f"{r['ci95'][0]:.0f}-{r['ci95'][1]:.0f}), n={r['n_engines']} engines / {r['n_failures']} failures")
        return "\n".join(L)

    # ------------------------------------------------------------------ export
    def export(self, outdir: str | Path, html: bool = True) -> dict[str, str]:
        out = Path(outdir)
        (out / "alert_data").mkdir(parents=True, exist_ok=True)
        paths = {}

        def w(name, obj):
            p = out / name
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(p, index=False)
            elif isinstance(obj, str):
                p.write_text(obj, encoding="utf-8")
            else:
                p.write_text(json.dumps(_jsonable({"framing": HONEST_FRAMING, **obj}), indent=2), encoding="utf-8")
            paths[name] = str(p)

        w("FRAMING.txt", HONEST_FRAMING + "\n")
        w("summary.txt", self.summary_text())
        al = self.analysis.alerts
        w("alerts.json", {"alerts": [a.to_dict() for a in al],
                          "suppressed": [{"alert": a.to_dict(), "why": why} for a, why in self.analysis.suppressed]})
        w("alerts.csv", self.analysis.alerts_frame().astype(str) if al else pd.DataFrame(columns=["alert_id"]))
        for a in al:
            w(f"alert_data/{a.alert_id}.csv", self.analysis.alert_data(a))
        w("trend.csv", self.trend_frame())
        w("learned_baselines.csv", self.baselines_frame())
        w("exposure.csv", _stringify(self.usage.exposure_frame()))
        w("metrics.json", {"engines": {e: u.metrics for e, u in self.usage.engines.items()},
                           "flags": self.usage.flags()})
        w("oil_service.csv", _stringify(self.usage.oil_history_frame()))
        iv = [u.oil.intervals for u in self.usage.engines.values() if len(u.oil.intervals)]
        w("oil_intervals.csv", _stringify(pd.concat(iv, ignore_index=True)) if iv else pd.DataFrame())
        w("transitions.csv", self.analysis.transitions)
        w("correlations.csv", _stringify(self.correlations.frame()))
        for eid, m in self.matrices.items():
            w(f"correlation_matrix_{eid}.csv", m.reset_index().rename(columns={"index": "signal"}))
        eps = [e for a in self.anomalies.values() for e in a.episodes]
        w("anomalies.csv", pd.DataFrame(eps).astype(str) if eps else pd.DataFrame(columns=["engine_id"]))
        rates = [a.rate_by_period.assign(engine_id=e) for e, a in self.anomalies.items() if len(a.rate_by_period)]
        w("anomaly_rates.csv", pd.concat(rates, ignore_index=True) if rates else pd.DataFrame())
        w("health.json", {"note": next(iter(self.health.values())).note if self.health else "",
                          "engines": {e: {"score": h.score, "components": h.components} for e, h in self.health.items()}})
        km = self.survival.details.get("kaplan_meier")
        w("labels.json", {"supervised": {"available": self.supervised.available, "message": self.supervised.message,
                                         "details": self.supervised.details},
                          "survival": {"available": self.survival.available, "message": self.survival.message,
                                       "details": {k: v for k, v in self.survival.details.items()
                                                   if k not in ("theta", "cov", "standardisation", "kaplan_meier")}},
                          "remaining_life": self.remaining_life})
        if isinstance(km, pd.DataFrame) and len(km):
            w("kaplan_meier.csv", km)
        if self.parse_report:
            w("parse_report.json", {"parse_report": self.parse_report})
        if html:
            from .html_report import render_html
            w("report.html", render_html(self))
        return paths


def _stringify(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    for c in out.columns:
        if out[c].map(lambda v: isinstance(v, (dict, list, tuple))).any():
            out[c] = out[c].map(lambda v: json.dumps(_jsonable(v)) if isinstance(v, (dict, list, tuple)) else v)
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.reset_index().to_dict("records") if not isinstance(o.index, pd.RangeIndex)
                         else o.to_dict("records"))
    if isinstance(o, pd.Series):
        return _jsonable(o.to_dict())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


# ---------------------------------------------------------------- label tables
def _period_features(usage: UsageReport, analysis) -> pd.DataFrame:
    rows = {}
    for eid, u in usage.engines.items():
        for _, r in u.exposure.iterrows():
            if not r["available"] or r["condition"] not in EXPOSURE_FEATURES:
                continue
            for key, v in (r["by_period"] or {}).items():
                a, b = key.replace(" h", "").split("-")
                rows.setdefault((eid, float(a), float(b)), {})[r["condition"]] = v
        sb = analysis.baselines.get((eid, "oil_pressure_kpa"))
        if sb is not None and sb.trend is not None:
            for (e, a, b), d in rows.items():
                if e == eid:
                    t = sb.trend[(sb.trend["hours"] >= a) & (sb.trend["hours"] < b)]
                    d["oil_pressure_drift_pct"] = float(t["drift_rel"].median() * 100) if len(t) else np.nan
    return pd.DataFrame([{"engine_id": e, "from_h": a, "to_h": b, **d} for (e, a, b), d in rows.items()])


def _engine_table(usage: UsageReport, failures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for eid, u in usage.engines.items():
        h0, h1 = u.metrics["time"]["engine_hours_span"]
        f = failures[failures["engine_id"] == eid]["engine_hours"]
        rec = {"engine_id": eid, "entry_h": h0, "exit_h": float(f.min()) if len(f) else h1, "failed": int(len(f) > 0)}
        ex = u.exposure.set_index("condition")
        for c in EXPOSURE_FEATURES:
            rec[c] = float(ex.loc[c, "hours_per_100_running_h"]) if c in ex.index and ex.loc[c, "available"] else np.nan
        idle = u.metrics.get("idle", {})
        rec["idle_pct"] = idle.get("idle_pct", np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_full(canonical: pd.DataFrame, engine_configs=None, *, dtc_episodes=None, events=None, feedback=None,
             params: dict | None = None, table=None, parse_report: dict | None = None,
             synthetic: bool = False) -> FullResult:
    params = params or load_defaults()
    table = table or load_spn_table()
    engine_configs = engine_configs or {}
    usage = usage_report(canonical, engine_configs, dtc_episodes=dtc_episodes, events=events, params=params,
                         table=table, feedback=feedback)
    analysis = usage.analysis
    corr = correlation_report(analysis, engine_configs, events=usage.events_used, params=params, table=table)
    anomalies, health, matrices = {}, {}, {}
    for eid, prep in analysis.prepared.items():
        sb = analysis.baselines.get((eid, "oil_pressure_kpa"))
        hours = prep.df["_hours"].to_numpy()
        ref_end = sb.learn_end_h if sb is not None and sb.learn_end_h else hours.min() + params["baseline"]["learning_period_h"]
        anomalies[eid] = anomaly_report(prep, params, ref_end, table)
        health[eid] = health_score(eid, analysis, params, anomalies[eid], corr)
        matrices[eid] = correlation_matrix(prep, ref_end, params)
    fails = failures_from_events(events, params["labels"]["failure_prefix"])
    sup = supervised(_period_features(usage, analysis), fails,
                     EXPOSURE_FEATURES + ["oil_pressure_drift_pct"], params)
    eng = _engine_table(usage, fails)
    surv = survival(eng, EXPOSURE_FEATURES + ["idle_pct"], params)
    rul = {}
    if surv.available:
        for _, r in eng[eng["failed"] == 0].iterrows():
            cv = {c: r[c] for c in surv.details["covariates"]}
            rul[r["engine_id"]] = remaining_life(surv, r["exit_h"], cv, params, int(params.get("seed", 0)))
    return FullResult(usage, corr, anomalies, health, sup, surv, rul, params, parse_report, synthetic, matrices)
