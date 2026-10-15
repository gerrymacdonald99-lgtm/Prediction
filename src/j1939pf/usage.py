"""Phase 3 orchestration: oil-change tracking, derived values, exposure report, flags."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysis import AnalysisResult, analyze
from .baseline import Prepared, prepare
from .config import load_defaults
from .derived import derived_metrics
from .engine_config import EngineConfig
from .exposure import exposure_report
from .framing import HONEST_FRAMING
from .oil_change import OilServiceResult, detect_oil_changes
from .spn_table import SpnTable, load_spn_table


@dataclass
class EngineUsage:
    engine_id: str
    metrics: dict
    exposure: pd.DataFrame
    flags: list[str]
    oil: OilServiceResult


@dataclass
class UsageReport:
    engines: dict[str, EngineUsage]
    analysis: AnalysisResult
    events_used: pd.DataFrame = field(default_factory=pd.DataFrame)

    def exposure_frame(self) -> pd.DataFrame:
        return pd.concat([u.exposure.assign(engine_id=e) for e, u in self.engines.items()], ignore_index=True)

    def oil_history_frame(self) -> pd.DataFrame:
        frames = [u.oil.history for u in self.engines.values() if len(u.oil.history)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def flags(self) -> dict[str, list[str]]:
        return {e: u.flags for e, u in self.engines.items()}

    def summary_text(self) -> str:
        L = [HONEST_FRAMING,
             "Exposure = time x severity of operation in each condition. It is NOT life consumed; no formula here "
             "converts exposure into life lost.", "=" * 78]
        for eid, u in self.engines.items():
            m = u.metrics
            L.append(f"Engine {eid}: {m['time']['running_h']:.1f} running h, {m['time']['starts']} starts, "
                     f"row interval {m['time']['row_interval_s']:g} s")
            L.append(f"  FLAGS: {', '.join(u.flags) if u.flags else 'none'}")
            for k, v in m.items():
                if k in ("time",):
                    continue
                if not v.get("available", False):
                    L.append(f"  {k}: unavailable - {v.get('reason', '')}")
                    continue
                shown = {a: b for a, b in v.items() if a not in ("available", "rpm_x_load_pct", "per_interval")}
                L.append(f"  {k}: " + ", ".join(f"{a}={_fmt(b)}" for a, b in shown.items()))
            L.append("  Exposure:")
            for _, r in u.exposure.iterrows():
                if not r["available"]:
                    L.append(f"    {r['condition']}: unavailable - {r['reason']}")
                else:
                    L.append(f"    {r['condition']}: {_fmt(r['hours'])} h, exposure {_fmt(r['exposure'])} {r['unit']}, "
                             f"{r['episodes']} episodes  [{r['definition']}]")
            L.append(f"  Oil service: hours since last oil change = {_fmt(u.oil.hours_since_change)} "
                     f"({u.oil.hours_since_change_source})")
            for _, h in u.oil.history.iterrows():
                L.append(f"    {h['engine_hours']:.1f} h  {h['event']:<16} {h['source']:<16} {h['status']}"
                         + (f"; pressure step {h['pressure_step_pct']:+.1f} % (z {h['pressure_step_z']:+.1f})"
                            if np.isfinite(h['pressure_step_pct']) else "")
                         + (f"; {', '.join(h['supporting'])}" if h["supporting"] else ""))
            for _, iv in u.oil.intervals.iterrows():
                txt = f"    interval {iv['from_h']:.0f}-{iv['to_h']:.0f} h:"
                if "pressure_trend_pct_per_100h" in iv and np.isfinite(iv.get("pressure_trend_pct_per_100h", np.nan)):
                    txt += f" oil pressure at fixed operating points {iv['pressure_trend_pct_per_100h']:+.1f} %/100 h"
                if "oil_level_trend_pct_per_100h" in iv and np.isfinite(iv.get("oil_level_trend_pct_per_100h", np.nan)):
                    txt += f"; oil level {iv['oil_level_trend_pct_per_100h']:+.2f} %-points/100 h"
                L.append(txt)
            for nnote in u.oil.notes:
                L.append(f"    note: {nnote}")
        return "\n".join(L)


def _fmt(v):
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.2f}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    return str(v)


def merge_events(logged: pd.DataFrame | None, oil_results: dict[str, OilServiceResult]) -> pd.DataFrame:
    """Logged maintenance events plus oil changes / sensor changes detected but NOT logged."""
    cols = ["engine_id", "t", "engine_hours", "event_type", "detail"]
    rows = [] if logged is None or len(logged) == 0 else [logged[cols]]
    for eid, r in oil_results.items():
        if len(r.history) == 0:
            continue
        det = r.history[(r.history["source"] == "detected") & (r.history["event"].isin(["oil_change", "sensor_change"]))]
        if len(det):
            rows.append(pd.DataFrame({"engine_id": eid, "t": det["t"].to_numpy(), "engine_hours": det["engine_hours"].to_numpy(),
                                      "event_type": det["event"].to_numpy(),
                                      "detail": "detected from J1939 signature (not in maintenance log)"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def usage_report(canonical: pd.DataFrame, engine_configs: dict[str, EngineConfig] | None = None, *,
                 dtc_episodes: pd.DataFrame | None = None, events: pd.DataFrame | None = None,
                 params: dict | None = None, table: SpnTable | None = None,
                 analysis: AnalysisResult | None = None, feedback=None) -> UsageReport:
    params = params or load_defaults()
    table = table or load_spn_table()
    engine_configs = engine_configs or {}
    res = table.by_canonical["oil_pressure_kpa"].scale if "oil_pressure_kpa" in table.by_canonical else 4.0

    # 1. oil-change tracking first, so detected (unlogged) changes can be confounders in the analysis
    preps: dict[str, Prepared] = {}
    oil: dict[str, OilServiceResult] = {}
    for eid, df_e in canonical.groupby("engine_id", sort=True):
        prep = prepare(df_e, engine_configs.get(str(eid)), params, table)
        preps[prep.engine_id] = prep
        oil[prep.engine_id] = detect_oil_changes(prep, params, events, res)
    merged = merge_events(events, oil)
    if analysis is None:
        analysis = analyze(canonical, engine_configs, dtc_episodes=dtc_episodes, events=merged, params=params,
                           table=table, feedback=feedback)

    engines = {}
    for eid, prep in preps.items():
        cfg = engine_configs.get(eid)
        metrics = derived_metrics(prep, cfg, params, dtc_episodes, oil[eid])
        sb = analysis.baselines.get((eid, "oil_pressure_kpa"))
        # the analysis prepared its own copy of the same rows in the same order: z aligns row for row
        expo = exposure_report(analysis.prepared.get(eid, prep), cfg, params, sb)
        flags = []
        idle = metrics.get("idle", {})
        if idle.get("flagged"):
            flags.append(f"HIGH IDLE: {idle['idle_pct']:.1f} % of running time at idle "
                         f"(> {idle['flag_threshold_pct']:.0f} % from engine config)")
        for label in ("coolant", "oil"):
            lt = metrics.get(f"low_temp_running_{label}", {})
            if lt.get("flagged_chronic_cold"):
                flags.append(f"CHRONIC COLD OPERATION ({label}): {lt['low_temp_running_pct_after_warmup']:.1f} % of "
                             f"running time below {lt['cutoff']} after the first {lt['warmup_exclusion_min']} min "
                             f"(> {lt['chronic_cold_threshold_pct']} %, program definition)")
        det = oil[eid].history
        if len(det):
            unl = det[(det["source"] == "detected") & (det["event"] == "oil_change")]
            if len(unl):
                flags.append(f"UNLOGGED OIL CHANGE(S) detected at {', '.join(f'{h:.1f}' for h in unl['engine_hours'])} "
                             f"engine h: add them to the maintenance log or check the detection")
        engines[eid] = EngineUsage(eid, metrics, expo, flags, oil[eid])
    return UsageReport(engines, analysis, merged)
