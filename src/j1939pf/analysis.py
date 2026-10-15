"""Per-engine analysis: layer (a) manual thresholds + learned layers (b)-(d), with sensor checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .alerts import Alert, check_next, explanation, fmt_value
from .baseline import Prepared, SignalBaseline, learn_signal, prepare
from .config import load_defaults
from .detection import drift_events, episode_rows, find_runs, merge_runs, operating_point
from .engine_config import EngineConfig
from .framing import HONEST_FRAMING
from .sensor_check import assess
from .spn_table import SpnTable, load_spn_table

LAYER_A_CHECKS = (
    # limit name, signal, direction
    ("oil_pressure_min_idle", "oil_pressure_kpa", "low"),
    ("oil_pressure_min_rated", "oil_pressure_kpa", "low"),
    ("oil_temp_max", "oil_temp_c", "high"),
    ("coolant_temp_max", "coolant_temp_c", "high"),
)


@dataclass
class AnalysisResult:
    alerts: list[Alert]
    status: dict[str, dict]
    baselines: dict[tuple[str, str], SignalBaseline]
    prepared: dict[str, Prepared] = field(default_factory=dict)

    def alerts_frame(self) -> pd.DataFrame:
        return pd.DataFrame([a.to_dict() for a in self.alerts])

    def for_engine(self, engine_id: str, layer: str | None = None) -> list[Alert]:
        return [a for a in self.alerts if a.engine_id == engine_id and (layer is None or a.layer == layer)]

    def alert_data(self, alert: Alert, pad_s: float = 300.0) -> pd.DataFrame:
        """The canonical rows behind an alert (with z / expected where learned)."""
        prep = self.prepared[alert.engine_id]
        df = prep.df
        t0, t1 = pd.Timestamp(alert.start_t), pd.Timestamp(alert.end_t)
        m = (df["t"] >= t0 - pd.Timedelta(seconds=pad_s)) & (df["t"] <= t1 + pd.Timedelta(seconds=pad_s))
        out = df.loc[m, [c for c in df.columns if not c.startswith("_")] + ["_hours", "_bin"]].copy()
        sb = self.baselines.get((alert.engine_id, alert.signal))
        if sb is not None and sb.z is not None:
            out["expected"] = sb.expected[m.to_numpy()]
            out["z"] = sb.z[m.to_numpy()]
        return out

    def summary_text(self) -> str:
        lines = [HONEST_FRAMING, "=" * 78]
        for eid, st in self.status.items():
            lines.append(f"Engine {eid}: {st['hours_observed'][0]:.1f}-{st['hours_observed'][1]:.1f} engine h "
                         f"({st['hours_source']}); config: {st['config']}")
            lines.append(f"  {st['learning_announcement']}")
            for k, v in st["layer_a"].items():
                lines.append(f"  layer (a) {k}: {v}")
            for n in st["notes"]:
                lines.append(f"  note: {n}")
            for a in self.for_engine(eid):
                lines.append("  " + a.headline())
                lines.append(f"      {a.explanation}")
                lines.append(f"      sensor-vs-engine: {a.sensor_verdict}")
                for ev in a.sensor_evidence:
                    lines.append(f"        - {ev}")
                for c in a.confounders:
                    lines.append(f"      confounder: {c}")
                for i, step in enumerate(a.check_next, 1):
                    lines.append(f"      check {i}: {step}")
            if not self.for_engine(eid):
                lines.append("  no alerts")
        return "\n".join(lines)


def _confounders(prep: Prepared, events: pd.DataFrame | None, h0: float, h1: float, ref: np.ndarray,
                 window: np.ndarray, alert_start_h: float) -> list[str]:
    """Maintenance events between the end of the reference window and the end of the alert."""
    out = []
    if events is not None and len(events):
        ev = events[(events["engine_id"] == prep.engine_id) & (events["engine_hours"] >= h0)
                    & (events["engine_hours"] <= h1)]
        for _, e in ev.iterrows():
            what = f"{e['event_type']} at {e['engine_hours']:.1f} engine h ({e.get('detail', '')})"
            if e["engine_hours"] <= alert_start_h:
                out.append(f"{what}, before the alert began: the deviation may be partly or wholly due to it")
            else:
                out.append(f"{what}, after the alert began: it cannot explain the onset, but may change the "
                           f"readings that follow")
    df = prep.df
    if "ambient_temp_c" in df and ref.any() and window.any():
        a = df["ambient_temp_c"].to_numpy(dtype=float)
        r, w = np.nanmedian(a[ref]), np.nanmedian(a[window])
        if np.isfinite(r) and np.isfinite(w) and abs(w - r) > 10:
            out.append(f"ambient temperature differs from the reference period ({r:.0f} C -> {w:.0f} C; season)")
    return out


def _whole_segments(prep: Prepared, window: np.ndarray) -> np.ndarray:
    """Widen a window to the full segments (days/sessions) it touches, so key-on/engine-off rows are included."""
    seg = prep.df["segment"].to_numpy()
    return np.isin(seg, np.unique(seg[window]))


def _finish(alert: Alert, prep: Prepared, sb: SignalBaseline | None, window: np.ndarray, ref_window: np.ndarray,
            params: dict, table: SpnTable, dtc_episodes, events, conf_h: tuple[float, float]) -> Alert:
    window = _whole_segments(prep, window)
    ref_window = _whole_segments(prep, ref_window) if ref_window.any() else ref_window
    d = table.by_canonical.get(alert.signal)
    res = d.scale if d else 1.0
    sa = assess(prep, sb, alert.signal, window, alert.direction, params, dtc_episodes,
                d.spn if d else None, res, ref_window)
    alert.sensor_verdict = sa.verdict
    alert.sensor_evidence = sa.evidence
    alert.confounders = _confounders(prep, events, conf_h[0], conf_h[1], ref_window, window, alert.start_h)
    pattern_text = next((e.split(": ", 1)[1] for e in sa.evidence
                         if e.startswith(("ENGINE: deviation", "ENGINE: pressure no longer rises",
                                          "SENSOR (weak): deviation"))), "")
    extra = ""
    if sa.verdict == "sensor_suspected":
        extra = "Sensor evidence is present: verify the reading before concluding the engine is at fault."
    if alert.limit and "PLACEHOLDER" in alert.limit:
        extra += " The limit is a PLACEHOLDER, not from this engine's manual."
    alert.explanation = explanation(alert, pattern_text, extra.strip())
    alert.check_next = check_next(alert.signal, sa.verdict, sa.pattern, alert.direction)
    t = prep.df["t"].to_numpy()
    idx = np.flatnonzero(window)
    alert.data_window = {"engine_id": prep.engine_id, "t_from": str(t[idx[0]]), "t_to": str(t[idx[-1]]),
                         "rows": int(idx.size)}
    return alert


def _layer_a(prep: Prepared, cfg: EngineConfig | None, params: dict) -> tuple[list[tuple], dict]:
    ap = params["alerts"]
    df = prep.df
    status, found = {}, []
    judge = df["_judgeable"].to_numpy()
    for name, signal, direction in LAYER_A_CHECKS:
        if cfg is None:
            status[name] = "disabled: no engine config for this engine (learned layers still run)"
            continue
        lim = cfg.limit(name)
        if lim is None:
            status[name] = f"disabled: `{name}` missing from {cfg.path or 'engine config'}"
            continue
        if signal not in df or df[signal].notna().sum() == 0:
            status[name] = f"disabled: {signal} not broadcast"
            continue
        scope = judge.copy()
        scope_txt = "all running speeds"
        if name == "oil_pressure_min_rated":
            rated = cfg.limit("rated_rpm")
            if rated is None:
                status[name] = "disabled: rated_rpm missing from engine config (needed to know where rated speed is)"
                continue
            scope &= df["engine_speed_rpm"].to_numpy(dtype=float) >= rated.value - ap["rated_speed_tolerance_rpm"]
            scope_txt = f">= {rated.value - ap['rated_speed_tolerance_rpm']:.0f} rpm"
        v = df[signal].to_numpy(dtype=float)
        bad = scope & ~np.isnan(v) & ((v < lim.value) if direction == "low" else (v > lim.value))
        status[name] = f"active ({scope_txt}): {lim.describe()}"
        runs = find_runs(bad, prep, ap["sustained_run_s"])
        for ep in merge_runs(runs, prep, ap["merge_gap_engine_h"]):
            found.append((name, signal, direction, lim, ep, bad))
    return found, status


def analyze(canonical: pd.DataFrame, engine_configs: dict[str, EngineConfig] | None = None, *,
            dtc_episodes: pd.DataFrame | None = None, events: pd.DataFrame | None = None,
            params: dict | None = None, table: SpnTable | None = None) -> AnalysisResult:
    params = params or load_defaults()
    table = table or load_spn_table()
    engine_configs = engine_configs or {}
    ap = params["alerts"]
    alerts: list[Alert] = []
    status: dict[str, dict] = {}
    baselines: dict[tuple[str, str], SignalBaseline] = {}
    prepared: dict[str, Prepared] = {}

    for engine_id, df_e in canonical.groupby("engine_id", sort=True):
        cfg = engine_configs.get(str(engine_id))
        prep = prepare(df_e, cfg, params, table)
        prepared[prep.engine_id] = prep
        df = prep.df
        n = len(df)
        hours = df["_hours"].to_numpy()
        st = {"hours_observed": (float(hours.min()), float(hours.max())), "hours_source": prep.hours_source,
              "row_interval_s": prep.dt_s, "config": cfg.path if cfg and cfg.path else ("in-memory" if cfg else "none"),
              "notes": list(prep.notes), "learned": {}}

        # ---- learned layers (b)-(d): per signal, from this engine's own rows only
        learn_ref = np.zeros(n, bool)
        for signal in params["baseline"]["signals"]:
            if signal not in df or df[signal].notna().sum() == 0:
                st["learned"][signal] = {"status": "no_data", "message": f"{signal} not broadcast"}
                continue
            res = table.by_canonical[signal].scale if signal in table.by_canonical else 1.0
            sb = learn_signal(prep, signal, params, res)
            baselines[(prep.engine_id, signal)] = sb
            st["learned"][signal] = {"status": sb.status, "message": sb.message, "trusted_bins": sb.trusted_bins,
                                     "capped_bins": len(sb.capped),
                                     "late_learned_bins": int(sb.reference["late_learned"].sum())
                                     if sb.reference is not None else 0}
            if sb.learn_start_h is not None:
                ref_w = (hours >= sb.learn_start_h) & (hours < sb.learn_end_h)
                learn_ref |= ref_w
            if sb.status != "active":
                continue
            ref_w = (hours >= sb.learn_start_h) & (hours < sb.learn_end_h)

            # layer (c)
            z = sb.z
            for direction, flag in (("low", z <= -ap["z_alert"]), ("high", z >= ap["z_alert"])):
                flag = np.nan_to_num(flag, nan=False).astype(bool)
                runs = find_runs(flag, prep, ap["sustained_run_s"])
                for ep in merge_runs(runs, prep, ap["merge_gap_engine_h"]):
                    rows = episode_rows(ep, n) & flag
                    zmed = float(np.median(z[rows]))
                    op = operating_point(prep, rows, params)
                    a = Alert(prep.engine_id, "c", signal, direction,
                              "warning" if abs(zmed) >= ap["z_severe"] else "watch",
                              df["t"].iloc[ep[0].i0], df["t"].iloc[ep[-1].i1], float(hours[ep[0].i0]),
                              float(hours[ep[-1].i1]), float(sum(r.duration_s for r in ep)), len(ep), op,
                              float(np.median(sb.expected[rows])), float(np.median(df[signal].to_numpy()[rows])), zmed)
                    window = np.zeros(n, bool)
                    window[ep[0].i0:ep[-1].i1 + 1] = True
                    alerts.append(_finish(a, prep, sb, window, ref_w, params, table, dtc_episodes, events,
                                          (sb.learn_end_h, float(hours[ep[-1].i1]))))

            # layer (d)
            tr = sb.trend
            for ev in drift_events(sb, params):
                first, last = tr.iloc[ev.first_idx], tr.iloc[ev.last_idx]
                window = (hours > float(last["hours"]) - ap["trend_window_h"]) & (hours <= float(last["hours"]))
                elig_w = window & df["_judgeable"].to_numpy() & np.isin(df["_bin"].to_numpy(),
                                                                        sb.reference.index[sb.reference["trusted"]])
                op = operating_point(prep, elig_w, params)
                k = op.get("bin")
                expected = float(sb.reference.loc[k, "median"]) if k in sb.reference.index else None
                actual = (float(np.nanmedian(df[signal].to_numpy()[elig_w & (df["_bin"].to_numpy() == k)]))
                          if k is not None else None)
                first_level = float(first["drift_rel"])
                op.update({"n_bins": int(last["n_bins"]), "drift_pct": ev.level * 100,
                           "drift_pct_at_first_detection": first_level * 100,
                           "slope_pct_per_100h": ev.slope * 1e4,
                           "slope_ci": f"{ev.lo * 1e4:+.1f}..{ev.hi * 1e4:+.1f} %/100 h",
                           "capped_frac": float(last["capped_frac"])})
                sev = "warning" if abs(ev.level) >= 2 * ap["drift_alert_rel"] else "watch"
                a = Alert(prep.engine_id, "d", signal, ev.direction, sev, first["t"], last["t"],
                          float(first["hours"]), float(last["hours"]), 0.0, ev.n_points, op, expected, actual)
                fin = _finish(a, prep, sb, window, ref_w, params, table, dtc_episodes, events,
                              (sb.learn_end_h, float(last["hours"])))
                if last["capped_frac"] > 0:
                    fin.explanation += (f" The online baseline hit its drift cap in {last['capped_frac'] * 100:.0f} % "
                                        f"of bins: the slow change was flagged, not absorbed.")
                alerts.append(fin)

        # ---- layer (a)
        found, st["layer_a"] = _layer_a(prep, cfg, params)
        for name, signal, direction, lim, ep, bad in found:
            rows = episode_rows(ep, n) & bad
            op = operating_point(prep, rows, params)
            a = Alert(prep.engine_id, "a", signal, direction, "critical", df["t"].iloc[ep[0].i0],
                      df["t"].iloc[ep[-1].i1], float(hours[ep[0].i0]), float(hours[ep[-1].i1]),
                      float(sum(r.duration_s for r in ep)), len(ep), op, lim.value,
                      float(np.median(df[signal].to_numpy()[rows])), limit=f"{name} = {lim.describe()}")
            window = np.zeros(n, bool)
            window[ep[0].i0:ep[-1].i1 + 1] = True
            sb = baselines.get((prep.engine_id, signal))
            alerts.append(_finish(a, prep, sb, window, learn_ref, params, table, dtc_episodes, events,
                                  (float(hours[ep[0].i0]), float(hours[ep[-1].i1]))))

        # ---- learning-period announcement
        msgs = [v["message"] for v in st["learned"].values() if v.get("status") in ("learning", "active", "no_data")]
        learning = [s for s, v in st["learned"].items() if v.get("status") == "learning"]
        if learning:
            st["learning_announcement"] = (f"NEW ENGINE / LEARNING PERIOD for {', '.join(learning)}: "
                                           + " | ".join(st["learned"][s]["message"] for s in learning)
                                           + " Layer (a) manual thresholds run meanwhile where configured.")
        else:
            st["learning_announcement"] = " | ".join(f"{s}: {v['message']}" for s, v in st["learned"].items()) \
                if msgs else "no learnable signals"
        st["dm1_guard"] = "active" if prep.dm1_known else "unavailable (DM1 not broadcast)"
        status[prep.engine_id] = st

    alerts.sort(key=lambda a: (a.engine_id, str(a.start_t), a.layer))
    return AnalysisResult(alerts, status, baselines, prepared)


__all__ = ["analyze", "AnalysisResult", "fmt_value"]
