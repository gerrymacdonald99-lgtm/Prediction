"""Health score 0-100, ALWAYS with its decomposition by signal.

The score summarises how far this engine currently is from its own learned behaviour and
its configured limits. It is NOT remaining life, NOT a failure probability, and two
engines' scores are comparable only as "distance from their own normal".

score = weighted mean of the available components' sub-scores (each 0-100);
points_lost(component) = weight / sum(available weights) x (100 - sub-score), so
100 - sum(points_lost) == score exactly. Unavailable components are listed with the
reason and weight 0. "Recent" = the last `trend_window_h` engine hours.

Sub-scores (program definitions, tunable in defaults.yaml `health:`):
  learned signal (oil pressure, filter dp): min(drift part, alert part)
      drift part: 100 inside +-drift_alert/2 of the reference, falling linearly to 0 at
      2 x drift_alert;  alert part: recent layer (a) -> 0, warning -> 40, watch -> 70
  limit-only signal (oil / coolant temperature): recent layer (a) breach -> 0, else 100
  joint anomaly: 100 until the recent anomaly rate exceeds 3 x its reference rate, then
      linearly to 0 at +10 points
  relationships: 100 - 50 per learned relationship currently BROKEN (unexplained)
  faults: 100 x (1 - share of recent rows with an active DM1 code)
  data quality: 100 x (1 - error-coded share / 5 %)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .framing import HONEST_FRAMING

NOTE = ("Health score = distance from this engine's own learned normal and its configured limits. It is not "
        "remaining life and not a failure probability.")


@dataclass
class Health:
    engine_id: str
    score: float | None
    components: list[dict] = field(default_factory=list)
    note: str = NOTE
    framing: str = HONEST_FRAMING

    def table(self) -> str:
        rows = [f"Health {self.engine_id}: {'n/a' if self.score is None else f'{self.score:.0f}/100'}  ({self.note})"]
        for c in self.components:
            if c["available"]:
                rows.append(f"  {c['component']:<22} sub-score {c['subscore']:5.1f}  weight {c['weight']:.1f}  "
                            f"-{c['points_lost']:.1f} pts  {c['reason']}")
            else:
                rows.append(f"  {c['component']:<22} unavailable: {c['reason']}")
        return "\n".join(rows)


def _recent_alerts(alerts, engine_id, signal, h_from):
    return [a for a in alerts if a.engine_id == engine_id and a.signal == signal and a.end_h >= h_from]


def health_score(engine_id: str, analysis, params: dict, anomaly=None, correlations=None) -> Health:
    hp, ap = params["health"], params["alerts"]
    w = hp["weights"]
    prep = analysis.prepared[engine_id]
    df = prep.df
    hours = df["_hours"].to_numpy()
    h_from = float(hours.max()) - ap["trend_window_h"]
    recent = hours >= h_from
    comps = []

    def add(name, weight, sub=None, reason=""):
        comps.append({"component": name, "weight": float(weight) if sub is not None else 0.0,
                      "subscore": None if sub is None else float(np.clip(sub, 0, 100)), "reason": reason,
                      "available": sub is not None})

    thr = ap["drift_alert_rel"]
    for sig in params["baseline"]["signals"]:
        sb = analysis.baselines.get((engine_id, sig))
        if sb is None or sb.status != "active" or sb.trend is None or len(sb.trend) == 0:
            add(sig, w.get(sig, 1), None, sb.message if sb is not None else f"{sig} not broadcast")
            continue
        d = float(np.median(sb.trend["drift_rel"].to_numpy()[-3:]))
        drift_part = 100 * (1 - max(0.0, abs(d) - thr / 2) / (1.5 * thr))
        al = _recent_alerts(analysis.alerts, engine_id, sig, h_from)
        worst = min((hp["alert_subscore"][a.severity] for a in al), default=100)
        why = f"drift {d * 100:+.1f} % vs reference"
        if al:
            why += f"; {len(al)} recent alert(s), worst {min(al, key=lambda a: hp['alert_subscore'][a.severity]).severity}"
            if any(a.sensor_verdict == "sensor_suspected" for a in al):
                why += " (sensor suspected: verify the reading)"
        add(sig, w.get(sig, 1), min(drift_part, worst), why)

    for sig in ("oil_temp_c", "coolant_temp_c"):
        layer_a = analysis.status[engine_id]["layer_a"]
        lim = {"oil_temp_c": "oil_temp_max", "coolant_temp_c": "coolant_temp_max"}[sig]
        if not str(layer_a.get(lim, "")).startswith("active"):
            add(sig, w.get(sig, 1), None, f"layer (a) {layer_a.get(lim, 'disabled')}")
            continue
        al = [a for a in _recent_alerts(analysis.alerts, engine_id, sig, h_from) if a.layer == "a"]
        add(sig, w.get(sig, 1), 0 if al else 100, f"{len(al)} recent limit breach(es)" if al else "within configured limit")

    if anomaly is None or not anomaly.available or anomaly.recent_rate is None:
        add("joint_anomaly", w["joint_anomaly"], None, anomaly.reason if anomaly is not None else "not run")
    else:
        excess = max(0.0, anomaly.recent_rate - 3 * anomaly.reference_rate)
        add("joint_anomaly", w["joint_anomaly"], 100 * (1 - excess / hp["anomaly_excess_zero"]),
            f"recent anomaly rate {anomaly.recent_rate * 100:.2f} % vs reference {anomaly.reference_rate * 100:.2f} %")

    if correlations is None:
        add("relationships", w["relationships"], None, "correlation report not run")
    else:
        mine = [r for r in correlations.relationships if r.engine_id == engine_id and r.kind == "curve" and r.available]
        if not mine:
            add("relationships", w["relationships"], None, "no learned relationships")
        else:
            broken = [r.name for r in mine if r.status == "BROKEN"]
            add("relationships", w["relationships"], 100 - 50 * len(broken),
                f"{len(broken)} of {len(mine)} learned relationships broken" + (f": {', '.join(broken)}" if broken else ""))

    if prep.dm1_known:
        frac = float(df["_dm1_active"].to_numpy()[recent].mean()) if recent.any() else 0.0
        add("faults", w["faults"], 100 * (1 - frac), f"DM1 code active in {frac * 100:.1f} % of recent rows")
    else:
        add("faults", w["faults"], None, "DM1 not broadcast")

    err_cols = [c for c in df.columns if c.endswith("__err")]
    if err_cols:
        n_err = float(df.loc[recent, err_cols].to_numpy().sum())
        base = [c[:-5] for c in err_cols if c[:-5] in df]
        n_ok = float(df.loc[recent, base].notna().to_numpy().sum())
        frac = n_err / max(n_err + n_ok, 1.0)
        add("data_quality", w["data_quality"], 100 * (1 - frac / hp["error_fraction_zero"]),
            f"{frac * 100:.2f} % of recent samples error-coded")
    else:
        add("data_quality", w["data_quality"], 100, "no error-coded samples")

    avail = [c for c in comps if c["available"]]
    total = sum(c["weight"] for c in avail)
    for c in comps:
        c["points_lost"] = (c["weight"] / total * (100 - c["subscore"])) if c["available"] and total else 0.0
    score = (100 - sum(c["points_lost"] for c in avail)) if total else None
    h = Health(engine_id, score, comps)
    learning = [s for s, v in analysis.status[engine_id]["learned"].items() if v.get("status") == "learning"]
    if learning:
        h.note += f" LEARNING PERIOD: {', '.join(learning)} not yet included."
    if score is None:
        h.note += " No component could be computed (see reasons)."
    return h
