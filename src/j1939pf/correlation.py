"""Correlation discovery (phase 4): learned relationships, their strength, and when they break.

Every relationship reports its strength (Spearman rho with a session-bootstrap 95 % CI),
sample size (rows AND independent sessions / periods / engines), the operating range it
is valid for, and its confounders. Correlation is not causation, and a strong correlation
learned on synthetic data says nothing about real engines.

  curve   y vs x inside an operating condition (e.g. oil pressure vs RPM with oil temperature
          held in a narrow band). The reference curve (binned medians) is learned on the same
          early-hours window as the baselines; each later period's median residual is compared
          with the reference spread. Two consecutive periods beyond `break_z` = BROKE.
          A break that coincides with a known change-point (grade change, sensor replacement)
          is reported as explained.
  trend   y vs engine hours (block medians), Theil-Sen slope with 95 % CI.
  period  within one engine, per period: idle % vs oil-pressure degradation, idle % vs
          inferred regeneration frequency, low-temperature running % vs wear trend.
  fleet   the same, one point per engine (needs several engines).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, theilslopes

from .baseline import Prepared
from .config import load_defaults
from .derived import detect_regens, idle_mask, starts_and_since
from .framing import HONEST_FRAMING


@dataclass
class Relationship:
    engine_id: str
    name: str
    kind: str
    x: str
    y: str
    available: bool
    reason: str = ""
    status: str = ""                    # stable | BROKEN | broken (explained) | significant trend | no trend | inconclusive
    rho: float | None = None
    rho_ci: tuple | None = None
    n_rows: int | None = None
    n_independent: int | None = None    # sessions, blocks, periods or engines
    slope: float | None = None
    slope_ci: tuple | None = None
    slope_unit: str = ""
    valid_range: str = ""
    condition: str = ""
    confounders: list = field(default_factory=list)
    breaks: list = field(default_factory=list)
    periods: list = field(default_factory=list)


@dataclass
class CorrelationReport:
    relationships: list[Relationship]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.relationships])

    def broken(self) -> list[Relationship]:
        return [r for r in self.relationships if r.status == "BROKEN"]

    def flags(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in self.broken():
            out.setdefault(r.engine_id, []).append(
                f"RELATIONSHIP BROKE: {r.name} ({r.y} vs {r.x}, learned rho {r.rho:+.2f}, n={r.n_rows}) "
                + "; ".join(b for b in r.breaks if "explained" not in b))
        return out

    def summary_text(self) -> str:
        L = [HONEST_FRAMING, "Correlations describe association in this engine's own data; they are not causes.",
             "=" * 78]
        for r in self.relationships:
            head = f"{r.engine_id} {r.kind:6s} {r.name}"
            if not r.available:
                L.append(f"{head}: unavailable - {r.reason}")
                continue
            ci = f" (95% CI {r.rho_ci[0]:+.2f}..{r.rho_ci[1]:+.2f})" if r.rho_ci else ""
            sl = (f"; slope {r.slope:+.4g} {r.slope_unit}" + (f" (CI {r.slope_ci[0]:+.3g}..{r.slope_ci[1]:+.3g})"
                                                             if r.slope_ci else "")) if r.slope is not None else ""
            L.append(f"{head}: {r.status.upper()} | rho {r.rho:+.2f}{ci}{sl} | n={r.n_rows} rows, "
                     f"{r.n_independent} independent | range {r.valid_range} | condition {r.condition}")
            for b in r.breaks:
                L.append(f"    break: {b}")
            if r.confounders:
                L.append(f"    confounders: {'; '.join(r.confounders)}")
        return "\n".join(L)


# ---------------------------------------------------------------- helpers
def _condition(prep: Prepared, name: str, params: dict, ref: np.ndarray) -> tuple[np.ndarray, str]:
    df = prep.df
    judge = df["_judgeable"].to_numpy()
    if name == "running":
        return judge, "running, past startup grace"
    if name == "after_warmup":
        _, since = starts_and_since(prep)
        w = params["derived"]["warmup_exclusion_min"]
        return judge & (since >= w * 60), f"running, > {w} min after start"
    if name == "steady":
        rpm = df["engine_speed_rpm"].to_numpy(dtype=float)
        seg = df["segment"].to_numpy()
        d = np.abs(np.diff(rpm, prepend=np.nan))
        same = np.r_[False, seg[1:] == seg[:-1]]
        return judge & same & (d <= 50), "steady (RPM change <= 50 rpm between rows)"
    if name == "high_load":
        if prep.load_col is None:
            return np.zeros(len(df), bool), "no load signal"
        return judge & (df[prep.load_col].to_numpy(dtype=float) >= 60), "load >= 60 %"
    if name == "warm_oil":
        ot = df["oil_temp_c"].to_numpy(dtype=float) if "oil_temp_c" in df else np.full(len(df), np.nan)
        base = judge & ref & np.isfinite(ot)
        if base.sum() < 50:
            return np.zeros(len(df), bool), "oil temperature unavailable"
        med = float(np.median(ot[base]))
        return judge & (np.abs(ot - med) <= 5), f"oil temperature {med - 5:.0f}-{med + 5:.0f} C"
    raise ValueError(f"unknown condition {name}")


def _col(prep: Prepared, name: str) -> np.ndarray | None:
    df = prep.df
    if name == "load":
        name = prep.load_col
    if not name or name not in df or df[name].notna().sum() == 0:
        return None
    return df[name].to_numpy(dtype=float)


def _boot_rho(x, y, groups, rng, n_boot):
    ug = np.unique(groups)
    if ug.size < 3:
        return None
    idx_by = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, ug.size, replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        if np.ptp(x[ii]) > 0 and np.ptp(y[ii]) > 0:
            vals.append(spearmanr(x[ii], y[ii]).statistic)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else None


def _events_near(events, engine_id, h0, h1):
    if events is None or len(events) == 0:
        return []
    e = events[(events["engine_id"] == engine_id) & (events["engine_hours"] >= h0) & (events["engine_hours"] <= h1)]
    return [f"{r['event_type']} at {r['engine_hours']:.1f} h" for _, r in e.iterrows()]


# ---------------------------------------------------------------- curves
def _curve(prep, spec, params, rng, events, resolution, ref_end):
    cp = params["correlations"]
    df = prep.df
    rel = Relationship(prep.engine_id, spec["name"], "curve", spec["x"], spec["y"], False,
                       confounders=list(spec.get("confounders", [])))
    x, y = _col(prep, spec["x"]), _col(prep, spec["y"])
    if x is None or y is None:
        rel.reason = f"{spec['x'] if x is None else spec['y']} not broadcast"
        return rel
    hours = df["_hours"].to_numpy()
    ref = hours < ref_end
    cond, ctext = _condition(prep, spec["condition"], params, ref)
    ok = cond & np.isfinite(x) & np.isfinite(y)
    rm = ok & ref
    if rm.sum() < cp["min_rows"]:
        rel.reason = f"only {int(rm.sum())} rows in the condition during the reference window"
        return rel
    seg = df["segment"].to_numpy()
    sub = np.flatnonzero(rm)
    if sub.size > cp["max_rows"]:
        sub = np.sort(rng.choice(sub, cp["max_rows"], replace=False))
    rel.available = True
    rel.condition = ctext
    rel.n_rows, rel.n_independent = int(rm.sum()), int(np.unique(seg[rm]).size)
    rel.rho = float(spearmanr(x[sub], y[sub]).statistic)
    rel.rho_ci = _boot_rho(x[sub], y[sub], seg[sub], rng, cp["bootstrap"])
    lo, hi = np.percentile(x[rm], [5, 95])
    rel.valid_range = f"{spec['x']} {lo:.4g}..{hi:.4g}"
    # reference curve: medians of y in x quantile bins
    edges = np.unique(np.percentile(x[rm], np.linspace(0, 100, 21)))
    if edges.size < 4:
        rel.status = "stable"
        rel.reason = "x barely varies; no curve"
        return rel
    b = np.clip(np.digitize(x[rm], edges[1:-1]), 0, edges.size - 2)
    cen = np.array([np.median(x[rm][b == i]) if (b == i).any() else np.nan for i in range(edges.size - 1)])
    med = np.array([np.median(y[rm][b == i]) if (b == i).any() else np.nan for i in range(edges.size - 1)])
    good = np.isfinite(cen) & np.isfinite(med)
    cen, med = cen[good], med[good]
    if cen.size >= 3:
        s, _, slo, shi = theilslopes(med, cen, 0.95)
        rel.slope, rel.slope_ci, rel.slope_unit = float(s), (float(slo), float(shi)), f"{spec['y']} per {spec['x']}"
    resid = lambda m: y[m] - np.interp(x[m], cen, med)  # noqa: E731
    r0 = resid(rm)
    spread = max(1.4826 * float(np.median(np.abs(r0 - np.median(r0)))), resolution)
    # later periods
    P = cp["period_h"]
    run = []
    after = ok & ~ref & (x >= lo) & (x <= hi)
    for p0 in np.arange(ref_end, hours.max(), P):
        pm = after & (hours >= p0) & (hours < p0 + P)
        if pm.sum() < 50:
            continue
        r = resid(pm)
        zp = float(np.median(r) / spread)
        rel.periods.append({"from_h": float(p0), "to_h": float(p0 + P), "median_residual": float(np.median(r)),
                            "z": zp, "rho": float(spearmanr(x[pm], y[pm]).statistic) if np.ptp(x[pm]) > 0 else None,
                            "n": int(pm.sum())})
        run = run + [p0] if abs(zp) >= cp["break_z"] else []
        if len(run) == cp["break_periods"]:
            start = run[0]
            first = next(q for q in rel.periods if q["from_h"] == start)
            ymed = float(np.median(y[rm]))
            txt = (f"from {start:.0f} h: {spec['y']} is {first['median_residual']:+.4g} "
                   f"({first['median_residual'] / ymed * 100:+.1f} % of typical) off the learned curve "
                   f"(z {first['z']:+.1f}, n={first['n']})")
            near = _events_near(events, prep.engine_id, start - P, start + P)
            if near:
                txt += f"; coincides with {', '.join(near)} (explained change-point)"
            rel.breaks.append(txt)
    if rel.breaks:
        rel.status = "BROKEN" if any("explained" not in b for b in rel.breaks) else "broken (explained)"
    else:
        rel.status = "stable"
    return rel


# ---------------------------------------------------------------- trends
def _trend(prep, spec, params, analysis, rng):
    df = prep.df
    rel = Relationship(prep.engine_id, spec["name"], "trend", "engine_hours", spec["y"], False,
                       confounders=list(spec.get("confounders", [])))
    if spec["y"] == "oil_pressure_drift":
        sb = analysis.baselines.get((prep.engine_id, "oil_pressure_kpa")) if analysis else None
        if sb is None or sb.trend is None or len(sb.trend) < 3:
            rel.reason = "oil pressure baseline not active (still learning or no data)"
            return rel
        h, v = sb.trend["hours"].to_numpy(), sb.trend["drift_rel"].to_numpy() * 100
        rel.condition, unit, n_rows = "learned baseline bins (layer d)", "% per 100 h", int(sb.trend["n_rows"].sum())
    else:
        y = _col(prep, spec["y"])
        if y is None:
            rel.reason = f"{spec['y']} not broadcast"
            return rel
        cond, rel.condition = _condition(prep, spec["condition"], params, np.ones(len(df), bool))
        m = cond & np.isfinite(y)
        if m.sum() < params["correlations"]["min_rows"]:
            rel.reason = f"only {int(m.sum())} rows in the condition"
            return rel
        hrs = df["_hours"].to_numpy()
        g = pd.DataFrame({"b": np.floor(hrs[m] / params["baseline"]["block_h"]), "h": hrs[m], "y": y[m]}) \
            .groupby("b").agg(h=("h", "median"), y=("y", "median"), n=("y", "size"))
        g = g[g["n"] >= 20]
        h, v, n_rows, unit = g["h"].to_numpy(), g["y"].to_numpy(), int(g["n"].sum()), f"{spec['y']} per 100 h"
    if len(h) < 3 or np.ptp(h) <= 0:
        rel.reason = "fewer than 3 blocks"
        return rel
    s, _, lo, hi = theilslopes(v, h, 0.95)
    rel.available = True
    rel.rho = float(spearmanr(h, v).statistic) if np.ptp(v) > 0 else 0.0
    rel.n_rows, rel.n_independent = n_rows, int(len(h))
    rel.slope, rel.slope_ci, rel.slope_unit = float(s * 100), (float(lo * 100), float(hi * 100)), unit
    rel.valid_range = f"{h.min():.0f}-{h.max():.0f} engine h"
    rel.status = "significant trend" if (lo > 0 or hi < 0) else "no trend"
    return rel


# ---------------------------------------------------------------- per-period / fleet
def _period_table(prep, cfg, params, analysis):
    df = prep.df
    P = params["derived"]["period_h"]
    hours = df["_hours"].to_numpy()
    running = df["_running"].to_numpy()
    idle, _ = idle_mask(prep, cfg)
    lim = cfg.limit("coolant_low_temp_cutoff") if cfg else None
    cool = df["coolant_temp_c"].to_numpy(dtype=float) if "coolant_temp_c" in df else None
    _, since = starts_and_since(prep)
    regens, _ = detect_regens(prep, params)
    sb = analysis.baselines.get((prep.engine_id, "oil_pressure_kpa")) if analysis else None
    tr = sb.trend if sb is not None and sb.trend is not None and len(sb.trend) else None
    rows = []
    for p0 in np.arange(hours.min(), hours.max(), P):
        pm = (hours >= p0) & (hours < p0 + P)
        rn = (running & pm).sum()
        if rn == 0:
            continue
        r = {"from_h": float(p0), "to_h": float(p0 + P), "running_h": rn * prep.dt_s / 3600}
        r["idle_pct"] = 100.0 * (idle & pm).sum() / rn if idle is not None else np.nan
        if lim is not None and cool is not None:
            aw = running & pm & (since >= params["derived"]["warmup_exclusion_min"] * 60)
            r["low_temp_pct"] = 100.0 * (aw & (cool < lim.value)).sum() / max(aw.sum(), 1)
        else:
            r["low_temp_pct"] = np.nan
        r["regens_per_100h"] = (100.0 * ((regens >= p0) & (regens < p0 + P)).sum() / r["running_h"]
                                if regens is not None else np.nan)
        if tr is not None:
            t = tr[(tr["hours"] >= p0) & (tr["hours"] < p0 + P)]
            r["oil_pressure_trend_pct_per_100h"] = (float(theilslopes(t["drift_rel"] * 100, t["hours"])[0] * 100)
                                                    if len(t) >= 3 and np.ptp(t["hours"]) > 0 else np.nan)
        else:
            r["oil_pressure_trend_pct_per_100h"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


_PERIOD_PAIRS = [
    ("idle_pct_vs_oil_pressure_degradation", "idle_pct", "oil_pressure_trend_pct_per_100h",
     ["oil age within the period", "load mix", "season"]),
    ("idle_pct_vs_regen_frequency", "idle_pct", "regens_per_100h",
     ["regen inference error (inferred from DPF dp)", "fuel sulphur / ash", "duty cycle"]),
    ("low_temp_pct_vs_wear_trend", "low_temp_pct", "oil_pressure_trend_pct_per_100h",
     ["ambient temperature", "oil age", "oil grade"]),
]


def _pairs_from_table(engine_id, kind, tab, params, rng):
    out = []
    need = params["correlations"]["min_points_conclusive"]
    for name, xa, ya, conf in _PERIOD_PAIRS:
        rel = Relationship(engine_id, name, kind, xa, ya, False, confounders=list(conf))
        if tab is None or len(tab) == 0 or xa not in tab or ya not in tab:
            rel.reason = "no data"
            out.append(rel)
            continue
        t = tab[[xa, ya]].dropna()
        if len(t) < 3:
            miss = xa if tab[xa].notna().sum() < 3 else ya
            rel.reason = f"{miss} unavailable or fewer than 3 points"
            out.append(rel)
            continue
        rel.available = True
        rel.n_rows = rel.n_independent = int(len(t))
        if np.ptp(t[xa]) == 0 or np.ptp(t[ya]) == 0:
            rel.rho, rel.status = 0.0, "inconclusive (no variation)"
        else:
            rel.rho = float(spearmanr(t[xa], t[ya]).statistic)
            boots = []
            for _ in range(params["correlations"]["bootstrap"]):
                ii = rng.integers(0, len(t), len(t))
                a, b = t[xa].to_numpy()[ii], t[ya].to_numpy()[ii]
                if np.ptp(a) > 0 and np.ptp(b) > 0:
                    boots.append(spearmanr(a, b).statistic)
            rel.rho_ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else None
            rel.status = (f"inconclusive (n={len(t)} < {need}: do not draw conclusions)" if len(t) < need
                          else "association" if rel.rho_ci and (rel.rho_ci[0] > 0 or rel.rho_ci[1] < 0)
                          else "no clear association")
        rel.valid_range = f"{xa} {t[xa].min():.3g}..{t[xa].max():.3g}"
        rel.condition = "per period" if kind == "period" else "one point per engine"
        out.append(rel)
    return out


def correlation_report(analysis, engine_configs=None, *, events: pd.DataFrame | None = None,
                       params: dict | None = None, table=None) -> CorrelationReport:
    from .spn_table import load_spn_table
    params = params or load_defaults()
    table = table or load_spn_table()
    engine_configs = engine_configs or {}
    rng = np.random.default_rng(params.get("seed", 0))
    rels: list[Relationship] = []
    fleet_rows = []
    for eid, prep in analysis.prepared.items():
        cfg = engine_configs.get(eid)
        sb = analysis.baselines.get((eid, "oil_pressure_kpa"))
        hours = prep.df["_hours"].to_numpy()
        ref_end = sb.learn_end_h if sb is not None and sb.learn_end_h else hours.min() + params["baseline"]["learning_period_h"]
        for spec in params["correlations"]["curves"]:
            d = table.by_canonical.get(spec["y"])
            rels.append(_curve(prep, spec, params, rng, events, d.scale if d else 0.0, ref_end))
        for spec in params["correlations"]["trends"]:
            rels.append(_trend(prep, spec, params, analysis, rng))
        tab = _period_table(prep, cfg, params, analysis)
        rels.extend(_pairs_from_table(eid, "period", tab, params, rng))
        if len(tab):
            agg = {"engine_id": eid}
            run = tab["running_h"].sum()
            for c in ("idle_pct", "low_temp_pct", "regens_per_100h"):
                agg[c] = float(np.average(tab[c], weights=tab["running_h"])) if tab[c].notna().all() else np.nan
            t = sb.trend if sb is not None and sb.trend is not None else None
            agg["oil_pressure_trend_pct_per_100h"] = (float(theilslopes(t["drift_rel"] * 100, t["hours"])[0] * 100)
                                                      if t is not None and len(t) >= 3 else np.nan)
            agg["running_h"] = run
            fleet_rows.append(agg)
    if len(fleet_rows) >= 3:
        rels.extend(_pairs_from_table("FLEET", "fleet", pd.DataFrame(fleet_rows), params, rng))
    else:
        for name, xa, ya, conf in _PERIOD_PAIRS:
            rels.append(Relationship("FLEET", name, "fleet", xa, ya, False,
                                     reason=f"needs at least 3 engines (have {len(fleet_rows)})", confounders=list(conf)))
    return CorrelationReport(rels)
