"""Exposure report: time x severity per operating condition (phase 3).

EXPOSURE IS NOT LIFE LOST. These numbers say how long, and how far, this engine
was operated in each condition. Without failure labels nothing here says that a
condition shortened the engine's life, and no formula converts exposure into
life consumed.

Severity units are physical and stated per row (z-hours, kPa-hours below a
limit, degC-hours, %-load-hours, counts). A condition whose definition needs a
value from the engine manual is "unavailable" until the engine config has it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import Prepared, SignalBaseline
from .derived import idle_mask, starts_and_since
from .detection import find_runs
from .engine_config import EngineConfig

COLUMNS = ["condition", "available", "reason", "hours", "exposure", "unit", "episodes", "hours_per_100_running_h",
           "definition", "by_period", "detail"]


def _row(condition, definition, unit, mask=None, severity=None, prep=None, params=None, episodes=None,
         exposure=None, detail=None):
    dt_h = prep.dt_s / 3600.0
    running_h = prep.df["_running"].sum() * dt_h
    hours = float(mask.sum() * dt_h) if mask is not None else float("nan")
    if exposure is None:
        exposure = float(np.nansum(severity[mask]) * dt_h) if severity is not None else hours
    if episodes is None and mask is not None:
        episodes = len(find_runs(mask, prep, 0.0))
    by_period = {}
    if mask is not None:
        h = prep.df["_hours"].to_numpy()
        P = params["derived"]["period_h"]
        per = np.floor((h - h.min()) / P)
        run = prep.df["_running"].to_numpy()
        for p in np.unique(per):
            inp = per == p
            rh = run[inp].sum() * dt_h
            if rh > 0:
                by_period[f"{h.min() + p * P:.0f}-{h.min() + (p + 1) * P:.0f} h"] = round(
                    100.0 * mask[inp].sum() * dt_h / rh, 3)
    return {"condition": condition, "available": True, "reason": "", "hours": hours, "exposure": exposure,
            "unit": unit, "episodes": episodes,
            "hours_per_100_running_h": 100.0 * hours / running_h if running_h and hours == hours else None,
            "definition": definition, "by_period": by_period, "detail": detail or {}}


def _na(condition, reason):
    return {"condition": condition, "available": False, "reason": reason, "hours": None, "exposure": None,
            "unit": "", "episodes": None, "hours_per_100_running_h": None, "definition": "", "by_period": {},
            "detail": {}}


def exposure_report(prep: Prepared, cfg: EngineConfig | None, params: dict,
                    oil_baseline: SignalBaseline | None = None) -> pd.DataFrame:
    dp = params["derived"]
    df = prep.df
    n = len(df)
    running = df["_running"].to_numpy()
    judge = df["_judgeable"].to_numpy()
    rpm = df["engine_speed_rpm"].to_numpy(dtype=float)
    load = df[prep.load_col].to_numpy(dtype=float) if prep.load_col else np.full(n, np.nan)
    col = lambda c: df[c].to_numpy(dtype=float) if c in df else np.full(n, np.nan)  # noqa: E731
    oilp, oilt, cool = col("oil_pressure_kpa"), col("oil_temp_c"), col("coolant_temp_c")
    lim = (lambda name: cfg.limit(name) if cfg else None)
    rows = []

    # 1. oil pressure below this engine's own learned baseline (the headline question, as exposure)
    name = "oil_pressure_below_own_baseline"
    if oil_baseline is None or oil_baseline.status != "active" or oil_baseline.z is None:
        why = oil_baseline.message if oil_baseline is not None else "no oil pressure baseline"
        rows.append(_na(name, why))
    else:
        z = oil_baseline.z
        m = np.nan_to_num(z <= -dp["exposure_z_min"], nan=False).astype(bool)
        sev = np.where(m, -z, 0.0)
        band = np.floor(rpm / 500) * 500
        detail = {}
        for b in np.unique(band[m]):
            mb = m & (band == b)
            detail[f"{b:.0f}-{b + 500:.0f} rpm"] = {"hours": float(mb.sum() * prep.dt_s / 3600),
                                                    "z_hours": float(sev[mb].sum() * prep.dt_s / 3600)}
        rows.append(_row(name, f"z <= -{dp['exposure_z_min']} against this engine's learned (RPM x oil temp x load) "
                               f"baseline; severity = -z", "z-hours", m, sev, prep, params,
                         detail={"by_rpm_band": detail}))

    # 2. oil pressure below the configured minimums
    for lname, scope_txt in (("oil_pressure_min_idle", "all running speeds"), ("oil_pressure_min_rated", "rated speed")):
        L = lim(lname)
        cname = f"oil_pressure_below_{lname.replace('oil_pressure_', '')}"
        if L is None:
            rows.append(_na(cname, f"{lname} missing from engine config"))
            continue
        scope = judge.copy()
        if lname == "oil_pressure_min_rated":
            r = lim("rated_rpm")
            if r is None:
                rows.append(_na(cname, "rated_rpm missing from engine config"))
                continue
            scope &= rpm >= r.value - params["alerts"]["rated_speed_tolerance_rpm"]
        m = scope & np.isfinite(oilp) & (oilp < L.value)
        rows.append(_row(cname, f"oil pressure < {L.describe()} at {scope_txt}; severity = kPa below the limit",
                         "kPa-hours", m, L.value - oilp, prep, params))

    # 3. over-temperature
    for lname, v, label in (("oil_temp_max", oilt, "oil"), ("coolant_temp_max", cool, "coolant")):
        L = lim(lname)
        if L is None:
            rows.append(_na(f"{label}_temp_above_max", f"{lname} missing from engine config"))
            continue
        m = running & np.isfinite(v) & (v > L.value)
        rows.append(_row(f"{label}_temp_above_max", f"{label} temperature > {L.describe()}; severity = C above",
                         "C-hours", m, v - L.value, prep, params))

    # 4. lugging
    L = lim("lugging_rpm_max")
    if L is None:
        rows.append(_na("lugging", "lugging_rpm_max missing from engine config (manual: below peak-torque speed)"))
    elif prep.load_col is None:
        rows.append(_na("lugging", "no load signal"))
    else:
        m = running & (rpm <= L.value) & (load >= dp["lugging_load_min_pct"])
        rows.append(_row("lugging", f"rpm <= {L.describe()} AND load >= {dp['lugging_load_min_pct']} % "
                                    f"(program definition); severity = load % above that", "%-load-hours",
                         m, load - dp["lugging_load_min_pct"], prep, params))

    # 5. cold oil at high RPM
    Lc, Lr = lim("oil_low_temp_cutoff"), lim("rated_rpm")
    if Lc is None or Lr is None:
        rows.append(_na("cold_oil_high_rpm", "needs oil_low_temp_cutoff and rated_rpm in engine config"))
    else:
        thr = dp["cold_high_rpm_frac_of_rated"] * Lr.value
        m = running & np.isfinite(oilt) & (oilt < Lc.value) & (rpm >= thr)
        rows.append(_row("cold_oil_high_rpm", f"oil temp < {Lc.describe()} AND rpm >= {dp['cold_high_rpm_frac_of_rated']}"
                                              f" x rated ({thr:.0f}); severity = C below cutoff", "C-hours",
                         m, Lc.value - oilt, prep, params))

    # 6. extended idling
    idle, why = idle_mask(prep, cfg)
    if idle is None:
        rows.append(_na("extended_idling", why))
    else:
        runs = [r for r in find_runs(idle, prep, dp["extended_idle_min"] * 60)]
        m = np.zeros(n, bool)
        beyond = 0.0
        for r in runs:
            m[r.i0:r.i1 + 1] = True
            beyond += r.duration_s - dp["extended_idle_min"] * 60
        rows.append(_row("extended_idling", f"idle runs longer than {dp['extended_idle_min']} min (idle per engine "
                                            f"config); exposure = idle time beyond that", "hours beyond threshold",
                         m, None, prep, params, episodes=len(runs), exposure=beyond / 3600.0))

    # 7. high load while cold
    Lc = lim("coolant_low_temp_cutoff")
    if Lc is None or prep.load_col is None:
        rows.append(_na("high_load_while_cold", "needs coolant_low_temp_cutoff in engine config and a load signal"))
    else:
        m = running & np.isfinite(cool) & (cool < Lc.value) & (load >= dp["cold_load_min_pct"])
        rows.append(_row("high_load_while_cold", f"load >= {dp['cold_load_min_pct']} % while coolant < {Lc.describe()}; "
                                                 f"severity = load % above that", "%-load-hours",
                         m, load - dp["cold_load_min_pct"], prep, params))

    # 8. hot shutdowns
    if prep.load_col is None:
        rows.append(_na("hot_shutdown", "no load signal"))
    else:
        t = df["_t_s"].to_numpy()
        seg = df["segment"].to_numpy()
        next_running = np.r_[running[1:], False]
        seg_ends = np.r_[seg[1:] != seg[:-1], True]
        stops = np.flatnonzero(running & (~next_running | seg_ends))
        hot = 0
        cool_s = dp["hot_shutdown_cooldown_min"] * 60
        for s in stops:
            j0 = int(np.searchsorted(t, t[s] - cool_s))
            if np.any(load[j0:s + 1] >= dp["hot_shutdown_load_pct"]):
                hot += 1
        rows.append({**_na("hot_shutdown", ""), "available": True, "reason": "", "exposure": float(hot),
                     "unit": "stops", "episodes": hot,
                     "definition": f"engine stopped within {dp['hot_shutdown_cooldown_min']} min of load >= "
                                   f"{dp['hot_shutdown_load_pct']} % (no cool-down idle)",
                     "detail": {"stops": int(stops.size), "hot_pct_of_stops": 100.0 * hot / max(stops.size, 1)}})

    # 9. low-temperature running after warm-up
    if Lc is None:
        rows.append(_na("low_temp_running_after_warmup", "coolant_low_temp_cutoff missing from engine config"))
    else:
        _, since = starts_and_since(prep)
        m = running & (since >= dp["warmup_exclusion_min"] * 60) & np.isfinite(cool) & (cool < Lc.value)
        rows.append(_row("low_temp_running_after_warmup", f"coolant < {Lc.describe()} more than "
                                                          f"{dp['warmup_exclusion_min']} min after start; severity = C below",
                         "C-hours", m, Lc.value - cool, prep, params))
    return pd.DataFrame(rows, columns=COLUMNS)
