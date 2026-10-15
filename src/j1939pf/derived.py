"""Derived usage values per engine (phase 3).

Every metric is either computed from this engine's J1939 data, or reported as
unavailable with the reason (signal not broadcast, limit missing from the engine
config, row interval too coarse). Engine-specific cut-offs come ONLY from the
engine config; the thresholds in `derived:` of defaults.yaml are program
definitions, labelled as such.

Time accounting: every canonical row stands for one row interval (the canonical
resample interval), so "hours" = rows x interval / 3600.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .baseline import Prepared
from .engine_config import EngineConfig


def unavailable(reason: str) -> dict:
    return {"available": False, "reason": reason}


def starts_and_since(prep: Prepared) -> tuple[np.ndarray, np.ndarray]:
    """Row indices where the engine starts running, and seconds since the current start (NaN when stopped)."""
    df = prep.df
    running = df["_running"].to_numpy()
    seg = df["segment"].to_numpy()
    t = df["_t_s"].to_numpy()
    prev = np.r_[False, running[:-1]]
    new_seg = np.r_[True, seg[1:] != seg[:-1]]
    start = running & (~prev | new_seg)
    rid = np.cumsum(start)
    st = pd.Series(t).groupby(rid).transform("first").to_numpy()
    return np.flatnonzero(start), np.where(running, t - st, np.nan)


def idle_mask(prep: Prepared, cfg: EngineConfig | None) -> tuple[np.ndarray | None, str]:
    """Idle = running AND idle_rpm_band.low <= RPM <= idle_rpm_band.high AND load < idle_load_cutoff."""
    if cfg is None:
        return None, "no engine config (idle band and idle load cutoff come from the manual)"
    if cfg.idle_rpm_band is None:
        return None, "idle_rpm_band missing from engine config"
    cut = cfg.limit("idle_load_cutoff")
    if cut is None:
        return None, "idle_load_cutoff missing from engine config"
    if prep.load_col is None:
        return None, "no load signal broadcast (SPN 92 / fallback)"
    df = prep.df
    rpm = df["engine_speed_rpm"].to_numpy(dtype=float)
    load = df[prep.load_col].to_numpy(dtype=float)
    m = (df["_running"].to_numpy() & (rpm >= cfg.idle_rpm_band.low) & (rpm <= cfg.idle_rpm_band.high)
         & np.isfinite(load) & (load < cut.value))
    return m, ""


def _hours(prep: Prepared, mask: np.ndarray) -> float:
    return float(mask.sum()) * prep.dt_s / 3600.0


def _trend(x: np.ndarray, y: np.ndarray, scale: float = 100.0) -> dict:
    if len(x) < 3 or np.ptp(x) <= 0:
        return {}
    s, _, lo, hi = theilslopes(y, x, 0.95)
    return {"slope_per_100h": s * scale, "ci95": (lo * scale, hi * scale), "points": int(len(x))}


def derived_metrics(prep: Prepared, cfg: EngineConfig | None, params: dict,
                    dtc_episodes: pd.DataFrame | None = None, oil=None) -> dict[str, dict]:
    dp = params["derived"]
    df = prep.df
    n = len(df)
    dt = prep.dt_s
    running = df["_running"].to_numpy()
    rpm = df["engine_speed_rpm"].to_numpy(dtype=float)
    hours = df["_hours"].to_numpy()
    load = df[prep.load_col].to_numpy(dtype=float) if prep.load_col else np.full(n, np.nan)
    starts, since = starts_and_since(prep)
    run_n = int(running.sum())
    out: dict[str, dict] = {}

    out["time"] = {"available": True, "row_interval_s": dt, "running_h": _hours(prep, running),
                   "key_on_engine_off_h": _hours(prep, ~running & np.isfinite(rpm)),
                   "engine_hours_span": (float(hours.min()), float(hours.max())), "starts": int(starts.size)}

    # ---- idle % and split -------------------------------------------------------------------
    idle, why = idle_mask(prep, cfg)
    if idle is None:
        out["idle"] = unavailable(why)
        out["usage_split"] = unavailable(why)
    else:
        idle_rows = int(idle.sum())
        unknown = int((running & ~np.isfinite(load)).sum())
        pct = 100.0 * idle_rows / run_n if run_n else float("nan")
        thr = cfg.limit("idle_pct_flag_threshold")
        out["idle"] = {"available": True, "idle_pct": pct, "idle_s": idle_rows * dt, "running_s": run_n * dt,
                       "load_unknown_running_pct": 100.0 * unknown / run_n if run_n else float("nan"),
                       "definition": (f"running AND {cfg.idle_rpm_band.low:.0f} <= rpm <= {cfg.idle_rpm_band.high:.0f} "
                                      f"AND {prep.load_col} < {cfg.limit('idle_load_cutoff').value:.0f} %"),
                       "flag_threshold_pct": thr.value if thr else None,
                       "flagged": bool(thr is not None and pct > thr.value)}
        out["usage_split"] = {"available": True,
                              "idle_pct": pct, "working_pct": 100.0 - pct if run_n else float("nan"),
                              "pto": "unavailable: no verified PTO status SPN in the table",
                              "driving": "not separated: vehicle speed is not decoded; 'working' = running, not idle"}

    # ---- cold starts, time to temperature, low-temp running ---------------------------------
    for sig, lim_name, label in (("coolant_temp_c", "coolant_low_temp_cutoff", "coolant"),
                                 ("oil_temp_c", "oil_low_temp_cutoff", "oil")):
        lim = cfg.limit(lim_name) if cfg else None
        if lim is None:
            out[f"cold_starts_{label}"] = unavailable(f"{lim_name} missing from engine config")
            out[f"low_temp_running_{label}"] = unavailable(f"{lim_name} missing from engine config")
            continue
        if sig not in df or df[sig].notna().sum() == 0:
            out[f"cold_starts_{label}"] = unavailable(f"{sig} not broadcast")
            out[f"low_temp_running_{label}"] = unavailable(f"{sig} not broadcast")
            continue
        v = df[sig].to_numpy(dtype=float)
        seg = df["segment"].to_numpy()
        n_cold, ttt, censored = 0, [], 0
        for s in starts:
            win = v[s:s + 3]
            win = win[np.isfinite(win)]
            if win.size == 0 or win[0] >= lim.value:
                continue
            n_cold += 1
            j = s
            while j < n and seg[j] == seg[s] and not (np.isfinite(v[j]) and v[j] >= lim.value):
                j += 1
            if j < n and seg[j] == seg[s]:
                ttt.append((df["_t_s"].iat[j] - df["_t_s"].iat[s]) / 60.0)
            else:
                censored += 1
        out[f"cold_starts_{label}"] = {
            "available": True, "cutoff": lim.describe(), "starts": int(starts.size), "cold_starts": n_cold,
            "cold_start_pct": 100.0 * n_cold / starts.size if starts.size else float("nan"),
            "time_to_temp_min_median": float(np.median(ttt)) if ttt else None,
            "time_to_temp_min_p90": float(np.percentile(ttt, 90)) if ttt else None,
            "never_reached_temp": censored}
        low = running & np.isfinite(v) & (v < lim.value)
        after = running & (since >= dp["warmup_exclusion_min"] * 60)
        low_after = low & after
        pct_after = 100.0 * low_after.sum() / after.sum() if after.sum() else float("nan")
        out[f"low_temp_running_{label}"] = {
            "available": True, "cutoff": lim.describe(),
            "low_temp_running_pct": 100.0 * low.sum() / run_n if run_n else float("nan"),
            "low_temp_running_pct_after_warmup": pct_after,
            "warmup_exclusion_min": dp["warmup_exclusion_min"],
            "chronic_cold_threshold_pct": dp["chronic_cold_pct"],
            "flagged_chronic_cold": bool(np.isfinite(pct_after) and pct_after > dp["chronic_cold_pct"])}

    # ---- load histogram -----------------------------------------------------------------------
    if prep.load_col:
        lr = load[running & np.isfinite(load)]
        edges = np.arange(0, 110, 10)
        h, _ = np.histogram(np.clip(lr, 0, 99.999), bins=edges)
        rb = np.floor(rpm[running & np.isfinite(load)] / 250) * 250
        lb = np.floor(np.clip(lr, 0, 99.999) / 25) * 25
        table = pd.crosstab(rb, lb, normalize="all") * 100 if lr.size else pd.DataFrame()
        out["load_histogram"] = {"available": True, "signal": prep.load_col,
                                 "pct_running_time": {f"{a}-{a + 10}%": float(100 * c / max(lr.size, 1))
                                                      for a, c in zip(edges[:-1], h)},
                                 "rpm_x_load_pct": table}
    else:
        out["load_histogram"] = unavailable("no load signal broadcast")

    # ---- fuel per power (BSFC proxy), compared at matched RPM x load bins ---------------------
    need = [c for c in ("fuel_rate_lph", "actual_torque_pct") if c not in df or df[c].notna().sum() == 0]
    if need or not prep.load_col:
        out["fuel_per_power"] = unavailable(f"missing {need or ['load signal']}")
    else:
        fuel = df["fuel_rate_lph"].to_numpy(dtype=float)
        tq = df["actual_torque_pct"].to_numpy(dtype=float)
        m = df["_judgeable"].to_numpy() & (load >= dp["bsfc_min_load_pct"]) & (tq > 5) & np.isfinite(fuel) & (rpm > 0)
        if m.sum() < 100:
            out["fuel_per_power"] = unavailable("too few steady high-load rows")
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                proxy = fuel / (tq / 100.0 * rpm / 1000.0)
            key = (np.floor(rpm / 100) * 1000 + np.floor(np.nan_to_num(load) / 10)).astype(np.int64)
            h0 = hours[m].min()
            L = params["baseline"]["learning_period_h"]
            ref_m = m & (hours < h0 + L)
            ref = pd.DataFrame({"k": key[ref_m], "p": proxy[ref_m]}).groupby("k")["p"].agg(["median", "size"])
            ref = ref[ref["size"] >= 20]["median"]
            blk = params["baseline"]["block_h"]
            pts = []
            for b0 in np.arange(h0 + L, hours.max(), blk):
                cm = m & (hours >= b0) & (hours < b0 + blk)
                g = pd.DataFrame({"k": key[cm], "p": proxy[cm]}).groupby("k")["p"].agg(["median", "size"])
                g = g[(g["size"] >= 10) & g.index.isin(ref.index)]
                if len(g) >= 3:
                    rel = (g["median"] / ref.reindex(g.index) - 1).to_numpy()
                    pts.append((b0 + blk / 2, float(np.average(rel, weights=g["size"].to_numpy())) * 100))
            if len(pts) < 3:
                out["fuel_per_power"] = unavailable("learning period not complete or too few comparable bins")
            else:
                x, y = np.array(pts).T
                out["fuel_per_power"] = {
                    "available": True,
                    "definition": "fuel rate / (actual torque % x rpm/1000) at load >= "
                                  f"{dp['bsfc_min_load_pct']} %, compared per RPM x load bin with the first "
                                  f"{L:g} engine h; a PROXY, not calibrated BSFC (SPN 183 accuracy unknown)",
                    "drift_pct_latest": float(np.median(y[-3:])), "trend": _trend(x, y, 100.0)}

    # ---- cranking -----------------------------------------------------------------------------
    crank_rows = (rpm > 0) & ~running
    if not starts.size:
        out["cranking"] = unavailable("no engine starts in the record")
    else:
        bat = df["battery_v"].to_numpy(dtype=float) if "battery_v" in df and df["battery_v"].notna().any() else None
        dur, sag, sag_h = [], [], []
        for s in starts:
            j = s - 1
            while j >= 0 and crank_rows[j]:
                j -= 1
            c0 = j + 1
            dur.append((s - c0) * dt)
            if bat is not None:
                pre = bat[max(0, c0 - int(300 / dt)):c0]
                pre = pre[np.isfinite(pre)]
                during = bat[c0:s + 1]
                during = during[np.isfinite(during)]
                if pre.size and during.size:
                    sag.append(float(np.median(pre) - during.min()))
                    sag_h.append(hours[s])
        rec = {"available": True, "starts": int(starts.size),
               "crank_duration_s_median": float(np.median(dur)),
               "resolution_note": f"crank duration is resolved only to the {dt:g} s row interval; voltage sag "
                                  f"is the minimum seen in that interval (true minimum may be lower)"}
        if bat is None:
            rec["voltage_sag"] = "unavailable: battery voltage (SPN 168) not broadcast"
        elif sag:
            rec.update({"voltage_sag_v_median": float(np.median(sag)), "voltage_sag_v_max": float(np.max(sag)),
                        "voltage_sag_trend": _trend(np.array(sag_h), np.array(sag), 100.0)})
        out["cranking"] = rec

    # ---- torque spikes ------------------------------------------------------------------------
    if "actual_torque_pct" not in df or df["actual_torque_pct"].notna().sum() == 0:
        out["torque_spikes"] = unavailable("actual torque (SPN 513) not broadcast")
    elif dt > dp["torque_spike_max_dt_s"]:
        out["torque_spikes"] = unavailable(f"row interval {dt:g} s is too coarse (needs <= "
                                           f"{dp['torque_spike_max_dt_s']} s) to see torque spikes")
    else:
        tq = df["actual_torque_pct"].to_numpy(dtype=float)
        t = df["_t_s"].to_numpy()
        spike = (np.diff(tq) >= dp["torque_spike_pct"]) & (np.diff(t) <= dp["torque_spike_max_dt_s"]) & running[1:]
        rh = _hours(prep, running)
        out["torque_spikes"] = {"available": True, "count": int(spike.sum()),
                                "per_100_running_h": 100.0 * spike.sum() / rh if rh else None,
                                "definition": f"rise >= {dp['torque_spike_pct']} %-points between rows "
                                              f"<= {dp['torque_spike_max_dt_s']} s apart"}

    # ---- oil consumption + hours since service (from oil-change tracking) --------------------
    if oil is None:
        out["oil_consumption"] = unavailable("oil-change tracking not run")
        out["hours_since_oil_change"] = unavailable("oil-change tracking not run")
    else:
        iv = oil.intervals
        if len(iv) and "oil_level_trend_pct_per_100h" in iv:
            rows = iv.dropna(subset=["oil_level_trend_pct_per_100h"])
            out["oil_consumption"] = {
                "available": True,
                "per_interval": rows[["from_h", "to_h", "oil_level_trend_pct_per_100h", "oil_level_trend_ci"]]
                .to_dict("records"),
                "note": "oil level (SPN 98) slope while running, in %-points per 100 engine h; the level unit is "
                        "the sender's % and is not converted to litres"}
        else:
            out["oil_consumption"] = unavailable("oil level (SPN 98) not broadcast or intervals too short")
        out["hours_since_oil_change"] = ({"available": True, "hours": oil.hours_since_change,
                                          "source": oil.hours_since_change_source}
                                         if oil.hours_since_change is not None
                                         else unavailable(oil.hours_since_change_source))

    # ---- faults ---------------------------------------------------------------------------------
    if dtc_episodes is None:
        out["faults"] = unavailable("no DTC episodes supplied (DM1 not decoded for this input)")
    else:
        ep = dtc_episodes[dtc_episodes["engine_id"] == prep.engine_id] if len(dtc_episodes) else dtc_episodes
        ep = ep[ep["kind"] == "DM1"] if len(ep) else ep
        rh = _hours(prep, running)
        if len(ep) == 0:
            out["faults"] = {"available": True, "episodes": 0, "per_100_running_h": 0.0, "by_code": {}, "clusters": []}
        else:
            tt = df["t"].to_numpy()
            pos = np.clip(np.searchsorted(tt, ep["first_t"].to_numpy(dtype=tt.dtype)), 0, n - 1)
            eh = hours[pos]
            by = ep.assign(code=ep["spn"].astype(str) + "/" + ep["fmi"].astype(str)).groupby("code").size()
            order = np.argsort(eh)
            clusters, cur = [], [order[0]]
            for a, b in zip(order[:-1], order[1:]):
                if eh[b] - eh[a] <= dp["fault_cluster_gap_h"]:
                    cur.append(b)
                else:
                    clusters.append(cur)
                    cur = [b]
            clusters.append(cur)
            cl = [{"start_h": float(eh[c[0]]), "codes": sorted({f"{ep['spn'].iat[i]}/{ep['fmi'].iat[i]}" for i in c})}
                  for c in clusters if len(c) > 1]
            out["faults"] = {"available": True, "episodes": int(len(ep)),
                             "per_100_running_h": 100.0 * len(ep) / rh if rh else None,
                             "by_code": {k: int(v) for k, v in by.items()}, "clusters": cl}

    # ---- fan, derate ----------------------------------------------------------------------------
    if "fan_speed_pct" in df and df["fan_speed_pct"].notna().any():
        f = df["fan_speed_pct"].to_numpy(dtype=float)
        m = running & np.isfinite(f)
        out["fan_on"] = {"available": True,
                         "fan_on_pct_of_running": 100.0 * (m & (f > dp["fan_on_pct"])).sum() / max(m.sum(), 1),
                         "definition": f"estimated fan speed (SPN 975) > {dp['fan_on_pct']} %"}
    else:
        out["fan_on"] = unavailable("fan speed (SPN 975, OEM-specific) not broadcast")
    out["derate"] = unavailable("no verified derate-status SPN in the SPN table")
    out["dpf_regens"] = regen_metric(prep, params)
    return out


def detect_regens(prep: Prepared, params: dict) -> tuple[np.ndarray | None, str]:
    """Infer active DPF regenerations from DPF differential pressure (no verified regen-status SPN).

    Soot index = DPF dp / fuel rate (fuel rate as an exhaust-flow proxy) on running rows at high flow (fuel rate
    >= `regen_min_flow_frac` x this engine's 95th percentile), where dp is well above its resolution. A regeneration is a drop of the index's 30-min median to below `regen_drop_frac` x the highest of
    the previous `regen_lookback` windows; detections closer than `regen_min_gap_h` engine hours are merged.
    Returns engine hours of inferred regenerations.
    """
    dp = params["derived"]
    df = prep.df
    if "dpf_dp_kpa" not in df or df["dpf_dp_kpa"].notna().sum() == 0:
        return None, "DPF differential pressure (SPN 3251, OEM-specific) not broadcast"
    if "fuel_rate_lph" not in df or df["fuel_rate_lph"].notna().sum() == 0:
        return None, "fuel rate (SPN 183) not broadcast: no exhaust-flow proxy"
    fuel = df["fuel_rate_lph"].to_numpy(dtype=float)
    d = df["dpf_dp_kpa"].to_numpy(dtype=float)
    run = df["_judgeable"].to_numpy() & np.isfinite(fuel) & np.isfinite(d)
    if run.sum() < 200:
        return None, "too few running rows"
    m = run & (fuel >= dp["regen_min_flow_frac"] * np.nanpercentile(fuel[run], 95))  # high exhaust flow only
    h = df["_hours"].to_numpy()[m]
    idx = d[m] / fuel[m]
    w = np.floor(h / 0.5)
    g = pd.DataFrame({"w": w, "h": h, "i": idx}).groupby("w").agg(h=("h", "median"), i=("i", "median"), n=("i", "size"))
    g = g[g["n"] >= 5]
    out, last = [], -np.inf
    vals, hs = g["i"].to_numpy(), g["h"].to_numpy()
    k = dp["regen_lookback"]
    for j in range(k, len(g) - 1):
        prev = vals[j - k:j].max()
        thr = dp["regen_drop_frac"] * prev
        # the drop must persist into the next window (a burnt-off filter stays clean; noise does not)
        if vals[j] < thr and vals[j + 1] < thr and hs[j] - last >= dp["regen_min_gap_h"]:
            out.append(hs[j])
            last = hs[j]
    return np.asarray(out), ""


def regen_metric(prep: Prepared, params: dict) -> dict:
    hrs, why = detect_regens(prep, params)
    if hrs is None:
        return unavailable(why)
    run_h = prep.df["_running"].sum() * prep.dt_s / 3600.0
    return {"available": True, "inferred_regens": int(hrs.size),
            "per_100_running_h": 100.0 * hrs.size / run_h if run_h else None,
            "regen_hours": [round(float(x), 2) for x in hrs],
            "note": "INFERRED from DPF dp drops (no verified regen-status SPN); missed or extra detections possible"}
