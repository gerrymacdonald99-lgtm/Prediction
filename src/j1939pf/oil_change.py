"""Oil-change detection from J1939 signatures, and oil service tracking.

An oil change needs the engine stopped, so candidates are engine-off boundaries
(between sessions, or a stop of at least `min_off_minutes`). At each boundary
this engine's own data from just before is compared with just after, at matched
operating points:

  primary    oil pressure at the same RPM x oil temperature x load bins: fresh oil
             restores viscosity lost to shear / dilution, so pressure steps UP
  supporting oil level steps up (SPN 98); filter dp steps down (new filter, SPN 99);
             the oil temperature vs RPM/load relationship; the slope of ln(pressure)
             vs oil temperature at fixed RPM (viscosity-temperature behaviour)
  exclusion  an engine-off (zero) reading that shifts at the same boundary means the
             pressure SENSOR changed, not the oil

Steps are judged against this engine's own boundary-to-boundary noise (robust z
over all boundaries), so the threshold calibrates itself per engine. A pressure
step DOWN is reported as an oil-system step (thinner grade? dilution?) and is not
counted as an oil change unless the maintenance log says so.

Detections are reconciled with the maintenance log: confirmed, logged-but-no-
signature (e.g. same oil after a short interval), or detected-but-unlogged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .baseline import KEY_RPM, KEY_TEMP, Prepared, eligible_mask

SIG = "oil_pressure_kpa"


@dataclass
class OilServiceResult:
    engine_id: str
    boundaries: pd.DataFrame          # every candidate boundary with its measured steps
    history: pd.DataFrame             # oil changes (logged and/or detected) + other oil-system steps
    intervals: pd.DataFrame           # per oil interval: pressure trend, oil consumption
    hours_now: float
    hours_since_change: float | None
    hours_since_change_source: str
    notes: list[str] = field(default_factory=list)


def _bin_means(k: np.ndarray, v: np.ndarray):
    if k.size == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    g = pd.DataFrame({"k": k, "v": v}).groupby("k")["v"]
    return g.mean(), g.size()


def _bin_step(keys, vals, pre, post, min_n, relative=True):
    """Weighted mean over common bins of (post - pre) per bin. pre/post: boolean masks or index arrays."""
    ka, va = keys[pre], vals[pre]
    kb, vb = keys[post], vals[post]
    if ka.size < min_n or kb.size < min_n:
        return np.nan, 0, np.nan
    ma_s, ca = _bin_means(ka, va)
    mb_s, cb = _bin_means(kb, vb)
    common = ca.index[(ca >= min_n)].intersection(cb.index[cb >= min_n])
    if len(common) == 0:
        return np.nan, 0, np.nan
    ma = ma_s.reindex(common).to_numpy()
    mb = mb_s.reindex(common).to_numpy()
    ok = np.abs(ma) > 1e-9 if relative else np.ones(len(common), bool)
    d = ((mb - ma) / np.abs(ma))[ok] if relative else (mb - ma)
    w = np.sqrt(np.minimum(ca.reindex(common).to_numpy(), cb.reindex(common).to_numpy()))[ok]
    if d.size == 0:
        return np.nan, 0, np.nan
    med = np.median(d)
    mad = np.median(np.abs(d - med)) * 1.4826 + 1e-12
    keep = np.abs(d - med) <= 3 * mad + 1e-9
    step = float(np.average(d[keep], weights=w[keep]))
    agree = float(np.mean(np.sign(d) == np.sign(step))) if step != 0 else 0.0
    return step, int(d.size), agree


def _pt_slope(p, t, r, rpm_w=200):
    """Median over RPM bands of the slope of ln(pressure) vs oil temperature (per 10 C)."""
    ok = (p > 0) & np.isfinite(t) & np.isfinite(r)
    p, t, band = p[ok], t[ok], np.floor(r[ok] / rpm_w)
    slopes = []
    for b in np.unique(band):
        m = band == b
        if m.sum() >= 30 and np.ptp(t[m]) >= 8:
            slopes.append(np.polyfit(t[m], np.log(p[m]), 1)[0] * 10)
    return float(np.median(slopes)) if slopes else np.nan


def _robust_z(x: np.ndarray, floor: float) -> np.ndarray:
    """z against this engine's own boundary-to-boundary spread; `floor` keeps quantized signals honest."""
    f = np.isfinite(x)
    if f.sum() < 5:
        return np.full(len(x), np.nan)
    med = np.median(x[f])
    scale = max(1.4826 * np.median(np.abs(x[f] - med)), floor)
    return (x - med) / scale


def detect_oil_changes(prep: Prepared, params: dict, events: pd.DataFrame | None = None,
                       resolution: float = 4.0) -> OilServiceResult:
    oc = params["oil_change"]
    bp = params["baseline"]
    df = prep.df
    n = len(df)
    hours = df["_hours"].to_numpy()
    t_s = df["_t_s"].to_numpy()
    running = df["_running"].to_numpy()
    seg = df["segment"].to_numpy()
    notes: list[str] = []
    empty = pd.DataFrame()
    if SIG not in df or df[SIG].notna().sum() == 0:
        return OilServiceResult(prep.engine_id, empty, empty, empty, float(hours.max()), None,
                                "unavailable", ["oil pressure not broadcast: no signature-based detection"])

    elig = eligible_mask(prep, SIG)
    keys = df["_bin"].to_numpy()
    vals = df[SIG].to_numpy(dtype=float)
    # operating point without oil temperature (for the oil-temperature-vs-RPM/load relationship)
    key_rl = (keys // KEY_RPM) * 1000 + (keys % KEY_TEMP)
    oil_t = df["oil_temp_c"].to_numpy(dtype=float) if "oil_temp_c" in df else np.full(n, np.nan)
    warm = df["_judgeable"].to_numpy()
    lvl = df["oil_level_pct"].to_numpy(dtype=float) if "oil_level_pct" in df else None
    dpv = df["oil_filter_dp_kpa"].to_numpy(dtype=float) if "oil_filter_dp_kpa" in df else None
    stopped = (~running) & (df["_stopped_s"].to_numpy() >= params["sensor_check"]["zero_check_settle_s"])

    # candidate boundaries: a running period that starts after a long enough stop
    prev_run = np.r_[False, running[:-1]]
    new_seg = np.r_[True, seg[1:] != seg[:-1]]
    starts = np.flatnonzero(running & (~prev_run | new_seg))
    run_idx = np.flatnonzero(running)
    W = float(oc["window_h"])
    # engine hours are non-decreasing in a sane record; a running maximum keeps searchsorted valid otherwise
    h_mono = np.maximum.accumulate(hours)
    rpm = df["engine_speed_rpm"].to_numpy(dtype=float)
    okt = np.isfinite(oil_t)
    okd = np.isfinite(dpv) if dpv is not None else None
    seg_first = pd.Series(np.arange(n)).groupby(seg).transform("first").to_numpy()
    # b starts the first running period of its segment?
    first_run_in_seg = pd.Series(np.where(running, np.arange(n), n)).groupby(seg).transform("min").to_numpy()
    new_seg_start = np.arange(n) == first_run_in_seg
    rows = []
    for b in starts:
        before = run_idx[run_idx < b]
        if before.size == 0:
            continue
        off_min = (t_s[b] - t_s[before[-1]]) / 60.0
        if off_min < oc["min_off_minutes"]:
            continue
        hb = hours[b]
        i0 = int(np.searchsorted(h_mono, hb - W, "left"))
        i1 = int(np.searchsorted(h_mono, hb + W, "left"))
        pre_r, post_r = np.arange(i0, b), np.arange(b, i1)
        pre, post = pre_r[elig[pre_r]], post_r[elig[post_r]]
        step, nb, agree = _bin_step(keys, vals, pre, post, oc["min_bin_samples"])
        rec = {"row": int(b), "engine_hours": float(hb), "t": df["t"].iloc[b], "off_minutes": float(off_min),
               "pressure_step_rel": step, "n_bins": nb, "sign_agreement": agree}
        pre_w, post_w = pre_r[warm[pre_r] & okt[pre_r]], post_r[warm[post_r] & okt[post_r]]
        rec["oil_temp_step_c"] = _bin_step(key_rl, oil_t, pre_w, post_w, oc["min_bin_samples"], relative=False)[0]
        pt = [p_[okt[p_]] for p_ in (pre, post)]
        rec["p_vs_t_slope_change"] = (_pt_slope(vals[pt[1]], oil_t[pt[1]], rpm[pt[1]])
                                      - _pt_slope(vals[pt[0]], oil_t[pt[0]], rpm[pt[0]]))
        if lvl is not None:
            a = lvl[pre_r[warm[pre_r]]]
            c = lvl[post_r[warm[post_r]]]
            a, c = a[np.isfinite(a)], c[np.isfinite(c)]
            rec["level_step_pct"] = float(np.median(c) - np.median(a)) if a.size >= 10 and c.size >= 10 else np.nan
        if dpv is not None:
            rec["filter_dp_step_rel"] = _bin_step(keys, dpv, pre[okd[pre]], post[okd[post]], oc["min_bin_samples"])[0]
        # engine-off (zero) readings: the new session's key-on rows come BEFORE its first running row, so split
        # at the session start (or, for a stop inside a session, at the last running row before it)
        split = int(seg_first[b]) if new_seg_start[b] else int(before[-1]) + 1
        zr0, zr1 = np.arange(i0, split), np.arange(split, i1)
        z0 = vals[zr0[stopped[zr0]]]
        z1 = vals[zr1[stopped[zr1]]]
        z0, z1 = z0[np.isfinite(z0)], z1[np.isfinite(z1)]
        rec["zero_shift_kpa"] = float(np.median(z1) - np.median(z0)) if z0.size >= 3 and z1.size >= 3 else np.nan
        rows.append(rec)
    bnd = pd.DataFrame(rows)
    if bnd.empty:
        notes.append("no engine-off boundaries with data on both sides")
        return OilServiceResult(prep.engine_id, bnd, empty, empty, float(hours.max()), None, "unavailable", notes)

    valid = bnd["n_bins"] >= oc["min_bins"]
    bnd["pressure_step_z"] = np.nan
    bnd.loc[valid, "pressure_step_z"] = _robust_z(bnd.loc[valid, "pressure_step_rel"].to_numpy(), 0.003)
    if "filter_dp_step_rel" in bnd:
        bnd["filter_dp_step_z"] = _robust_z(bnd["filter_dp_step_rel"].to_numpy(), 0.005)
    if "level_step_pct" in bnd:
        bnd["level_step_z"] = _robust_z(bnd["level_step_pct"].to_numpy(), 0.4)  # SPN 98: 0.4 %/bit
    bnd["oil_temp_step_z"] = _robust_z(bnd["oil_temp_step_c"].to_numpy(), 0.5)
    if valid.sum() < 5:
        notes.append(f"only {int(valid.sum())} boundaries with >= {oc['min_bins']} comparable operating bins: "
                     f"too few to calibrate a per-engine noise level; detection disabled")

    tol = params["sensor_check"]["zero_check_tolerance_steps"] * resolution
    kinds, conf, support = [], [], []
    for _, r in bnd.iterrows():
        sup = []
        z = r["pressure_step_z"]
        step = r["pressure_step_rel"]
        if np.isfinite(r.get("zero_shift_kpa", np.nan)) and abs(r["zero_shift_kpa"]) > tol:
            kinds.append("pressure sensor change (engine-off reading shifted)")
            conf.append("medium")
            support.append([f"engine-off reading shifted {r['zero_shift_kpa']:+.1f} kPa"])
            continue
        eps = 1e-6  # quantized steps (0.4 % / 0.4 %) land exactly on thresholds
        level_up = (np.isfinite(r.get("level_step_z", np.nan)) and r["level_step_z"] >= oc["robust_z_min"] - eps
                    and r["level_step_pct"] >= oc["level_step_min_pct"] - eps)
        if valid.sum() < 5 or not np.isfinite(z) or abs(step) < oc["step_min_rel"] or abs(z) < oc["robust_z_min"]:
            if level_up:
                kinds.append("oil level rise (oil added or changed; no pressure signature)")
                conf.append("low")
                support.append([f"oil level up {r['level_step_pct']:+.1f} %",
                                f"pressure step {step * 100:+.1f} % (z {z:+.1f}) not significant"])
            else:
                kinds.append("")
                conf.append("")
                support.append([])
            continue
        if np.isfinite(r.get("level_step_pct", np.nan)) and r["level_step_pct"] >= oc["level_step_min_pct"] - 1e-6:
            sup.append(f"oil level up {r['level_step_pct']:+.1f} %")
        if np.isfinite(r.get("filter_dp_step_z", np.nan)) and r["filter_dp_step_z"] <= -oc["robust_z_min"]:
            sup.append(f"filter dp down {r['filter_dp_step_rel'] * 100:+.1f} % (new filter?)")
        if np.isfinite(r.get("oil_temp_step_z", np.nan)) and abs(r["oil_temp_step_z"]) >= oc["robust_z_min"]:
            sup.append(f"oil temperature vs RPM/load shifted {r['oil_temp_step_c']:+.1f} C")
        if step > 0:
            kinds.append("oil change (pressure signature)")
            conf.append("high" if sup else "medium")
        else:
            kinds.append("oil-system step down (not counted as an oil change)")
            conf.append("low")
        support.append(sup)
    bnd["detected_kind"] = kinds
    bnd["confidence"] = conf
    bnd["supporting"] = support
    # a boundary just after a real change sees the same step (its window straddles the change): keep the strongest
    det_idx = bnd.index[bnd["detected_kind"] != ""].tolist()
    for i in det_idx:
        if bnd.at[i, "detected_kind"] == "":
            continue
        close = [j for j in det_idx if j != i and bnd.at[j, "detected_kind"] != ""
                 and abs(bnd.at[j, "engine_hours"] - bnd.at[i, "engine_hours"]) < W]
        for j in close:
            weaker = j if abs(bnd.at[j, "pressure_step_z"]) <= abs(bnd.at[i, "pressure_step_z"]) else i
            bnd.at[weaker, "detected_kind"] = ""
            bnd.at[weaker, "confidence"] = ""
            if weaker == i:
                break

    # ---- reconcile with the maintenance log
    hist = []
    logged = pd.DataFrame()
    if events is not None and len(events):
        logged = events[(events["engine_id"] == prep.engine_id)].copy()
    det = bnd[bnd["detected_kind"] != ""]
    used = set()
    for _, ev in logged.iterrows():
        if ev["event_type"] not in ("oil_change", "oil_grade_change", "sensor_replacement"):
            continue
        near = bnd[(bnd["engine_hours"] - ev["engine_hours"]).abs() <= oc["match_window_h"]]
        m = det[(det["engine_hours"] - ev["engine_hours"]).abs() <= oc["match_window_h"]]
        if len(m):
            r = m.iloc[(m["engine_hours"] - ev["engine_hours"]).abs().argmin()]
            used.add(int(r["row"]))
            source = "logged+detected"
            expected = {"oil_change": "oil change", "sensor_replacement": "pressure sensor"}.get(ev["event_type"])
            status = ("logged; signature confirms" if expected is None or r["detected_kind"].startswith(expected)
                      else "logged; only the oil level rise is visible (pressure signature not significant)"
                      if r["detected_kind"].startswith("oil level rise")
                      else f"logged; signature found instead: {r['detected_kind']}")
        else:
            r = near.iloc[(near["engine_hours"] - ev["engine_hours"]).abs().argmin()] if len(near) else None
            source = "logged"
            status = "logged; no clear signature (e.g. same oil after a short interval, or too little data)"
        hist.append(_hist_row(prep.engine_id, ev["event_type"], source, float(ev["engine_hours"]), ev.get("t"),
                              status, r))
    for _, r in det.iterrows():
        if int(r["row"]) in used:
            continue
        kind = ("oil_change" if r["detected_kind"].startswith("oil change") else
                "sensor_change" if r["detected_kind"].startswith("pressure sensor") else
                "oil_level_rise" if r["detected_kind"].startswith("oil level rise") else "oil_system_step")
        hist.append(_hist_row(prep.engine_id, kind, "detected", float(r["engine_hours"]), r["t"],
                              "detected from J1939 signature; NOT in the maintenance log", r))
    history = pd.DataFrame(hist, columns=list(_hist_row("", "", "", 0.0, None, "", None)) + ["interval_h"])
    if len(history):
        history = history.sort_values("engine_hours", ignore_index=True)
        ch = history[history["event"] == "oil_change"]
        history["interval_h"] = np.nan
        history.loc[ch.index, "interval_h"] = ch["engine_hours"].diff().to_numpy()

    # ---- hours since the last oil change
    h_now = float(hours.max())
    since, src = None, "unknown (no oil change logged or detected in this record)"
    if len(history):
        ch = history[history["event"] == "oil_change"]
        if len(ch):
            last = ch.iloc[-1]
            since = h_now - float(last["engine_hours"])
            src = f"{last['source']} oil change at {last['engine_hours']:.1f} engine h"

    intervals = _intervals(prep, history, params, elig, keys, vals, lvl)
    return OilServiceResult(prep.engine_id, bnd, history, intervals, h_now, since, src, notes)


def _hist_row(engine_id, event, source, h, t, status, r):
    out = {"engine_id": engine_id, "event": event, "source": source, "engine_hours": h, "t": t, "status": status,
           "confidence": "", "pressure_step_pct": np.nan, "pressure_step_z": np.nan, "level_step_pct": np.nan,
           "filter_dp_step_pct": np.nan, "oil_temp_step_c": np.nan, "p_vs_t_slope_change": np.nan,
           "off_minutes": np.nan, "supporting": []}
    if r is not None:
        out.update({"confidence": r.get("confidence", ""), "pressure_step_pct": r["pressure_step_rel"] * 100,
                    "pressure_step_z": r["pressure_step_z"], "level_step_pct": r.get("level_step_pct", np.nan),
                    "filter_dp_step_pct": r.get("filter_dp_step_rel", np.nan) * 100,
                    "oil_temp_step_c": r.get("oil_temp_step_c", np.nan),
                    "p_vs_t_slope_change": r.get("p_vs_t_slope_change", np.nan), "off_minutes": r["off_minutes"],
                    "supporting": r.get("supporting", [])})
    return out


def _intervals(prep, history, params, elig, keys, vals, lvl) -> pd.DataFrame:
    """Per oil interval: how oil pressure at fixed operating points moved, and oil consumption."""
    df = prep.df
    hours = df["_hours"].to_numpy()
    cuts = [float(hours.min())]
    if len(history):
        cuts += [float(h) for h in history.loc[history["event"] == "oil_change", "engine_hours"]]
    cuts.append(float(hours.max()) + 1e-9)
    block = params["baseline"]["block_h"]
    out = []
    for h0, h1 in zip(cuts[:-1], cuts[1:]):
        m = elig & (hours >= h0) & (hours < h1)
        if (h1 - h0) < 3 * block or m.sum() < 100:
            continue
        first = m & (hours < h0 + block)
        pts = []
        for b0 in np.arange(h0 + block, h1 - block / 2, block):
            cur = m & (hours >= b0) & (hours < b0 + block)
            s, nb, _ = _bin_step(keys, vals, first, cur, params["oil_change"]["min_bin_samples"])
            if nb >= params["oil_change"]["min_bins"]:
                pts.append((b0 + block / 2 - h0, s))
        rec = {"engine_id": prep.engine_id, "from_h": h0, "to_h": h1, "length_h": h1 - h0}
        if len(pts) >= 3:
            x, y = np.array(pts).T
            sl, _, lo, hi = theilslopes(y, x, 0.95)
            rec.update({"pressure_trend_pct_per_100h": sl * 1e4, "pressure_trend_ci": (lo * 1e4, hi * 1e4)})
        if lvl is not None:
            lm = df["_judgeable"].to_numpy() & (hours >= h0) & (hours < h1) & np.isfinite(lvl)
            if lm.sum() >= 50 and np.ptp(hours[lm]) > 10:
                sub = np.flatnonzero(lm)[:: max(1, lm.sum() // 2000)]
                sl, _, lo, hi = theilslopes(lvl[sub], hours[sub], 0.95)
                rec.update({"oil_level_trend_pct_per_100h": sl * 100, "oil_level_trend_ci": (lo * 100, hi * 100)})
        out.append(rec)
    return pd.DataFrame(out)
