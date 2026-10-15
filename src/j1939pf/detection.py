"""Sustained-run detection and alert episodes for layers (a), (c), (d)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .baseline import Prepared, SignalBaseline, bin_label


@dataclass
class Run:
    i0: int
    i1: int          # inclusive
    duration_s: float


def find_runs(flag: np.ndarray, prep: Prepared, min_duration_s: float) -> list[Run]:
    """Consecutive flagged rows in one segment with no clock gap; kept if they last long enough."""
    df = prep.df
    t = df["_t_s"].to_numpy()
    seg = df["segment"].to_numpy()
    dt = prep.dt_s
    idx = np.flatnonzero(flag)
    if idx.size == 0:
        return []
    brk = np.r_[True, (np.diff(idx) != 1) | (seg[idx[1:]] != seg[idx[:-1]])
                | (np.diff(t[idx]) > 2.5 * dt)]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:] - 1, idx.size - 1]
    out = []
    for s, e in zip(starts, ends):
        i0, i1 = int(idx[s]), int(idx[e])
        dur = float(t[i1] - t[i0] + dt)
        if dur >= min_duration_s:
            out.append(Run(i0, i1, dur))
    return out


def merge_runs(runs: list[Run], prep: Prepared, gap_engine_h: float) -> list[list[Run]]:
    hours = prep.df["_hours"].to_numpy()
    episodes: list[list[Run]] = []
    for r in runs:
        if episodes and hours[r.i0] - hours[episodes[-1][-1].i1] <= gap_engine_h:
            episodes[-1].append(r)
        else:
            episodes.append([r])
    return episodes


def episode_rows(ep: list[Run], n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for r in ep:
        m[r.i0:r.i1 + 1] = True
    return m


def operating_point(prep: Prepared, rows: np.ndarray, params: dict) -> dict:
    df = prep.df
    op = {}
    for col, name in (("engine_speed_rpm", "rpm"), ("oil_temp_c", "oil_temp_c"), ("coolant_temp_c", "coolant_temp_c")):
        if col in df:
            op[name] = float(np.nanmedian(df[col].to_numpy(dtype=float)[rows])) if rows.any() else None
    if prep.load_col:
        op["load_pct"] = float(np.nanmedian(df[prep.load_col].to_numpy(dtype=float)[rows]))
    rpm = df["engine_speed_rpm"].to_numpy(dtype=float)[rows] if "engine_speed_rpm" in df else np.array([])
    if rpm.size:
        op["rpm_range"] = [float(np.nanpercentile(rpm, 10)), float(np.nanpercentile(rpm, 90))]
    keys = df["_bin"].to_numpy()[rows]
    keys = keys[keys >= 0]
    if keys.size:
        k = int(pd.Series(keys).mode().iloc[0])
        op["bin"] = k
        op["label"] = bin_label(k, params, prep.load_col)
    else:
        op["label"] = f"{op.get('rpm', 0):.0f} rpm"
    return op


# ---------------------------------------------------------------- layer (d)
@dataclass
class DriftEvent:
    direction: str
    first_idx: int        # index into trend frame
    last_idx: int
    level: float
    slope: float
    lo: float
    hi: float
    n_points: int


def drift_events(sb: SignalBaseline, params: dict) -> list[DriftEvent]:
    ap = params["alerts"]
    tr = sb.trend
    if tr is None or len(tr) < ap["trend_min_points"]:
        return []
    h = tr["hours"].to_numpy()
    d = tr["drift_rel"].to_numpy()
    thr = ap["drift_alert_rel"]
    events: dict[str, DriftEvent] = {}
    for i in range(len(tr)):
        w = (h > h[i] - ap["trend_window_h"]) & (np.arange(len(tr)) <= i)
        if w.sum() < ap["trend_min_points"] or np.ptp(h[w]) <= 0:
            continue
        slope, _, lo, hi = theilslopes(d[w], h[w], 0.95)
        level = float(np.median(d[max(0, i - 2):i + 1]))
        for direction, ok in (("low", level <= -thr and hi < 0), ("high", level >= thr and lo > 0)):
            if not ok:
                continue
            ev = events.get(direction)
            if ev is None:
                events[direction] = DriftEvent(direction, i, i, level, float(slope), float(lo), float(hi), int(w.sum()))
            else:
                ev.last_idx, ev.level, ev.slope, ev.lo, ev.hi = i, level, float(slope), float(lo), float(hi)
    return list(events.values())
