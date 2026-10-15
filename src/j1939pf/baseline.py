"""Per-engine learned baselines: layer (b) bins, layer (c) z-scores, layer (d) trend.

Everything here is learned from ONE engine's own history. Nothing is shared
between engines, and no engine specification is used.

  (b) Reference baseline: per (RPM x oil temp x load) bin, the median and a
      robust spread (1.4826 x MAD, floored at the bus resolution so 4 kPa
      quantization can't make z-scores explode), learned from the first
      `learning_period_h` engine hours of *eligible* rows only: engine running,
      past the startup grace, no active DM1, reading decoded OK.
  Online baseline: after learning, each bin's expected value follows new
      eligible data slowly (alpha per block) but only when that block is not
      itself anomalous, and never more than `drift_cap_rel` away from the
      reference, so a slow fault cannot be absorbed. Hitting the cap is recorded.
  (c) z = (actual - online expected) / reference spread, per row.
  (d) Every `block_h` engine hours: engine-level drift = weighted median over
      trusted bins of (block median - REFERENCE median) / reference median.
      The reference window is fixed, so drift is always measured against the
      early healthy hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine_config import EngineConfig
from .spn_table import SpnTable

KEY_RPM = 100_000
KEY_TEMP = 100


@dataclass
class Prepared:
    engine_id: str
    df: pd.DataFrame
    dt_s: float
    hours_source: str
    load_col: str | None
    dm1_known: bool
    notes: list[str] = field(default_factory=list)


def _load_column(df: pd.DataFrame, cfg: EngineConfig | None, table: SpnTable) -> tuple[str | None, str]:
    spns = [cfg.load_spn, cfg.load_fallback_spn] if cfg else [92, 513]
    for i, spn in enumerate(s for s in spns if s is not None):
        d = table.by_spn.get(spn)
        if d is not None and d.canonical in df and df[d.canonical].notna().any():
            note = f"load signal: SPN {spn} ({d.canonical})" + (" [fallback]" if i else "")
            return d.canonical, note
    return None, "load signal: none broadcast (bins use RPM x oil temp only)"


def prepare(df_e: pd.DataFrame, cfg: EngineConfig | None, params: dict, table: SpnTable) -> Prepared:
    bp = params["baseline"]
    df = df_e.sort_values("t", kind="stable").reset_index(drop=True).copy()
    engine_id = str(df["engine_id"].iloc[0])
    notes: list[str] = []
    t_s = (df["t"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
    df["_t_s"] = t_s
    seg = df["segment"].to_numpy() if "segment" in df else np.zeros(len(df), dtype=np.int64)
    df["segment"] = seg
    diffs = np.diff(t_s)
    same = seg[1:] == seg[:-1]
    dt_s = float(np.median(diffs[same])) if same.any() else 1.0

    rpm = df["engine_speed_rpm"].to_numpy(dtype=float) if "engine_speed_rpm" in df else np.full(len(df), np.nan)
    running = np.nan_to_num(rpm, nan=0.0) > bp["running_rpm_min"]
    df["_running"] = running

    if "engine_hours_h" in df and df["engine_hours_h"].notna().any():
        h = df["engine_hours_h"].ffill().bfill().to_numpy(dtype=float)
        if (np.diff(h) < -0.1).any():
            notes.append("engine hours went backwards at least once (ECU swap or reset?); trend uses them as reported")
        hours_source = "SPN 247 engine hours"
    else:
        h = np.cumsum(running * dt_s) / 3600.0
        hours_source = "derived from running time (SPN 247 not broadcast; hours are relative to log start)"
    df["_hours"] = h

    prev_running = np.r_[False, running[:-1]]
    new_seg = np.r_[True, seg[1:] != seg[:-1]]
    start = running & (~prev_running | new_seg)
    run_id = np.cumsum(start)
    start_t = pd.Series(t_s).groupby(run_id).transform("first").to_numpy()
    since = np.where(running, t_s - start_t, np.nan)
    df["_judgeable"] = running & (since >= bp["startup_grace_s"])

    # stopped-for: seconds since the engine stopped (for engine-off zero checks)
    stop = ~running & (prev_running | new_seg)
    stop_id = np.cumsum(stop)
    stop_t = pd.Series(t_s).groupby(stop_id).transform("first").to_numpy()
    df["_stopped_s"] = np.where(~running & (stop_id > 0), t_s - stop_t, np.nan)

    if "dm1_active_count" in df and df["dm1_active_count"].notna().any():
        dm1_known = True
        df["_dm1_active"] = df["dm1_active_count"].fillna(0).to_numpy() > 0
    else:
        dm1_known = False
        df["_dm1_active"] = False
        notes.append("DM1 not broadcast: learning cannot be guarded by active fault codes")

    load_col, load_note = _load_column(df, cfg, table)
    notes.append(load_note)
    rb = np.floor(np.nan_to_num(rpm, nan=-1) / bp["rpm_bin_width"]).astype(np.int64)
    if "oil_temp_c" in df and df["oil_temp_c"].notna().any():
        ot = df["oil_temp_c"].to_numpy(dtype=float)
        tb = np.floor(ot / bp["oil_temp_bin_width_c"])
        have_t = ~np.isnan(tb)
        tb = np.nan_to_num(tb, nan=0).astype(np.int64)
    else:
        have_t = np.zeros(len(df), bool)
        tb = np.zeros(len(df), dtype=np.int64)
        notes.append("oil temperature not broadcast: learned layers (b)-(d) cannot bin and are disabled")
    if load_col:
        lv = df[load_col].to_numpy(dtype=float)
        lb = np.floor(np.nan_to_num(lv, nan=0) / bp["load_bin_width_pct"]).astype(np.int64)
        have_l = ~np.isnan(lv)
    else:
        lb = np.zeros(len(df), dtype=np.int64)
        have_l = np.ones(len(df), bool)
    key = rb * KEY_RPM + (tb + 100) * KEY_TEMP + np.clip(lb, 0, 99)
    df["_bin"] = np.where(have_t & have_l & (rb >= 0), key, -1)
    df["_base_eligible"] = df["_judgeable"].to_numpy() & ~df["_dm1_active"].to_numpy() & (df["_bin"].to_numpy() >= 0)
    return Prepared(engine_id, df, dt_s, hours_source, load_col, dm1_known, notes)


def bin_label(key: int, params: dict, load_col: str | None) -> str:
    bp = params["baseline"]
    rb, rest = divmod(int(key), KEY_RPM)
    tb, lb = divmod(rest, KEY_TEMP)
    tb -= 100
    w, wt, wl = bp["rpm_bin_width"], bp["oil_temp_bin_width_c"], bp["load_bin_width_pct"]
    s = f"{rb * w}-{(rb + 1) * w} rpm, oil {tb * wt}-{(tb + 1) * wt} C"
    if load_col:
        s += f", load {lb * wl}-{(lb + 1) * wl} %"
    return s


def bin_rpm_mid(keys: np.ndarray, params: dict) -> np.ndarray:
    return (np.asarray(keys) // KEY_RPM + 0.5) * params["baseline"]["rpm_bin_width"]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2.0)])


@dataclass
class SignalBaseline:
    engine_id: str
    signal: str
    resolution: float
    status: str                       # "learning" | "active" | "no_data"
    message: str
    learn_start_h: float | None = None
    learn_end_h: float | None = None
    hours_eligible: float = 0.0
    reference: pd.DataFrame | None = None   # index bin; median, spread, n, trusted, late_learned
    online: pd.Series | None = None
    capped: set = field(default_factory=set)
    z: np.ndarray | None = None
    expected: np.ndarray | None = None
    trend: pd.DataFrame | None = None
    adjustments: list = field(default_factory=list)   # re-baselining at change-points
    transition_rows: np.ndarray | None = None          # rows not judged while a step is being measured

    @property
    def trusted_bins(self) -> int:
        return 0 if self.reference is None else int(self.reference["trusted"].sum())


def _fmt_step(mode: str, step: float) -> str:
    return f"{step * 100:+.1f} %" if mode == "mul" else f"{step:+.1f} kPa"


def eligible_mask(prep: Prepared, signal: str) -> np.ndarray:
    df = prep.df
    m = df["_base_eligible"].to_numpy() & df[signal].notna().to_numpy()
    err = signal + "__err"
    if err in df:
        m &= df[err].fillna(0).to_numpy() == 0
    return m


def learn_signal(prep: Prepared, signal: str, params: dict, resolution: float,
                 changepoints: list[dict] | None = None) -> SignalBaseline:
    """changepoints: [{hours, kind, mode: 'mul'|'add'|'none', label}] for THIS engine and signal."""
    bp = params["baseline"]
    changepoints = sorted(changepoints or [], key=lambda c: c["hours"])
    df = prep.df
    hours = df["_hours"].to_numpy()
    vals = df[signal].to_numpy(dtype=float)
    keys = df["_bin"].to_numpy()
    elig = eligible_mask(prep, signal)
    sb = SignalBaseline(prep.engine_id, signal, resolution, "no_data", "")
    if not elig.any():
        sb.message = f"{signal}: no eligible (running, no DM1, plausible) data; learned layers inactive"
        return sb
    h0 = float(hours[elig].min())
    L = float(bp["learning_period_h"])
    # a grade change / sensor replacement inside the learning window would mix two states into one reference:
    # learning restarts after it
    restarted = None
    for cp in changepoints:
        if cp["mode"] != "none" and h0 < cp["hours"] < h0 + L:
            h0, restarted = float(cp["hours"]), cp
    if restarted is not None:
        later = elig & (hours >= h0)
        h0 = float(hours[later].min()) if later.any() else h0
    sb.learn_start_h, sb.learn_end_h = h0, h0 + L
    # eligible engine hours actually observed
    sb.hours_eligible = float(elig.sum() * prep.dt_s / 3600.0)
    learn = elig & (hours < h0 + L)
    floor = bp["spread_floor_steps"] * resolution
    seg = df["segment"].to_numpy()
    learn_df = pd.DataFrame({"k": keys[learn], "v": vals[learn], "s": seg[learn]})
    g = learn_df.groupby("k")["v"]
    sessions = learn_df.groupby("k")["s"].nunique()
    med = g.median()
    mad = g.apply(lambda s: float(np.median(np.abs(s.to_numpy() - np.median(s.to_numpy())))))
    # Session-to-session variation of the bin's level (where in the bin each session sat, warm-up state...)
    sess_med = learn_df.groupby(["k", "s"])["v"].median()
    between = sess_med.groupby(level=0).apply(
        lambda s: 1.4826 * float(np.median(np.abs(s.to_numpy() - np.median(s.to_numpy())))) if len(s) >= 3 else 0.0)
    spread = np.sqrt((1.4826 * mad) ** 2 + between.reindex(med.index).fillna(0.0) ** 2)
    spread = np.maximum(spread, np.maximum(floor, bp["spread_floor_rel"] * med.abs()))
    ref = pd.DataFrame({"median": med, "spread": spread, "n": g.size()})
    ref["sessions"] = sessions
    ref["trusted"] = (ref["n"] >= bp["min_healthy_samples"]) & (ref["sessions"] >= bp["min_healthy_sessions"])
    ref["late_learned"] = False
    sb.reference = ref
    if float(hours.max()) < h0 + L:
        sb.status = "learning"
        sb.message = (f"LEARNING PERIOD: {float(hours.max()) - h0:.1f} of {L:.0f} engine hours observed since "
                      f"{h0:.1f} h; {int(ref['trusted'].sum())} operating bins trusted so far. Learned layers "
                      f"(b)-(d) are not judging {signal} on this engine yet.")
        return sb
    sb.status = "active"
    sb.message = (f"learning complete: reference window {h0:.1f}-{h0 + L:.1f} engine h, "
                  f"{int(ref['trusted'].sum())} trusted bins")
    if restarted is not None:
        sb.message += f" (learning restarted after {restarted['label']} at {restarted['hours']:.1f} h)"
    _run_online(sb, prep, signal, params, elig, hours, vals, keys,
                [c for c in changepoints if c["hours"] >= h0 + L])
    return sb


def _measure_step(mode, elig, hours, vals, keys, h, pre_h, post_h, bins, min_n):
    """Per-bin step across a change-point: weighted median of post/pre - 1 (mul) or post - pre (add)."""
    pre = elig & (hours >= h - pre_h) & (hours < h)
    post = elig & (hours >= h) & (hours < h + post_h)
    a = pd.DataFrame({"k": keys[pre], "v": vals[pre]}).groupby("k")["v"].agg(["mean", "size"])
    b = pd.DataFrame({"k": keys[post], "v": vals[post]}).groupby("k")["v"].agg(["mean", "size"])
    common = a.index[a["size"] >= min_n].intersection(b.index[b["size"] >= min_n]).intersection(bins)
    if len(common) == 0:
        return np.nan, 0
    ma, mb = a.loc[common, "mean"].to_numpy(), b.loc[common, "mean"].to_numpy()
    w = np.minimum(a.loc[common, "size"].to_numpy(), b.loc[common, "size"].to_numpy()).astype(float)
    d = (mb / ma - 1) if mode == "mul" else (mb - ma)
    return _weighted_median(d, w), int(len(common))


def _run_online(sb: SignalBaseline, prep: Prepared, signal: str, params: dict, elig, hours, vals, keys,
                changepoints: list[dict] | None = None) -> None:
    bp = params["baseline"]
    cpp = params["changepoints"]
    df = prep.df
    # rows inside a re-baselining transition are neither judged nor learned from
    pending = [dict(c) for c in (changepoints or []) if c["mode"] in ("mul", "add")]
    tw = np.zeros(len(df), bool)
    for c in pending:
        c["until"] = c["hours"] + cpp["relearn_h"]
        tw |= (hours >= c["hours"]) & (hours < c["until"])
    elig_measure = elig            # the step itself is measured on the transition rows
    elig = elig & ~tw
    sb.transition_rows = tw
    for c in changepoints or []:
        if c["mode"] == "none":
            sb.adjustments.append({"hours": c["hours"], "kind": c["kind"], "label": c["label"], "mode": "none",
                                   "accepted": None, "step": None, "n_bins": None,
                                   "note": "no re-baseline: " + c.get("why", "")})
    ref = sb.reference
    trusted = ref[ref["trusted"]]
    online = trusted["median"].copy()
    spread = trusted["spread"].copy()
    ref_med = trusted["median"].copy()
    judge = df["_judgeable"].to_numpy() & ~np.isnan(vals) & (keys >= 0) & ~tw
    z = np.full(len(df), np.nan)
    expected = np.full(len(df), np.nan)
    learn_end = sb.learn_end_h
    post = hours >= learn_end
    block = np.where(post, np.floor((hours - learn_end) / bp["block_h"]), -1).astype(np.int64)
    cap_rel = bp["drift_cap_rel"]
    late: dict[int, list] = {}
    trend_rows = []
    last_drift = 0.0
    def apply(c):
        step, nb = _measure_step(c["mode"], elig_measure, hours, vals, keys, c["hours"], cpp["pre_window_h"],
                                 c["until"] - c["hours"], online.index, cpp["min_bin_samples"])
        rec = {"hours": c["hours"], "kind": c["kind"], "label": c["label"], "mode": c["mode"], "step": step,
               "n_bins": nb, "accepted": False}
        limit = cpp["max_step_rel"] if c["mode"] == "mul" else cpp["max_step_add_kpa"]
        if nb < cpp["min_bins"] or not np.isfinite(step):
            rec["note"] = (f"NOT re-baselined: only {nb} comparable operating bins around the {c['label']}; "
                           f"judging continues against the previous baseline")
        elif abs(step) > limit:
            rec["note"] = (f"NOT re-baselined: step {_fmt_step(c['mode'], step)} is larger than a {c['label']} can "
                           f"explain (limit {_fmt_step(c['mode'], limit)}); the change stays visible to the alerts")
        else:
            rec["accepted"] = True
            if c["mode"] == "mul":
                online.loc[:] = online * (1 + step)
                ref_med.loc[:] = ref_med * (1 + step)
                spread.loc[:] = spread * (1 + step)
            else:
                online.loc[:] = online + step
                ref_med.loc[:] = ref_med + step
            late.clear()
            rec["note"] = (f"re-baselined: step {_fmt_step(c['mode'], step)} measured over {nb} operating bins "
                           f"({cpp['pre_window_h']:g} h before vs {c['until'] - c['hours']:g} h after); alerts were "
                           f"not raised while it was measured")
        sb.adjustments.append(rec)

    def judge_rows(mask):
        jr = np.flatnonzero(mask & judge & np.isin(keys, online.index.to_numpy()))
        if jr.size:
            k = keys[jr]
            e = online.reindex(k).to_numpy()
            expected[jr] = e
            z[jr] = (vals[jr] - e) / spread.reindex(k).to_numpy()

    for b in np.unique(block[block >= 0]):
        rows = block == b
        bend = learn_end + (b + 1) * bp["block_h"]
        # layer (c): judge against the online baseline as it stood BEFORE this block; a re-baselining step whose
        # measurement window ends inside the block is applied exactly at that point (causal: it only uses rows
        # before it)
        done_upto = -np.inf
        for c in sorted([c for c in pending if not c.get("done") and c["until"] < bend], key=lambda c: c["until"]):
            judge_rows(rows & (hours >= done_upto) & (hours < c["until"]))
            apply(c)
            c["done"] = True
            done_upto = c["until"]
        judge_rows(rows & (hours >= done_upto))
        eb = rows & elig
        if not eb.any():
            continue
        g = pd.DataFrame({"k": keys[eb], "v": vals[eb]}).groupby("k")["v"].agg(["median", "count"])
        gt = g[(g["count"] >= bp["min_block_samples"]) & g.index.isin(online.index)]
        if len(gt):
            rm = ref_med.reindex(gt.index)
            rel = ((gt["median"] - rm) / rm).to_numpy()
            absd = (gt["median"] - rm).to_numpy()
            w = gt["count"].to_numpy(dtype=float)
            last_drift = _weighted_median(rel, w)
            # online update, guarded: anomalous blocks are not learned; capped at +-cap of reference
            on = online.reindex(gt.index)
            zz = (gt["median"] - on) / spread.reindex(gt.index)
            upd = zz.abs() <= bp["learn_z_max"]
            new = on[upd] + bp["online_alpha"] * (gt["median"][upd] - on[upd])
            lo, hi = rm[upd] * (1 - cap_rel), rm[upd] * (1 + cap_rel)
            clipped = new.clip(lower=lo, upper=hi)
            online.loc[clipped.index] = clipped
            at_cap = (clipped - rm[upd]).abs() >= (rm[upd] * cap_rel) * 0.999
            sb.capped.update(int(x) for x in clipped.index[at_cap.to_numpy()])
            sb.capped.difference_update(int(x) for x in clipped.index[~at_cap.to_numpy()])
            trend_rows.append({
                "hours": float(np.median(hours[rows])), "t": df["t"].iloc[int(np.flatnonzero(rows)[-1])],
                "drift_rel": last_drift, "drift_abs": _weighted_median(absd, w), "n_bins": int(len(gt)),
                "n_rows": int(gt["count"].sum()), "capped_frac": len(sb.capped) / max(len(online), 1),
            })
        # late learning of bins never seen in the learning window, only while the engine is near its reference
        if abs(last_drift) < cap_rel:
            new_rows = eb & ~np.isin(keys, ref.index.to_numpy())
            seg = df["segment"].to_numpy()
            for k, v, sg in zip(keys[new_rows], vals[new_rows], seg[new_rows]):
                lst, segs = late.setdefault(int(k), ([], set()))
                if len(lst) < 2000:
                    lst.append(v)
                segs.add(int(sg))
            ready = [k for k, (lst, segs) in late.items()
                     if len(lst) >= bp["min_healthy_samples"] and len(segs) >= bp["min_healthy_sessions"]]
            for k in ready:
                lst, segs = late.pop(k)
                arr = np.asarray(lst)
                m = float(np.median(arr))
                s = max(1.4826 * float(np.median(np.abs(arr - m))), bp["spread_floor_steps"] * sb.resolution,
                        bp["spread_floor_rel"] * abs(m))
                ref.loc[k] = {"median": m, "spread": s, "n": len(arr), "sessions": len(segs), "trusted": True,
                              "late_learned": True}
                online.loc[k], spread.loc[k], ref_med.loc[k] = m, s, m
    for c in [c for c in pending if not c.get("done")]:
        if hours.max() >= c["until"]:
            apply(c)
        else:
            sb.adjustments.append({"hours": c["hours"], "kind": c["kind"], "label": c["label"], "mode": c["mode"],
                                   "step": None, "n_bins": None, "accepted": None,
                                   "note": f"TRANSITION IN PROGRESS: measuring the step after the {c['label']} "
                                           f"until {c['until']:.1f} h; not judged meanwhile"})
    sb.adjustments.sort(key=lambda a: a["hours"])
    sb.online = online
    sb.z = z
    sb.expected = expected
    sb.trend = pd.DataFrame(trend_rows, columns=["hours", "t", "drift_rel", "drift_abs", "n_bins", "n_rows",
                                                 "capped_frac"])
