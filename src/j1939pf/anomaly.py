"""Joint-signal anomaly detection (phase 5): robust Mahalanobis distance and isolation forest.

Both are fitted per engine on that engine's own reference window (the baseline learning
window: warmed-up running, no active DM1) and seeded. The operating point
(RPM, load) is part of the feature vector, so an anomaly is an unusual COMBINATION -
e.g. low boost for the fuel being burned - not merely an unusual operating point.

Thresholds are the `reference_quantile` of each method's reference scores, so by
construction about (1 - q) of healthy rows exceed them. A row is anomalous when BOTH
methods agree; an episode needs `sustained_run_s` of agreement. Degrades gracefully: missing
signals are dropped from the feature set, scikit-learn absent -> Mahalanobis only (numpy),
too few rows -> unavailable with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baseline import Prepared
from .derived import starts_and_since
from .detection import find_runs, merge_runs

try:  # scikit-learn is a dependency, but the layer must not crash without it
    from sklearn.covariance import MinCovDet
    from sklearn.ensemble import IsolationForest
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover - exercised only without scikit-learn
    HAVE_SKLEARN = False


@dataclass
class AnomalyResult:
    engine_id: str
    available: bool
    reason: str = ""
    features: list[str] = field(default_factory=list)
    dropped_features: dict = field(default_factory=dict)
    methods: dict = field(default_factory=dict)
    reference_rate: float | None = None
    rate_by_period: pd.DataFrame = field(default_factory=pd.DataFrame)
    episodes: list[dict] = field(default_factory=list)
    flag: np.ndarray | None = None
    recent_rate: float | None = None


def _resolve(prep: Prepared, name: str) -> str | None:
    return prep.load_col if name == "load" else name


def anomaly_report(prep: Prepared, params: dict, ref_end_h: float, table=None) -> AnomalyResult:
    ap = params["anomaly"]
    seed = int(params.get("seed", 0))
    df = prep.df
    res = AnomalyResult(prep.engine_id, False)
    # warmed-up running only: warm-up after a cold start is covered by the cold-start / exposure metrics and
    # would otherwise dominate the joint distribution's tails
    _, since = starts_and_since(prep)
    warm = since >= params["derived"]["warmup_exclusion_min"] * 60
    judge = df["_judgeable"].to_numpy() & ~df["_dm1_active"].to_numpy() & warm
    if judge.sum() < ap["min_rows"]:
        res.reason = f"only {int(judge.sum())} running rows"
        return res
    feats = []
    for name in ap["features"]:
        col = _resolve(prep, name)
        if not col or col not in df:
            res.dropped_features[name] = "not broadcast"
            continue
        frac = float(np.isfinite(df[col].to_numpy(dtype=float)[judge]).mean())
        if frac < ap["min_feature_coverage"]:
            res.dropped_features[name] = f"only {frac * 100:.0f} % of running rows have it"
            continue
        feats.append(col)
    if "engine_speed_rpm" not in feats or len(feats) < 3:
        res.reason = f"needs RPM and at least 3 joint signals (have {feats})"
        return res
    # operating-point residuals: each signal minus its own reference median at the same RPM x load bin, so the
    # joint models see deviations from this engine's learned behaviour, not the (nonlinear) operating map
    hours = df["_hours"].to_numpy()
    ctx = [c for c in feats if c in ("engine_speed_rpm", prep.load_col)]
    sig = [c for c in feats if c not in ctx]
    if len(sig) < 2:
        res.reason = f"needs at least 2 signals besides RPM/load (have {sig})"
        return res
    rpm = df["engine_speed_rpm"].to_numpy(dtype=float)
    load = df[prep.load_col].to_numpy(dtype=float) if prep.load_col in ctx else np.zeros(len(df))
    key = (np.floor(np.nan_to_num(rpm) / ap["rpm_bin"]) * 1000 + np.floor(np.nan_to_num(load) / ap["load_bin"])).astype(np.int64)
    X = df[sig].to_numpy(dtype=float)
    ok0 = judge & np.isfinite(X).all(axis=1) & np.isfinite(rpm) & np.isfinite(load)
    ref0 = ok0 & (hours < ref_end_h)
    g = pd.DataFrame(X[ref0], columns=sig).assign(k=key[ref0]).groupby("k")
    med, cnt = g.median(), g.size()
    med = med[cnt >= ap["min_bin_rows"]]
    have = np.isin(key, med.index.to_numpy())
    ok = ok0 & have
    ref = ok & (hours < ref_end_h)
    res.methods["operating_point_coverage"] = float(ok.sum() / max(ok0.sum(), 1))
    if ref.sum() < ap["min_rows"]:
        res.reason = f"only {int(ref.sum())} complete rows in the reference window (need {ap['min_rows']})"
        return res
    M = np.full_like(X, np.nan)
    M[have] = med.reindex(key[have]).to_numpy()
    R = X - M
    # ambient temperature is CONTEXT, not a signal: remove each residual's learned linear dependence on it
    if "ambient_temp_c" in df and df["ambient_temp_c"].notna().any():
        amb = df["ambient_temp_c"].to_numpy(dtype=float)
        ra = ref & np.isfinite(amb)
        if ra.sum() >= ap["min_rows"] and np.ptp(amb[ra]) >= 3:
            a0 = float(np.median(amb[ra]))
            da = np.where(np.isfinite(amb), amb - a0, 0.0)
            slopes = {}
            for j, c in enumerate(sig):
                b = float(np.polyfit(da[ra], R[ra, j], 1)[0])
                R[:, j] -= b * da
                slopes[c] = round(b, 3)
            res.methods["ambient_adjustment"] = {"reference_median_c": a0, "slope_per_c": slopes,
                                                 "reference_range_c": [float(amb[ra].min()), float(amb[ra].max())]}
        else:
            res.methods["ambient_adjustment"] = "not learnable: too little ambient variation in the reference window"
    else:
        res.methods["ambient_adjustment"] = "ambient temperature not broadcast"
    scale = 1.4826 * np.nanmedian(np.abs(R[ref] - np.nanmedian(R[ref], axis=0)), axis=0)
    for j, c in enumerate(sig):
        d = table.by_canonical.get(c) if table is not None else None
        scale[j] = max(scale[j], d.scale if d else 0.0, 1e-6)
    Z = np.where(ok[:, None], R / scale, 0.0)
    feats = sig
    rng = np.random.default_rng(seed)
    ref_idx = np.flatnonzero(ref)
    fit_idx = np.sort(rng.choice(ref_idx, min(ref_idx.size, ap["max_fit_rows"]), replace=False))
    q = ap["reference_quantile"]
    scores, flags = {}, {}
    # Mahalanobis
    if HAVE_SKLEARN:
        mcd = MinCovDet(random_state=seed, support_fraction=0.9).fit(Z[fit_idx])
        loc, prec = mcd.location_, mcd.get_precision()
        mname = "robust Mahalanobis (MCD)"
    else:  # pragma: no cover
        loc = Z[fit_idx].mean(axis=0)
        prec = np.linalg.pinv(np.cov(Z[fit_idx], rowvar=False))
        mname = "Mahalanobis (classical covariance; scikit-learn unavailable)"
    dz = np.where(ok[:, None], Z - loc, 0.0)
    d2 = np.einsum("ij,jk,ik->i", dz, prec, dz)
    thr_m = float(np.quantile(d2[ref], q))
    scores["mahalanobis"], flags["mahalanobis"] = d2, ok & (d2 > thr_m)
    res.methods["mahalanobis"] = {"method": mname, "threshold": thr_m, "status": "active"}
    # Isolation forest
    if HAVE_SKLEARN:
        iso = IsolationForest(n_estimators=ap["iforest_trees"], max_samples=min(512, fit_idx.size),
                              random_state=seed).fit(Z[fit_idx])
        s = np.zeros(len(df))
        s[ok] = -iso.score_samples(Z[ok])
        thr_i = float(np.quantile(s[ref], q))
        scores["iforest"], flags["iforest"] = s, ok & (s > thr_i)
        res.methods["iforest"] = {"method": "isolation forest", "threshold": thr_i, "status": "active",
                                  "random_state": seed}
        flag = flags["mahalanobis"] & flags["iforest"]
        res.methods["combined"] = "a row is anomalous only when BOTH methods agree"
    else:  # pragma: no cover
        res.methods["iforest"] = {"status": "unavailable: scikit-learn not installed"}
        flag = flags["mahalanobis"]
    res.available = True
    res.features = feats
    res.flag = flag
    res.reference_rate = float(flag[ref].mean())
    # rate per period
    P = ap["period_h"]
    rows = []
    for p0 in np.arange(hours.min(), hours.max(), P):
        m = ok & (hours >= p0) & (hours < p0 + P)
        if m.sum() >= 50:
            rows.append({"from_h": float(p0), "to_h": float(p0 + P), "rows": int(m.sum()),
                         "anomaly_rate": float(flag[m].mean())})
    res.rate_by_period = pd.DataFrame(rows)
    recent = ok & (hours > hours.max() - params["alerts"]["trend_window_h"])
    res.recent_rate = float(flag[recent].mean()) if recent.sum() >= 50 else None
    # sustained episodes, with the signals that deviate most
    runs = find_runs(flag, prep, params["alerts"]["sustained_run_s"])
    for ep in merge_runs(runs, prep, params["alerts"]["merge_gap_engine_h"]):
        m = np.zeros(len(df), bool)
        for r in ep:
            m[r.i0:r.i1 + 1] = True
        m &= flag
        zmed = np.median(Z[m], axis=0)
        order = np.argsort(-np.abs(zmed))[:3]
        res.episodes.append({
            "engine_id": prep.engine_id, "start_t": str(df["t"].iloc[ep[0].i0]), "end_t": str(df["t"].iloc[ep[-1].i1]),
            "start_h": float(hours[ep[0].i0]), "end_h": float(hours[ep[-1].i1]),
            "duration_s": float(sum(r.duration_s for r in ep)), "runs": len(ep),
            "top_signals": [f"{feats[j]} {zmed[j]:+.1f} robust-z" for j in order],
            "median_rpm": float(np.median(df["engine_speed_rpm"].to_numpy()[m]))})
    return res
