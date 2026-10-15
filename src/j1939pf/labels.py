"""Label-based layers (phase 5): supervised failure prediction and survival analysis.

These are the ONLY layers allowed to say anything about failure risk or life, and only
when the maintenance/failure log contains enough failures. Every output carries its
sample size and a confidence interval, and says plainly when data is insufficient.

Failure labels: events whose event_type is "failure" or starts with "failure_".

Supervised: one row per engine per period (exposure rates, drift, anomaly rate);
  label = a failure follows within `horizon_h` engine hours. Logistic regression with
  class_weight="balanced" (imbalance) and engine-grouped cross-validation (no leakage
  between periods of the same engine). Reports out-of-fold ROC AUC with an engine-level
  bootstrap CI, PR AUC against the base rate, and standardised coefficients. Skill is
  claimed only when the AUC CI lower bound is above 0.5. Balanced weights distort
  probabilities, so no calibrated failure probability is reported.

Survival: one row per engine (entry hours, exit hours, failed?, covariates). Kaplan-Meier
  with left truncation (engines enter observation at their first logged hours) and a
  Weibull accelerated-failure-time model fitted by maximum likelihood (scipy; lifelines is
  not required). Covariate effects are time ratios exp(beta) with 95 % CI: < 1 means
  shorter time to failure for higher exposure. Remaining-life (RUL) estimates are given
  only from a fitted survival model with enough failures, as a median with a CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False


@dataclass
class LayerResult:
    layer: str
    available: bool
    message: str
    details: dict = field(default_factory=dict)


def failures_from_events(events: pd.DataFrame | None, prefix: str = "failure") -> pd.DataFrame:
    cols = ["engine_id", "engine_hours", "event_type"]
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=cols)
    m = events["event_type"].astype(str).str.startswith(prefix)
    return events.loc[m, cols].sort_values(["engine_id", "engine_hours"]).reset_index(drop=True)


# ---------------------------------------------------------------- supervised
def supervised(periods: pd.DataFrame, failures: pd.DataFrame, features: list[str], params: dict) -> LayerResult:
    lp = params["labels"]
    if not HAVE_SKLEARN:  # pragma: no cover
        return LayerResult("supervised", False, "scikit-learn not installed")
    if len(failures) == 0:
        return LayerResult("supervised", False, "no failure labels in the event log: supervised prediction inactive "
                                                "(exposure and trend only)")
    rows = []
    for _, p in periods.iterrows():
        f = failures[failures["engine_id"] == p["engine_id"]]["engine_hours"].to_numpy()
        if (f <= p["to_h"]).any():
            continue  # periods after a failure belong to a repaired engine: excluded, not relabelled
        rows.append({**p.to_dict(), "y": int(((f > p["to_h"]) & (f <= p["to_h"] + lp["horizon_h"])).any())})
    d = pd.DataFrame(rows)
    feats = [c for c in features if c in d and d[c].notna().mean() >= 0.9] if len(d) else []
    if len(d):
        d = d.dropna(subset=feats)
    n_pos = int(d["y"].sum()) if len(d) else 0
    n_neg = int(len(d) - n_pos)
    failed = int(d.loc[d["y"] == 1, "engine_id"].nunique()) if len(d) else 0
    need = (f"needs >= {lp['min_positive_periods']} pre-failure periods from >= {lp['min_failed_engines']} failed "
            f"engines and >= {lp['min_negative_periods']} other periods")
    info = {"positive_periods": n_pos, "negative_periods": n_neg, "failed_engines": failed,
            "engines": int(d["engine_id"].nunique()) if len(d) else 0, "base_rate": n_pos / max(len(d), 1),
            "horizon_h": lp["horizon_h"], "features": feats}
    if n_pos < lp["min_positive_periods"] or failed < lp["min_failed_engines"] or n_neg < lp["min_negative_periods"] \
            or len(feats) == 0:
        return LayerResult("supervised", False, f"INSUFFICIENT DATA: {n_pos} pre-failure periods from {failed} failed "
                                                f"engines, {n_neg} other periods; {need}. No prediction made.", info)
    X, y, g = d[feats].to_numpy(float), d["y"].to_numpy(), d["engine_id"].to_numpy()
    k = min(5, len(np.unique(g)))
    oof = np.full(len(d), np.nan)
    seed = int(params.get("seed", 0))
    for tr, te in GroupKFold(n_splits=k).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    ok = np.isfinite(oof)
    auc = float(roc_auc_score(y[ok], oof[ok]))
    ap_ = float(average_precision_score(y[ok], oof[ok]))
    rng = np.random.default_rng(seed)
    eng = np.unique(g[ok])
    boots = []
    for _ in range(lp["bootstrap"]):
        pick = rng.choice(eng, eng.size, replace=True)
        ii = np.concatenate([np.flatnonzero(ok & (g == e)) for e in pick])
        if len(np.unique(y[ii])) == 2:
            boots.append(roc_auc_score(y[ii], oof[ii]))
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (np.nan, np.nan)
    sc = StandardScaler().fit(X)
    full = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed).fit(sc.transform(X), y)
    info.update({"roc_auc_oof": auc, "roc_auc_ci95": ci, "pr_auc_oof": ap_, "cv": f"GroupKFold by engine, k={k}",
                 "standardised_coefficients": {f: float(c) for f, c in zip(feats, full.coef_[0])},
                 "class_weight": "balanced (probabilities are NOT calibrated)"})
    skill = ci[0] > 0.5
    msg = (f"{'PREDICTIVE SKILL SHOWN' if skill else 'NO DEMONSTRATED SKILL'}: out-of-fold ROC AUC {auc:.2f} "
           f"(95 % CI {ci[0]:.2f}-{ci[1]:.2f}), PR AUC {ap_:.2f} vs base rate {info['base_rate']:.2f}; "
           f"n = {n_pos} pre-failure / {n_neg} other periods, {failed} failed of {info['engines']} engines.")
    return LayerResult("supervised", True, msg, {**info, "skill": skill})


# ---------------------------------------------------------------- survival
def kaplan_meier(entry, exit_, event) -> pd.DataFrame:
    """Kaplan-Meier with left truncation; Greenwood 95 % CI (log-log)."""
    entry, exit_, event = map(np.asarray, (entry, exit_, event))
    times = np.unique(exit_[event == 1])
    s, var, rows = 1.0, 0.0, []
    for t in times:
        at_risk = int(((entry < t) & (exit_ >= t)).sum())
        d = int(((exit_ == t) & (event == 1)).sum())
        if at_risk == 0:
            continue
        s *= 1 - d / at_risk
        if at_risk > d:
            var += d / (at_risk * (at_risk - d))
        if 0 < s < 1:
            se = np.sqrt(var) / abs(np.log(s))
            lo, hi = s ** np.exp(1.96 * se), s ** np.exp(-1.96 * se)
        else:
            lo = hi = s
        rows.append({"engine_hours": float(t), "at_risk": at_risk, "failures": d, "survival": s, "ci_low": lo,
                     "ci_high": hi})
    return pd.DataFrame(rows)


def _weibull_aft_nll(theta, X, t, e, entry):
    beta, log_sigma = theta[:-1], theta[-1]
    sigma = np.exp(log_sigma)
    mu = X @ beta
    z = (np.log(t) - mu) / sigma
    z0 = (np.log(np.maximum(entry, 1e-9)) - mu) / sigma
    log_f = z - np.exp(z) - np.log(sigma) - np.log(t)
    log_s = -np.exp(z)
    log_s0 = np.where(entry > 0, -np.exp(z0), 0.0)  # left truncation: condition on surviving to entry
    return -(np.sum(e * log_f + (1 - e) * log_s) - np.sum(log_s0))


def _num_hessian(f, x, eps=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp, xpm, xmp, xmm = (x.copy() for _ in range(4))
            xpp[i] += eps; xpp[j] += eps
            xpm[i] += eps; xpm[j] -= eps
            xmp[i] -= eps; xmp[j] += eps
            xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps * eps)
    return H


def survival(engines: pd.DataFrame, covariates: list[str], params: dict) -> LayerResult:
    """engines: engine_id, entry_h, exit_h, failed (0/1), covariate columns."""
    lp = params["labels"]
    if len(engines) == 0 or engines["failed"].sum() == 0:
        return LayerResult("survival", False, "no failure labels in the event log: survival analysis inactive "
                                              "(exposure and trend only)")
    d = engines.dropna(subset=["entry_h", "exit_h"]).copy()
    d = d[d["exit_h"] > d["entry_h"]]
    n_ev = int(d["failed"].sum())
    km = kaplan_meier(d["entry_h"], d["exit_h"], d["failed"])
    info = {"engines": int(len(d)), "failures": n_ev, "kaplan_meier": km}
    if n_ev < lp["min_survival_events"]:
        return LayerResult("survival", False, f"INSUFFICIENT DATA: {n_ev} failures among {len(d)} engines; needs >= "
                                              f"{lp['min_survival_events']} failures for a survival model. "
                                              f"Kaplan-Meier table only, no covariate effects, no remaining-life "
                                              f"estimate.", info)
    covs = [c for c in covariates if c in d and d[c].notna().all() and d[c].std() > 0]
    mean, sd = d[covs].mean(), d[covs].std()
    Xc = ((d[covs] - mean) / sd).to_numpy(float) if covs else np.zeros((len(d), 0))
    X = np.column_stack([np.ones(len(d)), Xc])
    t, e, en = d["exit_h"].to_numpy(float), d["failed"].to_numpy(float), d["entry_h"].to_numpy(float)
    x0 = np.r_[np.log(np.median(t)), np.zeros(len(covs)), np.log(0.5)]
    nll = lambda th: _weibull_aft_nll(th, X, t, e, en)  # noqa: E731
    opt = minimize(nll, x0, method="BFGS")
    if not opt.success and not np.isfinite(opt.fun):
        return LayerResult("survival", False, f"Weibull AFT fit failed ({opt.message}); Kaplan-Meier only", info)
    H = _num_hessian(nll, opt.x)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov, se = None, np.full(len(opt.x), np.nan)
    effects = {}
    for i, c in enumerate(covs, start=1):
        b, s_ = opt.x[i] / sd[c], se[i] / sd[c]          # per raw unit of the covariate
        effects[c] = {"time_ratio_per_unit": float(np.exp(b)),
                      "ci95": (float(np.exp(b - 1.96 * s_)), float(np.exp(b + 1.96 * s_))),
                      "significant": bool(abs(opt.x[i]) > 1.96 * se[i])}
    sigma = float(np.exp(opt.x[-1]))
    info.update({"model": "Weibull accelerated failure time (MLE, left-truncated, right-censored)",
                 "shape": 1 / sigma, "covariate_effects": effects, "theta": opt.x, "cov": cov,
                 "covariates": covs, "standardisation": (mean, sd)})
    eff_txt = "; ".join(f"{c}: time ratio {v['time_ratio_per_unit']:.3f} per unit (95 % CI {v['ci95'][0]:.3f}-"
                        f"{v['ci95'][1]:.3f}){'' if v['significant'] else ' [not significant]'}" for c, v in effects.items())
    return LayerResult("survival", True, f"Weibull AFT on {len(d)} engines with {n_ev} failures (shape "
                                         f"{1 / sigma:.2f}). {eff_txt or 'no covariates'}", info)


def remaining_life(fit: LayerResult, current_h: float, covariate_values: dict, params: dict, seed: int = 0) -> dict:
    """Conditional median remaining engine hours given survival to current_h, with a parameter-uncertainty CI."""
    if not fit.available or fit.details.get("cov") is None:
        return {"available": False, "reason": fit.message}
    d = fit.details
    mean, sd = d["standardisation"]
    x = np.r_[1.0, [(covariate_values[c] - mean[c]) / sd[c] for c in d["covariates"]]]
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(d["theta"], d["cov"], size=500)

    def med(th):
        mu, sigma = x @ th[:-1], np.exp(th[-1])
        s0 = np.exp(-np.exp((np.log(current_h) - mu) / sigma))
        # S(t) = 0.5 * S(t0)  ->  t = exp(mu + sigma * log(-log(0.5 * S(t0))))
        return float(np.exp(mu + sigma * np.log(-np.log(0.5 * s0))) - current_h)

    point = med(d["theta"])
    sims = np.array([med(th) for th in draws])
    return {"available": True, "median_remaining_h": point,
            "ci95": (float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))),
            "n_engines": d["engines"], "n_failures": d["failures"],
            "note": "model-based; valid only for engines and duty like those it was fitted on"}
