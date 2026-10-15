"""Sensor fault vs engine wear: weigh evidence before blaming the engine.

Evidence used (J1939 values only):
  strong sensor evidence
    * DM1 code for this SPN with a measurement-circuit FMI (2,3,4,5,6,8,10,13)
    * error/out-of-range encodings above a small fraction of samples
    * engine-off zero check: with the engine stopped and settled, a gauge-pressure
      sensor should read ~0. A shift there is an offset in the sensor, not the engine.
  weak sensor evidence
    * the deviation is the same number of kPa at every operating point (additive offset)
  engine evidence
    * the deviation is confined to high RPM with idle near baseline (flow-dependent:
      relief valve / pump capacity / filter restriction): a sensor offset would shift idle too
    * the deviation scales with pressure (proportional), or a related signal moves with it
      (oil filter dp rising, oil level falling)
The verdict is "undetermined" whenever evidence is missing or conflicting. With one
pressure sensor, a slowly drifting-low sensor and uniform leakage can look alike: the
program says so instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baseline import Prepared, SignalBaseline, bin_rpm_mid, eligible_mask

FMI_TEXT = {2: "data erratic/intermittent", 3: "voltage above normal / shorted high",
            4: "voltage below normal / shorted low", 5: "current below normal / open circuit",
            6: "current above normal / grounded", 8: "abnormal frequency/pulse",
            10: "abnormal rate of change", 13: "out of calibration",
            1: "data valid but below operational range (most severe)", 17: "below normal (least severe)",
            18: "below normal (moderately severe)", 0: "above operational range (most severe)",
            15: "above normal (least severe)", 16: "above normal (moderately severe)"}


@dataclass
class SensorAssessment:
    verdict: str                     # sensor_suspected | engine_suspected | undetermined
    pattern: str                     # high_rpm_only | additive | proportional | ambiguous | none | unknown
    evidence: list[str] = field(default_factory=list)


def deviation_pattern(prep: Prepared, sb: SignalBaseline, window: np.ndarray, params: dict):
    """Per trusted bin: window median vs reference median."""
    if sb.reference is None:
        return "unknown", "no learned baseline", None
    df = prep.df
    m = window & eligible_mask(prep, sb.signal)
    ref = sb.reference[sb.reference["trusted"]]
    keys = df["_bin"].to_numpy()[m]
    vals = df[sb.signal].to_numpy(dtype=float)[m]
    g = pd.DataFrame({"k": keys, "v": vals}).groupby("k")["v"].agg(["median", "count"])
    g = g[(g["count"] >= params["baseline"]["min_block_samples"]) & g.index.isin(ref.index)]
    if len(g) < 3:
        return "unknown", "too few operating bins in the window to judge the deviation pattern", None
    rm = ref["median"].reindex(g.index).to_numpy()
    dev = g["median"].to_numpy() - rm
    rpm = bin_rpm_mid(g.index.to_numpy(), params)
    res = sb.resolution
    if np.abs(np.median(dev)) < res and np.abs(dev).max() < 3 * res:
        return "none", "no consistent deviation across bins", dev
    lo_cut, hi_cut = np.quantile(rpm, [1 / 3, 0.8])
    dev_lo = np.median(dev[rpm <= lo_cut])
    dev_hi = np.median(dev[rpm >= hi_cut])  # top quintile of operating bins by RPM
    if abs(dev_hi) > 3 * max(abs(dev_lo), res) and np.sign(dev_hi) == np.sign(np.median(dev)):
        return ("high_rpm_only",
                f"deviation is confined to high RPM ({dev_hi:+.0f} kPa at high RPM vs {dev_lo:+.0f} kPa near "
                f"idle): flow-dependent, as with a relief valve, pump capacity or filter restriction; a sensor "
                f"offset would shift idle by the same amount", dev)
    add_res = np.median(np.abs(dev - np.median(dev)))
    rel = np.median(dev / rm)
    mul_res = np.median(np.abs(dev - rel * rm))
    if add_res < 0.6 * mul_res:
        return ("additive", f"deviation is about the same in kPa at every operating point "
                            f"({np.median(dev):+.0f} kPa, spread {add_res:.1f}): consistent with a sensor offset", dev)
    if mul_res < 0.6 * add_res:
        return ("proportional", f"deviation scales with pressure ({rel * 100:+.1f} % across bins): consistent "
                                f"with a change in the oil system rather than a fixed sensor offset", dev)
    return "ambiguous", "deviation pattern does not separate a sensor offset from an oil-system change", dev


def assess(prep: Prepared, sb: SignalBaseline | None, signal: str, window: np.ndarray, direction: str,
           params: dict, dtc_episodes: pd.DataFrame | None, spn: int | None, resolution: float,
           ref_window: np.ndarray | None = None) -> SensorAssessment:
    sc = params["sensor_check"]
    df = prep.df
    strong, weak, engine = [], [], []
    info: list[str] = []
    t0, t1 = df["t"][window].min(), df["t"][window].max()

    # 1. DTCs for this SPN overlapping the window
    if dtc_episodes is not None and len(dtc_episodes) and spn is not None:
        ep = dtc_episodes[(dtc_episodes["engine_id"] == prep.engine_id) & (dtc_episodes["spn"] == spn)
                          & (dtc_episodes["kind"] == "DM1")
                          & (dtc_episodes["last_t"] >= t0) & (dtc_episodes["first_t"] <= t1)]
        for _, e in ep.iterrows():
            txt = f"DM1 SPN {spn} FMI {e['fmi']} ({FMI_TEXT.get(int(e['fmi']), 'see J1939-73')}) active"
            if int(e["fmi"]) in sc["sensor_fmis"]:
                strong.append(txt + ": a measurement-circuit code")
            else:
                info.append(txt + ": the ECU judges the same sensor, so this is not independent confirmation")
    if prep.dm1_known and df["_dm1_active"].to_numpy()[window].any():
        info.append("DM1 active codes present during the window (rows excluded from learning)")

    # 2. error encodings
    err = signal + "__err"
    run = window & df["_running"].to_numpy()
    if err in df and run.any():
        n_err = float(df[err].to_numpy()[run].sum())
        n_ok = float(df[signal].notna().to_numpy()[run].sum())
        frac = n_err / max(n_err + n_ok, 1.0)
        if frac > sc["err_fraction_suspect"]:
            strong.append(f"{frac * 100:.1f} % of samples were error/out-of-range encodings (sensor or wiring)")

    # 3. engine-off zero check
    if signal in sc["zero_check_signals"]:
        off = (~df["_running"].to_numpy()) & (df["_stopped_s"].to_numpy() >= sc["zero_check_settle_s"]) \
            & df[signal].notna().to_numpy()
        tol = sc["zero_check_tolerance_steps"] * resolution
        rw = ref_window if ref_window is not None else np.zeros(len(df), bool)
        z_ref = df[signal].to_numpy()[off & rw]
        z_win = df[signal].to_numpy()[off & window]
        if z_win.size >= 3:
            base = float(np.median(z_ref)) if z_ref.size >= 3 else 0.0
            shift = float(np.median(z_win)) - base
            if abs(shift) > tol:
                strong.append(f"engine-off zero check: stopped-engine reading shifted {shift:+.1f} kPa from "
                              f"{base:.1f} kPa (a stopped engine produces no oil pressure: this is a sensor offset)")
            elif direction == "high":
                engine.append("engine-off zero check passed (no upward sensor offset)")
            else:
                info.append("engine-off zero check passed, but readings clip at 0 kPa so a NEGATIVE sensor offset "
                            "cannot be seen this way")
        else:
            info.append("engine-off zero check unavailable (no key-on/engine-off samples in the window)")

    # 4. deviation pattern
    pattern, ptxt = "unknown", ""
    if sb is not None and sb.status == "active":
        pattern, ptxt, _ = deviation_pattern(prep, sb, window, params)
        if pattern == "high_rpm_only":
            engine += [ptxt, "(pattern weight x2)"]
        elif pattern == "proportional":
            engine += [ptxt, "(pattern weight x2)"]
        elif pattern == "additive":
            weak.append(ptxt)
        elif ptxt:
            info.append(ptxt)

    # 5. corroborating signals
    if signal == "oil_pressure_kpa" and ref_window is not None:
        hi = df["_judgeable"].to_numpy()
        for col, label, sign in (("oil_filter_dp_kpa", "oil filter dp", 1), ("oil_level_pct", "oil level", -1)):
            if col in df and df[col].notna().any():
                a = df[col].to_numpy(dtype=float)
                r, w = np.nanmedian(a[hi & ref_window]) if (hi & ref_window).any() else np.nan, \
                    np.nanmedian(a[hi & window]) if (hi & window).any() else np.nan
                if np.isfinite(r) and np.isfinite(w) and r > 0:
                    ch = (w - r) / r
                    if (sign > 0 and ch > 0.3) or (sign < 0 and ch < -0.1):
                        engine.append(f"{label} moved {ch * 100:+.0f} % vs the reference period, consistent "
                                      f"with an oil-system cause")

    s_score = 2 * len(strong) + len(weak)
    e_score = len([e for e in engine if not e.startswith("(")]) + engine.count("(pattern weight x2)")
    if s_score >= 2 and e_score < 2:
        verdict = "sensor_suspected"
    elif e_score >= 2 and s_score < 2:
        verdict = "engine_suspected"
    else:
        verdict = "undetermined"
        if s_score >= 2 and e_score >= 2:
            info.append("sensor and engine evidence conflict")
    evidence = ([f"SENSOR: {x}" for x in strong] + [f"SENSOR (weak): {x}" for x in weak]
                + [f"ENGINE: {x}" for x in engine if not x.startswith("(")] + [f"NOTE: {x}" for x in info])
    return SensorAssessment(verdict, pattern, evidence)
