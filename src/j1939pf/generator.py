"""Seeded synthetic engine generator (canonical-series level, optionally rendered to candump).

SYNTHETIC-DATA CAVEAT: the pressure/temperature model here is a plausible toy,
not an engine model. Every coefficient below is a generator parameter, NOT an
engine specification. Detecting faults this generator scripted proves the
plumbing works; it says nothing about real-world accuracy.

Scenarios (fault onset is in engine hours *into the record*):
  healthy             normal duty cycle
  bearing_wear        slow pressure loss, larger at idle (leakage grows)
  clogged_filter      filter restriction grows; pressure lost at high flow (high RPM)
  stuck_relief_valve  relief valve stuck partly open: pressure capped low at high RPM, idle normal
  failing_pump        accelerating loss, worse at high RPM (pump cannot keep up)
  sensor_drift        oil pressure SENSOR reading drifts (engine is fine)
  high_idle           healthy engine, idle-dominated duty cycle
  cold_operation      healthy engine run chronically cold (thermostat stuck open, cold ambient, light load)
  abusive             healthy engine, abusive use: lugging, high load straight after a cold start,
                      shutdown straight from high load (no cool-down idle)
  boost_leak          charge-air leak: boost (and turbo speed) low for the fuel being burned
Other synthetic signals (toy models, for correlation discovery): boost, turbo speed, intake manifold and
exhaust temperature, fuel delivery pressure, crankcase pressure (slow blow-by growth), coolant pressure and
level, DPF differential pressure from a soot model with active regenerations (more soot at idle / light
load, passive burn-off at high load), DPF ash load.
Events (maintenance log): oil change (+filter), oil-grade change, sensor replacement. Events take effect
at the start of a session (the engine is stopped for maintenance). Oil ageing is modelled: used oil
slowly loses viscosity (pressure at a fixed RPM/oil temperature sags), oil level falls with consumption
and the filter loads slightly; an oil change restores all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .spn_table import SpnTable, load_spn_table

SCENARIOS = ("healthy", "bearing_wear", "clogged_filter", "stuck_relief_valve", "failing_pump",
             "sensor_drift", "high_idle", "cold_operation", "abusive", "boost_leak")
NOISE_MODELS = ("gaussian", "student_t", "ar1")
START = pd.Timestamp("2024-01-01T00:00:00Z")

_DUTY = {  # mode probabilities: idle, low, mid, rated, lug
    "normal": (0.20, 0.20, 0.35, 0.25, 0.0),
    "high_idle": (0.65, 0.15, 0.12, 0.08, 0.0),
    "light": (0.40, 0.40, 0.15, 0.05, 0.0),
    "abusive": (0.10, 0.15, 0.30, 0.25, 0.20),
}
_MODES = ["idle", "low", "mid", "rated", "lug"]

_DEFAULT_FAULT = {
    "bearing_wear": {"onset_h": 80.0, "rate_per_h": 0.0012},       # fraction of pressure lost per hour
    "clogged_filter": {"onset_h": 80.0, "growth_per_h": 0.05},     # filter restriction growth per hour
    "stuck_relief_valve": {"onset_h": 80.0, "cap_frac_of_rated": 0.5},
    "failing_pump": {"onset_h": 80.0, "full_loss_h": 300.0, "max_loss": 0.6},
    "sensor_drift": {"onset_h": 80.0, "kpa_per_h": 0.15},
    "boost_leak": {"onset_h": 80.0, "boost_frac": 0.8},
}


@dataclass
class EngineProfile:
    """Synthetic generator parameters (NOT engine specifications)."""
    idle_rpm: float
    rated_rpm: float
    p_idle_kpa: float
    p_rated_kpa: float
    p_relief_kpa: float
    visc_k: float
    thermostat_c: float
    filter_dp_rated_kpa: float
    noise_model: str
    noise_kpa: float
    start_hours: float


@dataclass
class SyntheticEngine:
    engine_id: str
    scenario: str
    canonical: pd.DataFrame
    events: pd.DataFrame
    truth: dict
    profile: EngineProfile


def random_profile(rng: np.random.Generator, noise_model: str | None = None) -> EngineProfile:
    idle = float(rng.uniform(620, 760))
    rated = float(rng.choice([1800, 1900, 2000, 2100]))
    p_idle = float(rng.uniform(170, 250))
    p_rated = float(rng.uniform(380, 470))
    return EngineProfile(
        idle_rpm=idle, rated_rpm=rated, p_idle_kpa=p_idle, p_rated_kpa=p_rated,
        p_relief_kpa=p_rated * float(rng.uniform(1.08, 1.2)), visc_k=float(rng.uniform(0.006, 0.010)),
        thermostat_c=float(rng.uniform(84, 92)), filter_dp_rated_kpa=float(rng.uniform(25, 45)),
        noise_model=noise_model or str(rng.choice(NOISE_MODELS)), noise_kpa=float(rng.uniform(2.5, 5.0)),
        start_hours=float(rng.uniform(200, 6000)),
    )


def _quantize(table: SpnTable, canonical: str, x: np.ndarray) -> np.ndarray:
    d = table.by_canonical[canonical]
    raw = np.round((np.asarray(x, dtype=float) - d.offset) / d.scale)
    lo = np.ceil((d.valid_min - d.offset) / d.scale)
    hi = np.floor((d.valid_max - d.offset) / d.scale)
    out = np.clip(raw, lo, hi) * d.scale + d.offset
    out[np.isnan(x)] = np.nan
    return out


def _noise(rng, model: str, n: int, sigma: float) -> np.ndarray:
    if model == "gaussian":
        return rng.normal(0, sigma, n)
    if model == "student_t":
        return rng.standard_t(4, n) * sigma / np.sqrt(2.0)  # t4 has variance 2
    if model == "ar1":
        phi = 0.8
        e = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), n)
        return lfilter([1.0], [1.0, -phi], e)
    raise ValueError(f"unknown noise model {model}")


def _day_blocks(rng, duty: str, session_h: float):
    """List of (mode, seconds). mode in idle/low/mid/rated/lug/off."""
    probs = _DUTY[duty]
    abusive = duty == "abusive"
    blocks = [("off", float(rng.uniform(60, 180)))]  # key on, engine not yet started
    if abusive:
        blocks.append(("rated", float(rng.uniform(300, 900))))  # straight to high load from a cold start
    else:
        blocks.append(("idle", float(rng.uniform(120, 300))))   # warm-up idle

    def stop():
        if not abusive:
            blocks.append(("idle", float(rng.uniform(180, 300))))  # cool-down idle before shutdown
        elif blocks[-1][0] != "rated":
            blocks.append(("rated", float(rng.uniform(120, 300))))  # shut down straight from high load

    elapsed = 0.0
    while elapsed < session_h * 3600:
        if rng.random() < 0.03:
            stop()
            blocks.append(("off", float(rng.uniform(300, 1200))))  # mid-shift stop, key on
        mode = rng.choice(_MODES, p=probs)
        dur = float(np.clip(rng.exponential(480), 60, 1800))
        blocks.append((str(mode), dur))
        elapsed += dur
    stop()
    blocks.append(("off", float(rng.uniform(60, 120))))
    return blocks


def generate_engine(engine_id: str = "SA00", scenario: str = "healthy", *, hours: float = 300.0,
                    dt_s: float = 10.0, seed: int = 0, noise_model: str | None = None,
                    fault: dict | None = None, oil_change_interval_h: float | None = 250.0,
                    grade_change_at_h: float | None = None, grade_factor: float = 0.93,
                    sensor_replacement_at_h: float | None = None, sensor_offset_step_kpa: float = 6.0,
                    missing_frac: float = 0.002, profile: EngineProfile | None = None,
                    oil_degradation: float = 0.03, oil_consumption_pct_per_h: float = 0.012,
                    table: SpnTable | None = None, start: pd.Timestamp = START) -> SyntheticEngine:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}; choose from {SCENARIOS}")
    table = table or load_spn_table()
    rng = np.random.default_rng(seed)
    prof = profile or random_profile(rng, noise_model)
    if noise_model:
        prof.noise_model = noise_model
    fp = dict(_DEFAULT_FAULT.get(scenario, {}))
    fp.update(fault or {})
    duty = {"high_idle": "high_idle", "cold_operation": "light", "abusive": "abusive"}.get(scenario, "normal")
    cold = scenario == "cold_operation"

    mode_rpm = {"off": lambda: 0.0,
                "idle": lambda: prof.idle_rpm + rng.normal(0, 10),
                "low": lambda: prof.idle_rpm + (prof.rated_rpm - prof.idle_rpm) * rng.uniform(0.2, 0.45),
                "mid": lambda: prof.idle_rpm + (prof.rated_rpm - prof.idle_rpm) * rng.uniform(0.45, 0.85),
                "rated": lambda: prof.rated_rpm - rng.uniform(0, 60),
                "lug": lambda: prof.idle_rpm + (prof.rated_rpm - prof.idle_rpm) * rng.uniform(0.05, 0.2)}
    mode_load = {"off": (0, 0), "idle": (4, 12), "low": (20, 40), "mid": (40, 75), "rated": (78, 100),
                 "lug": (88, 100)}

    t_parts, rpm_parts, load_parts, seg_parts, amb_parts = [], [], [], [], []
    run_s = 0.0
    day = 0
    while run_s < hours * 3600:
        session_h = float(rng.uniform(6, 10))
        blocks = _day_blocks(rng, duty, session_h)
        n_per = [max(1, int(round(d / dt_s))) for _, d in blocks]
        rpm_t = np.concatenate([np.full(n, mode_rpm[m]()) for (m, _), n in zip(blocks, n_per)])
        lo_hi = [mode_load[m] for m, _ in blocks]
        load_t = np.concatenate([np.full(n, rng.uniform(*lh)) for lh, n in zip(lo_hi, n_per)])
        a = np.exp(-dt_s / 12.0)  # rpm responds within ~12 s
        rpm = lfilter([1 - a], [1, -a], rpm_t, zi=[0.0])[0]
        rpm[rpm_t == 0] = 0.0
        rpm = np.where(rpm_t > 0, rpm + rng.normal(0, 4, rpm.size), 0.0)
        crank = (rpm_t > 0) & (np.r_[0.0, rpm_t[:-1]] == 0)
        rpm[crank] = rng.uniform(120, 200, int(crank.sum()))  # cranking row
        load = np.where(rpm > 0, load_t + rng.normal(0, 2, rpm.size), 0.0)
        t0 = start + pd.Timedelta(days=day, hours=float(rng.uniform(5.5, 7.5)))
        t_parts.append(t0.value // 1000 + (np.arange(rpm.size) * dt_s * 1e6).astype(np.int64))
        rpm_parts.append(np.clip(rpm, 0, None))
        load_parts.append(np.clip(load, 0, 100))
        seg_parts.append(np.full(rpm.size, day))
        season = 12 * np.sin(2 * np.pi * (day - 100) / 365.0)
        amb = (-12.0 if cold else 15.0) + (0 if cold else season) + rng.normal(0, 3)
        amb_parts.append(np.full(rpm.size, amb))
        run_s += float((rpm > 300).sum()) * dt_s
        day += 1

    t_us = np.concatenate(t_parts)
    rpm = np.concatenate(rpm_parts)
    load = np.concatenate(load_parts)
    seg = np.concatenate(seg_parts)
    amb = np.concatenate(amb_parts)
    n = rpm.size
    running = rpm > 300
    rel_h = np.cumsum(running * dt_s) / 3600.0
    hours_abs = prof.start_hours + rel_h

    # --- temperatures (first-order lags; reset to ambient each day) ---------------
    coolant = np.empty(n)
    oil_t = np.empty(n)
    soot = np.empty(n)
    soot_g = float(rng.uniform(5, 20))
    regen_left = 0.0
    regen_rows = []
    a_c, a_o, a_off = np.exp(-dt_s / 300.0), np.exp(-dt_s / 600.0), np.exp(-dt_s / 3600.0)
    thermo = prof.thermostat_c
    for i in range(n):
        if i == 0 or seg[i] != seg[i - 1]:
            c, o = amb[i], amb[i]
        lf = load[i] / 100.0
        if running[i]:
            tc = min(thermo + 5 * lf, amb[i] + 25 + 40 * lf) if cold else thermo + 5 * lf
            to = tc + 6 + 10 * lf
            c = a_c * c + (1 - a_c) * tc
            o = a_o * o + (1 - a_o) * to
        else:
            c = a_off * c + (1 - a_off) * amb[i]
            o = a_off * o + (1 - a_off) * amb[i]
        coolant[i], oil_t[i] = c, o
        # DPF soot (grams, toy model): accumulates more at idle / light load, burns off passively at high load;
        # an active regeneration starts at 40 g and lasts ~20 min
        if running[i]:
            if regen_left > 0:
                soot_g = max(2.0, soot_g - 38.0 * dt_s / 1200.0)
                regen_left -= dt_s
            else:
                rate = 1.8 if lf < 0.3 else 0.7 if lf < 0.7 else -0.8
                soot_g = max(0.0, soot_g + rate * dt_s / 3600.0)
                if soot_g >= 40.0:
                    regen_left = 1200.0
                    regen_rows.append(i)
        soot[i] = soot_g

    # --- events: maintenance happens with the engine stopped, i.e. at a session start --------
    idx = np.arange(n)
    seg_start = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])

    def at_session(h):
        cand = seg_start[rel_h[seg_start] >= h]
        return int(cand[0]) if cand.size and cand[0] > 0 else None

    events = []  # (type, row index, detail)
    change_idx = []
    if oil_change_interval_h:
        k = 1
        while k * oil_change_interval_h < rel_h[-1]:
            i = at_session(k * oil_change_interval_h)
            if i is not None and i not in change_idx:
                change_idx.append(i)
                events.append(("oil_change", i, "oil and filter changed"))
            k += 1
    grade_idx = at_session(grade_change_at_h) if grade_change_at_h is not None else None
    if grade_idx is not None:
        events.append(("oil_grade_change", grade_idx, f"oil grade changed (synthetic viscosity factor {grade_factor})"))
    sensor_idx = at_session(sensor_replacement_at_h) if sensor_replacement_at_h is not None else None
    if sensor_idx is not None:
        events.append(("sensor_replacement", sensor_idx, "oil pressure sensor replaced"))
    events.sort(key=lambda e: e[1])

    # oil age (engine hours since the last change); the record starts part-way through an interval
    oil_age = rel_h + float(rng.uniform(0, 60))
    for i in change_idx:
        oil_age = np.where(idx >= i, rel_h - rel_h[i], oil_age)

    # --- oil pressure (true) ---------------------------------------------------------
    rf = np.clip((rpm - prof.idle_rpm) / (prof.rated_rpm - prof.idle_rpm), 0, 1)
    visc = np.clip(np.exp(-prof.visc_k * (oil_t - 95.0)), 0.7, 1.9)
    # used oil thins (shear, dilution); fresh oil is slightly thicker for its first tens of hours
    visc = visc * (1 - oil_degradation * (1 - np.exp(-oil_age / 150.0))) * (1 + 0.015 * np.exp(-oil_age / 30.0))
    if grade_idx is not None:
        visc = visc * np.where(idx >= grade_idx, grade_factor, 1.0)
    p = (prof.p_idle_kpa + (prof.p_rated_kpa - prof.p_idle_kpa) * rf ** 0.8) * visc
    p = np.minimum(p, prof.p_relief_kpa)

    # filter differential pressure (flow ~ rpm, viscosity) and restriction growth
    since_change = oil_age
    flow = 0.25 + 0.75 * rf ** 1.3
    restriction = 1.0 + 0.0001 * oil_age  # a healthy filter loads slightly over its interval
    onset = fp.get("onset_h", np.inf)
    after = np.clip(rel_h - onset, 0, None)
    if scenario == "clogged_filter":
        # restriction grows from onset and is reset by filter changes after onset
        grow_from = np.maximum(onset, rel_h - since_change)
        restriction += fp["growth_per_h"] * np.clip(rel_h - grow_from, 0, None)
    dp = prof.filter_dp_rated_kpa * flow * visc * restriction
    p = p - np.clip(dp - prof.filter_dp_rated_kpa * flow * visc, 0, None) * 0.9

    if scenario == "bearing_wear":
        loss = fp["rate_per_h"] * after
        p = p * (1 - loss * (1 - 0.4 * rf))
    elif scenario == "stuck_relief_valve":
        cap = fp["cap_frac_of_rated"] * prof.p_rated_kpa * visc
        p = np.where(rel_h >= onset, np.minimum(p, cap), p)
    elif scenario == "failing_pump":
        loss = fp["max_loss"] * np.clip(after / fp["full_loss_h"], 0, 1) ** 2
        p = p * (1 - loss * (0.5 + 0.5 * rf))

    p = np.where(running, p, np.where(rpm > 0, p * rpm / 300.0, 0.0))
    dp = np.where(running, dp, 0.0)

    # --- sensor reading ---------------------------------------------------------------
    offset = np.zeros(n)
    if scenario == "sensor_drift":
        offset += fp["kpa_per_h"] * after
        if sensor_idx is not None:
            offset = np.where(idx >= sensor_idx, 0.0, offset)
    if sensor_idx is not None:
        offset += np.where(idx >= sensor_idx, sensor_offset_step_kpa, 0.0)
    noise = _noise(rng, prof.noise_model, n, prof.noise_kpa)
    reading = np.clip(p + offset + np.where(running, noise, 0.3 * noise), 0, None)
    dp_reading = np.clip(dp + rng.normal(0, 1.0, n) * np.where(running, 1, 0.3), 0, None)

    # --- other signals -----------------------------------------------------------------
    fuel = np.where(running, 2.0 + 0.00006 * rpm * (10 + load) * 3, 0.0)
    torque = np.where(running, load * 0.95, 0.0)
    # charge air / turbo / exhaust / fuel system / crankcase / cooling system (toy relationships)
    boost = np.where(running, 5.5 * np.clip(fuel - 2.0, 0, None), 0.0)
    if scenario == "boost_leak":
        boost = np.where(rel_h >= onset, boost * fp["boost_frac"], boost)
    boost = np.clip(boost + np.where(running, rng.normal(0, 3, n), 0.0), 0, None)
    turbo = np.where(running, 25000 + 420 * boost + rng.normal(0, 800, n), 0.0)
    imt = amb + np.where(running, 12 + 0.12 * boost, 0.0) + rng.normal(0, 1, n)
    exh = np.where(running, oil_t + 140 + 3.2 * load, oil_t) + rng.normal(0, 5, n)
    fuel_p = np.where(running, 560 - 2.2 * fuel, 0.0) + rng.normal(0, 6, n)
    crank_p = np.where(running, 0.25 + 0.006 * load + 0.0008 * rel_h, 0.0) + rng.normal(0, 0.05, n)
    cool_p = np.where(running, 20 + 0.9 * np.clip(coolant - 20, 0, None), 0.5 * np.clip(coolant - 20, 0, None))
    cool_p = np.clip(cool_p + rng.normal(0, 2, n), 0, None)
    cool_lvl = 95.0 - 0.003 * rel_h + rng.normal(0, 0.3, n)
    flow_f = 0.3 + 0.7 * np.clip(fuel / 40.0, 0, 1.2)
    dpf = np.where(running, (0.8 + 0.12 * soot) * flow_f, 0.0) + rng.normal(0, 0.15, n)
    ash = 5.0 + 0.01 * rel_h
    cranking = (rpm > 0) & ~running
    batt = np.where(running, 14.1, np.where(cranking, 10.3, 12.6)) + rng.normal(0, 0.08, n)
    oil_level = 80.0 - oil_consumption_pct_per_h * oil_age + rng.normal(0, 0.3, n)
    df = pd.DataFrame({
        "engine_id": engine_id,
        "t": pd.to_datetime(t_us, unit="us", utc=True),
        "segment": seg.astype(np.int64),
        "actual_torque_pct": _quantize(table, "actual_torque_pct", torque),
        "engine_speed_rpm": _quantize(table, "engine_speed_rpm", rpm),
        "load_at_speed_pct": _quantize(table, "load_at_speed_pct", load),
        "oil_level_pct": _quantize(table, "oil_level_pct", oil_level),
        "battery_v": _quantize(table, "battery_v", batt),
        "oil_pressure_kpa": _quantize(table, "oil_pressure_kpa", reading),
        "oil_filter_dp_kpa": _quantize(table, "oil_filter_dp_kpa", dp_reading),
        "coolant_temp_c": _quantize(table, "coolant_temp_c", coolant),
        "oil_temp_c": _quantize(table, "oil_temp_c", oil_t),
        "fuel_rate_lph": _quantize(table, "fuel_rate_lph", fuel),
        "ambient_temp_c": _quantize(table, "ambient_temp_c", amb),
        "engine_hours_h": _quantize(table, "engine_hours_h", hours_abs),
        "boost_kpa": _quantize(table, "boost_kpa", boost),
        "turbo_speed_rpm": _quantize(table, "turbo_speed_rpm", turbo),
        "intake_manifold_temp_c": _quantize(table, "intake_manifold_temp_c", imt),
        "exhaust_temp_c": _quantize(table, "exhaust_temp_c", exh),
        "fuel_delivery_pressure_kpa": _quantize(table, "fuel_delivery_pressure_kpa", fuel_p),
        "crankcase_pressure_kpa": _quantize(table, "crankcase_pressure_kpa", crank_p),
        "coolant_pressure_kpa": _quantize(table, "coolant_pressure_kpa", cool_p),
        "coolant_level_pct": _quantize(table, "coolant_level_pct", cool_lvl),
        "dpf_dp_kpa": _quantize(table, "dpf_dp_kpa", np.clip(dpf, 0, None)),
        "dpf_ash_load_pct": _quantize(table, "dpf_ash_load_pct", ash),
        "dm1_active_count": 0.0, "dm1_mil": 0.0, "dm1_rsl": 0.0, "dm1_awl": 0.0, "dm1_pl": 0.0,
    })
    # missing data: drop random oil-pressure samples (lost frames)
    if missing_frac:
        df.loc[rng.random(n) < missing_frac, "oil_pressure_kpa"] = np.nan

    ev = pd.DataFrame(
        [{"engine_id": engine_id, "t": df["t"].iloc[i], "engine_hours": float(hours_abs[i]), "event_type": e,
          "detail": d} for e, i, d in events],
        columns=["engine_id", "t", "engine_hours", "event_type", "detail"])
    truth = {"scenario": scenario, "kind": {"sensor_drift": "sensor", "high_idle": "usage",
                                            "cold_operation": "usage", "abusive": "usage", "boost_leak": "engine",
                                            "healthy": "none"}.get(scenario, "engine"),
             "oil_change_engine_hours": [float(hours_abs[i]) for i in change_idx],
             "regen_engine_hours": [float(hours_abs[i]) for i in regen_rows],
             "onset_engine_hours": prof.start_hours + onset if np.isfinite(onset) else None,
             "fault_params": fp, "noise_model": prof.noise_model, "start_hours": prof.start_hours}
    return SyntheticEngine(engine_id, scenario, df, ev, truth, prof)


def generate_fleet(specs: list[dict], seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """specs: [{engine_id, scenario, **kwargs}]. Each engine gets its own derived seed."""
    engines = []
    for i, spec in enumerate(specs):
        spec = dict(spec)
        spec.setdefault("seed", seed * 1000 + i)
        engines.append(generate_engine(**spec))
    canonical = pd.concat([e.canonical for e in engines], ignore_index=True)
    events = pd.concat([e.events for e in engines], ignore_index=True)
    return canonical, events, {e.engine_id: e for e in engines}


def render_candump(df: pd.DataFrame, path, table: SpnTable | None = None, sa_of: dict | None = None) -> int:
    """Write the canonical rows back out as J1939 frames (one frame per PGN per row)."""
    from .encode import NA, candump_line, dm_payload, encode_pgn
    from .ids import make_can_id

    table = table or load_spn_table()
    pgns: dict[int, list] = {}
    for col in df.columns:
        d = table.by_canonical.get(col)
        if d is not None and d.attribute_to == "source":
            pgns.setdefault(d.pgn, []).append(d)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in df.itertuples(index=False):
            r = row._asdict()
            sa = (sa_of or {}).get(r["engine_id"], int(str(r["engine_id"])[2:], 16))
            t_us = r["t"].value // 1000
            for k, (pgn, defs) in enumerate(sorted(pgns.items())):
                vals = {d.spn: (NA if pd.isna(r[d.canonical]) else r[d.canonical]) for d in defs}
                fh.write(candump_line(t_us + 1000 * (k + 1), make_can_id(6, pgn, sa), encode_pgn(table, pgn, vals)))
                n += 1
            if "dm1_active_count" in r and not pd.isna(r["dm1_active_count"]):
                fh.write(candump_line(t_us + 900, make_can_id(6, 65226, sa), dm_payload([])))
                n += 1
    return n
