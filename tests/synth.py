"""Shared synthetic fixtures for phase 2+ tests (cached: generation is deterministic)."""

from functools import lru_cache

from j1939pf.analysis import analyze
from j1939pf.engine_config import parse_engine_config
from j1939pf.generator import generate_engine


def synthetic_config(e, **drop):
    """The synthetic engine's 'manual': on-highway PLACEHOLDER pressure minimums (as the spec allows for
    examples) plus the generator's own rated speed. Pass name=None to leave a limit out."""
    limits = {
        "oil_pressure_min_idle": {"value": 15, "unit": "psi", "placeholder": True, "source": "PLACEHOLDER on-highway ~15 psi"},
        "oil_pressure_min_rated": {"value": 35, "unit": "psi", "placeholder": True, "source": "PLACEHOLDER on-highway ~35 psi"},
        "oil_temp_max": {"value": 248, "unit": "degF", "placeholder": True, "source": "PLACEHOLDER unverified ~248 F"},
        "coolant_temp_max": {"value": 107, "unit": "degC", "source": "synthetic engine definition"},
        "rated_rpm": {"value": e.profile.rated_rpm, "unit": "rpm", "source": "synthetic engine definition"},
    }
    for k, v in drop.items():
        if v is None:
            limits.pop(k, None)
    return parse_engine_config({"engine_id": e.engine_id, "limits": limits})


@lru_cache(maxsize=64)
def engine(scenario, seed=0, noise_model=None, hours=300.0, engine_id="SA00"):
    return generate_engine(engine_id, scenario, hours=hours, seed=seed, noise_model=noise_model)


@lru_cache(maxsize=64)
def analyzed(scenario, seed=0, noise_model=None, hours=300.0):
    e = engine(scenario, seed, noise_model, hours)
    return e, analyze(e.canonical, {e.engine_id: synthetic_config(e)}, events=e.events)


def usage_config(e, **drop):
    """synthetic_config plus the usage cut-offs a manual (or fleet policy) would supply."""
    from j1939pf.engine_config import parse_engine_config
    c = synthetic_config(e)
    p = e.profile
    extra = {
        "idle_rpm_band": {"low": p.idle_rpm - 100, "high": p.idle_rpm + 100, "unit": "rpm", "source": "synthetic engine"},
        "idle_load_cutoff": {"value": 15, "unit": "%", "source": "synthetic engine"},
        "coolant_low_temp_cutoff": {"value": 60, "unit": "degC", "source": "synthetic engine"},
        "oil_low_temp_cutoff": {"value": 60, "unit": "degC", "source": "synthetic engine"},
        "idle_pct_flag_threshold": {"value": 45, "unit": "%", "source": "synthetic fleet policy"},
        "lugging_rpm_max": {"value": p.idle_rpm + 0.25 * (p.rated_rpm - p.idle_rpm), "unit": "rpm",
                            "source": "synthetic engine"},
    }
    for k, v in drop.items():
        if v is None:
            extra.pop(k, None)
            c.limits.pop(k, None)
    x = parse_engine_config({"engine_id": e.engine_id, "limits": extra})
    c.limits.update(x.limits)
    c.idle_rpm_band = x.idle_rpm_band
    return c


@lru_cache(maxsize=64)
def usage(scenario, seed=0, noise_model=None, with_log=True, oil_interval=250.0, hours=300.0):
    from j1939pf.generator import generate_engine
    from j1939pf.usage import usage_report
    e = generate_engine("SA00", scenario, hours=hours, seed=seed, noise_model=noise_model,
                        oil_change_interval_h=oil_interval)
    return e, usage_report(e.canonical, {"SA00": usage_config(e)}, events=e.events if with_log else None)
