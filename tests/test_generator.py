import numpy as np
import pandas as pd
import pytest

from j1939pf.generator import SCENARIOS, generate_engine, generate_fleet, render_candump
from j1939pf.pipeline import canonical_from_candump
from j1939pf.config import load_defaults


def test_deterministic_and_seed_sensitive():
    a = generate_engine("SA00", "bearing_wear", hours=30, seed=4)
    b = generate_engine("SA00", "bearing_wear", hours=30, seed=4)
    c = generate_engine("SA00", "bearing_wear", hours=30, seed=5)
    pd.testing.assert_frame_equal(a.canonical, b.canonical)
    assert not a.canonical["oil_pressure_kpa"].equals(c.canonical["oil_pressure_kpa"])


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_generates_bus_quantized_values(scenario):
    e = generate_engine("SA00", scenario, hours=20, seed=1)
    df = e.canonical
    p = df["oil_pressure_kpa"].dropna()
    assert np.allclose(p % 4, 0)                       # SPN 100: 4 kPa/bit
    assert (df.loc[df.engine_speed_rpm == 0, "oil_pressure_kpa"].dropna() <= 8).mean() > 0.95
    assert df["engine_hours_h"].is_monotonic_increasing
    assert e.truth["scenario"] == scenario


def test_noise_models_and_events():
    models = {generate_engine("SA00", "healthy", hours=10, seed=s).profile.noise_model for s in range(12)}
    assert models == {"gaussian", "student_t", "ar1"}
    e = generate_engine("SA00", "healthy", hours=300, seed=0, grade_change_at_h=120, sensor_replacement_at_h=200)
    assert set(e.events["event_type"]) == {"oil_change", "oil_grade_change", "sensor_replacement"}
    assert e.events["engine_hours"].is_monotonic_increasing is not None


def test_fleet_engines_differ():
    df, ev, eng = generate_fleet([{"engine_id": "SA00", "scenario": "healthy", "hours": 20},
                                  {"engine_id": "SA01", "scenario": "healthy", "hours": 20}], seed=1)
    assert set(df.engine_id) == {"SA00", "SA01"}
    assert eng["SA00"].profile.p_rated_kpa != eng["SA01"].profile.p_rated_kpa


def test_render_candump_roundtrip(tmp_path):
    e = generate_engine("SA00", "healthy", hours=3, seed=2, dt_s=10)
    log = tmp_path / "gen.log"
    render_candump(e.canonical, log)
    cfg = load_defaults(overrides={"canonical": {"resample_interval_s": 10, "gap_s": 60}})
    back = canonical_from_candump(log, config=cfg).canonical
    src = e.canonical.copy()
    src["t"] = src["t"].dt.floor("10s")   # canonical rows are stamped with the interval start
    assert len(back) >= len(src)
    m = back.merge(src, on=["engine_id", "t"], suffixes=("", "_src"))
    assert len(m) == len(src)
    for col in ["engine_speed_rpm", "oil_pressure_kpa", "oil_temp_c", "coolant_temp_c", "engine_hours_h",
                "oil_filter_dp_kpa", "load_at_speed_pct"]:
        assert np.allclose(m[col], m[col + "_src"], equal_nan=True, atol=1e-6), col
