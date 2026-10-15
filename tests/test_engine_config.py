import pytest

from j1939pf.config import CONFIG_DIR
from j1939pf.engine_config import load_engine_configs, parse_engine_config
from j1939pf.units import PSI_TO_KPA


def test_example_configs_are_placeholders_with_no_invented_industrial_values():
    cfgs = load_engine_configs(CONFIG_DIR / "engines")
    assert set(cfgs) == {"EXAMPLE_CUMMINS", "EXAMPLE_CAT", "EXAMPLE_CUMMINS_ISB"}  # TEMPLATE is skipped
    for c in cfgs.values():
        assert c.limits, c.engine_id
        for lim in c.limits.values():
            assert lim.placeholder and "PLACEHOLDER" in lim.source
        assert c.limit("rated_rpm") is None and c.idle_rpm_band is None   # must come from the manual
    assert cfgs["EXAMPLE_CUMMINS"].limit("oil_pressure_min_idle").value == pytest.approx(15 * PSI_TO_KPA)
    assert "UNVERIFIED" in cfgs["EXAMPLE_CUMMINS"].limit("oil_temp_max").source
    assert cfgs["EXAMPLE_CUMMINS"].limit("oil_temp_max").value == pytest.approx(120.0)


def test_limits_need_unit_and_source():
    with pytest.raises(ValueError, match="source"):
        parse_engine_config({"engine_id": "x", "limits": {"oil_temp_max": {"value": 120, "unit": "degC"}}})
    with pytest.raises(ValueError, match="value, unit"):
        parse_engine_config({"engine_id": "x", "limits": {"oil_temp_max": 120}})
    with pytest.raises(ValueError, match="unknown limits"):
        parse_engine_config({"engine_id": "x", "limits": {"oil_life_pct": {"value": 1, "unit": "%", "source": "?"}}})


def test_missing_limits_stay_missing():
    c = parse_engine_config({"engine_id": "x", "limits": {}})
    assert c.limits == {} and c.limit("oil_pressure_min_idle") is None
    assert (c.load_spn, c.load_fallback_spn) == (92, 513)
