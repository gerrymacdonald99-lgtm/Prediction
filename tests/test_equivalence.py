"""candump and CSV inputs must produce the same canonical series."""

import numpy as np
import pandas as pd
import pytest

from busbuilder import build_frames, csv_mapping, write_candump, write_csv
from j1939pf.pipeline import canonical_from_candump, canonical_from_csv


def assert_same(a: pd.DataFrame, b: pd.DataFrame):
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b) and len(a) > 0
    assert a["engine_id"].tolist() == b["engine_id"].tolist()
    assert (a["t"] == b["t"]).all() and (a["segment"] == b["segment"]).all()
    for c in a.columns[3:]:
        x, y = a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float)
        assert np.array_equal(np.isnan(x), np.isnan(y)), c
        assert np.allclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True), c


@pytest.mark.parametrize("layout,ts_format", [("wide", "iso8601"), ("long", "iso8601"), ("wide", "epoch_s")])
def test_candump_and_csv_same_canonical(tmp_path, table, cfg, layout, ts_format):
    frames = build_frames(table, 40, sas=(0x00, 0x01), seed=3)
    log, csvp = tmp_path / "bus.log", tmp_path / "bus.csv"
    write_candump(frames, log)
    write_csv(frames, table, csvp, layout=layout, ts_format=ts_format)
    a = canonical_from_candump(log, table=table, config=cfg)
    b = canonical_from_csv(csvp, csv_mapping(table, layout, ts_format), table=table, config=cfg)
    assert_same(a.canonical, b.canonical)
    # The fixture contains sensor-error frames and DM1; both paths must carry them identically.
    assert a.canonical["oil_pressure_kpa__err"].sum() > 0
    assert a.canonical["dm1_active_count"].max() == 1
    assert sorted(a.canonical["engine_id"].unique()) == ["SA00", "SA01"]


def test_csv_unit_conversion_and_bad_cells(tmp_path, table, cfg):
    p = tmp_path / "x.csv"
    p.write_text("ts,oil_psi,cool_f,fuel_gph\n"
                 "2024-01-01 00:00:00.5,43.5113,212,10\n"
                 "2024-01-01 00:00:01.5,ERR,junk,NA\n"
                 "2024-01-01 00:00:02.5,999,,\n")
    mp = {"timestamp": {"column": "ts", "format": "%Y-%m-%d %H:%M:%S.%f", "timezone": "America/Chicago"},
          "na_values": ["NA"], "error_values": ["ERR"],
          "columns": {"oil_psi": {"spn": 100, "unit": "psi"}, "cool_f": {"spn": 110, "unit": "degF"},
                      "fuel_gph": {"spn": 183, "unit": "gal/h"}}}
    r = canonical_from_csv(p, mp, table=table, config=cfg)
    df = r.canonical
    assert df["t"].iloc[0] == pd.Timestamp("2024-01-01 06:00:00", tz="UTC")  # CST = UTC-6
    assert df["oil_pressure_kpa"].iloc[0] == pytest.approx(300.0, abs=1e-3)
    assert df["coolant_temp_c"].iloc[0] == pytest.approx(100.0)
    assert df["fuel_rate_lph"].iloc[0] == pytest.approx(37.85411784)
    assert np.isnan(df["oil_pressure_kpa"].iloc[1]) and np.isnan(df["oil_pressure_kpa"].iloc[2])
    assert df["oil_pressure_kpa__err"].tolist() == [0, 1, 1]  # ERR token, then out of range (6888 kPa)
    assert df["coolant_temp_c__err"].tolist() == [0, 1, 0]   # non-numeric junk
    sc = r.report["csv"]["status_counts"]
    assert sc["SA00/fuel_rate_lph"] == {"ok": 1, "not_available": 1}


def test_csv_mapping_requires_units(tmp_path, table, cfg):
    p = tmp_path / "x.csv"
    p.write_text("ts,oil\n0,1\n")
    with pytest.raises(ValueError, match="unit"):
        canonical_from_csv(p, {"timestamp": {"column": "ts", "format": "epoch_s"},
                               "columns": {"oil": {"spn": 100}}}, table=table, config=cfg)
