import math

import numpy as np
import pytest

from j1939pf.canonical import CanonicalBuilder
from j1939pf.decoder import Sample
from j1939pf.encode import ERR, NA, candump_line, encode_pgn
from j1939pf.ids import make_can_id
from j1939pf.pipeline import canonical_from_candump
from j1939pf.spn_table import ERROR, NOT_AVAILABLE, OK, build_registry

T0 = 1_700_000_000_000_000


def builder(table, **kw):
    return CanonicalBuilder(build_registry(table), **kw)


def test_rate_mismatch_mean_and_hold(table):
    b = builder(table)
    for i in range(250):  # 20 ms engine speed for 5 s, ramp
        b.add(Sample(T0 + i * 20_000, "E", "engine_speed_rpm", 1000.0 + i, OK))
        if i == 5:
            b.add(Sample(T0 + 100_000, "E", "coolant_temp_c", 80.0, OK))  # slow signal, sent once
    df = b.finalize()
    assert len(df) == 5
    assert df["engine_speed_rpm"].tolist() == [1024.5, 1074.5, 1124.5, 1174.5, 1224.5]
    # held for up to max(3 s, 3 x 1 s) = 3 s after the bin where it was seen, then NULL
    assert df["coolant_temp_c"].tolist()[:4] == [80.0] * 4
    assert math.isnan(df["coolant_temp_c"].iloc[4])


def test_missing_frames_inside_segment_produce_rows(table):
    b = builder(table)
    for s in (0, 1, 4):  # nothing at s=2,3 (gap 3 s < gap_s 5 s)
        b.add(Sample(T0 + s * 1_000_000, "E", "oil_pressure_kpa", 300.0, OK))
    df = b.finalize()
    assert len(df) == 5 and df["segment"].nunique() == 1
    assert df["oil_pressure_kpa"].tolist() == [300.0] * 5  # oil pressure hold 3 s


def test_clock_gap_starts_new_segment_without_fill(table):
    b = builder(table)
    b.add(Sample(T0, "E", "oil_pressure_kpa", 300.0, OK))
    b.add(Sample(T0 + 60_000_000, "E", "engine_speed_rpm", 900.0, OK))
    df = b.finalize()
    assert len(df) == 2 and df["segment"].tolist() == [0, 1]
    assert math.isnan(df["oil_pressure_kpa"].iloc[1])  # not held across the gap
    assert b.stats.segments["E"] == 2


def test_error_and_na_are_null_never_held(table):
    b = builder(table)
    b.add(Sample(T0, "E", "oil_pressure_kpa", 300.0, OK))
    b.add(Sample(T0 + 1_000_000, "E", "oil_pressure_kpa", None, ERROR))
    b.add(Sample(T0 + 2_000_000, "E", "oil_pressure_kpa", None, NOT_AVAILABLE))
    b.add(Sample(T0 + 3_000_000, "E", "oil_pressure_kpa", 296.0, OK))
    b.add(Sample(T0 + 3_100_000, "E", "oil_pressure_kpa", None, ERROR))
    df = b.finalize()
    v = df["oil_pressure_kpa"].tolist()
    assert v[0] == 300.0 and math.isnan(v[1]) and math.isnan(v[2]) and v[3] == 296.0
    assert df["oil_pressure_kpa__err"].tolist() == [0, 1, 0, 1]


def test_late_samples_dropped_and_counted(table):
    b = builder(table, lateness_s=2.0)
    b.add(Sample(T0 + 10_000_000, "E", "engine_speed_rpm", 900.0, OK))
    b.add(Sample(T0 + 9_500_000, "E", "engine_speed_rpm", 950.0, OK))  # within lateness: kept
    b.add(Sample(T0, "E", "engine_speed_rpm", 1.0, OK))  # clock stepped back 10 s
    df = b.finalize()
    assert df["engine_speed_rpm"].tolist() == [950.0, 900.0]
    assert b.stats.late_samples_dropped == 1 and b.stats.first_late_t_us == T0


def test_multi_source_addresses_are_separate_engines(table, cfg):
    lines = []
    for i in range(100):
        t = T0 + i * 20_000
        lines.append(candump_line(t, make_can_id(3, 61444, 0x00), encode_pgn(table, 61444, {190: 1500})))
        lines.append(candump_line(t + 1000, make_can_id(3, 61444, 0x01), encode_pgn(table, 61444, {190: 800})))
    r = canonical_from_candump(lines, table=table, config=cfg)
    df = r.canonical
    assert sorted(df["engine_id"].unique()) == ["SA00", "SA01"]
    assert set(df[df.engine_id == "SA00"]["engine_speed_rpm"]) == {1500.0}
    assert set(df[df.engine_id == "SA01"]["engine_speed_rpm"]) == {800.0}
    assert r.report["decode"]["source_addresses"] == {"0x00": [61444], "0x01": [61444]}

    cfg["sources"]["engine_map"] = {0: "port_main", 1: "stbd_main"}
    df2 = canonical_from_candump(lines, table=table, config=cfg).canonical
    assert sorted(df2["engine_id"].unique()) == ["port_main", "stbd_main"]


def test_tsc1_attributed_to_destination_engine(table, cfg):
    # transmission (SA 0x03) commands engine 0x00
    lines = [candump_line(T0, make_can_id(3, 0, 0x03, 0x00), encode_pgn(table, 0, {898: 1800, 518: 50}))]
    df = canonical_from_candump(lines, table=table, config=cfg).canonical
    assert df["engine_id"].tolist() == ["SA00"]
    assert df["tsc1_requested_speed_rpm"].iloc[0] == 1800.0


def test_na_and_error_encodings_in_frames_are_ignored(table, cfg):
    lines = [
        candump_line(T0, make_can_id(6, 65263, 0), encode_pgn(table, 65263, {100: 300})),
        candump_line(T0 + 1_000_000, make_can_id(6, 65263, 0), encode_pgn(table, 65263, {100: ERR})),
        candump_line(T0 + 2_000_000, make_can_id(6, 65263, 0), encode_pgn(table, 65263, {100: NA})),
    ]
    r = canonical_from_candump(lines, table=table, config=cfg)
    v = r.canonical["oil_pressure_kpa"].to_numpy()
    assert v[0] == 300.0 and np.isnan(v[1]) and np.isnan(v[2])
    # Values never decoded into readings (e.g. 0xFE * 4 = 1016 kPa or 0xFF * 4 = 1020 kPa)
    assert not (r.canonical.filter(like="kpa").to_numpy() > 1000).any()
    sc = r.report["decode"]["status_counts"]["SA00/oil_pressure_kpa"]
    assert sc == {"ok": 1, "error": 1, "not_available": 1}
    # fields never set in the frame are "not available" everywhere -> all-NULL columns
    assert r.canonical["coolant_level_pct"].isna().all()


def test_bin_edges_use_integer_microseconds(table):
    b = builder(table)
    b.add(Sample(T0 + 999_999, "E", "engine_speed_rpm", 1.0, OK))
    b.add(Sample(T0 + 1_000_000, "E", "engine_speed_rpm", 2.0, OK))
    df = b.finalize()
    assert df["engine_speed_rpm"].tolist() == [1.0, 2.0]


def test_dtc_episodes_split_by_gap(table, cfg):
    from j1939pf.dm import PGN_DM1, Dtc
    from j1939pf.encode import dm_payload
    lines = []
    for s in list(range(0, 5)) + list(range(30, 33)):
        lines.append(candump_line(T0 + s * 1_000_000, make_can_id(6, PGN_DM1, 0),
                                  dm_payload([Dtc(100, 18, 4, 0)], awl=1)))
    r = canonical_from_candump(lines, table=table, config=cfg)
    ep = r.dtc_episodes
    assert len(ep) == 2 and ep["messages"].tolist() == [5, 3] and set(ep["fmi"]) == {18}
    assert (r.canonical["dm1_active_count"].dropna() == 1).all()


@pytest.mark.parametrize("interval", [0.5, 2.0])
def test_interval_configurable(table, interval):
    b = builder(table, resample_interval_s=interval, gap_s=10)
    for i in range(400):
        b.add(Sample(T0 + i * 20_000, "E", "engine_speed_rpm", 1000.0, OK))
    assert len(b.finalize()) == int(8 / interval)
