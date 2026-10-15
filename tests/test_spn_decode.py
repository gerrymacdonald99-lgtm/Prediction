import math

import pytest
import yaml

from j1939pf.encode import ERR, NA, encode_pgn
from j1939pf.spn_table import (ERROR, NOT_AVAILABLE, OK, OUT_OF_RANGE, RESERVED, SHORT_PAYLOAD,
                               load_spn_table)


def test_every_entry_unverified_and_complete(table):
    assert table.defs, "table must not be empty"
    assert all(d.verified is False for d in table.defs)
    for d in table.defs:
        assert d.unit and d.valid_max > d.valid_min


def test_oil_pressure_4kpa_per_bit(table):
    d = table.by_spn[100]
    assert d.decode(bytes([0xFF, 0xFF, 0xFF, 75, 0xFF, 0xFF, 0xFF, 0xFF])) == (300.0, OK)


@pytest.mark.parametrize("raw,status", [(0xFF, NOT_AVAILABLE), (0xFE, ERROR), (0xFB, RESERVED),
                                        (0xFD, RESERVED), (0xFA, OK)])
def test_one_byte_encodings(table, raw, status):
    v, s = table.by_spn[110].decode(bytes([raw] + [0xFF] * 7))
    assert s == status
    assert (v is None) == (status != OK)


@pytest.mark.parametrize("hi,status", [(0xFF, NOT_AVAILABLE), (0xFE, ERROR), (0xFC, RESERVED), (0x40, OK)])
def test_two_byte_encodings(table, hi, status):
    data = bytes([0xFF, 0xFF, 0xFF, 0x00, hi, 0xFF, 0xFF, 0xFF])  # SPN 190 bytes 4-5
    v, s = table.by_spn[190].decode(data)
    assert s == status
    if status == OK:
        assert v == (hi << 8) * 0.125


def test_four_byte_and_two_bit(table):
    hrs = table.by_spn[247]
    assert hrs.decode(b"\xff" * 8)[1] == NOT_AVAILABLE
    assert hrs.decode(bytes([0, 0, 0, 0xFE, 0, 0, 0, 0]))[1] == ERROR
    assert hrs.decode(bytes([0x10, 0x27, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF])) == (500.0, OK)
    wif = table.by_spn[97]
    assert wif.decode(bytes([0b11111101] + [0xFF] * 7)) == (1.0, OK)
    assert wif.decode(bytes([0b11111110] + [0xFF] * 7))[1] == ERROR
    assert wif.decode(bytes([0xFF] * 8))[1] == NOT_AVAILABLE


def test_offsets_and_short_payload(table):
    payload = encode_pgn(table, 65262, {110: -40, 175: 100.0})
    assert table.by_spn[110].decode(payload) == (-40.0, OK)
    assert table.by_spn[175].decode(payload) == (100.0, OK)
    assert table.by_spn[175].decode(payload[:3]) == (None, SHORT_PAYLOAD)


def test_out_of_range_is_null(table):
    # Pedal 0.4 %/bit, valid 0..100 %: raw 0xFA = 100 % OK; nothing above is OK either
    d = table.by_spn[91]
    assert d.decode(bytes([0xFF, 0xFA] + [0xFF] * 6)) == (100.0, OK)
    d92 = table.by_spn[92]
    assert d92.decode(bytes([0xFF, 0xFF, 0xFA] + [0xFF] * 5)) == (250.0, OK)
    # Fan drive state valid 0..13; raw 13 OK
    fd = table.by_spn[977]
    assert fd.decode(bytes([0xFF, 0xFD] + [0xFF] * 6))[1] == OK


def test_out_of_range_detected_with_custom_table(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump({"signals": [dict(
        pgn=65262, spn=110, name="x", canonical="x", start_byte=1, start_bit=1, length_bits=8, scale=1.0,
        offset=-40.0, unit="degC", valid_min=-40, valid_max=150, verified=False)]}))
    t = load_spn_table(p)
    assert t.by_spn[110].decode(bytes([200] + [0xFF] * 7)) == (None, OUT_OF_RANGE)


def test_encode_na_and_err_roundtrip(table):
    payload = encode_pgn(table, 65263, {100: NA, 94: ERR})
    assert table.by_spn[100].decode(payload)[1] == NOT_AVAILABLE
    assert table.by_spn[94].decode(payload)[1] == ERROR
    v, s = table.by_spn[101].decode(encode_pgn(table, 65263, {101: 1.5}))
    assert s == OK and math.isclose(v, 1.5)


def _bad_table(tmp_path, **override):
    e = dict(pgn=65262, spn=110, name="x", canonical="x", start_byte=1, start_bit=1, length_bits=8,
             scale=1.0, offset=-40.0, unit="degC", valid_min=-40, valid_max=210, verified=False)
    e.update(override)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"signals": [e]}))
    return p


def test_table_validation(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        p = tmp_path / "m.yaml"
        p.write_text(yaml.safe_dump({"signals": [{"pgn": 1, "spn": 2}]}))
        load_spn_table(p)
    with pytest.raises(ValueError, match="verified"):
        load_spn_table(_bad_table(tmp_path, verified="no"))
    with pytest.raises(ValueError, match="unknown unit"):
        load_spn_table(_bad_table(tmp_path, unit="furlongs"))
    p = tmp_path / "o.yaml"
    base = dict(pgn=65262, name="x", start_bit=1, length_bits=16, scale=1.0, offset=0.0, unit="degC",
                valid_min=0, valid_max=10, verified=False)
    p.write_text(yaml.safe_dump({"signals": [dict(base, spn=1, canonical="a", start_byte=1),
                                             dict(base, spn=2, canonical="b", start_byte=2)]}))
    with pytest.raises(ValueError, match="overlaps"):
        load_spn_table(p)
