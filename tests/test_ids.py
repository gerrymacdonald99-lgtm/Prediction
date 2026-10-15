import pytest

from j1939pf.ids import GLOBAL_ADDRESS, make_can_id, parse_can_id


def test_pdu2_broadcast():
    j = parse_can_id(0x18FEEE00)  # ET1 from SA 0x00, priority 6
    assert (j.priority, j.pgn, j.sa, j.da, j.is_pdu1) == (6, 65262, 0x00, GLOBAL_ADDRESS, False)


def test_pdu1_destination_not_part_of_pgn():
    j = parse_can_id(0x0C000003 | (0x00 << 8))  # TSC1 to DA 0x00 from SA 0x03
    assert (j.priority, j.pgn, j.sa, j.da, j.is_pdu1) == (3, 0, 0x03, 0x00, True)
    j = parse_can_id(0x1CEC00F9)  # TP.CM to DA 0x00 from SA 0xF9
    assert (j.pgn, j.da, j.sa) == (60416, 0x00, 0xF9)


def test_eec1_and_dp_edp_bits():
    assert parse_can_id(0x0CF00400).pgn == 61444
    assert parse_can_id(0x0DF00400).pgn == 61444 + 0x10000  # data page bit


@pytest.mark.parametrize("prio,pgn,sa,da", [(3, 61444, 0, 255), (6, 65263, 1, 255), (3, 0, 3, 0),
                                            (7, 60416, 0x3D, 0xF9), (7, 60160, 0, 255)])
def test_roundtrip(prio, pgn, sa, da):
    j = parse_can_id(make_can_id(prio, pgn, sa, da))
    assert (j.priority, j.pgn, j.sa) == (prio, pgn, sa)
    assert j.da == (da if j.is_pdu1 else GLOBAL_ADDRESS)


def test_rejects_non_29_bit():
    with pytest.raises(ValueError):
        parse_can_id(1 << 29)
