"""29-bit J1939 identifier handling (J1939-21)."""

from __future__ import annotations

from dataclasses import dataclass

GLOBAL_ADDRESS = 0xFF
PDU2_THRESHOLD = 240  # PF >= 240 -> PDU2 (broadcast, PS is group extension)


@dataclass(frozen=True, slots=True)
class J1939Id:
    priority: int
    pgn: int
    sa: int
    da: int  # GLOBAL_ADDRESS for PDU2
    pf: int
    is_pdu1: bool


def parse_can_id(can_id: int) -> J1939Id:
    """Split a 29-bit identifier into priority, PGN, source and destination address.

    PDU1 (PF < 240): PS is the destination address and is NOT part of the PGN.
    PDU2 (PF >= 240): PS is the group extension and IS part of the PGN.
    """
    if not 0 <= can_id < (1 << 29):
        raise ValueError(f"not a 29-bit identifier: {can_id:#x}")
    priority = (can_id >> 26) & 0x7
    edp_dp = (can_id >> 24) & 0x3
    pf = (can_id >> 16) & 0xFF
    ps = (can_id >> 8) & 0xFF
    sa = can_id & 0xFF
    if pf < PDU2_THRESHOLD:
        return J1939Id(priority, (edp_dp << 16) | (pf << 8), sa, ps, pf, True)
    return J1939Id(priority, (edp_dp << 16) | (pf << 8) | ps, sa, GLOBAL_ADDRESS, pf, False)


def make_can_id(priority: int, pgn: int, sa: int, da: int = GLOBAL_ADDRESS) -> int:
    """Inverse of parse_can_id. For PDU1 PGNs `da` fills the PS byte."""
    pf = (pgn >> 8) & 0xFF
    if pf < PDU2_THRESHOLD:
        if pgn & 0xFF:
            raise ValueError(f"PDU1 PGN {pgn} must have a zero low byte")
        pgn = pgn | (da & 0xFF)
    return ((priority & 0x7) << 26) | ((pgn & 0x3FFFF) << 8) | (sa & 0xFF)
