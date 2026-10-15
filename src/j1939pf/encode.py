"""Inverse of the decoder: physical values -> J1939 frames / candump lines.

Used to build test fixtures (and, from phase 2, by the synthetic generator).
"""

from __future__ import annotations

from collections.abc import Iterator

from .dm import Dtc
from .ids import GLOBAL_ADDRESS, make_can_id
from .spn_table import SpnDef, SpnTable
from .transport import CM_BAM, CM_CTS, CM_EOM_ACK, CM_RTS, PGN_TP_CM, PGN_TP_DT

NA = object()      # sentinel: encode the "not available" pattern
ERR = object()     # sentinel: encode the "error indicator" pattern


def raw_for(d: SpnDef, value) -> int:
    n = d.length_bits
    if value is NA or value is ERR:
        if n == 2:
            return 3 if value is NA else 2
        if n == 4:
            return 15 if value is NA else 14
        if n % 8 == 0:
            top = 0xFF if value is NA else 0xFE
            return (top << (n - 8)) | ((1 << (n - 8)) - 1 if value is NA else 0)
        return d.mask if value is NA else d.mask - 1
    raw = round((float(value) - d.offset) / d.scale)
    if not 0 <= raw <= d.mask:
        raise ValueError(f"SPN {d.spn}: value {value} does not fit {n} bits")
    return raw


def encode_pgn(table: SpnTable, pgn: int, values: dict[int, object], length: int = 8) -> bytes:
    """values: SPN -> physical value | NA | ERR. Unspecified bits are 1 (not available)."""
    payload = (1 << (length * 8)) - 1
    defs = {d.spn: d for d in table.by_pgn.get(pgn, [])}
    for spn, v in values.items():
        d = defs[spn]
        payload &= ~(d.mask << d.bit_offset)
        payload |= raw_for(d, v) << d.bit_offset
    return payload.to_bytes(length, "little")


def physical(d: SpnDef, value) -> float:
    """The value a decoder will report for `value` after quantization."""
    return raw_for(d, value) * d.scale + d.offset


def dm_payload(dtcs: list[Dtc], mil=0, rsl=0, awl=0, pl=0) -> bytes:
    lamps = (mil << 6) | (rsl << 4) | (awl << 2) | pl
    body = bytearray([lamps, 0xFF])
    if not dtcs:
        body += bytes([0, 0, 0, 0, 0xFF, 0xFF])
    for t in dtcs:
        body += bytes([t.spn & 0xFF, (t.spn >> 8) & 0xFF, ((t.spn >> 16) & 0x7) << 5 | (t.fmi & 0x1F),
                       ((t.cm & 1) << 7) | (t.oc & 0x7F)])
    if len(body) == 6:
        body += b"\xff\xff"
    return bytes(body)


def candump_line(t_us: int, can_id: int, data: bytes, iface: str = "can0", screen: bool = False) -> str:
    ts = f"({t_us // 1_000_000}.{t_us % 1_000_000:06d})"
    if screen:
        return f"{ts}  {iface}  {can_id:08X}   [{len(data)}]  {' '.join(f'{b:02X}' for b in data)}\n"
    return f"{ts} {iface} {can_id:08X}#{data.hex().upper()}\n"


def _dt_frames(data: bytes) -> list[bytes]:
    n = (len(data) + 6) // 7
    out = []
    for i in range(n):
        chunk = data[i * 7:(i + 1) * 7]
        out.append(bytes([i + 1]) + chunk + b"\xff" * (7 - len(chunk)))
    return out


def _cm(ctrl: int, size: int, n: int, fifth: int, pgn: int) -> bytes:
    return bytes([ctrl, size & 0xFF, size >> 8, n, fifth, pgn & 0xFF, (pgn >> 8) & 0xFF, pgn >> 16])


def bam_frames(t_us: int, pgn: int, sa: int, data: bytes, gap_us: int = 50_000,
               priority: int = 7) -> Iterator[tuple[int, int, bytes]]:
    """Yield (t_us, can_id, payload) for a BAM transfer."""
    n = (len(data) + 6) // 7
    yield t_us, make_can_id(priority, PGN_TP_CM, sa, GLOBAL_ADDRESS), _cm(CM_BAM, len(data), n, 0xFF, pgn)
    for i, f in enumerate(_dt_frames(data), start=1):
        yield t_us + i * gap_us, make_can_id(priority, PGN_TP_DT, sa, GLOBAL_ADDRESS), f


def rts_cts_frames(t_us: int, pgn: int, sa: int, da: int, data: bytes, per_cts: int = 2,
                   drop: set[int] = frozenset(), resend_dropped: bool = True,
                   gap_us: int = 10_000) -> Iterator[tuple[int, int, bytes]]:
    """Yield frames for an RTS/CTS transfer as seen on the bus.

    `drop` packet numbers are lost on their first transmission; if
    `resend_dropped`, the receiver sends a CTS asking for them again.
    """
    dts = _dt_frames(data)
    n = len(dts)
    t = t_us
    cm_s = make_can_id(7, PGN_TP_CM, sa, da)
    cm_r = make_can_id(7, PGN_TP_CM, da, sa)
    dt_s = make_can_id(7, PGN_TP_DT, sa, da)
    yield t, cm_s, _cm(CM_RTS, len(data), n, per_cts, pgn)
    seq = 1
    while seq <= n:
        t += gap_us
        count = min(per_cts, n - seq + 1)
        yield t, cm_r, bytes([CM_CTS, count, seq, 0xFF, 0xFF, pgn & 0xFF, (pgn >> 8) & 0xFF, pgn >> 16])
        lost = []
        for k in range(seq, seq + count):
            t += gap_us
            if k in drop:
                lost.append(k)
                continue
            yield t, dt_s, dts[k - 1]
        if lost and resend_dropped:
            for k in lost:
                t += gap_us
                yield t, cm_r, bytes([CM_CTS, 1, k, 0xFF, 0xFF, pgn & 0xFF, (pgn >> 8) & 0xFF, pgn >> 16])
                t += gap_us
                yield t, dt_s, dts[k - 1]
        elif lost:
            return  # receiver gave up; sender goes silent -> observer must time out
        seq += count
    t += gap_us
    yield t, cm_r, _cm(CM_EOM_ACK, len(data), n, 0xFF, pgn)
