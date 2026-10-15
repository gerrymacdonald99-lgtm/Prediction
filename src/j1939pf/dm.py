"""DM1 / DM2 diagnostic message decoding (J1939-73, SPN conversion method 4)."""

from __future__ import annotations

from dataclasses import dataclass

PGN_DM1 = 65226
PGN_DM2 = 65227

_LAMP_SHIFT = {"mil": 6, "rsl": 4, "awl": 2, "pl": 0}


@dataclass(frozen=True, slots=True)
class Dtc:
    spn: int
    fmi: int
    oc: int
    cm: int


@dataclass(frozen=True, slots=True)
class DmMessage:
    lamps: dict[str, int | None]  # 1 on, 0 off, None error / not available
    dtcs: tuple[Dtc, ...]
    old_conversion_method: bool   # CM=1 seen: SPN may be mis-decoded (pre-version-4 layout)
    malformed: bool


def decode_dm(payload: bytes) -> DmMessage:
    if len(payload) < 2:
        return DmMessage({k: None for k in _LAMP_SHIFT}, (), False, True)
    lamps: dict[str, int | None] = {}
    for name, shift in _LAMP_SHIFT.items():
        raw = (payload[0] >> shift) & 0x3
        lamps[name] = raw if raw in (0, 1) else None
    dtcs = []
    old_cm = False
    body = payload[2:]
    malformed = len(body) % 4 not in (0, 2)  # single-frame DM1 pads 2 bytes with 0xFF
    for i in range(0, len(body) - 3, 4):
        b0, b1, b2, b3 = body[i:i + 4]
        spn = b0 | (b1 << 8) | ((b2 >> 5) << 16)
        fmi = b2 & 0x1F
        cm, oc = b3 >> 7, b3 & 0x7F
        if (spn == 0 and fmi == 0) or (b0, b1, b2, b3) == (0xFF, 0xFF, 0xFF, 0xFF):
            continue  # "no active DTC" placeholder or padding
        if spn == 0x7FFFF:
            continue
        old_cm |= cm == 1
        dtcs.append(Dtc(spn, fmi, oc, cm))
    return DmMessage(lamps, tuple(dtcs), old_cm, malformed)
