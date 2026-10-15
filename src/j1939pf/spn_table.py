"""Data-driven SPN decoding from config/spn_table.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import DEFAULT_SPN_TABLE
from .units import normalize_unit

# Sample status values. Anything other than OK decodes to NULL.
OK = "ok"
NOT_AVAILABLE = "not_available"
ERROR = "error"
RESERVED = "reserved"
OUT_OF_RANGE = "out_of_range"
SHORT_PAYLOAD = "short_payload"
MALFORMED = "malformed"
# Statuses that indicate a bad reading (as opposed to "signal not supported").
BAD_STATUSES = frozenset({ERROR, RESERVED, OUT_OF_RANGE, SHORT_PAYLOAD, MALFORMED})

_REQUIRED = ("pgn", "spn", "name", "canonical", "start_byte", "start_bit", "length_bits",
             "scale", "offset", "unit", "valid_min", "valid_max", "verified")
_AGGREGATES = ("mean", "last", "max", "min")


@dataclass(frozen=True)
class SpnDef:
    pgn: int
    spn: int
    name: str
    canonical: str
    start_byte: int
    start_bit: int
    length_bits: int
    scale: float
    offset: float
    unit: str
    valid_min: float
    valid_max: float
    verified: bool
    na_handling: str = "standard"
    na_raw: tuple[int, ...] = ()
    error_raw: tuple[int, ...] = ()
    aggregate: str = "mean"
    rate_ms: float | None = None
    max_hold_s: float | None = None
    attribute_to: str = "source"
    pgn_acronym: str = ""
    oem_specific: bool = False
    notes: str = ""
    bit_offset: int = field(init=False)
    mask: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "bit_offset", (self.start_byte - 1) * 8 + (self.start_bit - 1))
        object.__setattr__(self, "mask", (1 << self.length_bits) - 1)

    def classify_raw(self, raw: int) -> str:
        n = self.length_bits
        if self.na_handling == "explicit":
            if raw in self.na_raw:
                return NOT_AVAILABLE
            if raw in self.error_raw:
                return ERROR
            return OK
        if n == 1:
            return OK
        if n == 2:
            return {3: NOT_AVAILABLE, 2: ERROR}.get(raw, OK)
        if n == 4:
            return {15: NOT_AVAILABLE, 14: ERROR}.get(raw, OK)
        if n % 8 == 0:
            top = raw >> (n - 8)
            if top == 0xFF:
                return NOT_AVAILABLE
            if top == 0xFE:
                return ERROR
            if top >= 0xFB:
                return RESERVED
            return OK
        if raw == self.mask:
            return NOT_AVAILABLE
        if raw == self.mask - 1:
            return ERROR
        return OK

    def decode_int(self, payload_int: int, payload_bits: int) -> tuple[float | None, str]:
        if self.bit_offset + self.length_bits > payload_bits:
            return None, SHORT_PAYLOAD
        raw = (payload_int >> self.bit_offset) & self.mask
        status = self.classify_raw(raw)
        if status != OK:
            return None, status
        value = raw * self.scale + self.offset
        if not (self.valid_min - 1e-9 <= value <= self.valid_max + 1e-9):
            return None, OUT_OF_RANGE
        return value, OK

    def decode(self, data: bytes) -> tuple[float | None, str]:
        return self.decode_int(int.from_bytes(data, "little"), len(data) * 8)


class SpnTable:
    def __init__(self, defs: list[SpnDef], source: str = "<memory>"):
        self.defs = defs
        self.source = source
        self.by_pgn: dict[int, list[SpnDef]] = {}
        self.by_spn: dict[int, SpnDef] = {}
        self.by_canonical: dict[str, SpnDef] = {}
        for d in defs:
            if d.spn in self.by_spn:
                raise ValueError(f"{source}: SPN {d.spn} defined twice")
            if d.canonical in self.by_canonical:
                raise ValueError(f"{source}: canonical name {d.canonical!r} used twice")
            self.by_pgn.setdefault(d.pgn, []).append(d)
            self.by_spn[d.spn] = d
            self.by_canonical[d.canonical] = d
        for pgn, group in self.by_pgn.items():
            _check_overlaps(pgn, group, source)

    @property
    def unverified(self) -> list[SpnDef]:
        return [d for d in self.defs if not d.verified]


def _check_overlaps(pgn: int, group: list[SpnDef], source: str) -> None:
    used: dict[int, int] = {}
    for d in group:
        for bit in range(d.bit_offset, d.bit_offset + d.length_bits):
            if bit in used:
                raise ValueError(f"{source}: PGN {pgn}: SPN {d.spn} overlaps SPN {used[bit]} at bit {bit}")
            used[bit] = d.spn


def _parse_entry(e: dict, source: str, idx: int) -> SpnDef:
    missing = [k for k in _REQUIRED if k not in e]
    if missing:
        raise ValueError(f"{source}: entry #{idx} (SPN {e.get('spn')}) missing {missing}")
    if not isinstance(e["verified"], bool):
        raise ValueError(f"{source}: SPN {e['spn']}: `verified` must be true/false")
    if not (1 <= int(e["start_byte"]) and 1 <= int(e["start_bit"]) <= 8 and int(e["length_bits"]) >= 1):
        raise ValueError(f"{source}: SPN {e['spn']}: bad start_byte/start_bit/length_bits")
    agg = e.get("aggregate", "mean")
    if agg not in _AGGREGATES:
        raise ValueError(f"{source}: SPN {e['spn']}: aggregate must be one of {_AGGREGATES}")
    nah = e.get("na_handling", "standard")
    if nah not in ("standard", "explicit"):
        raise ValueError(f"{source}: SPN {e['spn']}: na_handling must be standard|explicit")
    if e.get("attribute_to", "source") not in ("source", "destination"):
        raise ValueError(f"{source}: SPN {e['spn']}: attribute_to must be source|destination")
    normalize_unit(e["unit"])
    return SpnDef(
        pgn=int(e["pgn"]), spn=int(e["spn"]), name=str(e["name"]), canonical=str(e["canonical"]),
        start_byte=int(e["start_byte"]), start_bit=int(e["start_bit"]), length_bits=int(e["length_bits"]),
        scale=float(e["scale"]), offset=float(e["offset"]), unit=normalize_unit(e["unit"]),
        valid_min=float(e["valid_min"]), valid_max=float(e["valid_max"]), verified=e["verified"],
        na_handling=nah, na_raw=tuple(int(x) for x in e.get("na_raw", ())),
        error_raw=tuple(int(x) for x in e.get("error_raw", ())), aggregate=agg,
        rate_ms=e.get("rate_ms"), max_hold_s=e.get("max_hold_s"),
        attribute_to=e.get("attribute_to", "source"), pgn_acronym=str(e.get("pgn_acronym", "")),
        oem_specific=bool(e.get("oem_specific", False)), notes=str(e.get("notes", "")),
    )


def load_spn_table(path: str | Path | None = None) -> SpnTable:
    path = Path(path) if path else DEFAULT_SPN_TABLE
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    entries = doc.get("signals") or []
    return SpnTable([_parse_entry(e, str(path), i) for i, e in enumerate(entries)], str(path))


@dataclass(frozen=True)
class SignalInfo:
    """A canonical column: either an SPN or a value derived by fixed-structure decoding (DM1)."""
    name: str
    unit: str
    aggregate: str
    rate_ms: float | None
    max_hold_s: float | None
    valid_min: float
    valid_max: float
    spn: int | None = None


DM1_SIGNALS = (
    # Lamp states: 1 on, 0 off, NULL when error / not available. DM1 is sent once per second.
    SignalInfo("dm1_active_count", "count", "max", 1000, None, 0, 1e6),
    SignalInfo("dm1_mil", "state", "max", 1000, None, 0, 1),
    SignalInfo("dm1_rsl", "state", "max", 1000, None, 0, 1),
    SignalInfo("dm1_awl", "state", "max", 1000, None, 0, 1),
    SignalInfo("dm1_pl", "state", "max", 1000, None, 0, 1),
)


def build_registry(table: SpnTable) -> dict[str, SignalInfo]:
    reg = {
        d.canonical: SignalInfo(d.canonical, d.unit, d.aggregate, d.rate_ms, d.max_hold_s,
                                d.valid_min, d.valid_max, d.spn)
        for d in table.defs
    }
    for s in DM1_SIGNALS:
        reg[s.name] = s
    return reg
