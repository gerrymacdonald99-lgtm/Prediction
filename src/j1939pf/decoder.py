"""Frames -> decoded samples, per engine. Handles duplicates, TP, DM1/DM2, multi-SA buses."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .candump import RawFrame
from .config import engine_id_for_sa
from .dm import PGN_DM1, PGN_DM2, decode_dm
from .ids import parse_can_id
from .spn_table import NOT_AVAILABLE, OK, SpnTable
from .transport import PGN_TP_CM, PGN_TP_DT, TransportReassembler


@dataclass(slots=True)
class Sample:
    t_us: int
    engine: str
    name: str
    value: float | None
    status: str


@dataclass
class DtcEpisode:
    engine: str
    source_address: int
    kind: str  # DM1 (active) | DM2 (previously active)
    spn: int
    fmi: int
    first_t_us: int
    last_t_us: int
    max_oc: int
    messages: int


@dataclass
class DecodeStats:
    standard_id_frames: int = 0
    duplicate_frames: int = 0
    ignored_source_frames: int = 0
    undecoded_pgn_frames: Counter = field(default_factory=Counter)
    dm_old_conversion_method: int = 0
    dm_malformed: int = 0
    # source address -> PGNs it sent; lets the user decide how to map SAs to engines
    source_addresses: dict[int, set] = field(default_factory=lambda: defaultdict(set))
    # (engine, signal) -> Counter(status)
    status_counts: dict[tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))

    def as_dict(self) -> dict:
        return {
            "standard_id_frames": self.standard_id_frames,
            "duplicate_frames": self.duplicate_frames,
            "ignored_source_frames": self.ignored_source_frames,
            "undecoded_pgn_frames": dict(self.undecoded_pgn_frames),
            "dm_old_conversion_method": self.dm_old_conversion_method,
            "dm_malformed": self.dm_malformed,
            "source_addresses": {f"0x{k:02X}": sorted(v) for k, v in sorted(self.source_addresses.items())},
            "status_counts": {f"{e}/{n}": dict(c) for (e, n), c in sorted(self.status_counts.items())},
        }


class J1939Decoder:
    def __init__(self, table: SpnTable, *, engine_map: dict | None = None,
                 ignore_source_addresses: Iterable[int] = (), duplicate_window_s: float = 0.0005,
                 transport: dict | None = None, dtc_episode_gap_s: float = 10.0):
        self.table = table
        self.engine_map = engine_map or {}
        self.ignore = {int(x) for x in ignore_source_addresses}
        self.dup_us = int(duplicate_window_s * 1e6)
        self.tp = TransportReassembler(**(transport or {}))
        self.stats = DecodeStats()
        self._last: dict[tuple[str, int], tuple[int, bytes]] = {}
        self._episode_gap_us = int(dtc_episode_gap_s * 1e6)
        self._open_eps: dict[tuple, DtcEpisode] = {}
        self.dtc_episodes: list[DtcEpisode] = []

    def engine_id(self, sa: int) -> str:
        return engine_id_for_sa(sa, self.engine_map)

    def iter_samples(self, frames: Iterable[RawFrame]) -> Iterator[Sample]:
        for f in frames:
            yield from self.feed(f)
        self.finish()

    def feed(self, f: RawFrame) -> list[Sample]:
        if not f.extended:
            self.stats.standard_id_frames += 1
            return []
        key = (f.iface, f.can_id)
        prev = self._last.get(key)
        if prev is not None and prev[1] == f.data and 0 <= f.t_us - prev[0] <= self.dup_us:
            self.stats.duplicate_frames += 1
            return []
        self._last[key] = (f.t_us, f.data)
        jid = parse_can_id(f.can_id)
        if jid.pgn in (PGN_TP_CM, PGN_TP_DT):
            msg = self.tp.feed(f.t_us, jid.pgn, jid.sa, jid.da, f.data)
            if msg is None:
                return []
            return self._payload(msg.t_us, msg.pgn, msg.sa, msg.da, msg.data)
        self.tp.expire(f.t_us)
        return self._payload(f.t_us, jid.pgn, jid.sa, jid.da, f.data)

    def finish(self) -> None:
        self.tp.finish()
        self.dtc_episodes.extend(self._open_eps.values())
        self._open_eps.clear()

    # ----------------------------------------------------------------------
    def _payload(self, t_us: int, pgn: int, sa: int, da: int, data: bytes) -> list[Sample]:
        self.stats.source_addresses[sa].add(pgn)
        if pgn in (PGN_DM1, PGN_DM2):
            if sa in self.ignore:
                self.stats.ignored_source_frames += 1
                return []
            return self._dm(t_us, pgn, sa, data)
        defs = self.table.by_pgn.get(pgn)
        if not defs:
            self.stats.undecoded_pgn_frames[pgn] += 1
            return []
        payload_int = int.from_bytes(data, "little")
        nbits = len(data) * 8
        out = []
        for d in defs:
            owner = da if d.attribute_to == "destination" else sa
            if owner in self.ignore or owner == 0xFF:
                self.stats.ignored_source_frames += 1
                continue
            engine = self.engine_id(owner)
            value, status = d.decode_int(payload_int, nbits)
            self.stats.status_counts[(engine, d.canonical)][status] += 1
            out.append(Sample(t_us, engine, d.canonical, value, status))
        return out

    def _dm(self, t_us: int, pgn: int, sa: int, data: bytes) -> list[Sample]:
        msg = decode_dm(data)
        engine = self.engine_id(sa)
        if msg.malformed:
            self.stats.dm_malformed += 1
        if msg.old_conversion_method:
            self.stats.dm_old_conversion_method += 1
        kind = "DM1" if pgn == PGN_DM1 else "DM2"
        for dtc in msg.dtcs:
            k = (engine, sa, kind, dtc.spn, dtc.fmi)
            ep = self._open_eps.get(k)
            if ep is not None and t_us - ep.last_t_us > self._episode_gap_us:
                self.dtc_episodes.append(ep)
                ep = None
            if ep is None:
                self._open_eps[k] = DtcEpisode(engine, sa, kind, dtc.spn, dtc.fmi, t_us, t_us, dtc.oc, 1)
            else:
                ep.last_t_us = t_us
                ep.max_oc = max(ep.max_oc, dtc.oc)
                ep.messages += 1
        if kind == "DM2":
            return []  # previously-active codes describe history, not the current state
        out = [Sample(t_us, engine, "dm1_active_count", float(len(msg.dtcs)), OK)]
        for lamp, state in msg.lamps.items():
            name = f"dm1_{lamp}"
            status = OK if state is not None else NOT_AVAILABLE
            out.append(Sample(t_us, engine, name, None if state is None else float(state), status))
        return out
