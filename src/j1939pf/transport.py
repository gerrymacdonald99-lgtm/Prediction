"""J1939-21 transport protocol (TP.CM / TP.DT) reassembly for a passive bus observer.

Supports BAM (broadcast) and RTS/CTS (connection mode). A logger only *observes*
the bus, so it never sends CTS itself; it follows the CTS frames the real
receiver sends, which lets it accept retransmitted packets.

Rules (conservative: never emit a payload that might be corrupt):
  * a message is emitted only when every packet 1..N has been seen;
  * BAM: a sequence gap (lost packet) aborts the session immediately; a repeated
    packet with identical data is ignored, with different data aborts;
  * RTS/CTS: packets may be re-sent after a CTS; the latest copy wins;
  * timeouts (T1 between packets, T2 after CTS, T3 after last packet) are
    evaluated against frame timestamps, so offline logs behave like live ones;
  * a new BAM/RTS on a busy connection replaces the old session (old one counted
    as aborted), as J1939-21 specifies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ids import GLOBAL_ADDRESS

PGN_TP_CM = 60416
PGN_TP_DT = 60160

CM_RTS = 16
CM_CTS = 17
CM_EOM_ACK = 19
CM_BAM = 32
CM_ABORT = 255

MAX_TP_SIZE = 1785  # 255 packets * 7 bytes


@dataclass(slots=True)
class TpMessage:
    t_us: int
    pgn: int
    sa: int
    da: int
    data: bytes
    mode: str  # "BAM" | "RTS/CTS"


@dataclass
class TpStats:
    completed: int = 0
    aborted_by_sender: int = 0
    replaced: int = 0
    timed_out: int = 0
    missing_packet: int = 0
    duplicate_packets: int = 0
    conflicting_packets: int = 0
    orphan_packets: int = 0
    bad_announce: int = 0
    incomplete_at_end: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class _Session:
    mode: str
    pgn: int
    sa: int
    da: int
    size: int
    n_packets: int
    last_t_us: int
    timeout_us: int
    packets: dict[int, bytes] = field(default_factory=dict)
    next_seq: int = 1


class TransportReassembler:
    def __init__(self, t1_s: float = 0.75, t2_s: float = 1.25, t3_s: float = 1.25, t4_s: float = 1.05):
        self.t1 = int(t1_s * 1e6)
        self.t2 = int(t2_s * 1e6)
        self.t3 = int(t3_s * 1e6)
        self.t4 = int(t4_s * 1e6)
        self.sessions: dict[tuple[int, int], _Session] = {}
        self.stats = TpStats()

    # -- public -------------------------------------------------------------
    def feed(self, t_us: int, pgn: int, sa: int, da: int, data: bytes) -> TpMessage | None:
        """Feed one TP.CM or TP.DT frame. Returns a completed message or None."""
        self.expire(t_us)
        if pgn == PGN_TP_CM:
            return self._on_cm(t_us, sa, da, data)
        if pgn == PGN_TP_DT:
            return self._on_dt(t_us, sa, da, data)
        return None

    def expire(self, t_us: int) -> None:
        if not self.sessions:
            return
        for key in [k for k, s in self.sessions.items() if t_us - s.last_t_us > s.timeout_us]:
            del self.sessions[key]
            self.stats.timed_out += 1

    def finish(self) -> None:
        self.stats.incomplete_at_end += len(self.sessions)
        self.sessions.clear()

    # -- internals ----------------------------------------------------------
    def _on_cm(self, t_us: int, sa: int, da: int, d: bytes) -> TpMessage | None:
        if len(d) < 8:
            self.stats.bad_announce += 1
            return None
        ctrl = d[0]
        pgn = d[5] | (d[6] << 8) | (d[7] << 16)
        if ctrl in (CM_BAM, CM_RTS):
            size = d[1] | (d[2] << 8)
            n = d[3]
            if ctrl == CM_BAM and da != GLOBAL_ADDRESS:
                self.stats.bad_announce += 1
                return None
            if not (9 <= size <= MAX_TP_SIZE) or n != (size + 6) // 7:
                self.stats.bad_announce += 1
                return None
            key = (sa, da)
            if key in self.sessions:
                self.stats.replaced += 1
            mode = "BAM" if ctrl == CM_BAM else "RTS/CTS"
            timeout = self.t1 if ctrl == CM_BAM else self.t3  # RTS waits for the receiver's CTS
            self.sessions[key] = _Session(mode, pgn, sa, da, size, n, t_us, timeout)
            return None
        if ctrl == CM_CTS:
            # Sent by the receiver (sa) back to the originator (da).
            s = self.sessions.get((da, sa))
            if s and s.mode == "RTS/CTS" and s.pgn == pgn:
                n_allowed, next_seq = d[1], d[2]
                s.last_t_us = t_us
                if n_allowed == 0:
                    s.timeout_us = self.t4  # receiver asked to hold
                else:
                    s.timeout_us = self.t2
                    s.next_seq = next_seq  # retransmission request if next_seq <= already seen
            return None
        if ctrl == CM_EOM_ACK:
            return None  # completion is decided by the data we saw, not by the ack
        if ctrl == CM_ABORT:
            # Abort may come from either side of the connection.
            for key in ((sa, da), (da, sa)):
                s = self.sessions.get(key)
                if s and s.pgn == pgn:
                    del self.sessions[key]
                    self.stats.aborted_by_sender += 1
            return None
        self.stats.bad_announce += 1
        return None

    def _on_dt(self, t_us: int, sa: int, da: int, d: bytes) -> TpMessage | None:
        key = (sa, da)
        s = self.sessions.get(key)
        if s is None or len(d) < 2:
            self.stats.orphan_packets += 1
            return None
        seq, chunk = d[0], bytes(d[1:8])
        if not 1 <= seq <= s.n_packets:
            self.stats.orphan_packets += 1
            return None
        if seq in s.packets:
            if s.packets[seq] == chunk:
                self.stats.duplicate_packets += 1
                s.last_t_us = t_us
                return None
            if s.mode == "BAM":
                self.stats.conflicting_packets += 1
                del self.sessions[key]
                return None
            # RTS/CTS retransmission after CTS: the newest copy wins.
        elif s.mode == "BAM" and seq != s.next_seq:
            self.stats.missing_packet += 1
            del self.sessions[key]
            return None
        s.packets[seq] = chunk
        s.next_seq = max(s.next_seq, seq + 1)
        s.last_t_us = t_us
        s.timeout_us = self.t1
        if len(s.packets) == s.n_packets:
            del self.sessions[key]
            self.stats.completed += 1
            data = b"".join(s.packets[i] for i in range(1, s.n_packets + 1))[: s.size]
            return TpMessage(t_us, s.pgn, s.sa, s.da, data, s.mode)
        return None
