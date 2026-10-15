"""Streaming candump reader. Never loads a whole log into memory.

Accepted line formats (timestamps required):
  (1609459200.123456) can0 18FEEE00#7D7D0000FFFFFFFF          candump -l / log format
  (1609459200.123456)  can0  18FEEE00   [8]  7D 7D 00 00 ...  candump -ta screen format
Files ending in .gz are decompressed on the fly.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

_LOG_RE = re.compile(r"^\s*\((\d+)\.(\d+)\)\s+(\S+)\s+([0-9A-Fa-f]+)(##?)(\S*)\s*$")
_SCREEN_RE = re.compile(
    r"^\s*\((\d+)\.(\d+)\)\s+(\S+)\s+([0-9A-Fa-f]+)\s+\[(\d+)\]\s*((?:[0-9A-Fa-f]{2}\s*)*)(.*)$"
)


@dataclass(slots=True)
class RawFrame:
    t_us: int
    iface: str
    can_id: int
    extended: bool
    data: bytes
    line_no: int


@dataclass
class ReadStats:
    lines: int = 0
    frames: int = 0
    blank_or_comment: int = 0
    malformed: int = 0
    remote_frames: int = 0
    fd_frames: int = 0
    malformed_examples: list[str] = field(default_factory=list)

    def bad(self, line_no: int, line: str) -> None:
        self.malformed += 1
        if len(self.malformed_examples) < 5:
            self.malformed_examples.append(f"line {line_no}: {line.strip()[:80]}")

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _t_us(sec: str, frac: str) -> int:
    # Integer microseconds straight from the text: avoids float rounding at bin edges.
    return int(sec) * 1_000_000 + int((frac + "000000")[:6])


def _open_lines(source) -> Iterator[str]:
    if isinstance(source, (str, Path)):
        p = Path(source)
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8", errors="replace") as fh:
            yield from fh
    else:
        yield from source


def iter_candump_frames(source: str | Path | Iterable[str], stats: ReadStats | None = None) -> Iterator[RawFrame]:
    stats = stats if stats is not None else ReadStats()
    for line_no, line in enumerate(_open_lines(source), start=1):
        stats.lines += 1
        s = line.strip()
        if not s or s.startswith("#"):
            stats.blank_or_comment += 1
            continue
        m = _LOG_RE.match(line)
        if m:
            sec, frac, iface, id_s, sep, payload = m.groups()
            if sep == "##":
                stats.fd_frames += 1
                continue
            if payload.upper().startswith("R"):
                stats.remote_frames += 1
                continue
            if len(payload) % 2:
                stats.bad(line_no, line)
                continue
        else:
            m = _SCREEN_RE.match(line)
            if not m:
                stats.bad(line_no, line)
                continue
            sec, frac, iface, id_s, dlc, payload, rest = m.groups()
            if "remote" in rest.lower():
                stats.remote_frames += 1
                continue
            payload = "".join(payload.split())
            if rest.strip() or len(payload) != 2 * int(dlc):
                stats.bad(line_no, line)
                continue
        try:
            data = bytes.fromhex(payload)
            can_id = int(id_s, 16)
        except ValueError:
            stats.bad(line_no, line)
            continue
        if len(data) > 8:
            stats.fd_frames += 1
            continue
        extended = len(id_s) > 3
        stats.frames += 1
        yield RawFrame(_t_us(sec, frac), iface, can_id, extended, data, line_no)
