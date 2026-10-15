"""Deterministic test-bus builder: the same scripted values rendered as candump and as CSV.

This is test plumbing for the parser only (phase 1); it is not the synthetic
engine-scenario generator.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

import numpy as np

from j1939pf.dm import PGN_DM1, Dtc
from j1939pf.encode import ERR, NA, candump_line, dm_payload, encode_pgn, physical
from j1939pf.ids import make_can_id
from j1939pf.spn_table import SpnTable
from j1939pf.units import GAL_TO_L, PSI_TO_KPA

START_US = 1_700_000_000_000_000  # 2023-11-14T22:13:20Z

# (pgn, priority, period_us)
SCHEDULE = [(61444, 3, 20_000), (61443, 3, 50_000), (65263, 6, 500_000), (65262, 6, 1_000_000)]


@dataclass
class Frame:
    t_us: int
    can_id: int
    data: bytes
    sa: int
    pgn: int
    values: dict = field(default_factory=dict)  # SPN -> physical | NA | ERR
    dm: dict | None = None


def _values(pgn: int, t_s: float, sa: int, rng) -> dict:
    rpm = 1200 + 500 * math.sin(t_s / 7.0 + sa) + rng.normal(0, 5)
    if pgn == 61444:
        tq = 40 + 20 * math.sin(t_s / 5.0)
        return {190: max(0.0, rpm), 513: tq, 512: tq + 2}
    if pgn == 61443:
        return {91: NA, 92: min(100.0, max(0.0, 45 + 30 * math.sin(t_s / 5.0)))}
    if pgn == 65263:
        oil = 250 + 0.1 * (rpm - 1200) + rng.normal(0, 3)
        v = {100: oil, 98: 80.0, 94: 400.0}
        if int(t_s) % 17 == 3:
            v[100] = ERR  # sensor error indicator, must decode to NULL and be counted
        return v
    if pgn == 65262:
        return {110: 85.0 + (t_s % 3), 175: 95.0 + 0.01 * t_s, 174: 40.0}
    raise KeyError(pgn)


def build_frames(table: SpnTable, duration_s: float, sas=(0x00,), seed: int = 0,
                 jitter_us: int = 2_000, dm1: bool = True, start_us: int = START_US) -> list[Frame]:
    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    end = start_us + int(duration_s * 1e6)
    for sa in sas:
        for pgn, prio, period in SCHEDULE:
            t = start_us + int(rng.integers(0, period))
            while t < end:
                tj = t + int(rng.integers(-jitter_us, jitter_us + 1))
                vals = _values(pgn, (tj - start_us) / 1e6, sa, rng)
                frames.append(Frame(tj, make_can_id(prio, pgn, sa), encode_pgn(table, pgn, vals), sa, pgn, vals))
                t += period
        if dm1:
            t = start_us + 300_000
            while t < end:
                k = (t - start_us) // 1_000_000
                dtcs = [Dtc(100, 1, 3, 0)] if 5 <= k < 9 else []
                dm = {"count": len(dtcs), "mil": 0, "rsl": 0, "awl": 1 if dtcs else 0, "pl": 0}
                frames.append(Frame(t, make_can_id(6, PGN_DM1, sa), dm_payload(dtcs, awl=dm["awl"]), sa,
                                    PGN_DM1, dm=dm))
                t += 1_000_000
    frames.sort(key=lambda f: (f.t_us, f.can_id))
    return frames


def write_candump(frames, path, screen: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for f in frames:
            fh.write(candump_line(f.t_us, f.can_id, f.data, screen=screen))


# CSV columns: some in non-canonical units to exercise conversion.
ALT_UNITS = {100: ("psi", lambda kpa: kpa / PSI_TO_KPA), 175: ("degF", lambda c: c * 9 / 5 + 32),
             110: ("degF", lambda c: c * 9 / 5 + 32)}
DM_COLS = {"dm1_active_count": "count", "dm1_mil": "mil", "dm1_rsl": "rsl", "dm1_awl": "awl", "dm1_pl": "pl"}


def csv_mapping(table: SpnTable, layout: str = "wide", ts_format: str = "iso8601") -> dict:
    entries = {}
    for pgn, _, _ in SCHEDULE:
        for d in table.by_pgn[pgn]:
            unit = ALT_UNITS.get(d.spn, (d.unit,))[0]
            entries[f"spn{d.spn}"] = {"spn": d.spn, "unit": unit}
    for col in DM_COLS:
        entries[col] = {"canonical": col, "unit": "count" if col.endswith("count") else "state"}
    mp = {
        "layout": layout,
        "timestamp": {"column": "time", "format": ts_format, "timezone": "UTC"},
        "engine": {"column": "sa", "kind": "source_address"},
        "na_values": ["NA"], "error_values": ["ERR"], "chunksize": 997,
    }
    if layout == "wide":
        mp["columns"] = entries
    else:
        mp.update({"signal_column": "signal", "value_column": "value", "signals": entries})
    return mp


def _fmt_time(t_us: int, ts_format: str) -> str:
    if ts_format == "epoch_s":
        return f"{t_us // 1_000_000}.{t_us % 1_000_000:06d}"
    # ISO with a +02:00 offset, so timezone handling is exercised.
    import datetime as dt
    tz = dt.timezone(dt.timedelta(hours=2))
    return dt.datetime.fromtimestamp(t_us // 1_000_000, tz).replace(microsecond=t_us % 1_000_000).isoformat()


def write_csv(frames, table: SpnTable, path, layout: str = "wide", ts_format: str = "iso8601") -> None:
    mp = csv_mapping(table, layout, ts_format)
    cols = list((mp.get("columns") or mp.get("signals")).keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "sa"] + cols if layout == "wide" else ["time", "sa", "signal", "value"])
        for f in frames:
            cells: dict[str, str] = {}
            if f.dm is not None:
                for col, key in DM_COLS.items():
                    cells[col] = repr(float(f.dm[key]))
            else:
                for d in table.by_pgn[f.pgn]:
                    v = f.values.get(d.spn, NA)
                    if v is NA:
                        cells[f"spn{d.spn}"] = "NA"
                    elif v is ERR:
                        cells[f"spn{d.spn}"] = "ERR"
                    else:
                        phys = physical(d, v)
                        conv = ALT_UNITS.get(d.spn, (None, lambda x: x))[1]
                        cells[f"spn{d.spn}"] = repr(conv(phys))
            ts = _fmt_time(f.t_us, ts_format)
            if layout == "wide":
                w.writerow([ts, f.sa] + [cells.get(c, "") for c in cols])
            else:
                for c, v in cells.items():
                    w.writerow([ts, f.sa, c, v])


__all__ = ["build_frames", "write_candump", "write_csv", "csv_mapping", "START_US", "GAL_TO_L"]
