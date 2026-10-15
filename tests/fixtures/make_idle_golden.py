"""Builds idle_golden.log (committed). Re-run only if the scenario below is deliberately changed.

Timeline (1 s canonical rows, EEC1 + EEC2 every 100 ms, ET1 every 1 s), t0 = 1700000000 s:
  s  0- 9  rpm   0, load  0   key on, engine off           -> not running
  s 10-28  rpm 700, load  8   idle                          -> 19 s idle
  s 29     rpm 700, load NA   load "not available"          -> running, load unknown, NOT idle
  s 30-39  rpm 700, load 30   load >= cutoff                -> not idle
  s 40-49  rpm 1500, load 10  rpm above idle band           -> not idle
  s 50-54  rpm 800, load 14   rpm on the band's upper edge  -> 5 s idle (band is inclusive)
  s 55-59  rpm 750, load 15   load == cutoff (needs < )     -> not idle
  s 60-69  rpm 650, load 5    idle                          -> 10 s idle
One EEC1 frame at s 12 is captured twice (duplicate, must be dropped).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from j1939pf.encode import NA, candump_line, encode_pgn  # noqa: E402
from j1939pf.ids import make_can_id  # noqa: E402
from j1939pf.spn_table import load_spn_table  # noqa: E402

T0 = 1_700_000_000_000_000
PLAN = [(0, 10, 0, 0), (10, 29, 700, 8), (29, 30, 700, NA), (30, 40, 700, 30), (40, 50, 1500, 10),
        (50, 55, 800, 14), (55, 60, 750, 15), (60, 70, 650, 5)]

table = load_spn_table()
lines = []
for a, b, rpm, load in PLAN:
    for s in range(a, b):
        for k in range(10):
            t = T0 + s * 1_000_000 + k * 100_000
            eec1 = candump_line(t + 1_000, make_can_id(3, 61444, 0), encode_pgn(table, 61444, {190: rpm, 513: 0 if load is NA else load}))
            lines.append(eec1)
            if s == 12 and k == 3:
                lines.append(eec1)  # duplicate capture
            lines.append(candump_line(t + 2_000, make_can_id(3, 61443, 0), encode_pgn(table, 61443, {92: load})))
        lines.append(candump_line(T0 + s * 1_000_000 + 500_000, make_can_id(6, 65262, 0),
                                  encode_pgn(table, 65262, {110: 85, 175: 95})))
Path(__file__).with_name("idle_golden.log").write_text("".join(lines))
