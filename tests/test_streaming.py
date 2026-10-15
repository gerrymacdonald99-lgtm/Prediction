"""A large log must be parsed as a stream, not loaded into memory."""

import tracemalloc

from j1939pf.encode import candump_line, encode_pgn
from j1939pf.ids import make_can_id
from j1939pf.pipeline import canonical_from_candump

T0 = 1_700_000_000_000_000
DURATION_S = 2400


def _write_big_log(path, table):
    eec1 = [encode_pgn(table, 61444, {190: 600 + (k % 200) * 8, 513: 30, 512: 32}) for k in range(64)]
    et1 = encode_pgn(table, 65262, {110: 85, 175: 95, 174: 40})
    efl = encode_pgn(table, 65263, {100: 300})
    id_eec1, id_et1, id_efl = make_can_id(3, 61444, 0), make_can_id(6, 65262, 0), make_can_id(6, 65263, 0)
    n = 0
    with open(path, "w") as fh:
        for i in range(DURATION_S * 50):
            t = T0 + i * 20_000
            fh.write(candump_line(t, id_eec1, eec1[i % 64]))
            n += 1
            if i % 25 == 3:
                fh.write(candump_line(t + 1000, id_efl, efl))
                n += 1
            if i % 50 == 7:
                fh.write(candump_line(t + 2000, id_et1, et1))
                n += 1
    return n


def test_large_file_streams_with_bounded_memory(tmp_path, table, cfg):
    log = tmp_path / "big.log"
    n_lines = _write_big_log(log, table)
    size = log.stat().st_size
    assert n_lines > 120_000 and size > 5_000_000
    tracemalloc.start()
    try:
        r = canonical_from_candump(log, table=table, config=cfg)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert r.report["read"]["frames"] == n_lines
    assert len(r.canonical) == DURATION_S
    assert r.canonical["oil_pressure_kpa"].eq(300.0).all()
    # Holding the file (or one object per frame) would need far more than this.
    assert peak < size / 4, f"peak {peak/1e6:.1f} MB vs file {size/1e6:.1f} MB"
