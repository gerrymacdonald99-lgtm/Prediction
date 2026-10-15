import gzip

from j1939pf.candump import ReadStats, iter_candump_frames
from j1939pf.config import load_defaults
from j1939pf.pipeline import canonical_from_candump

LINES = [
    "(1700000000.000100) can0 0CF00400#FFFFFF6012FFFFFF\n",
    "(1700000000.020100)  can0  0CF00400   [8]  FF FF FF 60 12 FF FF FF\n",
    "\n",
    "# comment\n",
    "garbage line\n",
    "(1700000000.030000) can0 18FEEE00#R\n",
    "(1700000000.040000) can0 123#DEADBEEF\n",
    "(1700000000.050000) can0 18FEEE00##1112233\n",
    "(1700000000.060000) can0 18FEEE00#ABC\n",
]


def test_formats_and_bad_lines():
    st = ReadStats()
    frames = list(iter_candump_frames(LINES, st))
    assert [(f.t_us, f.can_id, f.data, f.extended) for f in frames[:2]] == [
        (1_700_000_000_000_100, 0x0CF00400, bytes.fromhex("FFFFFF6012FFFFFF"), True),
        (1_700_000_000_020_100, 0x0CF00400, bytes.fromhex("FFFFFF6012FFFFFF"), True),
    ]
    assert frames[2].extended is False and frames[2].can_id == 0x123
    assert st.remote_frames == 1 and st.fd_frames == 1 and st.blank_or_comment == 2
    assert st.malformed == 2 and st.malformed_examples


def test_reader_is_lazy():
    def lines():
        yield LINES[0]
        raise AssertionError("reader consumed past the first frame")
    it = iter_candump_frames(lines())
    assert next(it).can_id == 0x0CF00400


def test_duplicates_removed_and_standard_ids_counted(table):
    lines = [
        "(1700000000.000100) can0 0CF00400#FFFFFF6012FFFFFF\n",
        "(1700000000.000100) can0 0CF00400#FFFFFF6012FFFFFF\n",   # duplicate capture
        "(1700000000.000300) can0 0CF00400#FFFFFF6012FFFFFF\n",   # duplicate within window
        "(1700000000.020100) can0 0CF00400#FFFFFF6012FFFFFF\n",   # real next frame, same data
        "(1700000000.030000) can0 123#00\n",
    ]
    r = canonical_from_candump(lines, table=table)
    assert r.report["decode"]["duplicate_frames"] == 2
    assert r.report["decode"]["standard_id_frames"] == 1
    assert r.report["decode"]["status_counts"]["SA00/engine_speed_rpm"]["ok"] == 2
    assert r.canonical["engine_speed_rpm"].iloc[0] == 0x1260 * 0.125


def test_gzip_and_screen_format_equivalent(tmp_path, table):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from busbuilder import build_frames, write_candump
    frames = build_frames(table, 12, sas=(0x00, 0x01))
    log, scr = tmp_path / "a.log", tmp_path / "b.log"
    write_candump(frames, log)
    write_candump(frames, scr, screen=True)
    gz = tmp_path / "a.log.gz"
    with open(log, "rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())
    cfg = load_defaults()
    a = canonical_from_candump(log, table=table, config=cfg).canonical
    b = canonical_from_candump(scr, table=table, config=cfg).canonical
    c = canonical_from_candump(gz, table=table, config=cfg).canonical
    assert len(a) > 0
    assert a.equals(b) and a.equals(c)
