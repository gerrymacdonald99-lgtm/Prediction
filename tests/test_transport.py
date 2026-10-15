from j1939pf.candump import RawFrame
from j1939pf.decoder import J1939Decoder
from j1939pf.dm import PGN_DM1, Dtc, decode_dm
from j1939pf.encode import bam_frames, dm_payload, rts_cts_frames
from j1939pf.ids import parse_can_id
from j1939pf.transport import PGN_TP_CM, TransportReassembler

PAYLOAD = bytes(range(1, 24))  # 23 bytes -> 4 packets
T0 = 1_000_000_000


def run(frames, tp=None):
    tp = tp or TransportReassembler()
    out = []
    for t, can_id, data in frames:
        j = parse_can_id(can_id)
        m = tp.feed(t, j.pgn, j.sa, j.da, data)
        if m:
            out.append(m)
    return out, tp


def test_bam_reassembly():
    msgs, tp = run(bam_frames(T0, 65259, 0x00, PAYLOAD))
    assert len(msgs) == 1
    m = msgs[0]
    assert (m.pgn, m.sa, m.da, m.data, m.mode) == (65259, 0x00, 0xFF, PAYLOAD, "BAM")
    assert tp.stats.completed == 1


def test_bam_missing_packet_never_emits():
    frames = [f for i, f in enumerate(bam_frames(T0, 65259, 0x00, PAYLOAD)) if i != 2]
    msgs, tp = run(frames)
    assert msgs == []
    assert tp.stats.missing_packet == 1


def test_bam_last_packet_lost_times_out():
    frames = list(bam_frames(T0, 65259, 0x00, PAYLOAD))[:-1]
    frames.append((frames[-1][0] + 2_000_000, 0x18FEEE00, b"\x00" * 8))  # later unrelated traffic
    msgs, tp = run(frames)
    assert msgs == [] and tp.stats.timed_out == 1


def test_bam_duplicate_packet_ignored_and_conflict_aborts():
    frames = list(bam_frames(T0, 65259, 0x00, PAYLOAD))
    dup = (frames[2][0] + 1000, frames[2][1], frames[2][2])
    msgs, tp = run(frames[:3] + [dup] + frames[3:])
    assert [m.data for m in msgs] == [PAYLOAD] and tp.stats.duplicate_packets == 1

    bad = (frames[2][0] + 1000, frames[2][1], bytes([2]) + b"\x99" * 7)
    msgs, tp = run(frames[:3] + [bad] + frames[3:])
    assert msgs == [] and tp.stats.conflicting_packets == 1


def test_rts_cts_reassembly():
    msgs, tp = run(rts_cts_frames(T0, 65259, 0x00, 0xF9, PAYLOAD))
    assert len(msgs) == 1
    assert (msgs[0].data, msgs[0].da, msgs[0].mode) == (PAYLOAD, 0xF9, "RTS/CTS")


def test_rts_cts_lost_packet_retransmitted():
    msgs, tp = run(rts_cts_frames(T0, 65259, 0x00, 0xF9, PAYLOAD, drop={2}))
    assert [m.data for m in msgs] == [PAYLOAD]


def test_rts_cts_lost_packet_not_resent_times_out():
    frames = list(rts_cts_frames(T0, 65259, 0x00, 0xF9, PAYLOAD, drop={2}, resend_dropped=False))
    frames.append((frames[-1][0] + 5_000_000, 0x18FEEE00, b"\x00" * 8))
    msgs, tp = run(frames)
    assert msgs == [] and tp.stats.timed_out == 1


def test_abort_and_orphans():
    frames = list(bam_frames(T0, 65259, 0x00, PAYLOAD))
    from j1939pf.ids import make_can_id
    abort = (frames[1][0] + 1, make_can_id(7, PGN_TP_CM, 0x00, 0xFF),
             bytes([255, 1, 0xFF, 0xFF, 0xFF, 0xEB, 0xFE, 0x00]))
    msgs, tp = run(frames[:2] + [abort] + frames[2:])
    assert msgs == [] and tp.stats.aborted_by_sender == 1 and tp.stats.orphan_packets >= 1


def test_concurrent_sessions_from_two_sources():
    a = list(bam_frames(T0, 65226, 0x00, PAYLOAD))
    b = list(bam_frames(T0 + 5_000, 65226, 0x01, PAYLOAD[::-1]))
    msgs, _ = run(sorted(a + b))
    assert {(m.sa, m.data) for m in msgs} == {(0x00, PAYLOAD), (0x01, PAYLOAD[::-1])}


def test_dm1_multi_frame_through_decoder(table):
    dtcs = [Dtc(100, 1, 5, 0), Dtc(110, 0, 2, 0), Dtc(524287 - 1, 31, 1, 0)]
    payload = dm_payload(dtcs, mil=1, awl=1)
    assert len(payload) > 8
    dec = J1939Decoder(table)
    samples = []
    for t, cid, d in bam_frames(T0, PGN_DM1, 0x00, payload):
        samples += dec.feed(RawFrame(t, "can0", cid, True, d, 0))
    dec.finish()
    by = {s.name: s.value for s in samples}
    assert by["dm1_active_count"] == 3 and by["dm1_mil"] == 1 and by["dm1_awl"] == 1 and by["dm1_rsl"] == 0
    assert {(e.spn, e.fmi) for e in dec.dtc_episodes} == {(100, 1), (110, 0), (524286, 31)}


def test_dm1_single_frame_no_faults():
    m = decode_dm(dm_payload([]))
    assert m.dtcs == () and m.lamps["mil"] == 0 and not m.malformed
