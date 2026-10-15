"""Command line interface.

  j1939pf parse LOG [--csv-mapping MAP] --out canonical.csv
  j1939pf analyze INPUT [--csv-mapping MAP] [--engines DIR|FILE ...] [--events EVENTS.csv]
                  [--feedback FEEDBACK.json] [--defaults FILE] [--synthetic] --out OUTDIR
  j1939pf feedback {confirm,dismiss} --store FEEDBACK.json --alerts OUTDIR/alerts.json --alert-id ID --reason TEXT
  j1939pf explain --out OUTDIR --alert-id ID
  j1939pf generate --scenario NAME [--hours H] [--seed N] [--engine-id SA00] --out FILE [--format candump|canonical]
                   [--events-out EVENTS.csv]

INPUT for analyze: a candump log (.log/.txt/.gz), a canonical CSV written by `parse`, or a telematics
CSV with --csv-mapping. Nothing is fetched from the network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .framing import HONEST_FRAMING


def _configs(paths):
    from .engine_config import load_engine_configs
    return load_engine_configs(paths) if paths else {}


def _parse_input(args, params):
    from .io import is_canonical_csv, load_canonical
    from .pipeline import canonical_from_candump, canonical_from_csv
    src = Path(args.input)
    if args.csv_mapping:
        r = canonical_from_csv(src, args.csv_mapping, config=params)
        return r.canonical, None, r.report
    if src.suffix.lower() == ".csv" and is_canonical_csv(src):
        return load_canonical(src), None, {"source": str(src), "kind": "canonical csv"}
    if src.suffix.lower() == ".csv":
        raise SystemExit(f"{src} looks like a telematics CSV: pass --csv-mapping (see config/csv_mapping.example.yaml)")
    r = canonical_from_candump(src, config=params)
    return r.canonical, r.dtc_episodes, r.report


def cmd_parse(args):
    from .config import load_defaults
    params = load_defaults(args.defaults)
    args.input = args.log
    canon, dtc, rep = _parse_input(args, params)
    canon.to_csv(args.out, index=False)
    if dtc is not None and len(dtc):
        dtc.to_csv(Path(args.out).with_suffix(".dtc.csv"), index=False)
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"{len(canon)} canonical rows for {canon['engine_id'].nunique() if len(canon) else 0} engine(s) -> {args.out}")


def cmd_analyze(args):
    from .config import load_defaults
    from .feedback import FeedbackStore
    from .io import load_events
    from .report import run_full
    params = load_defaults(args.defaults)
    canon, dtc, rep = _parse_input(args, params)
    events = load_events(args.events) if args.events else None
    fb = FeedbackStore(args.feedback) if args.feedback else None
    res = run_full(canon, _configs(args.engines), dtc_episodes=dtc, events=events, feedback=fb, params=params,
                   parse_report=rep, synthetic=args.synthetic)
    paths = res.export(args.out)
    print(HONEST_FRAMING)
    for eid, h in res.health.items():
        n = len(res.analysis.for_engine(eid))
        print(f"{eid}: health {'n/a' if h.score is None else f'{h.score:.0f}'}/100, {n} alert(s)")
    print(f"wrote {len(paths)} files to {args.out} (open report.html)")


def cmd_feedback(args):
    from .alerts import Alert
    from .feedback import FeedbackStore
    doc = json.loads(Path(args.alerts).read_text())
    found = [a for a in doc["alerts"] if a["alert_id"] == args.alert_id]
    found += [s["alert"] for s in doc.get("suppressed", []) if s["alert"]["alert_id"] == args.alert_id]
    if not found:
        raise SystemExit(f"alert {args.alert_id} not found in {args.alerts}")
    rec = FeedbackStore(args.store).record(Alert.from_dict(found[0]), args.action, args.reason)
    print(f"{args.action} recorded for {rec['pattern_key']} -> {args.store}")


def cmd_explain(args):
    out = Path(args.out)
    doc = json.loads((out / "alerts.json").read_text())
    a = next((x for x in doc["alerts"] if x["alert_id"] == args.alert_id), None)
    if a is None:
        raise SystemExit(f"alert {args.alert_id} not found")
    print(json.dumps({k: a[k] for k in ("alert_id", "engine_id", "layer", "signal", "direction", "severity",
                                         "operating_point", "expected", "actual", "explanation", "sensor_verdict",
                                         "sensor_evidence", "confounders", "check_next")}, indent=2, default=str))
    print(f"data behind the alert: {out / 'alert_data' / (args.alert_id + '.csv')}")


def cmd_generate(args):
    from .generator import generate_engine, render_candump
    e = generate_engine(args.engine_id, args.scenario, hours=args.hours, seed=args.seed)
    if args.format == "candump":
        n = render_candump(e.canonical, args.out)
        print(f"SYNTHETIC candump: {n} frames -> {args.out} (parse with resample_interval_s >= 10)")
    else:
        e.canonical.to_csv(args.out, index=False)
        print(f"SYNTHETIC canonical CSV: {len(e.canonical)} rows -> {args.out}")
    if args.events_out:
        e.events.to_csv(args.events_out, index=False)
    print(json.dumps({k: v for k, v in e.truth.items() if k != "fault_params"}, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(prog="j1939pf", description=HONEST_FRAMING)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("parse", help="decode a log into the canonical series")
    a.add_argument("log")
    a.add_argument("--csv-mapping")
    a.add_argument("--defaults")
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_parse)
    a = sub.add_parser("analyze", help="run every layer and write reports/exports")
    a.add_argument("input")
    a.add_argument("--csv-mapping")
    a.add_argument("--engines", nargs="*", help="engine config files or a directory of them")
    a.add_argument("--events", help="maintenance / failure log CSV")
    a.add_argument("--feedback", help="feedback JSON (created if missing)")
    a.add_argument("--defaults", help="override file for config/defaults.yaml")
    a.add_argument("--synthetic", action="store_true", help="label outputs as synthetic data")
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_analyze)
    a = sub.add_parser("feedback", help="confirm or dismiss an alert (reason required)")
    a.add_argument("action", choices=["confirm", "dismiss"])
    a.add_argument("--store", required=True)
    a.add_argument("--alerts", required=True)
    a.add_argument("--alert-id", required=True)
    a.add_argument("--reason", required=True)
    a.set_defaults(func=cmd_feedback)
    a = sub.add_parser("explain", help="show one alert and where its data is")
    a.add_argument("--out", required=True)
    a.add_argument("--alert-id", required=True)
    a.set_defaults(func=cmd_explain)
    a = sub.add_parser("generate", help="write SYNTHETIC data for trying the program")
    from .generator import SCENARIOS
    a.add_argument("--scenario", choices=SCENARIOS, default="healthy")
    a.add_argument("--hours", type=float, default=300)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--engine-id", default="SA00")
    a.add_argument("--format", choices=["candump", "canonical"], default="canonical")
    a.add_argument("--events-out")
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_generate)
    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
