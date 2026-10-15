"""Self-contained HTML report (no network, no JavaScript libraries): inline SVG charts + tables.

Colours follow the reference data-viz palette (categorical slots 1-2, status colours only
with a text label), light and dark mode via CSS custom properties. Every chart has a
single y-axis, native SVG tooltips on its points, and the underlying numbers are in the
CSV exports next to the report.
"""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd

from .framing import HONEST_FRAMING

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ef;--border:#dedcd6;--text-primary:#0b0b0b;
--text-secondary:#52514e;--text-muted:#7a7873;--series-1:#2a78d6;--series-2:#eb6834;--good:#0ca30c;--warning:#fab219;
--serious:#ec835a;--critical:#d03b3b;--div-pos:42,120,214;--div-neg:227,73,72}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--surface-2:#242422;--border:#3a3936;--text-primary:#f0efec;--text-secondary:#c3c2b7;
--text-muted:#9a998f;--series-1:#3987e5;--series-2:#d95926;--div-pos:57,135,229;--div-neg:230,103,103}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242422;--border:#3a3936;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9a998f;--series-1:#3987e5;--series-2:#d95926;
--div-pos:57,135,229;--div-neg:230,103,103}
body{margin:0;background:#fcfcfb}
@media (prefers-color-scheme:dark){body{background:#1a1a19}}
.viz-root{background:var(--surface-1);color:var(--text-primary);font:14px/1.45 system-ui,-apple-system,"Segoe UI",
sans-serif;max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0}h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--border);
padding-bottom:4px}h3{font-size:15px;margin:18px 0 6px}
.banner{background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin:8px 0;
color:var(--text-secondary)}.muted{color:var(--text-muted)}.small{font-size:12px}
table{border-collapse:collapse;width:100%;margin:6px 0;font-size:13px}th,td{border-bottom:1px solid var(--border);
padding:4px 6px;text-align:left;vertical-align:top}th{color:var(--text-secondary);font-weight:600}
.tablewrap{overflow-x:auto}.card{border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin:10px 0;
background:var(--surface-1)}.hero{font-size:44px;font-weight:700;line-height:1}
.badge{display:inline-block;border-radius:10px;padding:1px 8px;font-size:12px;font-weight:600;
border:1px solid var(--border);color:var(--text-primary)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
.bar{height:10px;border-radius:4px;background:var(--series-1)}.barbg{background:var(--surface-2);border-radius:4px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--text-secondary);margin:4px 0}
.swatch{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:4px}
svg text{fill:var(--text-secondary);font-size:11px}svg .axis{stroke:var(--border)}svg .grid{stroke:var(--border);
stroke-opacity:.5}svg .ref{stroke:var(--text-muted);stroke-dasharray:4 4}
details summary{cursor:pointer;color:var(--text-secondary)}
"""

SEV = {"critical": ("--critical", "Critical"), "warning": ("--serious", "Warning"), "watch": ("--warning", "Watch")}


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _nice_ticks(lo, hi, n=5):
    if not np.isfinite(lo) or not np.isfinite(hi):
        return []
    if hi <= lo:
        hi = lo + 1
    step = 10 ** math.floor(math.log10((hi - lo) / n))
    for m in (1, 2, 5, 10):
        if (hi - lo) / (step * m) <= n:
            step *= m
            break
    start = math.ceil(lo / step) * step
    return [start + i * step for i in range(int((hi - start) / step) + 1)]


def line_chart(series, xlabel, ylabel, hlines=(), vlines=(), width=720, height=240, title=""):
    """series: [{name, x, y, slot(1|2), dash}] - one y-axis, native tooltips on points."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 14, 36
    xs = np.concatenate([np.asarray(s["x"], float) for s in series if len(s["x"])] or [np.array([0.0, 1.0])])
    ys = np.concatenate([np.asarray(s["y"], float) for s in series if len(s["y"])] or [np.array([0.0, 1.0])])
    ys = np.concatenate([ys, [h[0] for h in hlines]]) if hlines else ys
    ok = np.isfinite(xs)
    x0, x1 = (float(xs[ok].min()), float(xs[ok].max())) if ok.any() else (0.0, 1.0)
    oky = np.isfinite(ys)
    y0, y1 = (float(ys[oky].min()), float(ys[oky].max())) if oky.any() else (0.0, 1.0)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y0, y1 = y0 - 1, y1 + 1
    yr = y1 - y0
    y0, y1 = y0 - 0.05 * yr, y1 + 0.05 * yr
    W, H = width - pad_l - pad_r, height - pad_t - pad_b
    X = lambda v: pad_l + (v - x0) / (x1 - x0) * W  # noqa: E731
    Y = lambda v: pad_t + (1 - (v - y0) / (y1 - y0)) * H  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{_e(title or ylabel)}">']
    for t in _nice_ticks(y0, y1):
        out.append(f'<line class="grid" x1="{pad_l}" x2="{width - pad_r}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                   f'<text x="{pad_l - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in _nice_ticks(x0, x1, 6):
        out.append(f'<text x="{X(t):.1f}" y="{height - pad_b + 16}" text-anchor="middle">{t:g}</text>')
    out.append(f'<line class="axis" x1="{pad_l}" x2="{width - pad_r}" y1="{pad_t + H}" y2="{pad_t + H}"/>')
    out.append(f'<text x="{pad_l + W / 2:.0f}" y="{height - 4}" text-anchor="middle">{_e(xlabel)}</text>')
    out.append(f'<text x="12" y="{pad_t + H / 2:.0f}" text-anchor="middle" transform="rotate(-90 12 {pad_t + H / 2:.0f})">'
               f'{_e(ylabel)}</text>')
    for yv, lab in hlines:
        out.append(f'<line class="ref" x1="{pad_l}" x2="{width - pad_r}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}"/>'
                   f'<text x="{width - pad_r - 2}" y="{Y(yv) - 3:.1f}" text-anchor="end">{_e(lab)}</text>')
    for xv, lab in vlines:
        if x0 <= xv <= x1:
            out.append(f'<line class="ref" x1="{X(xv):.1f}" x2="{X(xv):.1f}" y1="{pad_t}" y2="{pad_t + H}">'
                       f'<title>{_e(lab)}</title></line>')
    for s in series:
        x, y = np.asarray(s["x"], float), np.asarray(s["y"], float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) > 400:
            idx = np.linspace(0, len(x) - 1, 400).astype(int)
            x, y = x[idx], y[idx]
        col = f"var(--series-{s.get('slot', 1)})"
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(x, y))
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"{dash} points="{pts}"/>')
        if len(x) <= 120:
            for a, b in zip(x, y):
                out.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="5" fill="transparent">'
                           f'<title>{_e(s["name"])}: {xlabel} {a:.4g}, {ylabel} {b:.4g}</title></circle>')
    out.append("</svg>")
    legend = ""
    if len(series) >= 2:
        legend = '<div class="legend">' + "".join(
            f'<span><span class="swatch" style="background:var(--series-{s.get("slot", 1)})"></span>{_e(s["name"])}'
            f'{" (dashed)" if s.get("dash") else ""}</span>' for s in series) + "</div>"
    return f'<div class="chart">{f"<h3>{_e(title)}</h3>" if title else ""}{legend}{"".join(out)}</div>'


def _table(df: pd.DataFrame, max_rows=40) -> str:
    if df is None or len(df) == 0:
        return '<p class="muted">none</p>'
    d = df.head(max_rows)
    head = "".join(f"<th>{_e(c)}</th>" for c in d.columns)
    body = "".join("<tr>" + "".join(f"<td>{_e(_fmt(v))}</td>" for v in r) + "</tr>" for r in d.itertuples(index=False))
    more = f'<p class="muted small">{len(df) - max_rows} more rows in the CSV export</p>' if len(df) > max_rows else ""
    return f'<div class="tablewrap"><table><tr>{head}</tr>{body}</table></div>{more}'


def _fmt(v):
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return "; ".join(_fmt(x) for x in v)
    return v


def _heatmap(m: pd.DataFrame) -> str:
    if m is None or m.empty:
        return '<p class="muted">not enough data</p>'
    cols = list(m.columns)
    h = ['<div class="tablewrap"><table><tr><th></th>' + "".join(f'<th class="small">{_e(c)}</th>' for c in cols) + "</tr>"]
    for r in cols:
        cells = []
        for c in cols:
            v = m.loc[r, c]
            if not np.isfinite(v):
                cells.append("<td></td>")
                continue
            rgb = "var(--div-pos)" if v >= 0 else "var(--div-neg)"
            cells.append(f'<td title="{_e(r)} vs {_e(c)}: Spearman {v:+.2f}" '
                         f'style="background:rgba({rgb},{min(abs(v), 1) * 0.55:.2f})">{v:+.2f}</td>')
        h.append(f'<tr><th class="small">{_e(r)}</th>{"".join(cells)}</tr>')
    h.append("</table></div><p class=\"muted small\">Spearman rank correlation on the reference window; blue = "
             "positive, red = negative, gray-to-white = none. Association, not cause.</p>")
    return "".join(h)


def _bars(items) -> str:
    """items: [(label, value, max_value, text)]"""
    rows = []
    for label, v, vmax, text in items:
        w = 0 if not vmax or v is None or not np.isfinite(v) else max(0.0, min(1.0, v / vmax)) * 100
        rows.append(f'<tr><td style="width:28%">{_e(label)}</td><td style="width:42%"><div class="barbg">'
                    f'<div class="bar" style="width:{w:.1f}%"></div></div></td><td>{_e(text)}</td></tr>')
    return f'<table>{"".join(rows)}</table>'


def _health_section(h) -> str:
    score = "n/a" if h.score is None else f"{h.score:.0f}"
    items = []
    vmax = max([c["points_lost"] for c in h.components if c["available"]] + [1.0])
    for c in h.components:
        if c["available"]:
            items.append((c["component"], c["points_lost"], vmax,
                          f"sub-score {c['subscore']:.0f}, -{c['points_lost']:.1f} pts: {c['reason']}"))
        else:
            items.append((c["component"], None, 1, f"unavailable: {c['reason']}"))
    return (f'<div class="card"><div class="hero">{score}<span class="muted" style="font-size:18px">/100</span></div>'
            f'<p class="small">{_e(h.note)}</p><h3>Decomposition (points lost per component)</h3>{_bars(items)}</div>')


def _alert_card(res, a) -> str:
    var, label = SEV.get(a.severity, ("--warning", a.severity))
    data = res.analysis.alert_data(a, pad_s=0)
    t = (data["t"] - data["t"].iloc[0]).dt.total_seconds() / 3600 if len(data) else pd.Series(dtype=float)
    win = t <= 2.0  # first two hours of the episode
    series = [{"name": "actual", "x": t[win], "y": data.loc[win, a.signal], "slot": 1}]
    hl = []
    if "expected" in data and data["expected"].notna().any():
        series.append({"name": "learned expected", "x": t[win], "y": data.loc[win, "expected"], "slot": 2, "dash": True})
    if a.layer == "a" and a.expected is not None:
        hl.append((a.expected, "configured limit"))
    chart = line_chart(series, "hours from episode start", a.signal, hl, (), height=200) if win.any() else ""
    ev = "".join(f"<li>{_e(x)}</li>" for x in a.sensor_evidence)
    conf = "".join(f"<li>{_e(x)}</li>" for x in a.confounders)
    steps = "".join(f"<li>{_e(x)}</li>" for x in a.check_next)
    fb = f'<p><b>Feedback:</b> {_e(a.feedback_note)}</p>' if a.feedback_note else ""
    return (f'<div class="card"><span class="badge"><span class="dot" style="background:var({var})"></span>{label}</span> '
            f'<b>{_e(a.headline().split("] ", 1)[-1])}</b><p class="small muted">id {a.alert_id} | layer ({a.layer}) | '
            f'{a.start_h:.1f}-{a.end_h:.1f} engine h | data: alert_data/{a.alert_id}.csv</p>'
            f'<p>{_e(a.explanation)}</p><p><b>Sensor vs engine:</b> {_e(a.sensor_verdict)}</p><ul>{ev}</ul>'
            + (f"<p><b>Confounders</b></p><ul>{conf}</ul>" if conf else "")
            + f'<p><b>What to check next (cheapest first)</b></p><ol>{steps}</ol>{fb}'
            f'<details><summary>Data behind this alert (first 2 h)</summary>{chart}</details></div>')


def _learned_curve(res, eid) -> str:
    sb = res.analysis.baselines.get((eid, "oil_pressure_kpa"))
    if sb is None or sb.reference is None or sb.status != "active":
        return '<p class="muted">no learned oil pressure baseline yet (learning period)</p>'
    ref = sb.reference[sb.reference["trusted"]].copy()
    ref["rpm"] = (ref.index // 100_000 + 0.5) * res.params["baseline"]["rpm_bin_width"]
    ref["temp_bin"] = (ref.index % 100_000) // 100 - 100
    tb = ref.groupby("temp_bin")["n"].sum().idxmax()
    sel = ref[ref["temp_bin"] == tb]
    g = sel.groupby("rpm").apply(lambda d: pd.Series({"ref": np.average(d["median"], weights=d["n"]),
                                                     "online": np.average(sb.online.reindex(d.index).fillna(d["median"]),
                                                                          weights=d["n"])}), include_groups=False)
    w = res.params["baseline"]["oil_temp_bin_width_c"]
    return line_chart([{"name": "learned reference", "x": g.index, "y": g["ref"], "slot": 1},
                       {"name": "current expected (online, drift-capped)", "x": g.index, "y": g["online"], "slot": 2,
                        "dash": True}], "engine speed (rpm)", "oil pressure (kPa)",
                      title=f"What it learned: oil pressure vs RPM at oil {tb * w}-{(tb + 1) * w} C")


def render_html(res) -> str:
    a = res.analysis
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
             f'content="width=device-width,initial-scale=1"><title>Engine Health Report</title><style>{CSS}</style>'
             f'</head><body><div class="viz-root"><h1>J1939 engine exposure &amp; trend report</h1>'
             f'<div class="banner"><b>Read this first.</b> {_e(HONEST_FRAMING)}</div>']
    if res.synthetic:
        parts.append('<div class="banner"><b>SYNTHETIC DATA.</b> These results show the program works end to end; '
                     'they say nothing about real-world accuracy.</div>')
    if res.parse_report and res.parse_report.get("unverified_spns"):
        parts.append(f'<div class="banner"><b>Unverified SPN table.</b> Decoded SPNs '
                     f'{_e(res.parse_report["unverified_spns"])} are not yet checked against J1939DA.</div>')
    # fleet overview
    rows = []
    for eid, h in res.health.items():
        al = a.for_engine(eid)
        sev = {s: sum(1 for x in al if x.severity == s) for s in ("critical", "warning", "watch")}
        worst = max((c for c in h.components if c["available"]), key=lambda c: c["points_lost"], default=None)
        rows.append({"engine": eid, "health": "n/a" if h.score is None else round(h.score),
                     "largest deduction": (f"{worst['component']} (-{worst['points_lost']:.1f})"
                                           if worst and worst["points_lost"] >= 0.05 else "none"),
                     "critical": sev["critical"], "warning": sev["warning"], "watch": sev["watch"],
                     "flags": "; ".join(res.usage.engines[eid].flags) if eid in res.usage.engines else ""})
    parts.append(f"<h2>Fleet overview</h2>{_table(pd.DataFrame(rows))}")
    for eid, h in res.health.items():
        st = a.status[eid]
        parts.append(f'<h2>Engine {_e(eid)}</h2><p class="muted small">{st["hours_observed"][0]:.1f}-'
                     f'{st["hours_observed"][1]:.1f} engine h ({_e(st["hours_source"])}); config: {_e(st["config"])}</p>'
                     f'<div class="banner">{_e(st["learning_announcement"])}</div>')
        parts.append("<h3>Health score</h3>" + _health_section(h))
        al = a.for_engine(eid)
        parts.append(f"<h3>Alerts ({len(al)})</h3>" + ("".join(_alert_card(res, x) for x in al) or
                                                        '<p class="muted">no alerts</p>'))
        sup = [(x, why) for x, why in a.suppressed if x.engine_id == eid]
        if sup:
            parts.append("<h3>Suppressed by feedback</h3>" + _table(pd.DataFrame(
                [{"alert": x.headline(), "why": why} for x, why in sup])))
        parts.append("<h3>Layer (a) manual thresholds</h3>" + _table(pd.DataFrame(
            [{"limit": k, "status": v} for k, v in st["layer_a"].items()])))
        # trend
        for sig in res.params["baseline"]["signals"]:
            sb = a.baselines.get((eid, sig))
            if sb is not None and sb.trend is not None and len(sb.trend):
                thr = res.params["alerts"]["drift_alert_rel"] * 100
                vl = [(adj["hours"], f"{adj['label']}: {adj.get('note', '')}") for adj in sb.adjustments]
                parts.append(line_chart([{"name": "drift", "x": sb.trend["hours"], "y": sb.trend["drift_rel"] * 100,
                                          "slot": 1}], "engine hours", "drift vs reference (%)",
                                        [(thr, f"+{thr:g} % trend alert"), (-thr, f"-{thr:g} % trend alert")], vl,
                                        title=f"Trend: {sig} vs its early-hours reference (dashed verticals = change-points)"))
        parts.append(_learned_curve(res, eid))
        if st.get("transitions"):
            parts.append("<h3>Transitions / re-baselining</h3><ul>" +
                         "".join(f"<li>{_e(t)}</li>" for t in st["transitions"]) + "</ul>")
        u = res.usage.engines.get(eid)
        if u is not None:
            ex = u.exposure
            parts.append("<h3>Exposure (time x severity; NOT life consumed)</h3>" + _table(
                ex[["condition", "available", "hours", "exposure", "unit", "episodes", "hours_per_100_running_h",
                    "reason"]]))
            m = u.metrics
            idle = m.get("idle", {})
            parts.append("<h3>Derived values</h3>" + _table(pd.DataFrame(
                [{"value": k, "result": (", ".join(f"{a_}={_fmt(b)}" for a_, b in v.items()
                                                   if a_ not in ("available", "rpm_x_load_pct", "per_interval"))
                                         if v.get("available") else f"unavailable: {v.get('reason', '')}")}
                 for k, v in m.items() if k != "time"])))
            if idle.get("available"):
                parts.append(f'<p class="small muted">Idle definition: {_e(idle["definition"])}</p>')
            parts.append("<h3>Oil service history (logged and detected)</h3>" + _table(
                u.oil.history[["engine_hours", "event", "source", "status", "confidence", "pressure_step_pct",
                               "level_step_pct"]] if len(u.oil.history) else u.oil.history))
            parts.append(f'<p>Hours since last oil change: <b>{_fmt(u.oil.hours_since_change)}</b> '
                         f'({_e(u.oil.hours_since_change_source)})</p>')
        an = res.anomalies.get(eid)
        if an is not None and an.available and len(an.rate_by_period):
            parts.append(line_chart([{"name": "anomaly rate", "x": an.rate_by_period["from_h"],
                                      "y": an.rate_by_period["anomaly_rate"] * 100, "slot": 1}], "engine hours",
                                    "joint anomaly rate (%)", [(an.reference_rate * 100, "reference rate")],
                                    title="Joint-signal anomalies (both methods agree)"))
            parts.append(_table(pd.DataFrame(an.episodes)[["start_h", "end_h", "duration_s", "top_signals"]]
                                if an.episodes else pd.DataFrame()))
        elif an is not None:
            parts.append(f'<p class="muted">Joint anomaly layer unavailable: {_e(an.reason)}</p>')
        rel = [r for r in res.correlations.relationships if r.engine_id == eid]
        parts.append("<h3>Learned relationships</h3>" + _table(pd.DataFrame(
            [{"relationship": r.name, "kind": r.kind, "status": r.status if r.available else f"unavailable: {r.reason}",
              "rho": r.rho, "rho 95% CI": r.rho_ci, "n rows": r.n_rows, "n independent": r.n_independent,
              "range": r.valid_range, "breaks": "; ".join(r.breaks)} for r in rel]), max_rows=60))
        parts.append("<h3>Correlation matrix (reference window)</h3>" + _heatmap(res.matrices.get(eid)))
    # label layers
    parts.append("<h2>Label-based layers</h2><p class=\"small muted\">Only these layers may speak about failure risk "
                 "or life, and only with sample sizes and confidence intervals.</p>"
                 f"<p><b>Supervised:</b> {_e(res.supervised.message)}</p><p><b>Survival:</b> {_e(res.survival.message)}</p>")
    km = res.survival.details.get("kaplan_meier")
    if isinstance(km, pd.DataFrame) and len(km) >= 2:
        parts.append(line_chart([{"name": "Kaplan-Meier", "x": km["engine_hours"], "y": km["survival"], "slot": 1}],
                                "engine hours", "survival probability", title="Kaplan-Meier (left-truncated)"))
    if res.remaining_life:
        parts.append(_table(pd.DataFrame([{"engine": k, **v} for k, v in res.remaining_life.items()])))
    parts.append('<p class="muted small">Generated by j1939pf. Numbers behind every chart are in the CSV/JSON exports '
                 'next to this file.</p></div></body></html>')
    return "".join(parts)
