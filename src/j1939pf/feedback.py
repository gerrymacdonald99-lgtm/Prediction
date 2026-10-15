"""Alert feedback loop: confirm / dismiss with a reason, persisted to a local JSON file.

* Every alert has a stable `pattern_key`: engine | layer | signal | direction | where
  (RPM band for layer (c), limit name for layer (a), "engine" for layer (d)).
* DISMISSED pattern: later alerts with the same key are suppressed (kept in
  `AnalysisResult.suppressed` with the reason) unless there is NEW EVIDENCE:
    - the deviation is materially larger than when it was dismissed (layer c: the larger of the
      episode's median |z| and its most recent 20 %; layer d: |drift %|; layer a: % beyond the
      limit) >= `new_evidence_factor` x the dismissed value, or
    - a DM1 code for the signal's SPN appeared after the dismissal.
  A re-raised alert says why it came back.
* CONFIRMED pattern: later alerts with the same key are escalated one severity level and
  cite the confirmation; detection for that engine/signal/direction runs at a lower
  threshold (`confirmed_threshold_factor`) so the confirmed fault is followed earlier.
Nothing is ever deleted: the file is an append-only log of decisions.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .alerts import Alert

ACTIONS = ("confirm", "dismiss")
_ESCALATE = {"watch": "warning", "warning": "critical", "critical": "critical"}


def pattern_key(a: Alert) -> str:
    if a.layer == "c":
        rpm = a.operating_point.get("rpm")
        where = f"{int(rpm // 500 * 500)}-{int(rpm // 500 * 500 + 500)}rpm" if rpm == rpm and rpm is not None else "?"
    elif a.layer == "a":
        where = (a.limit or "").split(" = ")[0]
    else:
        where = "engine"
    return f"{a.engine_id}|{a.layer}|{a.signal}|{a.direction}|{where}"


def alert_id(a: Alert) -> str:
    return hashlib.sha1(f"{pattern_key(a)}|{a.start_h:.1f}".encode()).hexdigest()[:12]


def magnitude(a: Alert) -> float:
    if a.layer == "c":
        return max(abs(a.z or 0.0), abs(a.operating_point.get("z_recent", 0.0)))
    if a.layer == "d":
        return abs(a.operating_point.get("drift_pct", 0.0))
    if a.expected and a.actual is not None:
        return abs(a.actual - a.expected) / abs(a.expected) * 100
    return 0.0


class FeedbackStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: list[dict] = []
        if self.path.exists():
            doc = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            self.records = list(doc.get("records", []))

    # -- writing --------------------------------------------------------------------
    def record(self, alert: Alert, action: str, reason: str, at: str | None = None) -> dict:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        if not reason or not reason.strip():
            raise ValueError("a reason is required for every confirm/dismiss")
        rec = {"alert_id": alert.alert_id or alert_id(alert), "pattern_key": alert.pattern_key or pattern_key(alert),
               "engine_id": alert.engine_id, "layer": alert.layer, "signal": alert.signal,
               "direction": alert.direction, "action": action, "reason": reason.strip(),
               "at": at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "alert_start_h": alert.start_h, "alert_end_h": alert.end_h, "alert_end_t": str(alert.end_t),
               "magnitude": magnitude(alert),
               "sensor_verdict": alert.sensor_verdict}
        self.records.append(rec)
        self.save()
        return rec

    def dismiss(self, alert: Alert, reason: str, at: str | None = None) -> dict:
        return self.record(alert, "dismiss", reason, at)

    def confirm(self, alert: Alert, reason: str, at: str | None = None) -> dict:
        return self.record(alert, "confirm", reason, at)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": 1, "records": self.records}, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    # -- reading --------------------------------------------------------------------
    def latest(self, key: str) -> dict | None:
        for rec in reversed(self.records):
            if rec["pattern_key"] == key:
                return rec
        return None

    def confirmed(self, engine_id: str, layer: str, signal: str, direction: str) -> list[dict]:
        out = {}
        for rec in self.records:
            if (rec["engine_id"], rec["layer"], rec["signal"], rec["direction"]) == (engine_id, layer, signal, direction):
                out[rec["pattern_key"]] = rec
        return [r for r in out.values() if r["action"] == "confirm"]


def apply_feedback(alerts: list[Alert], store: FeedbackStore | None, params: dict, dtc_episodes=None,
                   spn_of: dict | None = None) -> tuple[list[Alert], list[tuple[Alert, str]]]:
    if store is None:
        return alerts, []
    fp = params["feedback"]
    kept, suppressed = [], []
    for a in alerts:
        rec = store.latest(a.pattern_key)
        if rec is None:
            kept.append(a)
            continue
        if rec["action"] == "confirm":
            a.severity = _ESCALATE.get(a.severity, a.severity)
            a.feedback_note = (f"matches a CONFIRMED fault pattern ({rec['at']}: {rec['reason']}); severity escalated "
                               f"and detection threshold lowered for this pattern")
            kept.append(a)
            continue
        # dismissed: quiet unless there is new evidence
        new = []
        mag = magnitude(a)
        if rec["magnitude"] > 0 and mag >= fp["new_evidence_factor"] * rec["magnitude"]:
            new.append(f"deviation {mag:.1f} vs {rec['magnitude']:.1f} when dismissed")
        spn = (spn_of or {}).get(a.signal)
        if dtc_episodes is not None and len(dtc_episodes) and spn is not None:
            ep = dtc_episodes[(dtc_episodes["engine_id"] == a.engine_id) & (dtc_episodes["spn"] == spn)]
            import pandas as pd
            after = ep[pd.to_datetime(ep["first_t"], utc=True) > pd.Timestamp(rec["alert_end_t"])]
            if len(after):
                new.append(f"new DM1 code for SPN {spn} (FMI {', '.join(str(f) for f in sorted(set(after['fmi'])))}) "
                           f"after the dismissal")
        if new:
            a.feedback_note = (f"previously DISMISSED ({rec['at']}: {rec['reason']}); re-raised on new evidence: "
                               + "; ".join(new))
            kept.append(a)
        else:
            suppressed.append((a, f"dismissed {rec['at']}: {rec['reason']} (no new evidence since)"))
    return kept, suppressed
