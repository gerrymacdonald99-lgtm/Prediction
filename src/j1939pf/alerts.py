"""Alert record, falsifiable explanations and cheap-first "what to check next" lists.

Alerts describe deviations and exposure. They never claim a condition shortens
engine life, and never convert a deviation into "life lost".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .units import PSI_TO_KPA

SIGNAL_LABEL = {
    "oil_pressure_kpa": "Oil pressure",
    "oil_filter_dp_kpa": "Oil filter differential pressure",
    "oil_temp_c": "Oil temperature",
    "coolant_temp_c": "Coolant temperature",
}


def fmt_value(signal: str, v: float | None) -> str:
    if v is None or v != v:
        return "n/a"
    if signal.endswith("_kpa"):
        return f"{v:.0f} kPa ({v / PSI_TO_KPA:.1f} psi)"
    if signal.endswith("_c"):
        return f"{v:.1f} C ({v * 9 / 5 + 32:.0f} F)"
    return f"{v:.2f}"


@dataclass
class Alert:
    engine_id: str
    layer: str                      # a | c | d
    signal: str
    direction: str                  # low | high
    severity: str                   # watch | warning | critical
    start_t: object
    end_t: object
    start_h: float
    end_h: float
    duration_s: float               # total time the condition held (sum of sustained runs)
    n_runs: int
    operating_point: dict
    expected: float | None
    actual: float | None
    z: float | None = None
    limit: str | None = None
    explanation: str = ""
    check_next: list[str] = field(default_factory=list)
    sensor_verdict: str = "undetermined"
    sensor_evidence: list[str] = field(default_factory=list)
    confounders: list[str] = field(default_factory=list)
    data_window: dict = field(default_factory=dict)  # where to find the rows behind the alert

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_t"] = str(self.start_t)
        d["end_t"] = str(self.end_t)
        return d

    def headline(self) -> str:
        lbl = SIGNAL_LABEL.get(self.signal, self.signal)
        return (f"[{self.severity.upper()}] {self.engine_id} layer ({self.layer}) {lbl} {self.direction}: "
                f"actual {fmt_value(self.signal, self.actual)} vs "
                f"{'limit' if self.layer == 'a' else 'expected'} {fmt_value(self.signal, self.expected)}"
                f" @ {self.operating_point.get('label', '')}")


def check_next(signal: str, verdict: str, pattern: str, direction: str) -> list[str]:
    """Cheapest checks first: the reading, then oil level/grade, then filter, then mechanical."""
    gauge = ("Verify the reading: compare against a calibrated mechanical gauge at the same RPM and oil "
             "temperature; inspect the sensor connector, wiring and ground; review DM1 codes for this SPN.")
    if signal == "oil_filter_dp_kpa":
        steps = [gauge.replace("mechanical gauge", "differential gauge"),
                 "Check oil level and that the oil grade matches the configured grade (cold or thick oil raises filter dp).",
                 "Check hours since the last filter change; inspect/replace the oil filter.",
                 "If dp stays high with a new filter: look for contamination sources (fuel/coolant dilution, soot)."]
        return steps
    steps = [gauge]
    if verdict == "sensor_suspected":
        steps[0] = "FIRST (sensor evidence present): " + gauge + " Replace the sensor only if the gauge disagrees."
    steps.append("Check oil level (dipstick) and confirm the oil grade in the sump matches the configured grade.")
    if direction == "low":
        steps.append("Check the oil filter for restriction (filter dp trend if broadcast; hours since change).")
        if pattern in ("high_rpm_only", "flat_vs_rpm"):
            steps.append("Mechanical: pressure relief/regulator valve (sticking open), then pump capacity/pickup.")
        else:
            steps.append("Mechanical: relief/regulator valve, oil pump and pickup, then bearing clearances "
                         "(an oil analysis can corroborate wear; it is outside this program's inputs).")
    else:
        steps.append("High pressure: check for cold/thick oil and the relief/regulator valve sticking closed.")
    return steps


def explanation(alert: Alert, pattern_text: str, extra: str = "") -> str:
    if pattern_text:
        pattern_text = pattern_text[0].upper() + pattern_text[1:].rstrip(".") + "."
    lbl = SIGNAL_LABEL.get(alert.signal, alert.signal)
    op = alert.operating_point.get("label", "")
    hrs = f"{alert.start_h:.1f}-{alert.end_h:.1f} engine h"
    if alert.layer == "a":
        txt = (f"{lbl} was {'below' if alert.direction == 'low' else 'above'} the configured limit {alert.limit} for "
               f"{alert.duration_s / 60:.1f} min ({alert.n_runs} sustained run(s), {hrs}) at {op}; median "
               f"reading {fmt_value(alert.signal, alert.actual)}. Falsifiable: a calibrated gauge reading "
               f"within the limit at the same operating point means the sensor path, not the engine, is wrong.")
    elif alert.layer == "c":
        txt = (f"{lbl} at {op} read {fmt_value(alert.signal, alert.actual)} against this engine's own learned "
               f"{fmt_value(alert.signal, alert.expected)} (median z {alert.z:+.1f}) for {alert.duration_s / 60:.1f} "
               f"min over {alert.n_runs} sustained run(s), {hrs}. {pattern_text} Falsifiable: the deviation "
               f"should persist at this operating point on the next run; if a mechanical gauge reads the learned "
               f"value, the sensor path is at fault.")
    else:
        txt = (f"{lbl} across {alert.operating_point.get('n_bins', '?')} learned operating bins has drifted "
               f"{alert.operating_point.get('drift_pct', 0):+.1f} % from this engine's early-hours reference "
               f"(trend {alert.operating_point.get('slope_pct_per_100h', 0):+.1f} %/100 h, 95 % CI "
               f"{alert.operating_point.get('slope_ci', '')}), first detected at {alert.start_h:.1f} engine h. "
               f"This is a trend, not a threshold breach. {pattern_text} Falsifiable: if the drift is caused by "
               f"a confounder (oil grade/oil change, sensor replacement, season), it will step rather than ramp "
               f"and will line up with that event.")
    return (txt + (" " + extra if extra else "")).strip()
