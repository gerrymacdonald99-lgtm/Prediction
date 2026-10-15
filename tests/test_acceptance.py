"""Final acceptance checklist. Each item points at the test(s) that prove it; items not fully covered
elsewhere get a direct check here."""

import re
from pathlib import Path

import pytest
import yaml

from j1939pf.config import CONFIG_DIR
from j1939pf.framing import HONEST_FRAMING

ROOT = Path(__file__).resolve().parents[1]

CHECKLIST = {
    "candump and CSV produce the same canonical series": "test_equivalence.py::test_candump_and_csv_same_canonical",
    "not-available/error encodings ignored": "test_canonical.py::test_na_and_error_encodings_in_frames_are_ignored",
    "per-engine baseline; low-pressure-at-high-RPM flagged with operating point and explanation":
        "test_baseline_alerts.py::test_stuck_relief_valve_low_pressure_at_high_rpm",
    "slow trend detected at fixed operating point, flagged before any fixed threshold":
        "test_baseline_alerts.py::test_bearing_wear_drift_flagged_before_threshold",
    "no alerts on healthy synthetic data": "test_baseline_alerts.py::test_zero_alerts_on_healthy_engines",
    "health score always has its decomposition": "test_phase5.py::test_health_score_always_has_its_decomposition",
    "no invented specs or life-loss formulas": "test_acceptance.py (below)",
    "idle % matches hand-checked fixture": "test_usage.py::test_idle_pct_matches_hand_computed_golden_fixture",
    "cold-operation flagged, no false flags on warm data":
        "test_usage.py::test_cold_operation_is_flagged + test_no_cold_flags_on_warm_engines",
    "new engine states its learning period; baselines are its own":
        "test_baseline_alerts.py::test_new_engine_announces_learning_period_and_judges_only_layer_a + "
        "test_baselines_are_per_engine_and_never_copied",
    "oil change / sensor replacement re-baselines without alert flood":
        "test_phase4.py::test_no_alert_flood_after_sensor_replacement_or_grade_change",
    "dismissed alert logged with reason, no re-fire without new evidence":
        "test_phase4.py::test_dismissed_alert_stays_quiet_and_is_persisted + "
        "test_dismissed_pattern_refires_only_on_new_evidence",
    "correlation report includes the newer signals": "test_phase4.py::test_correlation_report_complete_with_newer_signals",
    "all tests pass; README documents limitations honestly": "this file + the whole suite",
}


def test_every_checklist_item_points_at_an_existing_test():
    for item, ref in CHECKLIST.items():
        for part in re.split(r"\s\+\s", ref):
            if "::" not in part:
                continue
            f, name = part.split("::")
            src = (ROOT / "tests" / f.strip()).read_text() if "/" not in f else ""
            assert f"def {name.strip()}" in src, (item, part)


LIFE_LOSS = re.compile(r"(life (lost|consumed|used up)|% of (engine )?life|percent of life|life[- ]?loss (formula|model))",
                       re.I)
DISCLAIMER = re.compile(r"(never|not|no |nothing|without|cannot|NOT|is not)", re.I)


def test_no_life_loss_formula_anywhere_in_code_docs_or_config():
    files = list((ROOT / "src").rglob("*.py")) + list(CONFIG_DIR.rglob("*.yaml")) + [ROOT / "README.md"]
    offenders = []
    for f in files:
        lines = f.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            context = (lines[i - 2] + " " + line) if i > 1 else line   # disclaimers wrap across lines
            if LIFE_LOSS.search(line) and not DISCLAIMER.search(context):
                offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert offenders == []


def test_no_invented_engine_specs():
    defaults = yaml.safe_load((CONFIG_DIR / "defaults.yaml").read_text())
    limit_words = ("oil_pressure_min", "temp_max", "rated_rpm", "idle_rpm_band", "lugging_rpm_max")
    flat = yaml.safe_dump(defaults)
    assert not any(w in flat for w in limit_words)                 # program defaults hold no engine limits
    for f in (CONFIG_DIR / "engines").glob("example_*.yaml"):
        doc = yaml.safe_load(f.read_text())
        for name, lim in (doc.get("limits") or {}).items():
            if lim is not None:
                assert lim.get("placeholder") is True and "PLACEHOLDER" in lim["source"], (f.name, name)
        assert "MUST come from the manual" in f.read_text() or "override" in f.read_text().lower()
    tmpl = yaml.safe_load((CONFIG_DIR / "engines" / "TEMPLATE.yaml").read_text())
    assert all(v is None for v in tmpl["limits"].values())
    spn = yaml.safe_load((CONFIG_DIR / "spn_table.yaml").read_text())
    assert all(e["verified"] is False for e in spn["signals"])


def test_readme_documents_framing_and_limitations():
    readme = (ROOT / "README.md").read_text()
    assert HONEST_FRAMING.split(".")[0] in readme
    for must in ("Synthetic-data caveat", "Honest limitations", "verify", "exposure", "not life", "learning period",
                 "j1939pf analyze", "engine config"):
        assert must.lower() in readme.lower(), must
