"""Readers for the event log and for canonical CSV exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KNOWN_EVENTS = ("oil_change", "oil_grade_change", "sensor_replacement")


def load_events(path: str | Path) -> pd.DataFrame:
    """Maintenance / failure log CSV.

    Required columns: engine_id, engine_hours, event_type. Optional: t (timestamp), detail, signal.
    event_type: oil_change | oil_grade_change | sensor_replacement | failure | failure_<component>.
    Other types are kept but not used.
    """
    df = pd.read_csv(path)
    miss = {"engine_id", "engine_hours", "event_type"} - set(df.columns)
    if miss:
        raise ValueError(f"{path}: event log is missing columns {sorted(miss)}")
    df["engine_id"] = df["engine_id"].astype(str)
    df["engine_hours"] = pd.to_numeric(df["engine_hours"], errors="raise")
    df["t"] = pd.to_datetime(df["t"], utc=True) if "t" in df else pd.NaT
    if "detail" not in df:
        df["detail"] = ""
    df["detail"] = df["detail"].fillna("")
    return df


def is_canonical_csv(path: str | Path) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            head = fh.readline().strip().split(",")
    except (UnicodeDecodeError, OSError):
        return False
    return head[:3] == ["engine_id", "t", "segment"]


def load_canonical(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["engine_id"] = df["engine_id"].astype(str)
    df["t"] = pd.to_datetime(df["t"], utc=True, format="ISO8601")
    return df
