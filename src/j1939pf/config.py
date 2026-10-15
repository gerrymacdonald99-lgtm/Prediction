"""Loading of program defaults (tunable, NOT engine specifications)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_SPN_TABLE = CONFIG_DIR / "spn_table.yaml"
DEFAULTS_PATH = CONFIG_DIR / "defaults.yaml"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_defaults(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    with open(DEFAULTS_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
    return _merge(cfg, overrides or {})


def engine_id_for_sa(sa: int, engine_map: dict | None) -> str:
    """Each source address is its own engine unless the config maps it."""
    if engine_map:
        for key in (sa, str(sa), f"0x{sa:02X}", f"0x{sa:02x}"):
            if key in engine_map:
                return str(engine_map[key])
    return f"SA{sa:02X}"
