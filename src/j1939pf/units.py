"""Unit conversion into the canonical (SPN table) unit. Only exact definitional factors."""

from __future__ import annotations

import numpy as np

PSI_TO_KPA = 6.894757293168361
GAL_TO_L = 3.785411784  # US gallon

_ALIASES = {
    "kpa": "kPa", "psi": "psi", "bar": "bar",
    "degc": "degC", "c": "degC", "°c": "degC",
    "degf": "degF", "f": "degF", "°f": "degF", "k": "K",
    "rpm": "rpm", "%": "%", "pct": "%", "percent": "%",
    "l/h": "L/h", "lph": "L/h", "gal/h": "gal/h", "gph": "gal/h",
    "l": "L", "gal": "gal", "v": "V", "a": "A", "h": "h", "hr": "h",
    "state": "state", "count": "count", "km/l": "km/L",
}

_TO_CANON = {
    ("psi", "kPa"): lambda x: x * PSI_TO_KPA,
    ("bar", "kPa"): lambda x: x * 100.0,
    ("degF", "degC"): lambda x: (x - 32.0) * 5.0 / 9.0,
    ("K", "degC"): lambda x: x - 273.15,
    ("gal/h", "L/h"): lambda x: x * GAL_TO_L,
    ("gal", "L"): lambda x: x * GAL_TO_L,
}


def normalize_unit(unit: str) -> str:
    key = str(unit).strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"unknown unit {unit!r}; known: {sorted(set(_ALIASES.values()))}")
    return _ALIASES[key]


def converter(from_unit: str, to_unit: str):
    """Return f(np.ndarray) -> np.ndarray converting from_unit to to_unit."""
    a, b = normalize_unit(from_unit), normalize_unit(to_unit)
    if a == b:
        return lambda x: x
    if (a, b) not in _TO_CANON:
        raise ValueError(f"no conversion from {a} to {b}")
    fn = _TO_CANON[(a, b)]
    return lambda x: fn(np.asarray(x, dtype=float))
