"""Per-engine configuration, written from THAT engine's manual.

No limit has a default. A missing limit disables layer (a) for that signal and
the analysis says so; it never substitutes a value from elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .units import converter

LIMIT_UNITS = {
    "oil_pressure_min_idle": "kPa",
    "oil_pressure_min_rated": "kPa",
    "oil_temp_max": "degC",
    "coolant_temp_max": "degC",
    "rated_rpm": "rpm",
    "idle_load_cutoff": "%",
    "coolant_low_temp_cutoff": "degC",
    "oil_low_temp_cutoff": "degC",
    "idle_pct_flag_threshold": "%",
    "lugging_rpm_max": "rpm",
}


@dataclass(frozen=True)
class Limit:
    value: float          # in the canonical unit (LIMIT_UNITS)
    unit: str             # canonical unit
    given: str            # value as written in the config, e.g. "15 psi"
    source: str
    placeholder: bool

    def describe(self) -> str:
        tag = " [PLACEHOLDER - not from this engine's manual]" if self.placeholder else ""
        return f"{self.given} ({self.source}){tag}"


@dataclass(frozen=True)
class Band:
    low: float
    high: float
    source: str
    placeholder: bool


@dataclass
class EngineConfig:
    engine_id: str
    limits: dict[str, Limit] = field(default_factory=dict)
    idle_rpm_band: Band | None = None
    oil_grade: str | None = None
    load_spn: int = 92
    load_fallback_spn: int | None = 513
    description: str = ""
    path: str = ""

    def limit(self, name: str) -> Limit | None:
        return self.limits.get(name)


def _limit(name: str, spec) -> Limit | None:
    if spec is None:
        return None
    if not isinstance(spec, dict) or "value" not in spec or "unit" not in spec:
        raise ValueError(f"limit {name}: needs {{value, unit, source}}")
    if not spec.get("source"):
        raise ValueError(f"limit {name}: `source` is required (manual reference or PLACEHOLDER note)")
    canon = LIMIT_UNITS[name]
    value = float(converter(spec["unit"], canon)(float(spec["value"])))
    return Limit(value, canon, f"{spec['value']} {spec['unit']}", str(spec["source"]),
                 bool(spec.get("placeholder", False)))


def parse_engine_config(doc: dict, path: str = "") -> EngineConfig:
    if "engine_id" not in doc:
        raise ValueError(f"{path or 'engine config'}: engine_id is required")
    raw = doc.get("limits") or {}
    unknown = set(raw) - set(LIMIT_UNITS) - {"idle_rpm_band"}
    if unknown:
        raise ValueError(f"{path}: unknown limits {sorted(unknown)}")
    limits = {}
    for name in LIMIT_UNITS:
        lim = _limit(name, raw.get(name))
        if lim is not None:
            limits[name] = lim
    band = None
    b = raw.get("idle_rpm_band")
    if b is not None:
        if not b.get("source"):
            raise ValueError(f"{path}: idle_rpm_band needs a source")
        conv = converter(b.get("unit", "rpm"), "rpm")
        band = Band(float(conv(b["low"])), float(conv(b["high"])), str(b["source"]), bool(b.get("placeholder")))
    ls = doc.get("load_signal") or {}
    return EngineConfig(
        engine_id=str(doc["engine_id"]), limits=limits, idle_rpm_band=band, oil_grade=doc.get("oil_grade"),
        load_spn=int(ls.get("spn", 92)),
        load_fallback_spn=None if ls.get("fallback_spn", 513) is None else int(ls.get("fallback_spn", 513)),
        description=str(doc.get("description", "")), path=path,
    )


def load_engine_config(path: str | Path) -> EngineConfig:
    with open(path, encoding="utf-8") as fh:
        return parse_engine_config(yaml.safe_load(fh) or {}, str(path))


def load_engine_configs(paths) -> dict[str, EngineConfig]:
    """Load configs from files or a directory (*.yaml). Duplicate engine ids are an error."""
    files: list[Path] = []
    for p in ([paths] if isinstance(paths, (str, Path)) else paths):
        p = Path(p)
        files += sorted(p.glob("*.yaml")) if p.is_dir() else [p]
    out: dict[str, EngineConfig] = {}
    for f in files:
        if f.name.upper().startswith("TEMPLATE"):
            continue
        cfg = load_engine_config(f)
        if cfg.engine_id in out:
            raise ValueError(f"engine {cfg.engine_id} configured twice ({out[cfg.engine_id].path}, {f})")
        out[cfg.engine_id] = cfg
    return out
