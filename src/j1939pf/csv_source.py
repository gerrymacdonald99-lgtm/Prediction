"""Telematics CSV -> the same Sample stream the candump path produces.

Read in chunks (never the whole file). Blank cells mean "no sample at this
timestamp"; explicit `na_values` tokens mean the logger reported "not
available"; `error_values` tokens and non-numeric junk count as bad readings.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import engine_id_for_sa
from .decoder import Sample
from .spn_table import ERROR, MALFORMED, NOT_AVAILABLE, OK, OUT_OF_RANGE, SignalInfo, SpnTable
from .units import converter


@dataclass
class CsvStats:
    rows: int = 0
    bad_timestamps: int = 0
    status_counts: dict[tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))

    def as_dict(self) -> dict:
        return {"rows": self.rows, "bad_timestamps": self.bad_timestamps,
                "status_counts": {f"{e}/{n}": dict(c) for (e, n), c in sorted(self.status_counts.items())}}


@dataclass
class _Target:
    file_name: str
    canonical: str
    convert: object
    info: SignalInfo


def load_mapping(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    with open(path_or_dict, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _targets(entries: dict, table: SpnTable, registry: dict[str, SignalInfo]) -> dict[str, _Target]:
    out = {}
    for file_name, spec in (entries or {}).items():
        if "spn" in spec:
            d = table.by_spn.get(int(spec["spn"]))
            if d is None:
                raise ValueError(f"CSV mapping {file_name!r}: SPN {spec['spn']} not in SPN table")
            canonical = d.canonical
        elif "canonical" in spec:
            canonical = spec["canonical"]
        else:
            raise ValueError(f"CSV mapping {file_name!r}: needs `spn` or `canonical`")
        if canonical not in registry:
            raise ValueError(f"CSV mapping {file_name!r}: unknown canonical signal {canonical!r}")
        info = registry[canonical]
        if "unit" not in spec:
            raise ValueError(f"CSV mapping {file_name!r}: `unit` is required (no unit guessing)")
        out[str(file_name)] = _Target(str(file_name), canonical, converter(spec["unit"], info.unit), info)
    return out


def _parse_timestamps(raw: pd.Series, ts_cfg: dict) -> np.ndarray:
    """Return int64 microseconds since epoch (UTC); -1 where unparseable."""
    fmt = ts_cfg.get("format", "iso8601")
    tz = ts_cfg.get("timezone", "UTC")
    if fmt in ("epoch_s", "epoch_ms", "epoch_us"):
        mult = {"epoch_s": 1_000_000, "epoch_ms": 1_000, "epoch_us": 1}[fmt]
        out = np.full(len(raw), -1, dtype=np.int64)
        for i, s in enumerate(raw.to_numpy()):
            try:
                s = str(s).strip()
                if "." in s and fmt != "epoch_us":
                    whole, frac = s.split(".", 1)
                    digits = {1_000_000: 6, 1_000: 3}[mult]
                    out[i] = int(whole) * mult + int((frac + "0" * digits)[:digits])
                else:
                    out[i] = int(s) * mult
            except (ValueError, KeyError):
                pass
        return out
    kwargs = {"format": "ISO8601"} if fmt == "iso8601" else {"format": fmt}
    parsed = pd.to_datetime(raw, errors="coerce", utc=False, **kwargs)
    if getattr(parsed.dt, "tz", None) is None:
        parsed = parsed.dt.tz_localize(tz)
    parsed = parsed.dt.tz_convert("UTC")
    ns = parsed.astype("datetime64[ns, UTC]").to_numpy(dtype="datetime64[ns]").astype(np.int64)
    us = ns // 1000
    us[parsed.isna().to_numpy()] = -1
    return us


def _engine_ids(chunk: pd.DataFrame, eng_cfg: dict, engine_map: dict | None) -> np.ndarray:
    col = eng_cfg.get("column")
    if not col:
        return np.full(len(chunk), str(eng_cfg.get("default", "SA00")), dtype=object)
    kind = eng_cfg.get("kind", "source_address")
    cache: dict[str, str] = {}
    out = np.empty(len(chunk), dtype=object)
    for i, v in enumerate(chunk[col].to_numpy()):
        v = str(v).strip()
        if v not in cache:
            if kind == "source_address":
                sa = int(v, 16) if v.lower().startswith("0x") else int(float(v))
                cache[v] = engine_id_for_sa(sa, engine_map)
            else:
                cache[v] = str((engine_map or {}).get(v, v))
        out[i] = cache[v]
    return out


def _classify(values: np.ndarray, target: _Target, na_set: set, err_set: set):
    """values: object array of strings. Returns (float array, status list)."""
    n = len(values)
    vals = np.full(n, np.nan)
    status = np.empty(n, dtype=object)
    for i, s in enumerate(values):
        s = "" if s is None else str(s).strip()
        if s == "":
            status[i] = None  # blank: no sample
        elif s in na_set:
            status[i] = NOT_AVAILABLE
        elif s in err_set:
            status[i] = ERROR
        else:
            try:
                vals[i] = float(s)
                status[i] = OK
            except ValueError:
                status[i] = MALFORMED
    ok = status == OK
    if ok.any():
        vals[ok] = target.convert(vals[ok])
        lo, hi = target.info.valid_min - 1e-9, target.info.valid_max + 1e-9
        bad = ok & ((vals < lo) | (vals > hi))
        status[bad] = OUT_OF_RANGE
        vals[bad] = np.nan
    return vals, status


def iter_csv_samples(path: str | Path, mapping, table: SpnTable, registry: dict[str, SignalInfo], *,
                     engine_map: dict | None = None, stats: CsvStats | None = None) -> Iterator[Sample]:
    mp = load_mapping(mapping)
    stats = stats if stats is not None else CsvStats()
    layout = mp.get("layout", "wide")
    ts_cfg = mp.get("timestamp") or {}
    if "column" not in ts_cfg:
        raise ValueError("CSV mapping: timestamp.column is required")
    eng_cfg = mp.get("engine") or {}
    na_set = {str(x) for x in mp.get("na_values", [])}
    err_set = {str(x) for x in mp.get("error_values", [])}
    chunksize = int(mp.get("chunksize", 100_000))
    if layout == "wide":
        targets = _targets(mp.get("columns"), table, registry)
    elif layout == "long":
        targets = _targets(mp.get("signals"), table, registry)
        sig_col, val_col = mp["signal_column"], mp["value_column"]
    else:
        raise ValueError("CSV mapping: layout must be wide|long")

    reader = pd.read_csv(path, chunksize=chunksize, dtype=str, keep_default_na=False)
    for chunk in reader:
        stats.rows += len(chunk)
        t_us = _parse_timestamps(chunk[ts_cfg["column"]], ts_cfg)
        engines = _engine_ids(chunk, eng_cfg, engine_map)
        good_t = t_us >= 0
        stats.bad_timestamps += int((~good_t).sum())
        if layout == "wide":
            missing = [c for c in targets if c not in chunk.columns]
            if missing:
                raise ValueError(f"CSV is missing mapped columns {missing}")
            decoded = {c: _classify(chunk[c].to_numpy(), tg, na_set, err_set) for c, tg in targets.items()}
            for i in range(len(chunk)):
                if not good_t[i]:
                    continue
                for c, tg in targets.items():
                    vals, st = decoded[c]
                    if st[i] is None:
                        continue
                    stats.status_counts[(engines[i], tg.canonical)][st[i]] += 1
                    yield Sample(int(t_us[i]), engines[i], tg.canonical,
                                 None if st[i] != OK else float(vals[i]), st[i])
        else:
            names = chunk[sig_col].to_numpy()
            values = chunk[val_col].to_numpy()
            for i in range(len(chunk)):
                tg = targets.get(str(names[i]).strip())
                if tg is None or not good_t[i]:
                    continue
                vals, st = _classify(np.array([values[i]], dtype=object), tg, na_set, err_set)
                if st[0] is None:
                    continue
                stats.status_counts[(engines[i], tg.canonical)][st[0]] += 1
                yield Sample(int(t_us[i]), engines[i], tg.canonical,
                             None if st[0] != OK else float(vals[0]), st[0])
