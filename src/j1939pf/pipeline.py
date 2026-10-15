"""Entry points: log file -> canonical series + DTC episodes + parse report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .candump import ReadStats, iter_candump_frames
from .canonical import CanonicalBuilder
from .config import load_defaults
from .csv_source import CsvStats, iter_csv_samples
from .decoder import J1939Decoder
from .spn_table import SpnTable, build_registry, load_spn_table


@dataclass
class ParseResult:
    canonical: pd.DataFrame
    dtc_episodes: pd.DataFrame
    report: dict = field(default_factory=dict)


def _builder(table: SpnTable, cfg: dict) -> CanonicalBuilder:
    return CanonicalBuilder(build_registry(table), **cfg["canonical"])


def _episodes_frame(episodes) -> pd.DataFrame:
    cols = ["engine_id", "source_address", "kind", "spn", "fmi", "first_t", "last_t", "max_oc", "messages"]
    rows = [{
        "engine_id": e.engine, "source_address": e.source_address, "kind": e.kind, "spn": e.spn,
        "fmi": e.fmi, "first_t": pd.Timestamp(e.first_t_us, unit="us", tz="UTC"),
        "last_t": pd.Timestamp(e.last_t_us, unit="us", tz="UTC"), "max_oc": e.max_oc,
        "messages": e.messages,
    } for e in episodes]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["engine_id", "first_t", "spn", "fmi"], ignore_index=True) if len(df) else df


def _unverified_note(table: SpnTable) -> dict:
    return {"spn_table": table.source, "unverified_spns": [d.spn for d in table.unverified],
            "warning": "Decoded values come from SPN entries not yet verified against J1939DA."
            if table.unverified else ""}


def canonical_from_candump(source, *, table: SpnTable | None = None, config: dict | None = None) -> ParseResult:
    cfg = config or load_defaults()
    table = table or load_spn_table()
    read_stats = ReadStats()
    dec = J1939Decoder(
        table,
        engine_map=cfg["sources"].get("engine_map"),
        ignore_source_addresses=cfg["sources"].get("ignore_source_addresses", ()),
        duplicate_window_s=cfg["frames"]["duplicate_window_s"],
        transport=cfg["transport"],
        dtc_episode_gap_s=cfg["dtc"]["episode_gap_s"],
    )
    builder = _builder(table, cfg)
    builder.add_many(dec.iter_samples(iter_candump_frames(source, read_stats)))
    df = builder.finalize()
    report = {
        "source": str(source) if isinstance(source, (str, Path)) else "<lines>",
        "kind": "candump",
        "read": read_stats.as_dict(),
        "decode": dec.stats.as_dict(),
        "transport": dec.tp.stats.as_dict(),
        "canonical": builder.stats.as_dict(),
        **_unverified_note(table),
    }
    return ParseResult(df, _episodes_frame(dec.dtc_episodes), report)


def canonical_from_csv(path, mapping, *, table: SpnTable | None = None, config: dict | None = None) -> ParseResult:
    cfg = config or load_defaults()
    table = table or load_spn_table()
    registry = build_registry(table)
    stats = CsvStats()
    builder = _builder(table, cfg)
    builder.add_many(iter_csv_samples(path, mapping, table, registry,
                                      engine_map=cfg["sources"].get("engine_map"), stats=stats))
    df = builder.finalize()
    report = {"source": str(path), "kind": "csv", "csv": stats.as_dict(),
              "canonical": builder.stats.as_dict(), **_unverified_note(table),
              "note": "CSV input carries no DM1 frames; DTC episodes are empty unless DM1 columns are mapped."}
    return ParseResult(df, _episodes_frame([]), report)
