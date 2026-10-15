"""Canonical time series: one row per resample interval per engine.

Both the candump path and the CSV path feed the same Sample stream into
CanonicalBuilder, so they produce the same series by construction.

Per interval and signal:
  * valid samples are combined with the signal's aggregate (mean of 20 ms EEC1
    samples, last for counters such as engine hours);
  * an interval with no sample holds the previous value for at most the
    signal's max hold (rate mismatch: 1 s ET1 vs 20 ms EEC1); beyond that NULL;
  * an interval whose only samples were error / not-available is NULL and is
    never filled by hold (a sensor error must not be papered over);
  * `<signal>__err` counts error / reserved / out-of-range samples in the
    interval (column present only when such samples occurred);
  * silence longer than gap_s ends a segment: no rows are produced inside the
    gap and nothing is held across it;
  * samples arriving later than lateness_s behind the newest timestamp (e.g.
    a clock stepping backwards) are dropped and counted, never silently merged.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .decoder import Sample
from .spn_table import BAD_STATUSES, OK, SignalInfo

_NAN = float("nan")


@dataclass
class CanonicalStats:
    rows: dict[str, int] = field(default_factory=dict)
    segments: dict[str, int] = field(default_factory=dict)
    late_samples_dropped: int = 0
    first_late_t_us: int | None = None
    unknown_signal_samples: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class _EngineStore:
    __slots__ = ("bins", "segment_ids", "cols", "n", "last_bin", "segment", "last_obs", "open")

    def __init__(self):
        self.bins = array("q")
        self.segment_ids = array("q")
        self.cols: dict[str, array] = {}
        self.n = 0
        self.last_bin: int | None = None
        self.segment = 0
        self.last_obs: dict[str, tuple[int, float]] = {}
        self.open: dict[int, dict[str, list]] = {}


class CanonicalBuilder:
    def __init__(self, registry: dict[str, SignalInfo], *, resample_interval_s: float = 1.0,
                 gap_s: float = 5.0, lateness_s: float = 2.0, default_max_hold_s: float = 3.0,
                 hold_rate_multiple: float = 3.0):
        self.registry = registry
        self.dt = int(round(resample_interval_s * 1e6))
        if self.dt <= 0:
            raise ValueError("resample_interval_s must be > 0")
        self.gap_bins = gap_s * 1e6 / self.dt
        self.lateness = int(lateness_s * 1e6)
        self.hold_bins: dict[str, float] = {}
        for name, info in registry.items():
            hold_s = info.max_hold_s
            if hold_s is None:
                rate_s = (info.rate_ms or 0) / 1000.0
                hold_s = max(default_max_hold_s, hold_rate_multiple * rate_s)
            self.hold_bins[name] = hold_s * 1e6 / self.dt
        self.order = list(registry)
        self.engines: dict[str, _EngineStore] = {}
        self.stats = CanonicalStats()
        self.max_t: int | None = None
        self._watermark_bin: int | None = None

    # -- input --------------------------------------------------------------
    def add(self, s: Sample) -> None:
        if s.name not in self.registry:
            self.stats.unknown_signal_samples += 1
            return
        if self.max_t is None or s.t_us > self.max_t:
            self.max_t = s.t_us
            wm = (self.max_t - self.lateness) // self.dt
            if self._watermark_bin is None or wm > self._watermark_bin:
                self._watermark_bin = wm
                self._flush(wm)
        b = s.t_us // self.dt
        st = self.engines.get(s.engine)
        if st is None:
            st = self.engines[s.engine] = _EngineStore()
        if (self._watermark_bin is not None and b < self._watermark_bin) or (
                st.last_bin is not None and b <= st.last_bin):
            self.stats.late_samples_dropped += 1
            if self.stats.first_late_t_us is None:
                self.stats.first_late_t_us = s.t_us
            return
        acc = st.open.setdefault(b, {}).get(s.name)
        if acc is None:
            # [sum, n_ok, last, min, max, n_bad, n_na]
            acc = st.open[b][s.name] = [0.0, 0, _NAN, math.inf, -math.inf, 0, 0]
        if s.status == OK and s.value is not None:
            v = float(s.value)
            acc[0] += v
            acc[1] += 1
            acc[2] = v
            if v < acc[3]:
                acc[3] = v
            if v > acc[4]:
                acc[4] = v
        elif s.status in BAD_STATUSES:
            acc[5] += 1
        else:
            acc[6] += 1

    def add_many(self, samples: Iterable[Sample]) -> None:
        for s in samples:
            self.add(s)

    # -- flushing -----------------------------------------------------------
    def _flush(self, below_bin: int | None) -> None:
        for engine, st in self.engines.items():
            if not st.open:
                continue
            ready = sorted(b for b in st.open if below_bin is None or b < below_bin)
            for b in ready:
                self._emit_bin(engine, st, b, st.open.pop(b))

    def _emit_bin(self, engine: str, st: _EngineStore, b: int, accs: dict[str, list]) -> None:
        if st.last_bin is not None:
            if b - st.last_bin > self.gap_bins:
                st.segment += 1
                st.last_obs.clear()
            else:
                for mb in range(st.last_bin + 1, b):
                    self._append_row(st, mb, {})
        self._append_row(st, b, accs)
        st.last_bin = b

    def _append_row(self, st: _EngineStore, b: int, accs: dict[str, list]) -> None:
        row: dict[str, float] = {}
        no_hold = set()
        for name, acc in accs.items():
            if acc[1] > 0:
                agg = self.registry[name].aggregate
                v = (acc[0] / acc[1] if agg == "mean" else acc[2] if agg == "last"
                     else acc[4] if agg == "max" else acc[3])
                row[name] = v
                st.last_obs[name] = (b, v)
            else:
                # Present column, but the only samples were error / not-available.
                row[name] = _NAN
                no_hold.add(name)
                st.last_obs.pop(name, None)
            if acc[5]:
                row[name + "__err"] = float(acc[5])
        for name, (lb, v) in list(st.last_obs.items()):
            if name in row or name in no_hold:
                continue
            if b - lb <= self.hold_bins[name]:
                row[name] = v
            else:
                del st.last_obs[name]
        n = st.n
        for name, v in row.items():
            col = st.cols.get(name)
            if col is None:
                fill = 0.0 if name.endswith("__err") else _NAN
                col = st.cols[name] = array("d", [fill]) * n
            col.append(v)
        for name, col in st.cols.items():
            if len(col) == n:
                col.append(0.0 if name.endswith("__err") else _NAN)
        st.bins.append(b)
        st.segment_ids.append(st.segment)
        st.n = n + 1

    # -- output -------------------------------------------------------------
    def finalize(self) -> pd.DataFrame:
        self._flush(None)
        frames = []
        all_cols: set[str] = set()
        for st in self.engines.values():
            all_cols.update(st.cols)
        sig_cols = [c for c in self.order if c in all_cols]
        err_cols = [c + "__err" for c in self.order if c + "__err" in all_cols]
        for engine in sorted(self.engines):
            st = self.engines[engine]
            if st.n == 0:
                continue
            data = {
                "engine_id": np.full(st.n, engine, dtype=object),
                "t": pd.to_datetime(np.frombuffer(st.bins, dtype=np.int64) * self.dt, unit="us", utc=True),
                "segment": np.frombuffer(st.segment_ids, dtype=np.int64).copy(),
            }
            for c in sig_cols:
                data[c] = np.frombuffer(st.cols[c], dtype=float).copy() if c in st.cols else np.full(st.n, np.nan)
            for c in err_cols:
                data[c] = (np.frombuffer(st.cols[c], dtype=float).astype(np.int64)
                           if c in st.cols else np.zeros(st.n, dtype=np.int64))
            frames.append(pd.DataFrame(data))
            self.stats.rows[engine] = st.n
            self.stats.segments[engine] = st.segment + 1
        if not frames:
            return pd.DataFrame(columns=["engine_id", "t", "segment"])
        return pd.concat(frames, ignore_index=True)
