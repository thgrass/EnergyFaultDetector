"""
IEEE C37.118.2 TCP/IP streaming sources.

This module provides DataStream implementations that connect to a PMU/PDC stream
and yield pandas DataFrames suitable for online evaluation in StreamingFaultDetector.

Optional dependency: `synchrophasor` (pyPMU, BSD-3-Clause; https://github.com/iicsys/pypmu).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import time

import pandas as pd

from .data_stream import DataStream


@dataclass(frozen=True)
class C37118ChannelMap:
    """Resolved channel naming for an incoming C37.118.2 stream."""

    phasor_names: Sequence[str]
    analog_names: Sequence[str]
    digital_names: Sequence[str]
    frequency_name: str = "frequency"
    rocof_name: str = "rocof"


class C37118TCPDataStream(DataStream):
    """
    Live IEEE C37.118.2 TCP stream.

    Uses the optional `synchrophasor` package (pyPMU) to connect to a PMU/PDC,
    obtain configuration, then read DATA frames and convert them into DataFrame rows.

    The yielded DataFrames are chunked by `frames_per_chunk` and indexed by timestamp.

    Notes:
      - If `synchrophasor` is not installed, this class raises an ImportError
        with a clear remediation message.
      - Mapping of channels (names/order) is derived from the configuration frame.
    """

    def __init__(
        self,
        pmu_ip: str,
        pmu_port: int,
        *,
        pdc_id: int = 1,
        frames_per_chunk: int = 50,
        poll_sleep_s: float = 0.0,
        include_raw: bool = False,
    ) -> None:
        """
        Args:
            pmu_ip: IP address of the PMU/PDC source (TCP).
            pmu_port: TCP port of the PMU/PDC source.
            pdc_id: C37.118 PDC ID used by client.
            frames_per_chunk: How many DATA frames to aggregate per yielded chunk.
            poll_sleep_s: Optional sleep between polls to reduce CPU load.
            include_raw: If True, include a `raw_frame` column with original objects.
        """
        self._pmu_ip = pmu_ip
        self._pmu_port = pmu_port
        self._pdc_id = int(pdc_id)
        self._frames_per_chunk = int(frames_per_chunk)
        self._poll_sleep_s = float(poll_sleep_s)
        self._include_raw = bool(include_raw)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """
        Connect to the stream and yield DataFrame chunks.

        Yields:
            DataFrames with a DateTimeIndex `timestamp` and columns for channels.
        """
        try:
            # pyPMU / synchrophasor
            from synchrophasor.pdc import Pdc  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "C37118TCPDataStream requires optional dependency 'synchrophasor'. "
                "Install with: pip install synchrophasor (or EnergyFaultDetector[c37118])."
            ) from e

        pdc = Pdc(pdc_id=self._pdc_id, pmu_ip=self._pmu_ip, pmu_port=self._pmu_port)
        pdc.run()

        # Attempt to fetch header/config.
        # `synchrophasor` supports these calls in the TinyPDC example. :contentReference[oaicite:1]{index=1}
        _ = pdc.get_header()
        cfg = pdc.get_config()

        chanmap = self._resolve_channel_map(cfg)
        pdc.start()

        rows: List[Dict[str, Any]] = []

        try:
            while True:
                frame = pdc.get()
                if not frame:
                    break

                row = self._frame_to_row(frame, chanmap)
                if self._include_raw:
                    row["raw_frame"] = frame

                rows.append(row)

                if len(rows) >= self._frames_per_chunk:
                    yield self._rows_to_df(rows)
                    rows = []

                if self._poll_sleep_s > 0.0:
                    time.sleep(self._poll_sleep_s)
        finally:
            # Ensure connection is closed cleanly
            try:
                pdc.quit()
            except Exception:
                pass

        if rows:
            yield self._rows_to_df(rows)

    def _rows_to_df(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of row dicts into a DataFrame with timestamp index."""
        df = pd.DataFrame(rows)
        # prefer timestamp as index if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        return df

    def _resolve_channel_map(self, cfg: Any) -> C37118ChannelMap:
        """
        Resolve channel names from a C37.118 configuration frame.

        This function is defensive: `synchrophasor` may represent config structures
        differently depending on version. If names cannot be extracted, it generates
        stable fallback names.
        """
        # Best-effort extraction of channel names
        ph = self._try_get_names(cfg, keys=("phasor_names", "phasors", "phnames"))
        an = self._try_get_names(cfg, keys=("analog_names", "analogs", "annames"))
        dg = self._try_get_names(cfg, keys=("digital_names", "digitals", "dgnames"))

        if not ph:
            ph = [
                f"phasor_{i}"
                for i in range(self._infer_count(cfg, "phasor", default=0))
            ]
        if not an:
            an = [
                f"analog_{i}"
                for i in range(self._infer_count(cfg, "analog", default=0))
            ]
        if not dg:
            dg = [
                f"digital_{i}"
                for i in range(self._infer_count(cfg, "digital", default=0))
            ]

        return C37118ChannelMap(phasor_names=ph, analog_names=an, digital_names=dg)

    def _try_get_names(self, cfg: Any, keys: Tuple[str, ...]) -> List[str]:
        """Try multiple attribute/dict keys to extract a list of channel names."""
        for k in keys:
            if isinstance(cfg, dict) and k in cfg and isinstance(cfg[k], (list, tuple)):
                return [str(x) for x in cfg[k]]
            if hasattr(cfg, k):
                v = getattr(cfg, k)
                if isinstance(v, (list, tuple)):
                    return [str(x) for x in v]
        return []

    def _infer_count(self, cfg: Any, kind: str, default: int = 0) -> int:
        """Try to infer channel counts from config object; fallback to default."""
        for k in (f"num_{kind}s", f"{kind}_count", f"n_{kind}"):
            if hasattr(cfg, k):
                try:
                    return int(getattr(cfg, k))
                except Exception:
                    pass
            if isinstance(cfg, dict) and k in cfg:
                try:
                    return int(cfg[k])
                except Exception:
                    pass
        return default

    def _frame_to_row(self, frame: Any, chanmap: C37118ChannelMap) -> Dict[str, Any]:
        """
        Convert a C37.118 DATA frame to a flat row.

        This extracts:
          - timestamp (UTC)
          - frequency and rocof (if present)
          - phasors as magnitude/angle pairs: <name>_mag, <name>_ang
          - analogs and digitals (if present)

        If the frame object exposes different attribute names, this function tries
        a set of common ones.
        """
        row: Dict[str, Any] = {}

        # Timestamp extraction (best effort)
        ts = (
            getattr(frame, "timestamp", None)
            or getattr(frame, "time", None)
            or getattr(frame, "soc", None)
        )
        row["timestamp"] = self._coerce_timestamp(ts)

        # Frequency / ROCOF
        freq = getattr(frame, "freq", None) or getattr(frame, "frequency", None)
        rocof = getattr(frame, "dfreq", None) or getattr(frame, "rocof", None)
        if freq is not None:
            row[chanmap.frequency_name] = float(freq)
        if rocof is not None:
            row[chanmap.rocof_name] = float(rocof)

        # Phasors (expect iterable of complex or (mag, ang) tuples)
        phasors = getattr(frame, "phasors", None) or getattr(frame, "phasor", None)
        if phasors is not None:
            self._add_phasors(row, phasors, chanmap.phasor_names)

        # Analogs / Digitals
        analogs = getattr(frame, "analogs", None) or getattr(frame, "analog", None)
        digitals = getattr(frame, "digitals", None) or getattr(frame, "digital", None)
        if analogs is not None:
            self._add_scalars(row, analogs, chanmap.analog_names, prefix="analog")
        if digitals is not None:
            self._add_scalars(row, digitals, chanmap.digital_names, prefix="digital")

        return row

    def _coerce_timestamp(self, ts: Any) -> pd.Timestamp:
        """Convert various timestamp representations into a UTC pandas Timestamp."""
        if ts is None:
            return pd.Timestamp.utcnow().tz_localize("UTC")
        if isinstance(ts, pd.Timestamp):
            return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        # seconds since epoch
        try:
            return pd.to_datetime(float(ts), unit="s", utc=True)
        except Exception:
            return pd.Timestamp.utcnow().tz_localize("UTC")

    def _add_phasors(
        self, row: Dict[str, Any], phasors: Any, names: Sequence[str]
    ) -> None:
        """Add phasor mag/ang columns, best-effort parsing of object types."""
        try:
            seq = list(phasors)
        except Exception:
            return

        for i, p in enumerate(seq):
            name = names[i] if i < len(names) else f"phasor_{i}"
            mag, ang = self._phasor_to_mag_ang(p)
            row[f"{name}_mag"] = mag
            row[f"{name}_ang"] = ang

    def _phasor_to_mag_ang(self, p: Any) -> Tuple[float, float]:
        """Convert complex or (mag, ang) into magnitude/angle (radians)."""
        # complex phasor
        if isinstance(p, complex):
            import math

            return float(abs(p)), float(math.atan2(p.imag, p.real))
        # tuple/list
        if isinstance(p, (tuple, list)) and len(p) >= 2:
            return float(p[0]), float(p[1])
        # unknown numeric -> magnitude only
        try:
            return float(p), 0.0
        except Exception:
            return 0.0, 0.0

    def _add_scalars(
        self, row: Dict[str, Any], values: Any, names: Sequence[str], prefix: str
    ) -> None:
        """Add scalar channels (analogs/digitals) using stable names."""
        try:
            seq = list(values)
        except Exception:
            return

        for i, v in enumerate(seq):
            name = names[i] if i < len(names) else f"{prefix}_{i}"
            try:
                row[name] = float(v)
            except Exception:
                row[name] = v
