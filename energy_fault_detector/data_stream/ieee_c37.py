"""Implementation of IEEE C37.118 synchrophasor data sources."""

from __future__ import annotations
from typing import Iterator, Optional, Dict, Any
import datetime

import pandas as pd

from .base import StreamDataSource, StreamChunk


# Optional dependency: synchrophasor (pyPMU)
try:
    from synchrophasor.pdc import Pdc  # type: ignore
except Exception:  # ImportError and friends
    Pdc = None  # type: ignore[assignment]


class IeeeC37StreamSource(StreamDataSource):
    """
    Stream IEEE C37.118 synchrophasor data from a PMU/PDC using the
    'synchrophasor' (pyPMU) library.

    This class is only usable if 'synchrophasor' is installed.
    Importing the module is always safe; an ImportError is raised lazily
    if you try to instantiate the class without the dependency.
    """

    def __init__(
        self,
        pmu_ip: str,
        pmu_port: int,
        pdc_id: int = 1,
        chunk_size: int = 100,
        timeout_s: Optional[float] = None,
    ) -> None:
        if Pdc is None:
            raise ImportError(
                "IeeeC37StreamSource requires the 'synchrophasor' package "
                "(pyPMU). Install it via:\n"
                "    pip install synchrophasor\n"
            )
        self.pmu_ip = pmu_ip
        self.pmu_port = pmu_port
        self.pdc_id = pdc_id
        self.chunk_size = int(chunk_size)
        self.timeout_s = timeout_s

    def __iter__(self) -> Iterator[StreamChunk]:
        # type: ignore[arg-type] because Pdc is Optional at type level
        pdc = Pdc(
            pdc_id=self.pdc_id,
            pmu_ip=self.pmu_ip,
            pmu_port=self.pmu_port,
            timeout=self.timeout_s,
            udp=False,
        )

        pdc.run()
        _cfg = pdc.get_config()
        pdc.start()

        rows: list[Dict[str, Any]] = []

        try:
            while True:
                frame = pdc.get()
                if frame is None:
                    break  # connection closed or timeout

                row: Dict[str, Any] = {}

                # These attributes depend on pyPMU version;
                # adjust once you inspect a real frame.
                ts = getattr(frame, "time", None)
                freq = getattr(frame, "freq", None)
                phasors = getattr(frame, "phasors", None)

                if isinstance(ts, datetime.datetime):
                    row["timestamp"] = ts
                elif ts is not None:
                    row["timestamp"] = ts

                if freq is not None:
                    try:
                        row["Frequency"] = float(freq)
                    except (TypeError, ValueError):
                        row["Frequency"] = freq

                if phasors is not None:
                    for i, ph in enumerate(phasors):
                        try:
                            mag, ang = ph
                        except Exception:
                            continue
                        row[f"phasor_{i + 1}_mag"] = float(mag)
                        row[f"phasor_{i + 1}_ang"] = float(ang)

                rows.append(row)

                if len(rows) >= self.chunk_size:
                    df = pd.DataFrame(rows)
                    yield StreamChunk(data=df)
                    rows = []
        finally:
            try:
                pdc.quit()
            except Exception:
                pass

        if rows:
            df = pd.DataFrame(rows)
            yield StreamChunk(data=df)
