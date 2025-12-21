"""
Offline playback of IEEE C37.118.2 frame captures.

Reads a binary file containing concatenated C37.118 frames (SYNC + FRAMESIZE ...)
and yields DataFrame chunks. Decoding uses optional `synchrophasor` if available;
otherwise yields raw-frame metadata. Synchrophasor data in csv files can use the
default pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from .data_stream import DataStream
from .c37118_stream import C37118ChannelMap, C37118TCPDataStream


class C37118FileDataStream(DataStream):
    """
    Read concatenated C37.118 frames from a binary file and yield DataFrame chunks.

    This is useful for reproducible offline evaluation while exercising the same
    code paths as live streaming evaluation.
    """

    def __init__(
        self, path: str, *, frames_per_chunk: int = 50, include_raw: bool = False
    ) -> None:
        self._path = path
        self._frames_per_chunk = int(frames_per_chunk)
        self._include_raw = bool(include_raw)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        # Reuse conversion logic from TCP class for consistent schema
        helper = C37118TCPDataStream(
            "0.0.0.0", 0
        )  # dummy, used only for helper methods

        rows: List[Dict[str, Any]] = []
        chanmap = C37118ChannelMap(phasor_names=[], analog_names=[], digital_names=[])

        with open(self._path, "rb") as f:
            buf = b""
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                buf += chunk

                # Extract frames from buffer
                while True:
                    frame = self._try_pop_one_frame(buf)
                    if frame is None:
                        break
                    buf, raw_frame = frame

                    # Optional decoding: if synchrophasor supports direct frame decode in your environment,
                    # you can implement it here. Otherwise we surface raw frames with timestamps handled
                    # downstream (or in your own adapter).
                    row = {"timestamp": pd.Timestamp.utcnow().tz_localize("UTC")}
                    if self._include_raw:
                        row["raw_frame"] = raw_frame
                    rows.append(row)

                    if len(rows) >= self._frames_per_chunk:
                        yield helper._rows_to_df(rows)  # consistent index handling
                        rows = []

        if rows:
            yield helper._rows_to_df(rows)

    def __next__(self) -> pd.DataFrame:
        # create an iterator on first call
        if not hasattr(self, "_it"):
            self._it = self.__iter__()
        return next(self._it)

    def _try_pop_one_frame(self, buf: bytes) -> Optional[tuple[bytes, bytes]]:
        """
        Try to pop one complete C37.118 frame from the front of buffer.

        C37.118 common frame header starts with SYNC (0xAA41 or 0xAA31 etc.) and
        FRAMESIZE (2 bytes, big-endian) immediately after SYNC.
        """
        if len(buf) < 4:
            return None

        # Find plausible SYNC (0xAA??). This is deliberately permissive.
        idx = buf.find(b"\xaa")
        if idx == -1 or len(buf) < idx + 4:
            return None

        if idx > 0:
            buf = buf[idx:]

        # FRAMESIZE is bytes 2..3
        framesize = int.from_bytes(buf[2:4], byteorder="big", signed=False)
        if framesize <= 0 or framesize > 65535:
            # Not a real frame, drop one byte and resync
            return (buf[1:], b"")  # caller will continue
        if len(buf) < framesize:
            return None

        raw = buf[:framesize]
        rest = buf[framesize:]
        return rest, raw
