"""
Generic windowing data stream.

This module implements a windowing adapter that converts a row‑oriented
data source into overlapping or strided windows suitable for
online fault detection.  Unlike the original synchrophasor
implementation, this version does not decode IEEE C37.118.2 frames
directly and is agnostic to the source of the underlying rows.  It
can wrap any :class:`~energy_fault_detector.streaming.data_stream.DataStream`
whose ``__next__`` method returns a :class:`pandas.DataFrame` with one
or more rows.  It can also wrap a :class:`pandas.DataFrame` or load
data from a CSV file and treat each row as an individual timestamped
sample.

The primary purpose of this class is to provide sliding (or strided)
windows over streaming data.  Many anomaly‑detection models require
fixed‑length time windows as input rather than single samples; this
adapter assembles such windows on the fly.

Example usage:

.. code:: python

    from energy_fault_detector import StreamingFaultDetector
    from energy_fault_detector.streaming.c37118_stream import C37118TCPDataStream
    from energy_fault_detector.streaming.windowed_data_stream import WindowedDataStream

    live_rows = C37118TCPDataStream("192.168.1.10", 4712, frames_per_chunk=1)
    # Wrap the live per‑row stream into fixed windows of 50 samples
    windowed = WindowedDataStream(source=live_rows, window_size=50, step_size=1)
    detector = StreamingFaultDetector.from_config("path/to/config.yaml")

    for window_df in windowed:
        result = detector.predict(window_df)
        print(result)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd

from .data_stream import DataStream


@dataclass
class WindowedDataStream(DataStream):
    """Windowing adapter for streaming data sources.

    A ``WindowedDataStream`` wraps an underlying row‑oriented source
    (either a :class:`pandas.DataFrame` or another
    :class:`~energy_fault_detector.streaming.data_stream.DataStream` that
    yields DataFrames) and produces overlapping or strided windows of
    fixed length.  This is useful for feeding models that operate on
    sequences of time‑indexed measurements.

    Parameters:
        source: A data source that yields DataFrames with rows of
            measurements.  Each call to ``next(source)`` should return
            a DataFrame containing one or more rows.  If ``source``
            is a DataFrame, it is treated as a static dataset and
            internally wrapped into a streaming source.
        window_size: Number of rows per output window.
        step_size: Stride between consecutive windows (defaults to 1,
            meaning the window slides by one row each time).
    """

    source: object
    window_size: int
    step_size: int = 1

    def __post_init__(self) -> None:
        # If a DataFrame is provided, wrap it into a simple stream
        if isinstance(self.source, pd.DataFrame):
            self.source = _DataFrameRowStream(self.source)
        # Validate that source behaves like a DataStream (iterable of DataFrames)
        if not hasattr(self.source, "__iter__"):
            raise TypeError(
                "source must be a DataFrame or DataStream yielding DataFrames"
            )
        self._source_iter = iter(self.source)
        # Internal buffer accumulating rows
        self._buffer: pd.DataFrame = pd.DataFrame()

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        window_size: int,
        step_size: int = 1,
        timestamp_col: Optional[str] = None,
        **kwargs,
    ) -> "WindowedDataStream":
        """Create a windowed stream from a CSV file.

        The CSV is loaded into a DataFrame, optionally parsing a
        specified timestamp column as the index, and wrapped into a
        stream.  Each row is treated as one timestamped sample.

        Args:
            file_path: Path to the CSV file.
            window_size: Number of rows per output window.
            step_size: Stride between windows.
            timestamp_col: Optional name of the column containing
                timestamps.  If provided, it is parsed as datetime and
                set as the DataFrame index.
            **kwargs: Additional arguments forwarded to
                :func:`pandas.read_csv`.

        Returns:
            WindowedDataStream: An instance configured for windowing.
        """
        df = pd.read_csv(file_path, **kwargs)
        if timestamp_col and timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col)
        return cls(source=df, window_size=window_size, step_size=step_size)

    def __iter__(self) -> Iterator[pd.DataFrame]:  # type: ignore[override]
        # Reset the underlying iterator and buffer each iteration
        if isinstance(self.source, _DataFrameRowStream):
            # Reset DataFrame source as well
            self.source.reset()
        self._source_iter = iter(self.source)
        self._buffer = pd.DataFrame()
        return self

    def __next__(self) -> pd.DataFrame:  # type: ignore[override]
        # Accumulate rows until we have enough for a full window
        while len(self._buffer) < self.window_size:
            try:
                df = next(self._source_iter)
            except StopIteration:
                # No more data from source
                if len(self._buffer) < self.window_size:
                    raise StopIteration
                break
            # Concatenate new rows; ensure index alignment
            self._buffer = pd.concat([self._buffer, df])
        # Slice the first window
        window = self._buffer.iloc[: self.window_size].copy()
        # Remove step_size rows from the buffer
        self._buffer = self._buffer.iloc[self.step_size :].copy()
        return window


class _DataFrameRowStream(DataStream):
    """Simple stream that yields one-row DataFrames from a DataFrame.

    This helper wraps a :class:`pandas.DataFrame` so that each call to
    ``__next__`` returns a DataFrame containing a single row (with the
    same index and columns).  It resets automatically when re‑iterated.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # Sort by index to preserve temporal order
        self.df = df.sort_index()
        self._cursor = 0

    def __iter__(self) -> "_DataFrameRowStream":
        self._cursor = 0
        return self

    def __next__(self) -> pd.DataFrame:
        if self._cursor >= len(self.df):
            raise StopIteration
        row = self.df.iloc[self._cursor : self._cursor + 1].copy()
        self._cursor += 1
        return row

    def reset(self) -> None:
        """Reset the internal cursor to the beginning of the DataFrame."""
        self._cursor = 0
