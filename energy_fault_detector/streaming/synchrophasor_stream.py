"""Streaming interface for synchrophasor measurement data.

This module provides a concrete implementation of
:class:`~energy_fault_detector.streaming.data_stream.DataStream` for
synchrophasor measurement data.  Synchrophasor data frames transmit a
time‑stamped set of measurements that include phasor estimates,
frequency deviation from the nominal power line frequency and the rate
of change of frequency.  Phasor data can be sent in either
rectangular (real and imaginary) or polar (magnitude and angle)
coordinates, and both phasor and frequency data may be represented as
32‑bit IEEE floating‑point numbers or as scaled integers.

The :class:`SynchrophasorStream` class does not attempt to decode the
binary C37.118 frames directly.  Instead, it consumes a
``pandas.DataFrame`` (or loads one from a CSV file) where columns
represent individual measurement channels (phasor components, frequency
deviation, etc.) and the index represents timestamps.  The stream
yields overlapping windows from this DataFrame to support online
evaluation of anomaly‑detection models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional
import pandas as pd

from .data_stream import DataStream


@dataclass
class SynchrophasorStream(DataStream):
    """Data stream for synchrophasor measurements.

    This stream iterates over a time‑indexed :class:`pandas.DataFrame`
    and returns sliding windows of fixed length.  Each window contains
    consecutive rows of data, preserving the index and columns.  The
    design allows fault detection models to process live data in
    batches rather than requiring the complete dataset at once.

    Args:
        data (pd.DataFrame): Synchrophasor measurements with a
            datetime index.  Columns may include phasor magnitude/angle
            pairs or rectangular components, frequency deviation,
            rate of change of frequency, and optional analog or digital
            channels.  When using scaled integer formats, values
            should be pre‑converted using the factors provided in
            configuration frames.
        window_size (int): Number of rows per window.
        step_size (int): Step size between windows.  Defaults to 1,
            corresponding to a stride of one frame.
    """

    data: pd.DataFrame
    window_size: int
    step_size: int = 1
    _cursor: int = 0

    def __post_init__(self) -> None:
        # Ensure the data is sorted by its index so sliding windows are contiguous.
        self.data = self.data.sort_index()

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        window_size: int,
        step_size: int = 1,
        timestamp_col: Optional[str] = None,
        **kwargs,
    ) -> "SynchrophasorStream":
        """Load synchrophasor data from a CSV file and create a stream.

        The CSV file should contain a header row.  If ``timestamp_col``
        is provided and present in the file, it will be parsed as a
        datetime column and used as the index.  All other columns are
        treated as measurement channels.

        Args:
            file_path (str): Path to a CSV file containing
                synchrophasor measurements.
            window_size (int): Number of rows per window.
            step_size (int, optional): Step size between windows.
            timestamp_col (Optional[str], optional): Name of the column
                containing timestamps.  If provided, that column is
                parsed as datetime and set as the DataFrame index.
            **kwargs: Additional keyword arguments passed to
                :func:`pandas.read_csv`.

        Returns:
            SynchrophasorStream: An instantiated data stream.
        """
        df = pd.read_csv(file_path, **kwargs)
        if timestamp_col is not None and timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col)
        return cls(data=df, window_size=window_size, step_size=step_size)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        # Reset the internal cursor each time iteration begins.
        self._cursor = 0
        return self

    def __next__(self) -> pd.DataFrame:
        """Return the next window of data.

        Raises:
            StopIteration: When the end of the data is reached.

        Returns:
            pandas.DataFrame: A copy of the current window of data.
        """
        if self._cursor + self.window_size > len(self.data):
            # No more full windows are available.
            raise StopIteration
        # Slice the DataFrame to obtain the next window.  Copy to
        # decouple it from the underlying DataFrame.
        window = self.data.iloc[self._cursor : self._cursor + self.window_size].copy()
        # Advance the cursor by the step size.
        self._cursor += self.step_size
        return window
