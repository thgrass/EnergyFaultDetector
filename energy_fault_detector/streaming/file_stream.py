"""Streaming interface for file-based sensor data.

This module implements a simple data stream that reads data from a file
in fixed-size chunks.  It is designed for situations where sensor data
is stored on disk and must be processed in batches that fit into
memory.  The stream yields successive ``pandas.DataFrame`` objects
containing ``chunk_size`` rows.  Optional timestamp parsing is supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional
import pandas as pd

from .data_stream import DataStream


@dataclass
class FileDataStream(DataStream):
    """Data stream that reads a CSV file in fixed-size chunks.

    Each call to :meth:`__next__` returns the next chunk of rows from the
    input file as a :class:`pandas.DataFrame`.  If ``timestamp_col`` is
    provided, that column is parsed as a datetime and set as the index.

    Args:
        file_path (str): Path to the CSV file containing sensor data.
        chunk_size (int): Number of rows to read per chunk.
        timestamp_col (Optional[str]): Name of the timestamp column.  If
            provided, this column is converted to datetime and used as
            the index of the returned DataFrames.
        **read_csv_kwargs: Additional keyword arguments passed to
            :func:`pandas.read_csv` optionally.
    """

    file_path: str
    chunk_size: int
    timestamp_col: Optional[str] = None
    read_csv_kwargs: dict = None
    _reader: Optional[Iterator[pd.DataFrame]] = None

    def __post_init__(self) -> None:
        # Set a default for read_csv_kwargs if none was provided.
        if self.read_csv_kwargs is None:
            self.read_csv_kwargs = {}

    def __iter__(self) -> Iterator[pd.DataFrame]:
        # Initialise the chunked CSV reader.
        self._reader = pd.read_csv(
            self.file_path, chunksize=self.chunk_size, **self.read_csv_kwargs
        )
        return self

    def __next__(self) -> pd.DataFrame:
        """Return the next chunk of data.

        Raises:
            StopIteration: When no further chunks are available.

        Returns:
            pandas.DataFrame: The next data chunk.
        """
        if self._reader is None:
            # Ensure the reader is initialised if __iter__ was not called.
            self.__iter__()
        try:
            chunk = next(self._reader)  # may raise StopIteration
        except StopIteration:
            # Propagate StopIteration to the caller.
            raise
        # Optionally parse the timestamp column and set it as the index.
        if self.timestamp_col is not None and self.timestamp_col in chunk.columns:
            chunk[self.timestamp_col] = pd.to_datetime(chunk[self.timestamp_col])
            chunk = chunk.set_index(self.timestamp_col)
        return chunk
