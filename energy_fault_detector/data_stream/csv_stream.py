"""CSV-files as streamed data sources. Useful for large files or data
directly from network connections."""

from __future__ import annotations
from typing import Iterator, Optional, Dict, Any

import pandas as pd

from .base import StreamDataSource, StreamChunk


class CsvStreamSource(StreamDataSource):
    """
    Stream data from a (potentiall large or unreliable) CSV file in chunks, as StreamChunk objects.

    This is the "streamed" analogue to the existing CSV-based batch loading in the main repo:
    - uses pandas.read_csv(chunksize=...)
    - yields one StreamChunk per chunk
    - aims to keep memory usage bounded

    Paramters
    ---------

    csv_path : str
        Path to the CSV file being read.
    chunk_size : int
        Number of rows per chunk to read.
    time_column : str, optional
        Name of the time column in the CSV (default: "timestamp").
        If present and parse_dates=True, this column is used as datetime.
    parse_dates : bool
        If true and time_column exists, parse it as datetime.
    read_csv_kwargs : dict
        Additional keyword arguments forwarded to pandas.read_csv.
    """

    def __init__(
        self,
        csv_path: str,
        chunk_size: int = 10_000,
        time_column: str = "timestamp",
        parse_dates: bool = True,
        **read_csv_kwargs: Any,
    ) -> None:
        self.csv_path = csv_path
        self.chunk_size = int(chunk_size)
        self.time_column = time_column
        self.parse_dates = parse_dates

        # build kwargs for pandas.read_csv
        self._read_csv_kwargs: Dict[str, Any] = dict(read_csv_kwargs)
        self._read_csv_kwargs["chunksize"] = self.chunk_size

        if self.parse_dates and self.time_column:
            # use existing config
            existing = self._read_csv_kwargs.get("parse_dates")
            if existing is None:
                self._read_csv_kwargs["parse_dates"] = existing
        # other cases treated by caller

    def __iter__(self) -> Iterator[StreamChunk]:
        """
        Iterate over a CSV file, yield StreamChunk objects.
        """

        # Pandas does the chunking.
        for chunk_df in pd.read_csv(self.csv_path, **self._read_csv_kwargs):
            # Ensure there is some form of time column for the downstream code
            if self.time_column and self.time_column not in chunk_df.columns:
                # Fallback: row index as synthetic time column
                chunk_df[self.time_column] = range(len(chunk_df))

            # Index not set here
            yield StreamChunk(data=chunk_df)
