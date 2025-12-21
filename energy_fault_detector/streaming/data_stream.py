"""Base classes for streaming data sources.

This module defines an abstract base class for streaming data sources.  A
streaming data source yields chunks of sensor data as pandas
DataFrames.  Such streams are useful when evaluatinganomaly‑detection
models on live or near real‑time data, wheremeasurements arrive
incrementally rather than being available all at once.

The abstract :class:`~energy_fault_detector.streaming.data_stream.DataStream`
class implements Python's iterator protocol and must be subclassed by
concrete stream implementations.
"""

from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod
from typing import Iterator


class DataStream(ABC):
    """Abstract base class for streaming data sources.

    A :class:`DataStream` yields successive chunks of sensor data as
    :class:`pandas.DataFrame` objects.  Each chunk is expected to
    contain rows of data indexed by a time stamp, matching the input
    requirements of the existing fault detection pipeline.  Subclasses
    must implement the iterator protocol by providing
    :meth:`__iter__` and :meth:`__next__` methods.
    """

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Return the iterator for this stream.

        Returns:
            Iterator[pd.DataFrame]: An iterator over data chunks.
        """
        return self

    @abstractmethod
    def __next__(self) -> pd.DataFrame:
        """Return the next chunk of data from the stream.

        Subclasses should raise ``StopIteration`` when no further
        chunks are available.

        Returns:
            pandas.DataFrame: The next data chunk.
        """
        raise StopIteration
