"""Abstract interface for streaming data sources."""

from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any

import pandas as pd


@dataclass
class StreamChunk:
    """
    One batch of time-ordered samples for a set of data sources/sensor channels.

    The pd DataFrame must be:
    - of wide format (columns = features)
    - containing a timestamp column or DateTimeIndex
    - contain only numeric or boolean columns beside time
    """

    data: pd.DataFrame
    meta: Optional[Dict[str, Any]] = (
        None  # meta like source info, dropped frames, warnings, ...
    )


class StreamDataSource(ABC):
    """
    Abstract base class for streamed data sources.

    Concrete implementations must handle reading from actual files, streams, etc and yield StreamChunk objects.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[StreamChunk]:
        """Yield successive StreamChunk objects from the source."""
        raise NotImplementedError

    def to_csv(self, csv_path: str, chunk_size: Optional[int] = None) -> None:
        """
        Convenience: iterate over the stream and append it to a single CSV file.

        - Writes header only on the first chunk.
        - Assumes each chunk's data dataframe has the same columns (as the first).
        """

        header_written = False
        for chunk in self:
            df = chunk.data
            mode = "w" if not header_written else "a"
            df.to_csv(csv_path, mode=mode, header=(not header_written), index=False)
            header_written = True
