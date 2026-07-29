"""Base classes for data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Iterator, Generic, TypeVar
from datetime import datetime
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger('energy_fault_detector.data_sources')


@dataclass
class StreamConfig:
    """Configuration for streaming data sources.
    
    Attributes:
        batch_size: Number of samples per batch (default: 1000)
        delay_seconds: Delay between batches in seconds (for synthetic streams, default: 0.1)
        buffer_size: Maximum number of batches to buffer (default: 10)
        timeout: Timeout for stream operations in seconds (default: 30)
        max_retries: Maximum number of retry attempts for failed reads (default: 3)
        metadata: Additional metadata as key-value pairs
    """
    batch_size: int = 1000
    delay_seconds: float = 0.1
    buffer_size: int = 10
    timeout: float = 30.0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataBatch:
    """A batch of data from a stream source.
    
    Attributes:
        data: The actual data as a pandas DataFrame
        timestamps: Array of timestamps for the data points
        batch_index: Index of this batch in the stream
        is_complete: Whether this is the final batch
        metadata: Additional metadata about this batch
    """
    data: pd.DataFrame
    timestamps: np.ndarray
    batch_index: int
    is_complete: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def shape(self) -> tuple:
        """Shape of the data (rows, columns)."""
        return self.data.shape


T = TypeVar('T')


class DataSource(ABC, Iterator[DataBatch]):
    """Abstract base class for all data sources.
    
    Data sources provide a unified interface for reading data from various
    sources (files, networks, synthetic generators) in a streaming fashion.
    
    All implementations must provide:
    - stream(): Returns an iterator over DataBatch objects
    - close(): Cleans up resources
    - reset(): Resets the source to the beginning
    
    Args:
        config: Stream configuration
        **kwargs: Additional source-specific parameters
    """
    
    def __init__(self, config: Optional[StreamConfig] = None, **kwargs):
        self.config = config or StreamConfig()
        self._is_open = False
        self._batch_index = 0
        self._kwargs = kwargs
        logger.debug(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def open(self) -> 'DataSource':
        """Open the data source and prepare for reading.
        
        Returns:
            self for method chaining
        """
        self._is_open = True
        self._batch_index = 0
        return self
    
    @abstractmethod
    def close(self) -> None:
        """Close the data source and release resources."""
        self._is_open = False
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the data source to the beginning."""
        self._batch_index = 0
    
    @abstractmethod
    def __next__(self) -> DataBatch:
        """Get the next batch of data.
        
        Returns:
            DataBatch containing the next batch of data
            
        Raises:
            StopIteration: When no more data is available
        """
        pass
    
    def __iter__(self) -> Iterator[DataBatch]:
        """Return self as an iterator."""
        self.open()
        return self
    
    def __enter__(self) -> 'DataSource':
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    @property
    def is_open(self) -> bool:
        """Whether the data source is currently open."""
        return self._is_open
    
    @property
    def batch_index(self) -> int:
        """Current batch index."""
        return self._batch_index
    
    def read_all(self) -> pd.DataFrame:
        """Read all data and return as a single DataFrame.
        
        This is useful for testing and when you need all data at once.
        
        Returns:
            DataFrame containing all data from the source
        """
        all_data = []
        with self:
            for batch in self:
                all_data.append(batch.data)
        return pd.concat(all_data, ignore_index=True)
    
    def take(self, n_batches: int) -> list[DataBatch]:
        """Take a specified number of batches.
        
        Args:
            n_batches: Number of batches to take
            
        Returns:
            List of DataBatch objects
        """
        batches = []
        with self:
            for _ in range(n_batches):
                try:
                    batches.append(next(self))
                except StopIteration:
                    break
        return batches
