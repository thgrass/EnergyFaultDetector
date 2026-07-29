"""CSV file stream data source.

Reads CSV files in batches to simulate streaming data.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

from energy_fault_detector.data_sources.base import DataSource, DataBatch, StreamConfig

logger = logging.getLogger('energy_fault_detector.data_sources.csv_stream')


class CSVStreamDataSource(DataSource):
    """Data source that reads a CSV file in batches to simulate streaming.
    
    This is useful for:
    - Testing streaming pipelines with real data
    - Simulating real-time data ingestion
    - Benchmarking streaming performance
    
    Args:
        file_path: Path to the CSV file
        config: Stream configuration (batch_size, delay, etc.)
        parse_dates: Column(s) to parse as dates (default: None)
        index_col: Column to use as index (default: None)
        timestamp_col: Column containing timestamps (default: None)
        dtype: Data types for columns (default: None)
        **kwargs: Additional arguments passed to pd.read_csv
    
    Example:
        >>> source = CSVStreamDataSource(
        ...     file_path="data.csv",
        ...     config=StreamConfig(batch_size=100, delay_seconds=0.01),
        ...     parse_dates=["timestamp"],
        ...     index_col="timestamp"
        ... )
        >>> for batch in source:
        ...     print(f"Batch {batch.batch_index}: {len(batch.data)} rows")
    """
    
    def __init__(
        self,
        file_path: str | Path,
        config: Optional[StreamConfig] = None,
        parse_dates: Optional[list[str] | str] = None,
        index_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        dtype: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        self.file_path = Path(file_path)
        self.parse_dates = parse_dates
        self.index_col = index_col
        self.timestamp_col = timestamp_col
        self.dtype = dtype
        self.read_csv_kwargs = kwargs
        
        # Will be set when opened
        self._data: Optional[pd.DataFrame] = None
        self._current_index = 0
        self._total_rows = 0
    
    def open(self) -> 'CSVStreamDataSource':
        """Open the CSV file and load it into memory."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        
        logger.info(f"Opening CSV file: {self.file_path}")
        
        # Read the entire CSV file
        read_kwargs = {
            'parse_dates': self.parse_dates,
            'index_col': self.index_col,
            'dtype': self.dtype,
            **self.read_csv_kwargs
        }
        
        self._data = pd.read_csv(self.file_path, **read_kwargs)
        self._total_rows = len(self._data)
        self._current_index = 0
        self._batch_index = 0
        self._is_open = True
        
        logger.info(f"Loaded {self._total_rows} rows from {self.file_path}")
        return self
    
    def close(self) -> None:
        """Close the data source."""
        self._data = None
        self._current_index = 0
        self._is_open = False
        logger.debug("CSV data source closed")
    
    def reset(self) -> None:
        """Reset to the beginning of the file."""
        self._current_index = 0
        self._batch_index = 0
        logger.debug("CSV data source reset")
    
    def __next__(self) -> DataBatch:
        """Get the next batch of data."""
        if not self._is_open:
            raise RuntimeError("Data source is not open. Call open() first.")
        
        if self._data is None:
            raise RuntimeError("No data loaded. Call open() first.")
        
        # Check if we've reached the end
        if self._current_index >= self._total_rows:
            raise StopIteration("Reached end of CSV file")
        
        # Calculate batch end index
        batch_size = self.config.batch_size
        end_index = min(self._current_index + batch_size, self._total_rows)
        
        # Extract batch data
        batch_data = self._data.iloc[self._current_index:end_index].copy()
        
        # Get timestamps
        if self.timestamp_col and self.timestamp_col in batch_data.columns:
            timestamps = batch_data[self.timestamp_col].values
        elif batch_data.index.name == 'timestamp' or isinstance(batch_data.index, pd.DatetimeIndex):
            timestamps = batch_data.index.values
        else:
            # Use a generated timestamp if none available
            timestamps = np.arange(self._current_index, end_index, dtype='datetime64[ns]')
        
        # Create metadata
        metadata = {
            'source_file': str(self.file_path),
            'total_rows': self._total_rows,
            'current_row': self._current_index,
            'end_row': end_index,
        }
        
        # Create and return batch
        batch = DataBatch(
            data=batch_data,
            timestamps=timestamps,
            batch_index=self._batch_index,
            is_complete=end_index >= self._total_rows,
            metadata=metadata
        )
        
        # Update indices
        self._current_index = end_index
        self._batch_index += 1
        
        # Apply delay if configured
        if self.config.delay_seconds > 0:
            time.sleep(self.config.delay_seconds)
        
        logger.debug(f"Read batch {batch.batch_index}: {len(batch.data)} rows")
        return batch
    
    @property
    def total_rows(self) -> int:
        """Total number of rows in the CSV file."""
        if self._data is None:
            return 0
        return self._total_rows
    
    @property
    def columns(self) -> list[str]:
        """Column names in the CSV file."""
        if self._data is None:
            return []
        return list(self._data.columns)
    
    @property
    def estimated_batches(self) -> int:
        """Estimated number of batches."""
        if self._data is None:
            return 0
        return (self._total_rows + self.config.batch_size - 1) // self.config.batch_size
