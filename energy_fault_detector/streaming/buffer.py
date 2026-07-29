"""Data buffering utilities for streaming fault detection.

Provides classes for buffering and windowing streaming data,
especially for sequence-based models.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger('energy_fault_detector.streaming.buffer')


@dataclass
class DataBuffer:
    """Simple buffer for storing recent data from a stream.
    
    This buffer stores data in a first-in-first-out (FIFO) manner,
    keeping only the most recent samples up to a maximum size.
    
    Args:
        max_size: Maximum number of samples to store
        dtype: Data type for the buffer (default: float32)
    """
    max_size: int = 1000
    dtype: np.dtype = np.float32
    
    def __init__(self, max_size: int = 1000, dtype: np.dtype = np.float32):
        self.max_size = max_size
        self.dtype = dtype
        self._data: deque = deque(maxlen=max_size)
        self._timestamps: deque = deque(maxlen=max_size)
        self._total_samples = 0
    
    def add(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> None:
        """Add new data to the buffer.
        
        Args:
            data: New data to add (2D array: samples x features)
            timestamps: Optional timestamps for the data
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Convert to specified dtype
        data = data.astype(self.dtype)
        
        # Add each sample
        for i in range(len(data)):
            self._data.append(data[i])
            if timestamps is not None:
                self._timestamps.append(timestamps[i])
            self._total_samples += 1
    
    def add_dataframe(self, df: pd.DataFrame) -> None:
        """Add data from a DataFrame.
        
        Args:
            df: DataFrame with data to add
        """
        self.add(df.values, df.index.values if isinstance(df.index, pd.DatetimeIndex) else None)
    
    @property
    def data(self) -> np.ndarray:
        """Get all data in the buffer as a numpy array."""
        if not self._data:
            return np.array([], dtype=self.dtype)
        return np.array(self._data, dtype=self.dtype)
    
    @property
    def timestamps(self) -> np.ndarray:
        """Get all timestamps in the buffer."""
        if not self._timestamps:
            return np.array([], dtype='datetime64[ns]')
        return np.array(self._timestamps)
    
    @property
    def size(self) -> int:
        """Current number of samples in the buffer."""
        return len(self._data)
    
    @property
    def is_full(self) -> bool:
        """Whether the buffer is full."""
        return self.size >= self.max_size
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._data.clear()
        self._timestamps.clear()
    
    def get_recent(self, n_samples: int) -> np.ndarray:
        """Get the most recent n samples.
        
        Args:
            n_samples: Number of samples to retrieve
            
        Returns:
            Array of the most recent samples
        """
        if n_samples <= 0:
            return np.array([], dtype=self.dtype)
        
        start = max(0, self.size - n_samples)
        return self.data[start:]
    
    def to_dataframe(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert buffer data to a DataFrame.
        
        Args:
            columns: Column names for the DataFrame
            
        Returns:
            DataFrame with buffer data
        """
        if self.size == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data)
        if columns:
            df.columns = columns
        if self._timestamps:
            df.index = pd.DatetimeIndex(self.timestamps)
        return df


@dataclass
class SlidingWindowBuffer:
    """Buffer that maintains sliding windows for sequence models.
    
    This buffer is designed for sequence-based autoencoders that require
    fixed-length sequences as input. It automatically creates and manages
    sliding windows from the incoming stream.
    
    Args:
        window_size: Size of each window (number of timesteps)
        stride: Stride between consecutive windows (default: 1)
        max_windows: Maximum number of windows to store (default: 100)
        pad_value: Value to use for padding when window is incomplete (default: 0.0)
    """
    window_size: int = 10
    stride: int = 1
    max_windows: int = 100
    pad_value: float = 0.0
    
    def __init__(
        self,
        window_size: int = 10,
        stride: int = 1,
        max_windows: int = 100,
        pad_value: float = 0.0
    ):
        self.window_size = window_size
        self.stride = stride
        self.max_windows = max_windows
        self.pad_value = pad_value
        
        # Internal storage
        self._data_buffer: deque = deque()
        self._window_buffer: deque = deque(maxlen=max_windows)
        self._timestamps: deque = deque()
        self._window_timestamps: deque = deque(maxlen=max_windows)
        
        # Statistics
        self._total_samples = 0
        self._total_windows = 0
    
    def add(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Add new data and generate windows.
        
        Args:
            data: New data to add (2D array: samples x features)
            timestamps: Optional timestamps for the data
            
        Returns:
            List of new windows that were created
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        new_windows = []
        
        # Add each sample
        for i in range(len(data)):
            self._data_buffer.append(data[i])
            if timestamps is not None:
                self._timestamps.append(timestamps[i])
            self._total_samples += 1
            
            # Check if we can create a new window
            if len(self._data_buffer) >= self.window_size:
                # Create window
                window_data = np.array(self._data_buffer)[-self.window_size:]
                new_windows.append(window_data)
                
                # Store window
                self._window_buffer.append(window_data)
                if self._timestamps:
                    window_ts = np.array(self._timestamps)[-self.window_size:]
                    self._window_timestamps.append(window_ts)
                self._total_windows += 1
        
        return new_windows
    
    def add_dataframe(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Add data from a DataFrame.
        
        Args:
            df: DataFrame with data to add
            
        Returns:
            List of new windows that were created
        """
        timestamps = df.index.values if isinstance(df.index, pd.DatetimeIndex) else None
        return self.add(df.values, timestamps)
    
    @property
    def windows(self) -> List[np.ndarray]:
        """Get all windows as a list of arrays."""
        return list(self._window_buffer)
    
    @property
    def window_timestamps(self) -> List[np.ndarray]:
        """Get timestamps for all windows."""
        return list(self._window_timestamps)
    
    @property
    def n_windows(self) -> int:
        """Number of windows currently stored."""
        return len(self._window_buffer)
    
    @property
    def buffer_size(self) -> int:
        """Current size of the raw data buffer."""
        return len(self._data_buffer)
    
    def clear(self) -> None:
        """Clear all buffers."""
        self._data_buffer.clear()
        self._window_buffer.clear()
        self._timestamps.clear()
        self._window_timestamps.clear()
    
    def get_window_dataframe(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert window buffer to a DataFrame.
        
        Note: This flattens the windows into a 2D array where each row
        is a flattened window.
        
        Args:
            columns: Column names for the DataFrame
            
        Returns:
            DataFrame with window data
        """
        if self.n_windows == 0:
            return pd.DataFrame()
        
        # Stack windows into a 2D array
        stacked = np.stack(self.windows)  # Shape: (n_windows, window_size, n_features)
        n_features = stacked.shape[2]
        
        # Reshape to (n_windows, window_size * n_features)
        flat_shape = (stacked.shape[0], stacked.shape[1] * stacked.shape[2])
        flat_data = stacked.reshape(flat_shape)
        
        # Generate column names if not provided
        if columns is None:
            columns = [f"t{i}_f{j}" for i in range(self.window_size) for j in range(n_features)]
        
        df = pd.DataFrame(flat_data, columns=columns)
        
        # Add timestamps (use the last timestamp of each window)
        if self._window_timestamps:
            last_timestamps = [ts[-1] for ts in self._window_timestamps]
            df.index = pd.DatetimeIndex(last_timestamps)
        
        return df
    
    def get_sequence_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Get data in format suitable for sequence models.
        
        Returns:
            Tuple of (X, timestamps) where X has shape (n_windows, window_size, n_features)
        """
        if self.n_windows == 0:
            return np.array([]), np.array([])
        
        X = np.stack(self.windows)  # Shape: (n_windows, window_size, n_features)
        
        if self._window_timestamps:
            # Use the last timestamp of each window
            timestamps = np.array([ts[-1] for ts in self._window_timestamps])
        else:
            timestamps = np.arange(self.n_windows)
        
        return X, timestamps
