"""Tests for data buffer classes."""

import pytest
import numpy as np
import pandas as pd

from energy_fault_detector.streaming import DataBuffer, SlidingWindowBuffer


class TestDataBuffer:
    """Tests for DataBuffer."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        buffer = DataBuffer(max_size=100)
        
        assert buffer.max_size == 100
        assert buffer.size == 0
        assert not buffer.is_full
    
    def test_add_data(self):
        """Test adding data to buffer."""
        buffer = DataBuffer(max_size=10)
        
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        buffer.add(data)
        
        assert buffer.size == 3
        assert buffer.data.shape == (3, 3)
    
    def test_add_single_sample(self):
        """Test adding a single sample."""
        buffer = DataBuffer(max_size=10)
        
        buffer.add(np.array([1, 2, 3]))
        
        assert buffer.size == 1
        assert buffer.data.shape == (1, 3)
    
    def test_add_dataframe(self):
        """Test adding DataFrame data."""
        buffer = DataBuffer(max_size=10)
        
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        buffer.add_dataframe(df)
        
        assert buffer.size == 3
        assert buffer.data.shape == (3, 2)
    
    def test_max_size_limit(self):
        """Test that buffer respects max size."""
        buffer = DataBuffer(max_size=5)
        
        # Add more data than max_size
        data = np.arange(20).reshape(10, 2)
        buffer.add(data)
        
        assert buffer.size == 5
        assert buffer.is_full
    
    def test_fifo_behavior(self):
        """Test FIFO behavior."""
        buffer = DataBuffer(max_size=3)
        
        buffer.add(np.array([1, 2]))
        buffer.add(np.array([3, 4]))
        buffer.add(np.array([5, 6]))
        buffer.add(np.array([7, 8]))  # This should push out [1, 2]
        
        assert buffer.size == 3
        assert buffer.data[0][0] == 3  # First element should be 3, not 1
    
    def test_clear(self):
        """Test clearing the buffer."""
        buffer = DataBuffer(max_size=10)
        
        buffer.add(np.arange(12).reshape(4, 3))
        assert buffer.size == 4
        
        buffer.clear()
        assert buffer.size == 0
    
    def test_get_recent(self):
        """Test getting recent samples."""
        buffer = DataBuffer(max_size=10)
        
        data = np.arange(30).reshape(10, 3)
        buffer.add(data)
        
        recent = buffer.get_recent(5)
        assert recent.shape == (5, 3)
        assert recent[0][0] == 15  # Should be the 6th sample (0-indexed)
    
    def test_get_recent_more_than_available(self):
        """Test getting more samples than available."""
        buffer = DataBuffer(max_size=10)
        
        buffer.add(np.arange(6).reshape(2, 3))
        
        recent = buffer.get_recent(10)  # Ask for more than available
        assert recent.shape == (2, 3)  # Should return all available
    
    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        buffer = DataBuffer(max_size=10)
        
        data = np.array([[1, 2], [3, 4], [5, 6]])
        buffer.add(data)
        
        df = buffer.to_dataframe(columns=['a', 'b'])
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ['a', 'b']
    
    def test_dtype_conversion(self):
        """Test dtype conversion."""
        buffer = DataBuffer(max_size=10, dtype=np.float32)
        
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        buffer.add(data)
        
        assert buffer.data.dtype == np.float32


class TestSlidingWindowBuffer:
    """Tests for SlidingWindowBuffer."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        buffer = SlidingWindowBuffer(window_size=5, stride=1)
        
        assert buffer.window_size == 5
        assert buffer.stride == 1
        assert buffer.max_windows == 100
    
    def test_add_data(self):
        """Test adding data and generating windows."""
        buffer = SlidingWindowBuffer(window_size=3, stride=1)
        
        data = np.array([[1], [2], [3], [4], [5]])
        new_windows = buffer.add(data)
        
        assert len(new_windows) == 3  # Should create 3 windows: [1,2,3], [2,3,4], [3,4,5]
        assert buffer.n_windows == 3
        assert buffer.buffer_size == 5
    
    def test_window_creation(self):
        """Test that windows are created correctly."""
        buffer = SlidingWindowBuffer(window_size=3, stride=1)
        
        data = np.array([[1], [2], [3], [4]])
        buffer.add(data)
        
        windows = buffer.windows
        assert len(windows) == 2  # [1,2,3] and [2,3,4]
        
        assert windows[0].shape == (3, 1)
        assert windows[0][0][0] == 1
        assert windows[0][1][0] == 2
        assert windows[0][2][0] == 3
    
    def test_window_stride(self):
        """Test window stride."""
        buffer = SlidingWindowBuffer(window_size=3, stride=2)
        
        data = np.array([[1], [2], [3], [4], [5], [6]])
        buffer.add(data)
        
        # With stride=2, windows should be: [1,2,3], [3,4,5]
        windows = buffer.windows
        assert len(windows) == 2
        
        assert windows[0][0][0] == 1
        assert windows[1][0][0] == 3
    
    def test_max_windows_limit(self):
        """Test max windows limit."""
        buffer = SlidingWindowBuffer(window_size=2, stride=1, max_windows=3)
        
        # Add enough data to create more than max_windows
        data = np.arange(20).reshape(20, 1)
        buffer.add(data)
        
        assert buffer.n_windows == 3  # Should be limited to max_windows
    
    def test_add_dataframe(self):
        """Test adding DataFrame data."""
        buffer = SlidingWindowBuffer(window_size=3, stride=1)
        
        df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        new_windows = buffer.add_dataframe(df)
        
        assert len(new_windows) == 2
        assert buffer.n_windows == 2
    
    def test_clear(self):
        """Test clearing the buffer."""
        buffer = SlidingWindowBuffer(window_size=3, stride=1)
        
        buffer.add(np.arange(10).reshape(10, 1))
        assert buffer.n_windows > 0
        
        buffer.clear()
        assert buffer.n_windows == 0
        assert buffer.buffer_size == 0
    
    def test_get_window_dataframe(self):
        """Test getting window data as DataFrame."""
        buffer = SlidingWindowBuffer(window_size=2, stride=1)
        
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        buffer.add_dataframe(df)
        
        window_df = buffer.get_window_dataframe()
        
        assert isinstance(window_df, pd.DataFrame)
        assert window_df.shape == (2, 4)  # 2 windows, 2*2 features
    
    def test_get_sequence_dataset(self):
        """Test getting sequence dataset."""
        buffer = SlidingWindowBuffer(window_size=3, stride=1)
        
        data = np.arange(12).reshape(6, 2)  # 6 samples, 2 features
        buffer.add(data)
        
        X, timestamps = buffer.get_sequence_dataset()
        
        assert X.shape == (4, 3, 2)  # 4 windows, 3 timesteps, 2 features
        assert len(timestamps) == 4
    
    def test_empty_buffer(self):
        """Test behavior with empty buffer."""
        buffer = SlidingWindowBuffer(window_size=3, stride=1)
        
        assert buffer.n_windows == 0
        assert buffer.buffer_size == 0
        
        windows = buffer.windows
        assert len(windows) == 0
        
        X, timestamps = buffer.get_sequence_dataset()
        assert X.size == 0
        assert len(timestamps) == 0
    
    def test_partial_window(self):
        """Test that partial windows are not created."""
        buffer = SlidingWindowBuffer(window_size=5, stride=1)
        
        # Add less data than window_size
        data = np.array([[1], [2], [3]])
        buffer.add(data)
        
        assert buffer.n_windows == 0  # No complete windows
        assert buffer.buffer_size == 3
