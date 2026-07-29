"""Tests for CSVStreamDataSource."""

import tempfile
import os
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from energy_fault_detector.data_sources import CSVStreamDataSource, StreamConfig, DataBatch


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample data."""
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'sensor_1': np.random.randn(100),
        'sensor_2': np.random.randn(100),
        'sensor_3': np.random.randn(100),
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def sample_csv_with_index():
    """Create a temporary CSV file with timestamp index."""
    data = {
        'sensor_1': np.random.randn(100),
        'sensor_2': np.random.randn(100),
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range('2024-01-01', periods=100, freq='1min')
    df.index.name = 'timestamp'
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


class TestCSVStreamDataSource:
    """Tests for CSVStreamDataSource."""
    
    def test_basic_initialization(self, sample_csv_file):
        """Test basic initialization."""
        source = CSVStreamDataSource(file_path=sample_csv_file)
        assert source.file_path == Path(sample_csv_file)
        assert source.config.batch_size == 1000
        assert source.config.delay_seconds == 0.1
    
    def test_custom_config(self, sample_csv_file):
        """Test with custom configuration."""
        config = StreamConfig(batch_size=10, delay_seconds=0.01)
        source = CSVStreamDataSource(file_path=sample_csv_file, config=config)
        assert source.config.batch_size == 10
        assert source.config.delay_seconds == 0.01
    
    def test_open_and_close(self, sample_csv_file):
        """Test opening and closing the source."""
        source = CSVStreamDataSource(file_path=sample_csv_file)
        
        assert not source.is_open
        source.open()
        assert source.is_open
        assert source.total_rows == 100
        
        source.close()
        assert not source.is_open
    
    def test_context_manager(self, sample_csv_file):
        """Test using as context manager."""
        with CSVStreamDataSource(file_path=sample_csv_file) as source:
            assert source.is_open
            assert source.total_rows == 100
        
        assert not source.is_open
    
    def test_iteration(self, sample_csv_file):
        """Test iterating through batches."""
        config = StreamConfig(batch_size=20, delay_seconds=0)
        source = CSVStreamDataSource(file_path=sample_csv_file, config=config)
        
        batches = list(source)
        
        assert len(batches) == 5  # 100 samples / 20 per batch = 5 batches
        assert all(isinstance(batch, DataBatch) for batch in batches)
        
        # Check batch sizes
        assert len(batches[0].data) == 20
        assert len(batches[-1].data) == 20  # Last batch should also be 20
    
    def test_iteration_with_remainder(self, sample_csv_file):
        """Test iteration with uneven division."""
        config = StreamConfig(batch_size=30, delay_seconds=0)
        source = CSVStreamDataSource(file_path=sample_csv_file, config=config)
        
        batches = list(source)
        
        assert len(batches) == 4  # 100 / 30 = 3 full + 1 partial
        assert len(batches[0].data) == 30
        assert len(batches[1].data) == 30
        assert len(batches[2].data) == 30
        assert len(batches[3].data) == 10  # Remainder
    
    def test_batch_metadata(self, sample_csv_file):
        """Test batch metadata."""
        config = StreamConfig(batch_size=10, delay_seconds=0)
        source = CSVStreamDataSource(file_path=sample_csv_file, config=config)
        
        first_batch = next(iter(source))
        
        assert first_batch.batch_index == 0
        assert not first_batch.is_complete
        assert 'source_file' in first_batch.metadata
        assert first_batch.metadata['total_rows'] == 100
    
    def test_read_all(self, sample_csv_file):
        """Test reading all data at once."""
        source = CSVStreamDataSource(file_path=sample_csv_file)
        
        df = source.read_all()
        
        assert len(df) == 100
        assert list(df.columns) == ['timestamp', 'sensor_1', 'sensor_2', 'sensor_3']
    
    def test_take(self, sample_csv_file):
        """Test taking a limited number of batches."""
        config = StreamConfig(batch_size=10, delay_seconds=0)
        source = CSVStreamDataSource(file_path=sample_csv_file, config=config)
        
        batches = source.take(3)
        
        assert len(batches) == 3
        assert all(isinstance(batch, DataBatch) for batch in batches)
    
    def test_reset(self, sample_csv_file):
        """Test resetting the source."""
        config = StreamConfig(batch_size=10, delay_seconds=0)
        source = CSVStreamDataSource(file_path=sample_csv_file, config=config)
        
        # Read first batch
        source.open()
        first_batch = next(source)
        assert first_batch.batch_index == 0
        
        # Reset and read again
        source.reset()
        source.open()
        first_batch_again = next(source)
        assert first_batch_again.batch_index == 0
    
    def test_with_timestamp_index(self, sample_csv_with_index):
        """Test with timestamp as index."""
        config = StreamConfig(batch_size=10, delay_seconds=0)
        source = CSVStreamDataSource(
            file_path=sample_csv_with_index,
            config=config,
            parse_dates=['timestamp'],
            index_col='timestamp'
        )
        
        batches = list(source)
        
        # Check that timestamps are properly handled
        first_batch = batches[0]
        assert len(first_batch.timestamps) == 10
        assert first_batch.data.index.name == 'timestamp'
    
    def test_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            source = CSVStreamDataSource(file_path="/nonexistent/file.csv")
            source.open()
    
    def test_columns_property(self, sample_csv_file):
        """Test columns property."""
        source = CSVStreamDataSource(file_path=sample_csv_file)
        source.open()
        
        assert set(source.columns) == {'timestamp', 'sensor_1', 'sensor_2', 'sensor_3'}
    
    def test_estimated_batches(self, sample_csv_file):
        """Test estimated batches calculation."""
        config = StreamConfig(batch_size=15, delay_seconds=0)
        source = CSVStreamDataSource(file_path=sample_csv_file, config=config)
        source.open()
        
        # 100 samples / 15 per batch = 7 batches (6 full + 1 partial)
        assert source.estimated_batches == 7
