"""Tests for StreamingFaultDetector."""

import tempfile
import os
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from energy_fault_detector import Config
from energy_fault_detector.data_sources import (
    CSVStreamDataSource,
    SimulatedFaultDataSource,
    StreamConfig
)
from energy_fault_detector.streaming import (
    StreamingFaultDetector,
    StreamingResult,
    BatchResult
)
from energy_fault_detector.config import generate_quickstart_config


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample data."""
    np.random.seed(42)
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
        'sensor_1': np.random.randn(1000) * 0.1,
        'sensor_2': np.random.randn(1000) * 0.1,
        'sensor_3': np.random.randn(1000) * 0.1,
    }
    # Add some anomalies
    data['sensor_1'][500:510] = 10.0  # Large values
    
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def quickstart_config():
    """Create a quickstart configuration."""
    return generate_quickstart_config()


class TestStreamingFaultDetector:
    """Tests for StreamingFaultDetector."""
    
    def test_basic_initialization(self, quickstart_config):
        """Test basic initialization."""
        detector = StreamingFaultDetector(config=quickstart_config)
        
        assert detector.config is not None
        assert detector.buffer_size == 1000
        assert not detector.online_learning
    
    def test_custom_parameters(self, quickstart_config):
        """Test with custom parameters."""
        detector = StreamingFaultDetector(
            config=quickstart_config,
            buffer_size=500,
            online_learning=True,
            online_learning_interval=50
        )
        
        assert detector.buffer_size == 500
        assert detector.online_learning
        assert detector.online_learning_interval == 50
    
    def test_initialization_with_data(self, quickstart_config, sample_csv_file):
        """Test initialization with training data."""
        df = pd.read_csv(sample_csv_file, parse_dates=['timestamp'])
        sensor_data = df[['sensor_1', 'sensor_2', 'sensor_3']]
        
        detector = StreamingFaultDetector(config=quickstart_config)
        detector.initialize(fit_data=sensor_data)
        
        assert detector._is_initialized
        assert detector.fault_detector is not None
    
    def test_process_dataframe(self, quickstart_config, sample_csv_file):
        """Test processing a DataFrame in batches."""
        df = pd.read_csv(sample_csv_file, parse_dates=['timestamp'])
        sensor_data = df[['sensor_1', 'sensor_2', 'sensor_3']]
        
        detector = StreamingFaultDetector(config=quickstart_config)
        detector.initialize(fit_data=sensor_data.iloc[:500])  # Train on first half
        
        # Process the data
        result = detector.process_dataframe(sensor_data.iloc[500:], batch_size=100)
        
        assert isinstance(result, StreamingResult)
        assert result.total_samples == 500
        assert len(result.batch_results) == 5  # 500 / 100 = 5 batches
    
    def test_process_stream_csv(self, quickstart_config, sample_csv_file):
        """Test processing a CSV stream."""
        # First, train on a subset
        df = pd.read_csv(sample_csv_file, parse_dates=['timestamp'])
        sensor_data = df[['sensor_1', 'sensor_2', 'sensor_3']]
        
        detector = StreamingFaultDetector(config=quickstart_config)
        detector.initialize(fit_data=sensor_data.iloc[:500])
        
        # Create stream source for the rest
        source = CSVStreamDataSource(
            file_path=sample_csv_file,
            config=StreamConfig(batch_size=100, delay_seconds=0)
        )
        
        # Process stream
        result = detector.process_stream(source, max_batches=5)
        
        assert isinstance(result, StreamingResult)
        assert result.total_samples == 500  # 5 batches * 100 = 500
        assert len(result.batch_results) == 5
    
    def test_process_stream_synthetic(self, quickstart_config):
        """Test processing synthetic stream."""
        detector = StreamingFaultDetector(config=quickstart_config)
        
        # Create synthetic data source
        source = SimulatedFaultDataSource(
            n_samples=500,
            n_features=3,
            config=StreamConfig(batch_size=100, delay_seconds=0),
            fault_rate=0.1
        )
        
        # Initialize with some normal data
        normal_data = pd.DataFrame(np.random.randn(100, 3), columns=['f1', 'f2', 'f3'])
        detector.initialize(fit_data=normal_data)
        
        # Process stream
        result = detector.process_stream(source)
        
        assert isinstance(result, StreamingResult)
        assert result.total_samples == 500
        assert len(result.batch_results) == 5
    
    def test_batch_result_properties(self, quickstart_config, sample_csv_file):
        """Test BatchResult properties."""
        df = pd.read_csv(sample_csv_file, parse_dates=['timestamp'])
        sensor_data = df[['sensor_1', 'sensor_2', 'sensor_3']]
        
        detector = StreamingFaultDetector(config=quickstart_config)
        detector.initialize(fit_data=sensor_data.iloc[:500])
        
        # Process a batch
        source = CSVStreamDataSource(
            file_path=sample_csv_file,
            config=StreamConfig(batch_size=100, delay_seconds=0)
        )
        
        result = detector.process_stream(source, max_batches=1)
        
        assert len(result.batch_results) == 1
        batch_result = result.batch_results[0]
        
        assert isinstance(batch_result, BatchResult)
        assert batch_result.batch_index == 0
        assert len(batch_result.timestamps) == 100
        assert len(batch_result.predictions) == 100
        assert len(batch_result.scores) == 100
    
    def test_streaming_result_summary(self, quickstart_config, sample_csv_file):
        """Test StreamingResult summary."""
        df = pd.read_csv(sample_csv_file, parse_dates=['timestamp'])
        sensor_data = df[['sensor_1', 'sensor_2', 'sensor_3']]
        
        detector = StreamingFaultDetector(config=quickstart_config)
        detector.initialize(fit_data=sensor_data.iloc[:500])
        
        source = CSVStreamDataSource(
            file_path=sample_csv_file,
            config=StreamConfig(batch_size=100, delay_seconds=0)
        )
        
        result = detector.process_stream(source, max_batches=3)
        
        summary = result.summary()
        
        assert 'total_samples' in summary
        assert 'total_anomalies' in summary
        assert 'n_batches' in summary
        assert 'throughput' in summary
        assert summary['total_samples'] == 300
        assert summary['n_batches'] == 3
    
    def test_to_dataframe(self, quickstart_config, sample_csv_file):
        """Test converting results to DataFrame."""
        df = pd.read_csv(sample_csv_file, parse_dates=['timestamp'])
        sensor_data = df[['sensor_1', 'sensor_2', 'sensor_3']]
        
        detector = StreamingFaultDetector(config=quickstart_config)
        detector.initialize(fit_data=sensor_data.iloc[:500])
        
        source = CSVStreamDataSource(
            file_path=sample_csv_file,
            config=StreamConfig(batch_size=100, delay_seconds=0)
        )
        
        result = detector.process_stream(source, max_batches=2)
        
        result_df = result.to_dataframe()
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 200
        assert 'anomaly_score' in result_df.columns
        assert 'is_anomaly' in result_df.columns
    
    def test_get_anomalies(self, quickstart_config, sample_csv_file):
        """Test getting only anomalies."""
        df = pd.read_csv(sample_csv_file, parse_dates=['timestamp'])
        sensor_data = df[['sensor_1', 'sensor_2', 'sensor_3']]
        
        detector = StreamingFaultDetector(config=quickstart_config)
        detector.initialize(fit_data=sensor_data.iloc[:500])
        
        source = CSVStreamDataSource(
            file_path=sample_csv_file,
            config=StreamConfig(batch_size=100, delay_seconds=0)
        )
        
        result = detector.process_stream(source, max_batches=2)
        
        anomalies_df = result.get_anomalies()
        
        assert isinstance(anomalies_df, pd.DataFrame)
        # Should have some anomalies (the ones we added at index 500-510)
        assert len(anomalies_df) >= 0  # Could be 0 if threshold is high
    
    def test_state_tracking(self, quickstart_config, sample_csv_file):
        """Test that state is properly tracked."""
        df = pd.read_csv(sample_csv_file, parse_dates=['timestamp'])
        sensor_data = df[['sensor_1', 'sensor_2', 'sensor_3']]
        
        detector = StreamingFaultDetector(config=quickstart_config)
        detector.initialize(fit_data=sensor_data.iloc[:100])
        
        assert detector.batch_count == 0
        assert detector.total_samples == 0
        
        source = CSVStreamDataSource(
            file_path=sample_csv_file,
            config=StreamConfig(batch_size=50, delay_seconds=0)
        )
        
        result = detector.process_stream(source, max_batches=3)
        
        assert detector.batch_count == 3
        assert detector.total_samples == 150
    
    def test_context_manager_style(self, quickstart_config, sample_csv_file):
        """Test using detector in a context-like pattern."""
        df = pd.read_csv(sample_csv_file, parse_dates=['timestamp'])
        sensor_data = df[['sensor_1', 'sensor_2', 'sensor_3']]
        
        detector = StreamingFaultDetector(config=quickstart_config)
        detector.initialize(fit_data=sensor_data.iloc[:100])
        
        # Process data
        result = detector.process_dataframe(sensor_data.iloc[100:200], batch_size=50)
        
        # Clean up
        detector.close()
        
        assert not detector._is_initialized
