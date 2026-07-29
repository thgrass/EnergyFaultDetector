"""Tests for synthetic data sources."""

import pytest
import numpy as np
import pandas as pd

from energy_fault_detector.data_sources import (
    SimulatedFaultDataSource,
    SineWaveDataSource,
    StreamConfig,
    DataBatch
)


class TestSimulatedFaultDataSource:
    """Tests for SimulatedFaultDataSource."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        source = SimulatedFaultDataSource(n_samples=1000, n_features=3)
        assert source.n_samples == 1000
        assert source.n_features == 3
        assert source.fault_rate == 0.1
    
    def test_custom_config(self):
        """Test with custom configuration."""
        config = StreamConfig(batch_size=50, delay_seconds=0.01)
        source = SimulatedFaultDataSource(
            n_samples=500,
            n_features=2,
            config=config,
            fault_rate=0.2,
            fault_magnitude=5.0
        )
        assert source.config.batch_size == 50
        assert source.config.delay_seconds == 0.01
        assert source.fault_rate == 0.2
        assert source.fault_magnitude == 5.0
    
    def test_open_and_close(self):
        """Test opening and closing the source."""
        source = SimulatedFaultDataSource(n_samples=100, n_features=2)
        
        assert not source.is_open
        source.open()
        assert source.is_open
        
        source.close()
        assert not source.is_open
    
    def test_iteration(self):
        """Test iterating through batches."""
        config = StreamConfig(batch_size=20, delay_seconds=0)
        source = SimulatedFaultDataSource(n_samples=100, n_features=3, config=config)
        
        batches = list(source)
        
        assert len(batches) == 5  # 100 samples / 20 per batch = 5 batches
        assert all(isinstance(batch, DataBatch) for batch in batches)
        
        # Check batch sizes
        for batch in batches:
            assert len(batch.data) == 20
    
    def test_batch_metadata(self):
        """Test batch metadata."""
        config = StreamConfig(batch_size=10, delay_seconds=0)
        source = SimulatedFaultDataSource(n_samples=50, n_features=2, config=config)
        
        first_batch = next(iter(source))
        
        assert first_batch.batch_index == 0
        assert 'type' in first_batch.metadata
        assert first_batch.metadata['type'] == 'synthetic'
        assert first_batch.metadata['n_samples'] == 50
        assert first_batch.metadata['n_features'] == 2
    
    def test_data_shape(self):
        """Test that generated data has correct shape."""
        config = StreamConfig(batch_size=10, delay_seconds=0)
        source = SimulatedFaultDataSource(n_samples=50, n_features=4, config=config)
        
        first_batch = next(iter(source))
        
        assert first_batch.data.shape == (10, 4)
        assert len(first_batch.timestamps) == 10
    
    def test_reproducibility(self):
        """Test that data is reproducible with same seed."""
        config = StreamConfig(batch_size=10, delay_seconds=0)
        
        source1 = SimulatedFaultDataSource(n_samples=50, n_features=2, config=config)
        source1.set_seed(42)
        source1.open()
        
        source2 = SimulatedFaultDataSource(n_samples=50, n_features=2, config=config)
        source2.set_seed(42)
        source2.open()
        
        batch1 = next(source1)
        batch2 = next(source2)
        
        np.testing.assert_array_equal(batch1.data.values, batch2.data.values)
    
    def test_reset(self):
        """Test resetting the source."""
        config = StreamConfig(batch_size=10, delay_seconds=0)
        source = SimulatedFaultDataSource(n_samples=50, n_features=2, config=config)
        
        # Read first batch
        source.open()
        first_batch = next(source)
        
        # Reset and read again
        source.reset()
        source.open()
        first_batch_again = next(source)
        
        # Should be the same (due to seed reset)
        np.testing.assert_array_equal(
            first_batch.data.values,
            first_batch_again.data.values
        )
    
    def test_remaining_samples(self):
        """Test remaining samples property."""
        config = StreamConfig(batch_size=10, delay_seconds=0)
        source = SimulatedFaultDataSource(n_samples=50, n_features=2, config=config)
        
        source.open()
        assert source.remaining_samples == 50
        
        next(source)  # Read 10 samples
        assert source.remaining_samples == 40
        
        next(source)  # Read another 10
        assert source.remaining_samples == 30
    
    def test_feature_names(self):
        """Test custom feature names."""
        config = StreamConfig(batch_size=10, delay_seconds=0)
        source = SimulatedFaultDataSource(
            n_samples=50,
            n_features=3,
            config=config,
            feature_names=['temp', 'pressure', 'vibration']
        )
        
        first_batch = next(iter(source))
        
        assert list(first_batch.data.columns) == ['temp', 'pressure', 'vibration']
    
    def test_fault_generation(self):
        """Test that faults are generated."""
        config = StreamConfig(batch_size=100, delay_seconds=0)
        source = SimulatedFaultDataSource(
            n_samples=1000,
            n_features=1,
            config=config,
            fault_rate=1.0,  # Always generate faults
            fault_magnitude=10.0
        )
        
        # With fault_rate=1.0, every batch should have faults
        batches = list(source)
        
        # Check that at least some values are large (faults)
        all_data = np.concatenate([b.data.values for b in batches])
        assert np.any(np.abs(all_data) > 5.0)  # Should have some large values


class TestSineWaveDataSource:
    """Tests for SineWaveDataSource."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        source = SineWaveDataSource(n_samples=1000, n_features=2)
        assert source.n_samples == 1000
        assert source.n_features == 2
        assert source.frequency == 1.0
        assert source.amplitude == 1.0
    
    def test_iteration(self):
        """Test iterating through batches."""
        config = StreamConfig(batch_size=20, delay_seconds=0)
        source = SineWaveDataSource(n_samples=100, n_features=2, config=config)
        
        batches = list(source)
        
        assert len(batches) == 5
        assert all(isinstance(batch, DataBatch) for batch in batches)
    
    def test_sine_wave_pattern(self):
        """Test that data follows sine wave pattern."""
        config = StreamConfig(batch_size=100, delay_seconds=0)
        source = SineWaveDataSource(
            n_samples=100,
            n_features=1,
            config=config,
            frequency=1.0,
            amplitude=2.0
        )
        
        batch = next(iter(source))
        data = batch.data.values.flatten()
        
        # Check that data is within expected range
        assert np.all(np.abs(data) <= 2.0)
        
        # Check that it's approximately a sine wave
        # The mean should be close to 0
        assert np.abs(np.mean(data)) < 0.1
    
    def test_multiple_features(self):
        """Test multiple features with phase differences."""
        config = StreamConfig(batch_size=100, delay_seconds=0)
        source = SineWaveDataSource(
            n_samples=100,
            n_features=3,
            config=config,
            frequency=1.0
        )
        
        batch = next(iter(source))
        
        assert batch.data.shape == (100, 3)
        
        # Each feature should have different phase
        # Check that they're not all identical
        for i in range(3):
            for j in range(i + 1, 3):
                assert not np.allclose(batch.data.iloc[:, i], batch.data.iloc[:, j])
