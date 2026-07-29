"""Synthetic data sources for testing streaming functionality.

Generates synthetic time series data with configurable patterns and anomalies.
"""

import time
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from energy_fault_detector.data_sources.base import DataSource, DataBatch, StreamConfig

logger = logging.getLogger('energy_fault_detector.data_sources.synthetic')


class SimulatedFaultDataSource(DataSource):
    """Generates synthetic time series data with configurable faults.
    
    This data source is useful for:
    - Testing streaming fault detection pipelines
    - Benchmarking performance
    - Demonstrating functionality without real data
    
    Args:
        n_samples: Total number of samples to generate (default: 10000)
        n_features: Number of features/sensors (default: 5)
        config: Stream configuration
        fault_rate: Probability of a fault occurring in each batch (default: 0.1)
        fault_magnitude: Magnitude of faults as multiple of std (default: 3.0)
        base_freq: Base frequency for seasonal patterns in Hz (default: 0.01)
        noise_level: Standard deviation of Gaussian noise (default: 0.1)
        timestamp_start: Start timestamp for the data (default: current time)
        feature_names: Names for the features (default: sensor_0, sensor_1, ...)
        fault_generator: Custom function to generate faults (optional)
    
    Example:
        >>> source = SimulatedFaultDataSource(
        ...     n_samples=10000,
        ...     n_features=3,
        ...     config=StreamConfig(batch_size=100, delay_seconds=0.01),
        ...     fault_rate=0.2
        ... )
        >>> for batch in source:
        ...     print(f"Batch {batch.batch_index}: {len(batch.data)} samples")
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        n_features: int = 5,
        config: Optional[StreamConfig] = None,
        fault_rate: float = 0.1,
        fault_magnitude: float = 3.0,
        base_freq: float = 0.01,
        noise_level: float = 0.1,
        timestamp_start: Optional[datetime] = None,
        feature_names: Optional[list[str]] = None,
        fault_generator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        
        self.n_samples = n_samples
        self.n_features = n_features
        self.fault_rate = fault_rate
        self.fault_magnitude = fault_magnitude
        self.base_freq = base_freq
        self.noise_level = noise_level
        self.timestamp_start = timestamp_start or datetime.now()
        self.feature_names = feature_names or [f"sensor_{i}" for i in range(n_features)]
        self.fault_generator = fault_generator
        
        # Internal state
        self._current_sample = 0
        self._rng = np.random.RandomState(42)  # Reproducible by default
    
    def open(self) -> 'SimulatedFaultDataSource':
        """Open the data source."""
        self._current_sample = 0
        self._batch_index = 0
        self._is_open = True
        logger.info(f"Opened synthetic data source with {self.n_samples} samples")
        return self
    
    def close(self) -> None:
        """Close the data source."""
        self._current_sample = 0
        self._is_open = False
        logger.debug("Synthetic data source closed")
    
    def reset(self) -> None:
        """Reset to the beginning."""
        self._current_sample = 0
        self._batch_index = 0
        self._rng = np.random.RandomState(42)  # Reset RNG for reproducibility
        logger.debug("Synthetic data source reset")
    
    def _generate_batch_data(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate a batch of synthetic data.
        
        Returns:
            Tuple of (data, timestamps)
        """
        start_idx = self._current_sample
        end_idx = min(self._current_sample + batch_size, self.n_samples)
        batch_len = end_idx - start_idx
        
        # Generate time values
        time_values = np.arange(start_idx, end_idx, dtype=float)
        
        # Generate base patterns (sine waves with different phases)
        data = np.zeros((batch_len, self.n_features))
        for i in range(self.n_features):
            # Each feature has a slightly different frequency
            freq = self.base_freq * (1 + i * 0.1)
            data[:, i] = np.sin(2 * np.pi * freq * time_values)
        
        # Add trends
        for i in range(self.n_features):
            trend = 0.01 * (i + 1) * time_values / self.n_samples
            data[:, i] += trend
        
        # Add noise
        data += self._rng.normal(0, self.noise_level, size=data.shape)
        
        # Add faults
        if self.fault_generator:
            data = self.fault_generator(data)
        elif self._rng.random() < self.fault_rate:
            # Add a fault to a random feature
            fault_feature = self._rng.randint(0, self.n_features)
            fault_start = self._rng.randint(0, batch_len)
            fault_duration = self._rng.randint(1, max(1, batch_len // 10))
            
            # Apply fault
            fault_indices = range(fault_start, min(fault_start + fault_duration, batch_len))
            data[fault_indices, fault_feature] += self.fault_magnitude * self._rng.randn(len(fault_indices))
        
        # Generate timestamps
        timestamps = np.array([
            self.timestamp_start + timedelta(seconds=t / self.base_freq)
            for t in time_values
        ], dtype='datetime64[ns]')
        
        return data, timestamps
    
    def __next__(self) -> DataBatch:
        """Get the next batch of synthetic data."""
        if not self._is_open:
            raise RuntimeError("Data source is not open. Call open() first.")
        
        if self._current_sample >= self.n_samples:
            raise StopIteration("Reached end of synthetic data")
        
        # Generate batch data
        batch_size = self.config.batch_size
        data, timestamps = self._generate_batch_data(batch_size)
        
        # Create DataFrame
        batch_df = pd.DataFrame(data, columns=self.feature_names)
        batch_df.index = pd.DatetimeIndex(timestamps)
        
        # Create metadata
        metadata = {
            'type': 'synthetic',
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'fault_rate': self.fault_rate,
            'current_sample': self._current_sample,
        }
        
        # Create and return batch
        batch = DataBatch(
            data=batch_df,
            timestamps=timestamps,
            batch_index=self._batch_index,
            is_complete=self._current_sample + batch_size >= self.n_samples,
            metadata=metadata
        )
        
        # Update state
        self._current_sample += batch_size
        self._batch_index += 1
        
        # Apply delay if configured
        if self.config.delay_seconds > 0:
            time.sleep(self.config.delay_seconds)
        
        logger.debug(f"Generated batch {batch.batch_index}: {len(batch.data)} samples")
        return batch
    
    @property
    def remaining_samples(self) -> int:
        """Number of samples remaining."""
        return max(0, self.n_samples - self._current_sample)
    
    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducible data generation."""
        self._rng = np.random.RandomState(seed)


class SineWaveDataSource(DataSource):
    """Simple sine wave data source for basic testing.
    
    Generates clean sine wave data without faults.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of sine waves to generate
        config: Stream configuration
        frequency: Frequency of the sine wave in Hz (default: 1.0)
        amplitude: Amplitude of the sine wave (default: 1.0)
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        n_features: int = 3,
        config: Optional[StreamConfig] = None,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        self.n_samples = n_samples
        self.n_features = n_features
        self.frequency = frequency
        self.amplitude = amplitude
        self._current_sample = 0
    
    def open(self) -> 'SineWaveDataSource':
        """Open the data source."""
        self._current_sample = 0
        self._batch_index = 0
        self._is_open = True
        return self
    
    def close(self) -> None:
        """Close the data source."""
        self._current_sample = 0
        self._is_open = False
    
    def reset(self) -> None:
        """Reset to the beginning."""
        self._current_sample = 0
        self._batch_index = 0
    
    def __next__(self) -> DataBatch:
        """Get the next batch of sine wave data."""
        if not self._is_open:
            raise RuntimeError("Data source is not open. Call open() first.")
        
        if self._current_sample >= self.n_samples:
            raise StopIteration("Reached end of data")
        
        batch_size = self.config.batch_size
        start_idx = self._current_sample
        end_idx = min(self._current_sample + batch_size, self.n_samples)
        batch_len = end_idx - start_idx
        
        # Generate time values
        time_values = np.linspace(0, 10 * np.pi, self.n_samples)[start_idx:end_idx]
        
        # Generate sine waves
        data = np.zeros((batch_len, self.n_features))
        for i in range(self.n_features):
            phase = 2 * np.pi * i / self.n_features
            data[:, i] = self.amplitude * np.sin(self.frequency * time_values + phase)
        
        # Create DataFrame
        feature_names = [f"sine_{i}" for i in range(self.n_features)]
        batch_df = pd.DataFrame(data, columns=feature_names)
        
        # Generate timestamps
        timestamps = np.array([
            datetime.now() + timedelta(seconds=t)
            for t in np.linspace(0, batch_len / self.frequency, batch_len)
        ], dtype='datetime64[ns]')
        
        batch = DataBatch(
            data=batch_df,
            timestamps=timestamps,
            batch_index=self._batch_index,
            is_complete=end_idx >= self.n_samples,
            metadata={'type': 'sine_wave'}
        )
        
        self._current_sample = end_idx
        self._batch_index += 1
        
        if self.config.delay_seconds > 0:
            time.sleep(self.config.delay_seconds)
        
        return batch
