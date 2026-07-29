"""Streaming fault detector for real-time anomaly detection.

This module provides the StreamingFaultDetector class which enables
real-time fault detection on streaming data sources.
"""

import time
import logging
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from energy_fault_detector.data_sources import DataSource, DataBatch
from energy_fault_detector.streaming.buffer import DataBuffer, SlidingWindowBuffer
from energy_fault_detector.streaming.result import StreamingResult, BatchResult
from energy_fault_detector.config import Config
from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.core.fault_detection_result import FaultDetectionResult

logger = logging.getLogger('energy_fault_detector.streaming')


class StreamingFaultDetector:
    """Fault detector for processing streaming data in real-time.
    
    This class wraps a FaultDetector and provides methods for processing
    data from stream sources batch by batch. It handles:
    - Incremental data preprocessing
    - Batch prediction
    - Result aggregation
    - Optional online learning
    
    Args:
        config: Configuration for the fault detector
        model_directory: Directory to save/load models
        buffer_size: Size of the data buffer (default: 1000)
        window_size: Window size for sequence models (default: None)
        online_learning: Whether to enable online learning (default: False)
        online_learning_interval: How often to update the model (default: 100 batches)
        
    Example:
        >>> from energy_fault_detector import Config
        >>> from energy_fault_detector.data_sources import CSVStreamDataSource, StreamConfig
        >>> from energy_fault_detector.streaming import StreamingFaultDetector
        >>> 
        >>> # Create config and data source
        >>> config = Config.from_file("config.yaml")
        >>> source = CSVStreamDataSource("data.csv", config=StreamConfig(batch_size=100))
        >>> 
        >>> # Create streaming detector
        >>> detector = StreamingFaultDetector(config=config)
        >>> 
        >>> # Process stream
        >>> result = detector.process_stream(source)
        >>> print(f"Detected {result.total_anomalies} anomalies")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        model_directory: Union[str, Path] = 'streaming_model',
        buffer_size: int = 1000,
        window_size: Optional[int] = None,
        online_learning: bool = False,
        online_learning_interval: int = 100,
        **kwargs
    ):
        self.config = config
        self.model_directory = Path(model_directory)
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.online_learning = online_learning
        self.online_learning_interval = online_learning_interval
        
        # Initialize components
        self._fault_detector: Optional[FaultDetector] = None
        self._data_buffer: Optional[DataBuffer] = None
        self._window_buffer: Optional[SlidingWindowBuffer] = None
        
        # State
        self._is_initialized = False
        self._batch_count = 0
        self._total_samples = 0
        self._last_online_learning_batch = 0
        
        # Results
        self._result: Optional[StreamingResult] = None
        
        # Additional kwargs for FaultDetector
        self._fault_detector_kwargs = kwargs
        
        logger.info("StreamingFaultDetector initialized")
    
    def initialize(self, fit_data: Optional[pd.DataFrame] = None, normal_index: Optional[pd.Series] = None) -> None:
        """Initialize the fault detector with training data.
        
        Args:
            fit_data: Training data for initializing the model
            normal_index: Boolean index indicating normal data points
        """
        if self.config is None:
            raise ValueError("Config must be provided for initialization")
        
        logger.info("Initializing StreamingFaultDetector...")
        
        # Create fault detector
        self._fault_detector = FaultDetector(
            config=self.config,
            model_directory=self.model_directory,
            **self._fault_detector_kwargs
        )
        
        # Initialize buffers
        self._data_buffer = DataBuffer(max_size=self.buffer_size)
        
        if self.window_size:
            self._window_buffer = SlidingWindowBuffer(window_size=self.window_size)
        
        # Train the model if data is provided
        if fit_data is not None:
            logger.info(f"Training model with {len(fit_data)} samples...")
            self._fault_detector.fit(
                sensor_data=fit_data,
                normal_index=normal_index,
                save_models=True
            )
        else:
            # Try to load existing model
            try:
                self._fault_detector.load_models(self.model_directory)
                logger.info(f"Loaded existing model from {self.model_directory}")
            except Exception as e:
                logger.warning(f"Could not load existing model: {e}")
        
        # Initialize result tracking
        self._result = StreamingResult()
        self._result.start_time = datetime.now()
        
        # Add model metadata
        if self._fault_detector:
            self._result.model_metadata = {
                'autoencoder': str(type(self._fault_detector.autoencoder).__name__),
                'anomaly_score': str(type(self._fault_detector.anomaly_score).__name__),
                'threshold_selector': str(type(self._fault_detector.threshold_selector).__name__),
            }
        
        self._is_initialized = True
        logger.info("StreamingFaultDetector initialized successfully")
    
    def _ensure_initialized(self) -> None:
        """Ensure the detector is initialized."""
        if not self._is_initialized:
            self.initialize()
    
    def process_batch(self, batch: DataBatch) -> BatchResult:
        """Process a single batch of data.
        
        Args:
            batch: DataBatch to process
            
        Returns:
            BatchResult with predictions and scores
        """
        self._ensure_initialized()
        
        start_time = time.time()
        
        # Extract data from batch
        data = batch.data
        timestamps = batch.timestamps
        
        # Add to buffer
        if self._data_buffer:
            self._data_buffer.add_dataframe(data)
        
        # Add to window buffer if configured
        if self._window_buffer:
            self._window_buffer.add_dataframe(data)
        
        # Prepare data for prediction
        # For now, use the raw batch data
        # In the future, we might use buffered/windowed data
        sensor_data = data
        
        # Make prediction
        try:
            result = self._fault_detector.predict(sensor_data=sensor_data)
            
            # Extract predictions and scores
            predictions = result.predicted_anomalies.values if hasattr(result.predicted_anomalies, 'values') else result.predicted_anomalies
            scores = result.anomaly_score.values if hasattr(result.anomaly_score, 'values') else result.anomaly_score
            
            # Handle case where predictions might be a single value
            if np.isscalar(predictions):
                predictions = np.array([predictions] * len(data))
            if np.isscalar(scores):
                scores = np.array([scores] * len(data))
            
            # Ensure predictions and scores have the right length
            if len(predictions) != len(data):
                logger.warning(f"Predictions length {len(predictions)} doesn't match data length {len(data)}")
                # Try to repeat or truncate
                if len(predictions) == 1:
                    predictions = np.repeat(predictions, len(data))
                else:
                    predictions = predictions[:len(data)]
            
            if len(scores) != len(data):
                logger.warning(f"Scores length {len(scores)} doesn't match data length {len(data)}")
                if len(scores) == 1:
                    scores = np.repeat(scores, len(data))
                else:
                    scores = scores[:len(data)]
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Return empty results on error
            predictions = np.zeros(len(data), dtype=bool)
            scores = np.zeros(len(data), dtype=float)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create batch result
        batch_result = BatchResult(
            batch_index=batch.batch_index,
            timestamps=timestamps,
            predictions=predictions,
            scores=scores,
            processing_time=processing_time,
            metadata={
                'batch_size': len(data),
                'data_shape': data.shape,
            }
        )
        
        # Update state
        self._batch_count += 1
        self._total_samples += len(data)
        
        # Online learning
        if self.online_learning and self._batch_count >= self._last_online_learning_batch + self.online_learning_interval:
            self._perform_online_learning()
            self._last_online_learning_batch = self._batch_count
        
        return batch_result
    
    def _perform_online_learning(self) -> None:
        """Perform online learning with buffered data."""
        if not self.online_learning or self._data_buffer is None or self._fault_detector is None:
            return
        
        logger.info(f"Performing online learning with {self._data_buffer.size} buffered samples...")
        
        try:
            # Get buffered data
            buffered_data = self._data_buffer.to_dataframe()
            
            if len(buffered_data) > 0:
                # For now, just refit the autoencoder (simplified approach)
                # In a production system, you might want more sophisticated online learning
                self._fault_detector.autoencoder.fit(
                    buffered_data.values,
                    buffered_data.values,  # Autoencoders learn to reconstruct their input
                    epochs=1,  # Just one epoch for online learning
                    verbose=0
                )
                logger.info("Online learning completed")
        except Exception as e:
            logger.error(f"Error during online learning: {e}")
    
    def process_stream(
        self,
        data_source: DataSource,
        max_batches: Optional[int] = None
    ) -> StreamingResult:
        """Process a complete data stream.
        
        Args:
            data_source: Data source to process
            max_batches: Maximum number of batches to process (None for all)
            
        Returns:
            StreamingResult with all predictions
        """
        self._ensure_initialized()
        
        # Reset result
        self._result = StreamingResult()
        self._result.start_time = datetime.now()
        self._batch_count = 0
        self._total_samples = 0
        
        # Add stream metadata
        self._result.stream_metadata = {
            'source_type': type(data_source).__name__,
            'batch_size': data_source.config.batch_size,
            'delay_seconds': data_source.config.delay_seconds,
        }
        
        logger.info(f"Starting to process stream from {type(data_source).__name__}...")
        
        try:
            with data_source:
                for batch in data_source:
                    # Check if we've reached max_batches
                    if max_batches is not None and self._batch_count >= max_batches:
                        logger.info(f"Reached maximum of {max_batches} batches")
                        break
                    
                    # Process batch
                    batch_result = self.process_batch(batch)
                    
                    # Add to result
                    self._result.add_batch_result(batch_result)
                    
                    # Log progress
                    if self._batch_count % 10 == 0:
                        logger.info(f"Processed batch {self._batch_count}: {batch_result.n_anomalies} anomalies")
                    
                    # Check for early termination
                    if batch.is_complete:
                        logger.info("Reached end of stream")
                        break
        
        except KeyboardInterrupt:
            logger.info("Stream processing interrupted by user")
        except Exception as e:
            logger.error(f"Error during stream processing: {e}")
            raise
        finally:
            self._result.end_time = datetime.now()
            logger.info(f"Stream processing completed. Total: {self._total_samples} samples, {self._result.total_anomalies} anomalies")
        
        return self._result
    
    def process_dataframe(self, df: pd.DataFrame, batch_size: int = 100) -> StreamingResult:
        """Process a DataFrame in batches.
        
        This is a convenience method for testing with existing DataFrames.
        
        Args:
            df: DataFrame to process
            batch_size: Batch size for processing
            
        Returns:
            StreamingResult with predictions
        """
        from energy_fault_detector.data_sources import DataBatch
        
        self._ensure_initialized()
        
        # Reset result
        self._result = StreamingResult()
        self._result.start_time = datetime.now()
        self._batch_count = 0
        self._total_samples = 0
        
        # Split DataFrame into batches
        n_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            
            batch_data = df.iloc[start_idx:end_idx]
            timestamps = batch_data.index.values if isinstance(batch_data.index, pd.DatetimeIndex) else np.arange(start_idx, end_idx)
            
            batch = DataBatch(
                data=batch_data,
                timestamps=timestamps,
                batch_index=i,
                is_complete=end_idx >= len(df)
            )
            
            # Process batch
            batch_result = self.process_batch(batch)
            self._result.add_batch_result(batch_result)
        
        self._result.end_time = datetime.now()
        return self._result
    
    @property
    def result(self) -> Optional[StreamingResult]:
        """Get the current streaming result."""
        return self._result
    
    @property
    def batch_count(self) -> int:
        """Number of batches processed."""
        return self._batch_count
    
    @property
    def total_samples(self) -> int:
        """Total number of samples processed."""
        return self._total_samples
    
    @property
    def fault_detector(self) -> Optional[FaultDetector]:
        """Get the underlying FaultDetector."""
        return self._fault_detector
    
    def save_models(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save the models to disk.
        
        Args:
            path: Path to save models (defaults to model_directory)
        """
        if self._fault_detector:
            save_path = path or self.model_directory
            self._fault_detector.save_models(save_path)
            logger.info(f"Models saved to {save_path}")
    
    def load_models(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load models from disk.
        
        Args:
            path: Path to load models from (defaults to model_directory)
        """
        self._ensure_initialized()
        if self._fault_detector:
            load_path = path or self.model_directory
            self._fault_detector.load_models(load_path)
            logger.info(f"Models loaded from {load_path}")
    
    def close(self) -> None:
        """Close the streaming detector and clean up resources."""
        if self._data_buffer:
            self._data_buffer.clear()
        if self._window_buffer:
            self._window_buffer.clear()
        self._is_initialized = False
        logger.info("StreamingFaultDetector closed")
