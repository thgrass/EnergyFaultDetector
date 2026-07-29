"""Window-based processor for continuous streaming analysis.

This module provides classes for processing streaming data with configurable
sliding windows, enabling continuous analysis with sequence models.
"""

import time
import logging
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from energy_fault_detector.data_sources import DataBatch
from energy_fault_detector.streaming.buffer import SlidingWindowBuffer
from energy_fault_detector.streaming.result import StreamingResult, BatchResult

logger = logging.getLogger('energy_fault_detector.streaming.window_processor')


@dataclass
class WindowConfig:
    """Configuration for window-based processing.
    
    Attributes:
        window_size: Number of samples per window (required)
        stride: Number of samples between window starts (default: 1)
        overlap: Overlap between windows as fraction (0.0 to 1.0, default: 0.0)
        max_windows: Maximum number of windows to store (default: 100)
        window_unit: Unit of window_size ('samples' or 'seconds', default: 'samples')
        sample_rate: Samples per second (required if window_unit='seconds')
        min_samples: Minimum samples before processing (default: window_size)
        
    Note:
        - If overlap > 0, stride is calculated as: stride = window_size * (1 - overlap)
        - If window_unit='seconds', window_size is converted to samples using sample_rate
    """
    window_size: int
    stride: int = 1
    overlap: float = 0.0
    max_windows: int = 100
    window_unit: str = 'samples'
    sample_rate: Optional[float] = None
    min_samples: Optional[int] = None
    
    def __post_init__(self):
        if self.window_unit == 'seconds':
            if self.sample_rate is None:
                raise ValueError("sample_rate must be provided when window_unit='seconds'")
            self.window_size = int(self.window_size * self.sample_rate)
        
        if self.overlap > 0:
            self.stride = max(1, int(self.window_size * (1 - self.overlap)))
        
        if self.min_samples is None:
            self.min_samples = self.window_size


@dataclass
class WindowResult:
    """Result for a single window analysis.
    
    Attributes:
        window_index: Index of this window
        start_sample: Starting sample index
        end_sample: Ending sample index
        data: Data for this window
        predictions: Anomaly predictions for this window
        scores: Anomaly scores for this window
        timestamp: Timestamp for this window
        processing_time: Time to process this window
        metadata: Additional metadata
    """
    window_index: int
    start_sample: int
    end_sample: int
    data: np.ndarray
    predictions: np.ndarray
    scores: np.ndarray
    timestamp: float
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class WindowProcessor:
    """Processor for window-based analysis of streaming data.
    
    This class maintains a sliding window buffer and processes each window
    as it becomes available. It's designed for sequence models that require
    fixed-length input sequences.
    
    Args:
        window_config: Configuration for window processing
        prediction_function: Function to make predictions on window data
        preprocess_function: Optional function to preprocess window data
        postprocess_function: Optional function to postprocess results
        
    Example:
        >>> from energy_fault_detector.streaming import WindowProcessor, WindowConfig
        >>> 
        >>> # Define prediction function
        >>> def predict_window(window_data):
        ...     # window_data shape: (window_size, n_features)
        ...     predictions = model.predict(window_data)
        ...     return predictions, scores
        >>> 
        >>> # Create processor
        >>> processor = WindowProcessor(
        ...     window_config=WindowConfig(window_size=10, stride=1),
        ...     prediction_function=predict_window
        ... )
        >>> 
        >>> # Process batches
        >>> for batch in data_source:
        ...     results = processor.process_batch(batch)
        ...     for window_result in results:
        ...         print(f"Window {window_result.window_index}: {window_result.scores.mean():.4f}")
    """
    
    def __init__(
        self,
        window_config: WindowConfig,
        prediction_function: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        preprocess_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        postprocess_function: Optional[Callable[[WindowResult], WindowResult]] = None
    ):
        self.window_config = window_config
        self.prediction_function = prediction_function
        self.preprocess_function = preprocess_function
        self.postprocess_function = postprocess_function
        
        # Initialize window buffer
        self._window_buffer = SlidingWindowBuffer(
            window_size=window_config.window_size,
            stride=window_config.stride,
            max_windows=window_config.max_windows
        )
        
        # State tracking
        self._total_samples = 0
        self._window_count = 0
        self._last_processing_time = 0.0
        
        logger.info(f"WindowProcessor initialized: window_size={window_config.window_size}, "
                   f"stride={window_config.stride}, overlap={window_config.overlap}")
    
    def process_batch(self, batch: DataBatch) -> List[WindowResult]:
        """Process a batch of data and generate window results.
        
        Args:
            batch: DataBatch to process
            
        Returns:
            List of WindowResult objects for each new window
        """
        start_time = time.time()
        
        # Extract data from batch
        data = batch.data.values if hasattr(batch.data, 'values') else batch.data
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Add to window buffer
        new_windows = self._window_buffer.add(data)
        
        # Process each new window
        results = []
        for i, window_data in enumerate(new_windows):
            window_result = self._process_window(window_data, batch, i)
            results.append(window_result)
        
        # Update state
        self._total_samples += len(data)
        self._last_processing_time = time.time() - start_time
        
        return results
    
    def _process_window(
        self,
        window_data: np.ndarray,
        batch: DataBatch,
        window_index_in_batch: int
    ) -> WindowResult:
        """Process a single window.
        
        Args:
            window_data: Window data (shape: window_size, n_features)
            batch: Original batch
            window_index_in_batch: Index of this window within the batch
            
        Returns:
            WindowResult object
        """
        start_time = time.time()
        
        # Calculate sample indices
        start_sample = self._total_samples + window_index_in_batch * self.window_config.stride
        end_sample = start_sample + self.window_config.window_size
        
        # Preprocess if function provided
        if self.preprocess_function:
            window_data = self.preprocess_function(window_data)
        
        # Make prediction
        try:
            predictions, scores = self.prediction_function(window_data)
            
            # Ensure predictions and scores have correct shape
            if predictions.ndim == 0:
                predictions = np.array([predictions])
            if scores.ndim == 0:
                scores = np.array([scores])
            
            # If predictions/scores are for the whole window, repeat for each sample
            if len(predictions) == 1 and len(window_data) > 1:
                predictions = np.repeat(predictions, len(window_data))
                scores = np.repeat(scores, len(window_data))
            
        except Exception as e:
            logger.error(f"Error during window prediction: {e}")
            predictions = np.zeros(len(window_data), dtype=bool)
            scores = np.zeros(len(window_data), dtype=float)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Determine timestamp (use the last timestamp in the window)
        if len(batch.timestamps) > 0:
            # Use the timestamp corresponding to the end of the window
            window_end_idx = min(start_sample + self.window_config.window_size - 1, len(batch.timestamps) - 1)
            timestamp = float(batch.timestamps[window_end_idx])
        else:
            timestamp = time.time()
        
        # Create window result
        window_result = WindowResult(
            window_index=self._window_count,
            start_sample=start_sample,
            end_sample=end_sample,
            data=window_data,
            predictions=predictions,
            scores=scores,
            timestamp=timestamp,
            processing_time=processing_time,
            metadata={
                'batch_index': batch.batch_index,
                'window_index_in_batch': window_index_in_batch,
                'n_features': window_data.shape[1] if window_data.ndim > 1 else 1
            }
        )
        
        # Postprocess if function provided
        if self.postprocess_function:
            window_result = self.postprocess_function(window_result)
        
        # Update window count
        self._window_count += 1
        
        return window_result
    
    def reset(self) -> None:
        """Reset the processor state."""
        self._window_buffer.clear()
        self._total_samples = 0
        self._window_count = 0
        self._last_processing_time = 0.0
        logger.debug("WindowProcessor reset")
    
    @property
    def total_samples(self) -> int:
        """Total number of samples processed."""
        return self._total_samples
    
    @property
    def window_count(self) -> int:
        """Total number of windows processed."""
        return self._window_count
    
    @property
    def current_windows(self) -> int:
        """Number of windows currently in buffer."""
        return self._window_buffer.n_windows
    
    @property
    def last_processing_time(self) -> float:
        """Processing time for the last batch."""
        return self._last_processing_time


class ContinuousStreamingFaultDetector:
    """Enhanced StreamingFaultDetector with window-based continuous analysis.
    
    This class extends the basic StreamingFaultDetector with support for:
    - Window-based prediction for sequence models
    - Configurable window parameters
    - Continuous analysis across batches
    - Stateful processing
    
    Args:
        config: Configuration for the fault detector
        window_config: Configuration for window processing (optional)
        model_directory: Directory to save/load models
        buffer_size: Size of the data buffer
        online_learning: Whether to enable online learning
        online_learning_interval: How often to update the model
        
    Example:
        >>> from energy_fault_detector import Config
        >>> from energy_fault_detector.streaming import ContinuousStreamingFaultDetector, WindowConfig
        >>> from energy_fault_detector.config import generate_quickstart_config
        >>> 
        >>> # Create config
        >>> config = generate_quickstart_config()
        >>> 
        >>> # Create window config
        >>> window_config = WindowConfig(window_size=10, stride=1, overlap=0.5)
        >>> 
        >>> # Create continuous detector
        >>> detector = ContinuousStreamingFaultDetector(
        ...     config=config,
        ...     window_config=window_config
        ... )
        >>> 
        >>> # Initialize and process
        >>> detector.initialize(fit_data=training_data)
        >>> result = detector.process_stream(data_source)
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        window_config: Optional[WindowConfig] = None,
        model_directory: str = 'streaming_model',
        buffer_size: int = 1000,
        online_learning: bool = False,
        online_learning_interval: int = 100,
        **kwargs
    ):
        # Import here to avoid circular imports
        from energy_fault_detector.streaming.streaming_fault_detector import StreamingFaultDetector
        
        self.window_config = window_config
        self._window_processor: Optional[WindowProcessor] = None
        self._window_results: List[WindowResult] = []
        
        # Create base streaming detector
        self._base_detector = StreamingFaultDetector(
            config=config,
            model_directory=model_directory,
            buffer_size=buffer_size,
            online_learning=online_learning,
            online_learning_interval=online_learning_interval,
            **kwargs
        )
        
        # State
        self._is_initialized = False
        self._total_window_samples = 0
        
        logger.info("ContinuousStreamingFaultDetector initialized")
    
    def initialize(self, fit_data: Optional[pd.DataFrame] = None, normal_index: Optional[pd.Series] = None) -> None:
        """Initialize the detector with training data."""
        # Initialize base detector
        self._base_detector.initialize(fit_data=fit_data, normal_index=normal_index)
        
        # Initialize window processor if configured
        if self.window_config:
            self._window_processor = WindowProcessor(
                window_config=self.window_config,
                prediction_function=self._predict_window
            )
            logger.info(f"Window processor initialized: {self.window_config}")
        
        self._is_initialized = True
    
    def _predict_window(self, window_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prediction function for window processing.
        
        Args:
            window_data: Window data (shape: window_size, n_features)
            
        Returns:
            Tuple of (predictions, scores)
        """
        # Convert to DataFrame for prediction
        if window_data.ndim == 1:
            window_data = window_data.reshape(1, -1)
        
        # Create DataFrame with appropriate column names
        n_features = window_data.shape[1] if window_data.ndim > 1 else 1
        columns = [f'feature_{i}' for i in range(n_features)]
        window_df = pd.DataFrame(window_data, columns=columns)
        
        # Make prediction using the base detector
        try:
            result = self._base_detector.fault_detector.predict(sensor_data=window_df)
            
            # Extract predictions and scores
            predictions = result.predicted_anomalies.values
            scores = result.anomaly_score.values
            
            return predictions, scores
        except Exception as e:
            logger.error(f"Error during window prediction: {e}")
            return np.zeros(len(window_data), dtype=bool), np.zeros(len(window_data), dtype=float)
    
    def process_batch(self, batch: DataBatch) -> List[WindowResult]:
        """Process a batch with window-based analysis.
        
        Args:
            batch: DataBatch to process
            
        Returns:
            List of WindowResult objects
        """
        if not self._is_initialized:
            self.initialize()
        
        # If window processor is configured, use it
        if self._window_processor:
            return self._window_processor.process_batch(batch)
        else:
            # Fall back to basic batch processing
            batch_result = self._base_detector.process_batch(batch)
            
            # Convert to WindowResult for consistency
            window_result = WindowResult(
                window_index=batch.batch_index,
                start_sample=0,
                end_sample=len(batch.data),
                data=batch.data.values,
                predictions=batch_result.predictions,
                scores=batch_result.scores,
                timestamp=float(batch.timestamps[0]) if len(batch.timestamps) > 0 else time.time(),
                processing_time=batch_result.processing_time,
                metadata={'batch_index': batch.batch_index}
            )
            
            return [window_result]
    
    def process_stream(self, data_source, max_batches: Optional[int] = None) -> 'ContinuousStreamingResult':
        """Process a complete data stream with window-based analysis.
        
        Args:
            data_source: Data source to process
            max_batches: Maximum number of batches to process
            
        Returns:
            ContinuousStreamingResult with all window results
        """
        if not self._is_initialized:
            self.initialize()
        
        # Reset results
        self._window_results = []
        self._total_window_samples = 0
        
        result = StreamingResult()
        result.start_time = datetime.now()
        
        try:
            with data_source:
                for batch in data_source:
                    if max_batches is not None and batch.batch_index >= max_batches:
                        break
                    
                    # Process batch
                    window_results = self.process_batch(batch)
                    
                    # Add to results
                    for window_result in window_results:
                        self._window_results.append(window_result)
                        self._total_window_samples += len(window_result.data)
                        
                        # Create a BatchResult for compatibility
                        batch_result = BatchResult(
                            batch_index=window_result.window_index,
                            timestamps=np.array([window_result.timestamp]),
                            predictions=window_result.predictions,
                            scores=window_result.scores,
                            processing_time=window_result.processing_time,
                            metadata=window_result.metadata
                        )
                        result.add_batch_result(batch_result)
                    
                    if batch.is_complete:
                        break
        
        except KeyboardInterrupt:
            logger.info("Stream processing interrupted")
        except Exception as e:
            logger.error(f"Error during stream processing: {e}")
            raise
        finally:
            result.end_time = datetime.now()
        
        return ContinuousStreamingResult(
            streaming_result=result,
            window_results=self._window_results
        )
    
    def process_dataframe(self, df: pd.DataFrame, batch_size: int = 100) -> 'ContinuousStreamingResult':
        """Process a DataFrame with window-based analysis.
        
        Args:
            df: DataFrame to process
            batch_size: Batch size for processing
            
        Returns:
            ContinuousStreamingResult with all window results
        """
        from energy_fault_detector.data_sources import DataBatch
        
        if not self._is_initialized:
            self.initialize()
        
        # Reset results
        self._window_results = []
        self._total_window_samples = 0
        
        result = StreamingResult()
        result.start_time = datetime.now()
        
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
            window_results = self.process_batch(batch)
            
            for window_result in window_results:
                self._window_results.append(window_result)
                self._total_window_samples += len(window_result.data)
                
                batch_result = BatchResult(
                    batch_index=window_result.window_index,
                    timestamps=np.array([window_result.timestamp]),
                    predictions=window_result.predictions,
                    scores=window_result.scores,
                    processing_time=window_result.processing_time,
                    metadata=window_result.metadata
                )
                result.add_batch_result(batch_result)
        
        result.end_time = datetime.now()
        
        return ContinuousStreamingResult(
            streaming_result=result,
            window_results=self._window_results
        )
    
    @property
    def base_detector(self):
        """Get the underlying StreamingFaultDetector."""
        return self._base_detector
    
    @property
    def window_results(self) -> List[WindowResult]:
        """Get all window results."""
        return self._window_results
    
    @property
    def total_window_samples(self) -> int:
        """Total number of samples in all windows."""
        return self._total_window_samples
    
    def save_models(self, path: Optional[str] = None) -> None:
        """Save models to disk."""
        self._base_detector.save_models(path)
    
    def load_models(self, path: Optional[str] = None) -> None:
        """Load models from disk."""
        self._base_detector.load_models(path)
    
    def close(self) -> None:
        """Close the detector and clean up resources."""
        self._base_detector.close()
        if self._window_processor:
            self._window_processor.reset()
        self._is_initialized = False


@dataclass
class ContinuousStreamingResult:
    """Result for continuous streaming with window-based analysis.
    
    Attributes:
        streaming_result: The underlying StreamingResult
        window_results: List of all WindowResult objects
    """
    streaming_result: Any  # StreamingResult
    window_results: List[WindowResult]
    
    @property
    def total_samples(self) -> int:
        """Total number of samples processed."""
        return self.streaming_result.total_samples
    
    @property
    def total_anomalies(self) -> int:
        """Total number of anomalies detected."""
        return self.streaming_result.total_anomalies
    
    @property
    def total_windows(self) -> int:
        """Total number of windows processed."""
        return len(self.window_results)
    
    @property
    def avg_window_score(self) -> float:
        """Average anomaly score across all windows."""
        if not self.window_results:
            return 0.0
        
        all_scores = np.concatenate([wr.scores for wr in self.window_results])
        return float(np.mean(all_scores))
    
    @property
    def max_window_score(self) -> float:
        """Maximum anomaly score across all windows."""
        if not self.window_results:
            return 0.0
        
        all_scores = np.concatenate([wr.scores for wr in self.window_results])
        return float(np.max(all_scores))
    
    def get_window_dataframe(self) -> pd.DataFrame:
        """Get all window data as a DataFrame."""
        if not self.window_results:
            return pd.DataFrame()
        
        # Collect all window data
        all_data = []
        all_metadata = []
        
        for wr in self.window_results:
            # Flatten window data
            if wr.data.ndim > 1:
                flat_data = wr.data.reshape(-1)
            else:
                flat_data = wr.data
            
            all_data.append(flat_data)
            
            # Add metadata
            metadata = {
                'window_index': wr.window_index,
                'start_sample': wr.start_sample,
                'end_sample': wr.end_sample,
                'timestamp': wr.timestamp,
                'processing_time': wr.processing_time,
                'n_anomalies': int(np.sum(wr.predictions)),
                'mean_score': float(np.mean(wr.scores))
            }
            all_metadata.append(metadata)
        
        # Create DataFrame
        df = pd.DataFrame(np.concatenate(all_data))
        
        # Add metadata columns
        metadata_df = pd.DataFrame(all_metadata)
        
        return pd.concat([df, metadata_df], axis=1)
    
    def get_anomaly_windows(self) -> List[WindowResult]:
        """Get all windows that contain anomalies."""
        return [wr for wr in self.window_results if np.any(wr.predictions)]
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the continuous streaming results."""
        summary = self.streaming_result.summary()
        
        summary.update({
            'total_windows': self.total_windows,
            'avg_window_score': self.avg_window_score,
            'max_window_score': self.max_window_score,
            'anomaly_windows': len(self.get_anomaly_windows()),
        })
        
        return summary
