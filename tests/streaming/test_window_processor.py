"""Tests for window-based streaming processor."""

import pytest
import numpy as np
import pandas as pd

from energy_fault_detector.streaming import (
    WindowProcessor,
    ContinuousStreamingFaultDetector,
    ContinuousStreamingResult,
    WindowConfig,
    WindowResult
)
from energy_fault_detector.data_sources import DataBatch


class TestWindowConfig:
    """Tests for WindowConfig."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        config = WindowConfig(window_size=10)
        
        assert config.window_size == 10
        assert config.stride == 1
        assert config.overlap == 0.0
        assert config.max_windows == 100
        assert config.window_unit == 'samples'
        assert config.min_samples == 10
    
    def test_overlap_calculation(self):
        """Test overlap calculation."""
        config = WindowConfig(window_size=10, overlap=0.5)
        
        # With 50% overlap, stride should be 5
        assert config.stride == 5
    
    def test_custom_stride(self):
        """Test custom stride overrides overlap."""
        config = WindowConfig(window_size=10, stride=3, overlap=0.5)
        
        # Custom stride should override overlap calculation
        assert config.stride == 3
    
    def test_seconds_unit(self):
        """Test window size in seconds."""
        config = WindowConfig(window_size=0.1, window_unit='seconds', sample_rate=100.0)
        
        # 0.1 seconds at 100 Hz = 10 samples
        assert config.window_size == 10
    
    def test_min_samples(self):
        """Test min_samples default."""
        config = WindowConfig(window_size=20, min_samples=15)
        
        assert config.min_samples == 15
    
    def test_min_samples_default(self):
        """Test min_samples defaults to window_size."""
        config = WindowConfig(window_size=20)
        
        assert config.min_samples == 20
    
    def test_seconds_without_sample_rate_raises(self):
        """Test that seconds unit without sample_rate raises error."""
        with pytest.raises(ValueError):
            WindowConfig(window_size=0.1, window_unit='seconds')


class TestWindowProcessor:
    """Tests for WindowProcessor."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        config = WindowConfig(window_size=10)
        
        def dummy_predict(window_data):
            return np.zeros(len(window_data), dtype=bool), np.zeros(len(window_data))
        
        processor = WindowProcessor(config=config, prediction_function=dummy_predict)
        
        assert processor.window_config.window_size == 10
        assert processor.window_count == 0
        assert processor.total_samples == 0
    
    def test_process_batch(self):
        """Test processing a batch."""
        config = WindowConfig(window_size=5, stride=1)
        
        def dummy_predict(window_data):
            return np.zeros(len(window_data), dtype=bool), np.zeros(len(window_data))
        
        processor = WindowProcessor(config=config, prediction_function=dummy_predict)
        
        # Create test batch
        data = np.arange(10).reshape(10, 1)
        batch = DataBatch(
            data=pd.DataFrame(data),
            timestamps=np.arange(10),
            batch_index=0,
            is_complete=False
        )
        
        # Process batch
        results = processor.process_batch(batch)
        
        # Should create windows: [0-4], [1-5], [2-6], [3-7], [4-8], [5-9] = 6 windows
        assert len(results) == 6
        assert all(isinstance(r, WindowResult) for r in results)
    
    def test_window_size(self):
        """Test window size."""
        config = WindowConfig(window_size=3, stride=1)
        
        def dummy_predict(window_data):
            return np.zeros(len(window_data), dtype=bool), np.zeros(len(window_data))
        
        processor = WindowProcessor(config=config, prediction_function=dummy_predict)
        
        data = np.arange(10).reshape(10, 1)
        batch = DataBatch(
            data=pd.DataFrame(data),
            timestamps=np.arange(10),
            batch_index=0
        )
        
        results = processor.process_batch(batch)
        
        # Each window should have 3 samples
        for result in results:
            assert len(result.data) == 3
    
    def test_stride(self):
        """Test stride between windows."""
        config = WindowConfig(window_size=4, stride=2)
        
        def dummy_predict(window_data):
            return np.zeros(len(window_data), dtype=bool), np.zeros(len(window_data))
        
        processor = WindowProcessor(config=config, prediction_function=dummy_predict)
        
        data = np.arange(10).reshape(10, 1)
        batch = DataBatch(
            data=pd.DataFrame(data),
            timestamps=np.arange(10),
            batch_index=0
        )
        
        results = processor.process_batch(batch)
        
        # With stride=2, windows: [0-3], [2-5], [4-7], [6-9] = 4 windows
        assert len(results) == 4
    
    def test_overlap(self):
        """Test overlapping windows."""
        config = WindowConfig(window_size=5, overlap=0.5)  # stride = 2.5 -> 2
        
        def dummy_predict(window_data):
            return np.zeros(len(window_data), dtype=bool), np.zeros(len(window_data))
        
        processor = WindowProcessor(config=config, prediction_function=dummy_predict)
        
        data = np.arange(10).reshape(10, 1)
        batch = DataBatch(
            data=pd.DataFrame(data),
            timestamps=np.arange(10),
            batch_index=0
        )
        
        results = processor.process_batch(batch)
        
        # With window_size=5, stride=2, windows: [0-4], [2-6], [4-8], [6-9] = 4 windows
        assert len(results) == 4
        
        # Check overlap
        assert results[0].end_sample == 4
        assert results[1].start_sample == 2  # Overlaps with previous window
    
    def test_preprocess_function(self):
        """Test preprocess function."""
        config = WindowConfig(window_size=3)
        
        def preprocess(window_data):
            return window_data * 2  # Double the values
        
        def predict(window_data):
            return np.zeros(len(window_data), dtype=bool), window_data.flatten()
        
        processor = WindowProcessor(
            config=config,
            prediction_function=predict,
            preprocess_function=preprocess
        )
        
        data = np.array([[1], [2], [3], [4], [5]])
        batch = DataBatch(
            data=pd.DataFrame(data),
            timestamps=np.arange(5),
            batch_index=0
        )
        
        results = processor.process_batch(batch)
        
        # Scores should be the preprocessed values (doubled)
        assert results[0].scores[0] == 2.0  # First sample * 2
    
    def test_postprocess_function(self):
        """Test postprocess function."""
        config = WindowConfig(window_size=3)
        
        def predict(window_data):
            return np.zeros(len(window_data), dtype=bool), np.zeros(len(window_data))
        
        def postprocess(window_result):
            window_result.metadata['postprocessed'] = True
            return window_result
        
        processor = WindowProcessor(
            config=config,
            prediction_function=predict,
            postprocess_function=postprocess
        )
        
        data = np.arange(6).reshape(6, 1)
        batch = DataBatch(
            data=pd.DataFrame(data),
            timestamps=np.arange(6),
            batch_index=0
        )
        
        results = processor.process_batch(batch)
        
        # Check that postprocess was applied
        for result in results:
            assert result.metadata.get('postprocessed') is True
    
    def test_reset(self):
        """Test resetting the processor."""
        config = WindowConfig(window_size=3)
        
        def dummy_predict(window_data):
            return np.zeros(len(window_data), dtype=bool), np.zeros(len(window_data))
        
        processor = WindowProcessor(config=config, prediction_function=dummy_predict)
        
        # Process some data
        data = np.arange(10).reshape(10, 1)
        batch = DataBatch(
            data=pd.DataFrame(data),
            timestamps=np.arange(10),
            batch_index=0
        )
        processor.process_batch(batch)
        
        # Reset
        processor.reset()
        
        assert processor.window_count == 0
        assert processor.total_samples == 0
        assert processor.current_windows == 0
    
    def test_state_tracking(self):
        """Test state tracking."""
        config = WindowConfig(window_size=3)
        
        def dummy_predict(window_data):
            return np.zeros(len(window_data), dtype=bool), np.zeros(len(window_data))
        
        processor = WindowProcessor(config=config, prediction_function=dummy_predict)
        
        assert processor.window_count == 0
        assert processor.total_samples == 0
        
        # Process batch
        data = np.arange(6).reshape(6, 1)
        batch = DataBatch(
            data=pd.DataFrame(data),
            timestamps=np.arange(6),
            batch_index=0
        )
        processor.process_batch(batch)
        
        assert processor.window_count > 0
        assert processor.total_samples == 6


class TestWindowResult:
    """Tests for WindowResult."""
    
    def test_basic_creation(self):
        """Test basic creation."""
        result = WindowResult(
            window_index=0,
            start_sample=0,
            end_sample=10,
            data=np.arange(10).reshape(10, 1),
            predictions=np.zeros(10, dtype=bool),
            scores=np.zeros(10),
            timestamp=1234567890.0
        )
        
        assert result.window_index == 0
        assert result.start_sample == 0
        assert result.end_sample == 10
        assert result.data.shape == (10, 1)
        assert len(result.predictions) == 10
        assert len(result.scores) == 10
        assert result.timestamp == 1234567890.0


class TestContinuousStreamingFaultDetector:
    """Tests for ContinuousStreamingFaultDetector."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        from energy_fault_detector.config import generate_quickstart_config
        
        config = generate_quickstart_config()
        window_config = WindowConfig(window_size=10)
        
        detector = ContinuousStreamingFaultDetector(
            config=config,
            window_config=window_config
        )
        
        assert detector.window_config.window_size == 10
        assert detector.base_detector is not None
    
    def test_without_window_config(self):
        """Test without window config."""
        from energy_fault_detector.config import generate_quickstart_config
        
        config = generate_quickstart_config()
        
        detector = ContinuousStreamingFaultDetector(config=config)
        
        assert detector.window_config is None
    
    def test_process_dataframe(self):
        """Test processing a DataFrame."""
        from energy_fault_detector.config import generate_quickstart_config
        
        config = generate_quickstart_config()
        window_config = WindowConfig(window_size=5, stride=1)
        
        detector = ContinuousStreamingFaultDetector(
            config=config,
            window_config=window_config
        )
        
        # Create training data
        training_data = pd.DataFrame(np.random.randn(100, 3))
        detector.initialize(fit_data=training_data)
        
        # Create test data
        test_data = pd.DataFrame(np.random.randn(20, 3))
        
        # Process
        result = detector.process_dataframe(test_data, batch_size=10)
        
        assert isinstance(result, ContinuousStreamingResult)
        assert result.total_samples == 20
        assert result.total_windows > 0
    
    def test_window_results(self):
        """Test window results."""
        from energy_fault_detector.config import generate_quickstart_config
        
        config = generate_quickstart_config()
        window_config = WindowConfig(window_size=3, stride=1)
        
        detector = ContinuousStreamingFaultDetector(
            config=config,
            window_config=window_config
        )
        
        training_data = pd.DataFrame(np.random.randn(100, 2))
        detector.initialize(fit_data=training_data)
        
        test_data = pd.DataFrame(np.random.randn(10, 2))
        result = detector.process_dataframe(test_data, batch_size=5)
        
        # Should have multiple windows
        assert len(result.window_results) > 0
        
        # Each window should have the correct size
        for wr in result.window_results:
            assert len(wr.data) == window_config.window_size
    
    def test_overlap_windows(self):
        """Test overlapping windows."""
        from energy_fault_detector.config import generate_quickstart_config
        
        config = generate_quickstart_config()
        window_config = WindowConfig(window_size=5, overlap=0.5)  # stride = 2
        
        detector = ContinuousStreamingFaultDetector(
            config=config,
            window_config=window_config
        )
        
        training_data = pd.DataFrame(np.random.randn(100, 2))
        detector.initialize(fit_data=training_data)
        
        test_data = pd.DataFrame(np.random.randn(15, 2))
        result = detector.process_dataframe(test_data, batch_size=10)
        
        # With overlap, should have more windows
        assert len(result.window_results) > 0
        
        # Check that windows overlap
        if len(result.window_results) > 1:
            first_window_end = result.window_results[0].end_sample
            second_window_start = result.window_results[1].start_sample
            assert second_window_start < first_window_end  # Overlap


class TestContinuousStreamingResult:
    """Tests for ContinuousStreamingResult."""
    
    def test_basic_creation(self):
        """Test basic creation."""
        from energy_fault_detector.streaming import StreamingResult
        
        streaming_result = StreamingResult()
        window_results = []
        
        result = ContinuousStreamingResult(
            streaming_result=streaming_result,
            window_results=window_results
        )
        
        assert result.total_samples == 0
        assert result.total_anomalies == 0
        assert result.total_windows == 0
    
    def test_with_data(self):
        """Test with data."""
        from energy_fault_detector.streaming import StreamingResult, BatchResult
        
        # Create streaming result
        streaming_result = StreamingResult()
        batch_result = BatchResult(
            batch_index=0,
            timestamps=np.array([0.0, 1.0, 2.0]),
            predictions=np.array([False, True, False]),
            scores=np.array([0.1, 0.9, 0.2]),
            processing_time=0.1
        )
        streaming_result.add_batch_result(batch_result)
        
        # Create window result
        window_result = WindowResult(
            window_index=0,
            start_sample=0,
            end_sample=5,
            data=np.arange(5).reshape(5, 1),
            predictions=np.array([False, True, False, False, True]),
            scores=np.array([0.1, 0.9, 0.2, 0.3, 0.8]),
            timestamp=1234567890.0
        )
        
        result = ContinuousStreamingResult(
            streaming_result=streaming_result,
            window_results=[window_result]
        )
        
        assert result.total_samples == 3
        assert result.total_anomalies == 1
        assert result.total_windows == 1
        assert result.avg_window_score == pytest.approx(0.66)
        assert result.max_window_score == pytest.approx(0.9)
    
    def test_get_window_dataframe(self):
        """Test get_window_dataframe."""
        from energy_fault_detector.streaming import StreamingResult
        
        streaming_result = StreamingResult()
        
        window_result = WindowResult(
            window_index=0,
            start_sample=0,
            end_sample=3,
            data=np.array([[1, 2], [3, 4], [5, 6]]),
            predictions=np.array([False, True, False]),
            scores=np.array([0.1, 0.9, 0.2]),
            timestamp=1234567890.0,
            processing_time=0.1
        )
        
        result = ContinuousStreamingResult(
            streaming_result=streaming_result,
            window_results=[window_result]
        )
        
        df = result.get_window_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 3 samples
        assert 'window_index' in df.columns
        assert 'start_sample' in df.columns
        assert 'mean_score' in df.columns
    
    def test_get_anomaly_windows(self):
        """Test get_anomaly_windows."""
        from energy_fault_detector.streaming import StreamingResult
        
        streaming_result = StreamingResult()
        
        # Window with anomalies
        window_with_anomaly = WindowResult(
            window_index=0,
            start_sample=0,
            end_sample=3,
            data=np.arange(3).reshape(3, 1),
            predictions=np.array([False, True, False]),
            scores=np.array([0.1, 0.9, 0.2]),
            timestamp=1234567890.0
        )
        
        # Window without anomalies
        window_without_anomaly = WindowResult(
            window_index=1,
            start_sample=3,
            end_sample=6,
            data=np.arange(3, 6).reshape(3, 1),
            predictions=np.array([False, False, False]),
            scores=np.array([0.1, 0.2, 0.1]),
            timestamp=1234567891.0
        )
        
        result = ContinuousStreamingResult(
            streaming_result=streaming_result,
            window_results=[window_with_anomaly, window_without_anomaly]
        )
        
        anomaly_windows = result.get_anomaly_windows()
        
        assert len(anomaly_windows) == 1
        assert anomaly_windows[0].window_index == 0
    
    def test_summary(self):
        """Test summary."""
        from energy_fault_detector.streaming import StreamingResult, BatchResult
        
        streaming_result = StreamingResult()
        batch_result = BatchResult(
            batch_index=0,
            timestamps=np.array([0.0, 1.0, 2.0]),
            predictions=np.array([False, True, False]),
            scores=np.array([0.1, 0.9, 0.2]),
            processing_time=0.1
        )
        streaming_result.add_batch_result(batch_result)
        
        window_result = WindowResult(
            window_index=0,
            start_sample=0,
            end_sample=3,
            data=np.arange(3).reshape(3, 1),
            predictions=np.array([False, True, False]),
            scores=np.array([0.1, 0.9, 0.2]),
            timestamp=1234567890.0
        )
        
        result = ContinuousStreamingResult(
            streaming_result=streaming_result,
            window_results=[window_result]
        )
        
        summary = result.summary()
        
        assert 'total_samples' in summary
        assert 'total_anomalies' in summary
        assert 'total_windows' in summary
        assert 'avg_window_score' in summary
        assert 'max_window_score' in summary
        assert 'anomaly_windows' in summary
