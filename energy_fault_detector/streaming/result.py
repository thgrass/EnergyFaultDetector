"""Result classes for streaming fault detection."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class BatchResult:
    """Result for a single batch of data.
    
    Attributes:
        batch_index: Index of the batch
        timestamps: Timestamps for the predictions
        predictions: Anomaly predictions (boolean array)
        scores: Anomaly scores (float array)
        reconstructed: Reconstructed data from autoencoder (optional)
        processing_time: Time taken to process this batch in seconds
        metadata: Additional metadata
    """
    batch_index: int
    timestamps: np.ndarray
    predictions: np.ndarray
    scores: np.ndarray
    reconstructed: Optional[np.ndarray] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_anomalies(self) -> int:
        """Number of anomalies detected in this batch."""
        return int(np.sum(self.predictions))
    
    @property
    def anomaly_indices(self) -> np.ndarray:
        """Indices of anomalies in this batch."""
        return np.where(self.predictions)[0]


@dataclass
class StreamingResult:
    """Aggregated result for streaming fault detection.
    
    Attributes:
        batch_results: List of BatchResult objects
        total_samples: Total number of samples processed
        total_anomalies: Total number of anomalies detected
        start_time: When processing started
        end_time: When processing ended
        model_metadata: Metadata about the model used
        stream_metadata: Metadata about the data stream
    """
    batch_results: List[BatchResult] = field(default_factory=list)
    total_samples: int = 0
    total_anomalies: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    stream_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_batch_result(self, batch_result: BatchResult) -> None:
        """Add a batch result to the aggregated results."""
        self.batch_results.append(batch_result)
        self.total_samples += len(batch_result.timestamps)
        self.total_anomalies += batch_result.n_anomalies
        
        # Update end time
        if self.start_time is None:
            self.start_time = datetime.now()
        self.end_time = datetime.now()
    
    @property
    def all_predictions(self) -> np.ndarray:
        """All predictions concatenated."""
        return np.concatenate([r.predictions for r in self.batch_results])
    
    @property
    def all_scores(self) -> np.ndarray:
        """All scores concatenated."""
        return np.concatenate([r.scores for r in self.batch_results])
    
    @property
    def all_timestamps(self) -> np.ndarray:
        """All timestamps concatenated."""
        return np.concatenate([r.timestamps for r in self.batch_results])
    
    @property
    def processing_times(self) -> np.ndarray:
        """Processing times for all batches."""
        return np.array([r.processing_time for r in self.batch_results])
    
    @property
    def avg_processing_time(self) -> float:
        """Average processing time per batch."""
        if not self.batch_results:
            return 0.0
        return float(np.mean(self.processing_times))
    
    @property
    def total_processing_time(self) -> float:
        """Total processing time."""
        return float(np.sum(self.processing_times))
    
    @property
    def throughput(self) -> float:
        """Samples per second throughput."""
        if self.total_processing_time <= 0:
            return 0.0
        return self.total_samples / self.total_processing_time
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        data = {
            'timestamp': self.all_timestamps,
            'anomaly_score': self.all_scores,
            'is_anomaly': self.all_predictions,
        }
        return pd.DataFrame(data)
    
    def get_anomalies(self) -> pd.DataFrame:
        """Get a DataFrame with only the anomaly samples."""
        df = self.to_dataframe()
        return df[df['is_anomaly']].copy()
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the streaming results."""
        return {
            'total_samples': self.total_samples,
            'total_anomalies': self.total_anomalies,
            'anomaly_rate': self.total_anomalies / self.total_samples if self.total_samples > 0 else 0,
            'n_batches': len(self.batch_results),
            'avg_processing_time': self.avg_processing_time,
            'total_processing_time': self.total_processing_time,
            'throughput': self.throughput,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
        }
