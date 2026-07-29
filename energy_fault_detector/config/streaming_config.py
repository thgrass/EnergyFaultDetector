"""Streaming configuration for fault detection.

This module provides configuration schemas and classes for streaming
fault detection settings.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from energy_fault_detector.config.base_config import BaseConfig


# Schema for streaming configuration
STREAMING_SCHEMA = {
    'batch_size': {'type': 'integer', 'required': False, 'default': 1000},
    'delay_seconds': {'type': 'float', 'required': False, 'default': 0.0},
    'buffer_size': {'type': 'integer', 'required': False, 'default': 1000},
    'window_size': {'type': 'integer', 'required': False},
    'online_learning': {'type': 'boolean', 'required': False, 'default': False},
    'online_learning_interval': {'type': 'integer', 'required': False, 'default': 100},
    'data_source': {
        'type': 'dict',
        'required': False,
        'schema': {
            'type': {'type': 'string', 'required': True, 'allowed': [
                'csv', 'synthetic', 'udp', 'tcp', 'mqtt'
            ]},
            'file_path': {'type': 'string', 'required': False},
            'n_samples': {'type': 'integer', 'required': False},
            'n_features': {'type': 'integer', 'required': False},
            'fault_rate': {'type': 'float', 'required': False},
            'host': {'type': 'string', 'required': False},
            'port': {'type': 'integer', 'required': False},
            'topic': {'type': 'string', 'required': False},
        }
    }
}


class StreamingConfig(BaseConfig):
    """Configuration for streaming fault detection.
    
    This class provides configuration settings specifically for streaming
    data processing and real-time fault detection.
    
    Args:
        config_filename: Path to a YAML configuration file
        config_dict: Dictionary with configuration settings
    
    Example:
        >>> config = StreamingConfig(config_dict={
        ...     'batch_size': 100,
        ...     'delay_seconds': 0.01,
        ...     'online_learning': True,
        ...     'online_learning_interval': 50
        ... })
        >>> print(config.batch_size)
        100
    """
    
    def __init__(
        self,
        config_filename: Optional[str | Path] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config_filename=config_filename, config_dict=config_dict)
        self._schema = STREAMING_SCHEMA
        self._extra_validation_checks = []
        self.read_config()
    
    @property
    def batch_size(self) -> int:
        """Number of samples per batch."""
        return self.config_dict.get('batch_size', 1000)
    
    @property
    def delay_seconds(self) -> float:
        """Delay between batches in seconds."""
        return self.config_dict.get('delay_seconds', 0.0)
    
    @property
    def buffer_size(self) -> int:
        """Maximum number of samples to buffer."""
        return self.config_dict.get('buffer_size', 1000)
    
    @property
    def window_size(self) -> Optional[int]:
        """Window size for sequence models."""
        return self.config_dict.get('window_size')
    
    @property
    def online_learning(self) -> bool:
        """Whether online learning is enabled."""
        return self.config_dict.get('online_learning', False)
    
    @property
    def online_learning_interval(self) -> int:
        """How often to perform online learning (in batches)."""
        return self.config_dict.get('online_learning_interval', 100)
    
    @property
    def data_source_config(self) -> Dict[str, Any]:
        """Configuration for the data source."""
        return self.config_dict.get('data_source', {})
    
    def to_stream_config(self) -> 'StreamConfig':
        """Convert to a StreamConfig object."""
        from energy_fault_detector.data_sources import StreamConfig
        
        return StreamConfig(
            batch_size=self.batch_size,
            delay_seconds=self.delay_seconds,
            buffer_size=self.buffer_size,
        )
