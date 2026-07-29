"""Streaming module for real-time fault detection.

This module provides classes and utilities for processing data streams
and performing real-time fault detection.
"""

from energy_fault_detector.streaming.streaming_fault_detector import StreamingFaultDetector
from energy_fault_detector.streaming.buffer import DataBuffer, SlidingWindowBuffer
from energy_fault_detector.streaming.result import StreamingResult, BatchResult

__all__ = [
    "StreamingFaultDetector",
    "DataBuffer",
    "SlidingWindowBuffer",
    "StreamingResult",
    "BatchResult",
]
