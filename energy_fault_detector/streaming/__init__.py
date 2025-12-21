"""Streaming subpackage.

This package defines the streaming abstractions used by the Energy Fault Detector.
Concrete stream implementations such as synchrophasor or file-based streams
are available in their respective modules but are not imported here.
"""

from .data_stream import DataStream  # base class for all streams
from .stream_fault_detector import (
    StreamingFaultDetector,
)  # fault detector that consumes streams

__all__ = ["DataStream", "StreamingFaultDetector"]
