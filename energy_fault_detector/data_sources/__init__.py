"""Data source module for streaming data support.

This module provides abstract and concrete implementations for various data sources,
including real-time streams (UDP, TCP, MQTT) and synthetic streams (CSV, Parquet).
"""

from energy_fault_detector.data_sources.base import DataSource, StreamConfig, DataBatch
from energy_fault_detector.data_sources.csv_stream import CSVStreamDataSource
from energy_fault_detector.data_sources.synthetic import (
    SimulatedFaultDataSource,
    SineWaveDataSource
)

# Real stream sources (will be added in Phase 2)
# from energy_fault_detector.data_sources.udp_stream import UDPPhasorDataSource
# from energy_fault_detector.data_sources.tcp_stream import TCPStreamDataSource

__all__ = [
    "DataSource",
    "StreamConfig", 
    "DataBatch",
    "CSVStreamDataSource",
    "SimulatedFaultDataSource",
    "SineWaveDataSource",
]
