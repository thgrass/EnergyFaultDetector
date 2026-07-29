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

# Network stream sources
from energy_fault_detector.data_sources.network import (
    UDPStreamDataSource,
    PhasorUDPDataSource,
    TCPStreamDataSource,
    LineDelimitedTCPDataSource,
    MQTTStreamDataSource,
    JSONMQTTDataSource
)

__all__ = [
    "DataSource",
    "StreamConfig", 
    "DataBatch",
    "CSVStreamDataSource",
    "SimulatedFaultDataSource",
    "SineWaveDataSource",
    # Network sources
    "UDPStreamDataSource",
    "PhasorUDPDataSource",
    "TCPStreamDataSource",
    "LineDelimitedTCPDataSource",
    "MQTTStreamDataSource",
    "JSONMQTTDataSource",
]
