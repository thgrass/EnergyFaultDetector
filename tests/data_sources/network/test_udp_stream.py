"""Tests for UDP stream data sources."""

import struct
import time
import threading
import socket
import pytest

import numpy as np
import pandas as pd

from energy_fault_detector.data_sources import (
    UDPStreamDataSource,
    PhasorUDPDataSource,
    StreamConfig,
    DataBatch
)


class TestUDPStreamDataSource:
    """Tests for UDPStreamDataSource."""
    
    @pytest.fixture
    def udp_server(self, port=5555):
        """Create a simple UDP server for testing."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('localhost', port))
        
        def send_data(data: bytes, count: int = 1):
            for _ in range(count):
                sock.sendto(data, ('localhost', port))
                time.sleep(0.01)
        
        yield send_data
        sock.close()
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        source = UDPStreamDataSource(host='localhost', port=5000)
        
        assert source.host == 'localhost'
        assert source.port == 5000
        assert source.buffer_size == 65536
        assert source.timeout == 1.0
    
    def test_custom_config(self):
        """Test with custom configuration."""
        config = StreamConfig(batch_size=10, timeout=0.5)
        source = UDPStreamDataSource(
            host='0.0.0.0',
            port=5000,
            config=config,
            buffer_size=32768
        )
        
        assert source.config.batch_size == 10
        assert source.config.timeout == 0.5
        assert source.buffer_size == 32768
    
    def test_open_and_close(self):
        """Test opening and closing the source."""
        source = UDPStreamDataSource(host='localhost', port=5556)
        
        assert not source.is_open
        source.open()
        assert source.is_open
        
        source.close()
        assert not source.is_open
    
    def test_context_manager(self):
        """Test using as context manager."""
        with UDPStreamDataSource(host='localhost', port=5557) as source:
            assert source.is_open
        
        assert not source.is_open
    
    def test_default_parser_float64(self):
        """Test default parser with float64 data."""
        source = UDPStreamDataSource(host='localhost', port=5558)
        
        # Create test data
        values = [1.0, 2.0, 3.0, 4.0]
        packet_data = struct.pack('4d', *values)
        
        # Parse the data
        data_dict, timestamps = source._default_parser(packet_data, time.time())
        
        assert len(data_dict) == 4
        assert 'value_0' in data_dict
        assert data_dict['value_0'][0] == 1.0
        assert len(timestamps) == 4
    
    def test_default_parser_single_value(self):
        """Test default parser with single value."""
        source = UDPStreamDataSource(host='localhost', port=5559)
        
        # Single float64
        packet_data = struct.pack('d', 42.0)
        
        data_dict, timestamps = source._default_parser(packet_data, time.time())
        
        assert len(data_dict) == 1
        assert 'value_0' in data_dict
        assert data_dict['value_0'][0] == 42.0
    
    def test_statistics(self):
        """Test statistics tracking."""
        source = UDPStreamDataSource(host='localhost', port=5560)
        source.open()
        
        # Simulate receiving some data
        source._packets_received = 100
        source._bytes_received = 10000
        source._start_time = time.time()
        
        stats = source.statistics
        
        assert stats['packets_received'] == 100
        assert stats['bytes_received'] == 10000
        assert 'packets_per_second' in stats
        assert 'bytes_per_second' in stats
        
        source.close()
    
    def test_reset(self):
        """Test resetting the source."""
        source = UDPStreamDataSource(host='localhost', port=5561)
        source.open()
        
        # Simulate some data
        source._packets_received = 50
        source._batch_index = 5
        
        source.reset()
        
        assert source._packets_received == 0
        assert source._batch_index == 0
        
        source.close()


class TestPhasorUDPDataSource:
    """Tests for PhasorUDPDataSource."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        source = PhasorUDPDataSource(host='localhost', port=4712, n_phasors=3)
        
        assert source.host == 'localhost'
        assert source.port == 4712
        assert source.n_phasors == 3
        assert source.n_analog == 0
        assert source.n_digital == 0
    
    def test_custom_phasor_config(self):
        """Test with custom phasor configuration."""
        source = PhasorUDPDataSource(
            host='0.0.0.0',
            port=4713,
            n_phasors=6,
            n_analog=4,
            n_digital=2
        )
        
        assert source.n_phasors == 6
        assert source.n_analog == 4
        assert source.n_digital == 2
    
    def test_phasor_parser(self):
        """Test phasor packet parsing."""
        source = PhasorUDPDataSource(host='localhost', port=4714, n_phasors=2)
        
        # Create a simplified phasor packet
        # Header (16 bytes) + phasor data (2 phasors * 2 floats * 4 bytes = 16 bytes)
        header = b'\x00' * 16
        
        # Phasor data: real and imaginary for each phasor (as float32)
        phasor_data = struct.pack('4f', 1.0, 0.0, 0.0, 1.0)  # (1+0j, 0+1j)
        
        packet_data = header + phasor_data
        timestamp = time.time()
        
        data_dict, timestamps = source._parse_phasor_packet(
            packet_data, timestamp, n_phasors=2, n_analog=0, n_digital=0
        )
        
        # Should have magnitude and phase for each phasor
        assert 'phasor_0_mag' in data_dict
        assert 'phasor_0_phase' in data_dict
        assert 'phasor_1_mag' in data_dict
        assert 'phasor_1_phase' in data_dict
        
        # Check magnitudes (should be 1.0 and 1.0)
        assert abs(data_dict['phasor_0_mag'][0] - 1.0) < 0.01
        assert abs(data_dict['phasor_1_mag'][0] - 1.0) < 0.01
        
        # Check phases (0 and 90 degrees)
        assert abs(data_dict['phasor_0_phase'][0] - 0.0) < 0.1
        assert abs(data_dict['phasor_1_phase'][0] - 90.0) < 0.1
