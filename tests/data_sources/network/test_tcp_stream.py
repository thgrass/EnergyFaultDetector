"""Tests for TCP stream data sources."""

import struct
import time
import threading
import socket
import pytest

import numpy as np
import pandas as pd

from energy_fault_detector.data_sources import (
    TCPStreamDataSource,
    LineDelimitedTCPDataSource,
    StreamConfig,
    DataBatch
)


class TestTCPStreamDataSource:
    """Tests for TCPStreamDataSource."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        source = TCPStreamDataSource(host='localhost', port=5000)
        
        assert source.host == 'localhost'
        assert source.port == 5000
        assert source.mode == 'server'
        assert source.buffer_size == 65536
        assert source.timeout == 1.0
    
    def test_client_mode(self):
        """Test client mode initialization."""
        source = TCPStreamDataSource(host='server_ip', port=5000, mode='client')
        
        assert source.mode == 'client'
    
    def test_custom_config(self):
        """Test with custom configuration."""
        config = StreamConfig(batch_size=20, timeout=0.5)
        source = TCPStreamDataSource(
            host='localhost',
            port=5000,
            config=config,
            buffer_size=32768
        )
        
        assert source.config.batch_size == 20
        assert source.config.timeout == 0.5
        assert source.buffer_size == 32768
    
    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError):
            TCPStreamDataSource(mode='invalid')
    
    def test_default_parser_text(self):
        """Test default parser with text data."""
        source = TCPStreamDataSource(host='localhost', port=5001)
        
        # Text with space-separated values
        message_data = b"1.0 2.0 3.0 4.0"
        
        data_dict, timestamps = source._default_parser(message_data, time.time())
        
        assert len(data_dict) == 4
        assert 'value_0' in data_dict
        assert data_dict['value_0'][0] == 1.0
        assert len(timestamps) == 4
    
    def test_default_parser_comma_separated(self):
        """Test default parser with comma-separated values."""
        source = TCPStreamDataSource(host='localhost', port=5002)
        
        message_data = b"1.0,2.0,3.0,4.0"
        
        data_dict, timestamps = source._default_parser(message_data, time.time())
        
        assert len(data_dict) == 4
        assert 'value_0' in data_dict
        assert data_dict['value_0'][0] == 1.0
    
    def test_default_parser_single_value(self):
        """Test default parser with single value."""
        source = TCPStreamDataSource(host='localhost', port=5003)
        
        message_data = b"42.0"
        
        data_dict, timestamps = source._default_parser(message_data, time.time())
        
        assert len(data_dict) == 1
        assert 'value' in data_dict
        assert data_dict['value'][0] == 42.0
    
    def test_default_parser_binary(self):
        """Test default parser with binary float64 data."""
        source = TCPStreamDataSource(host='localhost', port=5004)
        
        values = [1.0, 2.0, 3.0]
        message_data = struct.pack('3d', *values)
        
        data_dict, timestamps = source._default_parser(message_data, time.time())
        
        assert len(data_dict) == 3
        assert 'value_0' in data_dict
        assert data_dict['value_0'][0] == 1.0
    
    def test_default_parser_json(self):
        """Test default parser with JSON data."""
        source = TCPStreamDataSource(host='localhost', port=5005)
        
        message_data = b'{"temperature": 25.5, "humidity": 60.0}'
        
        data_dict, timestamps = source._default_parser(message_data, time.time())
        
        assert 'message' in data_dict  # Falls back to message for non-numeric
    
    def test_statistics(self):
        """Test statistics tracking."""
        source = TCPStreamDataSource(host='localhost', port=5006)
        source.open()
        
        # Simulate receiving some data
        source._messages_received = 100
        source._bytes_received = 10000
        source._start_time = time.time()
        
        stats = source.statistics
        
        assert stats['messages_received'] == 100
        assert stats['bytes_received'] == 10000
        assert 'messages_per_second' in stats
        assert 'bytes_per_second' in stats
        
        source.close()
    
    def test_reset(self):
        """Test resetting the source."""
        source = TCPStreamDataSource(host='localhost', port=5007)
        source.open()
        
        # Simulate some data
        source._messages_received = 50
        source._batch_index = 5
        
        source.reset()
        
        assert source._messages_received == 0
        assert source._batch_index == 0
        
        source.close()


class TestLineDelimitedTCPDataSource:
    """Tests for LineDelimitedTCPDataSource."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        source = LineDelimitedTCPDataSource(host='localhost', port=5008)
        
        assert source.host == 'localhost'
        assert source.port == 5008
        assert source.delimiter == b'\n'
    
    def test_custom_delimiter(self):
        """Test with custom delimiter."""
        source = LineDelimitedTCPDataSource(
            host='localhost',
            port=5009,
            delimiter=b'|'
        )
        
        assert source.delimiter == b'|'
