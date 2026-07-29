"""Tests for MQTT stream data sources."""

import json
import time
import pytest

import numpy as np
import pandas as pd

# Check if MQTT is available
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

from energy_fault_detector.data_sources import (
    MQTTStreamDataSource,
    JSONMQTTDataSource,
    StreamConfig,
    DataBatch
)


@pytest.mark.skipif(not MQTT_AVAILABLE, reason="paho-mqtt not installed")
class TestMQTTStreamDataSource:
    """Tests for MQTTStreamDataSource."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        source = MQTTStreamDataSource(broker='localhost', port=1883)
        
        assert source.broker == 'localhost'
        assert source.port == 1883
        assert source.topics == ['#']
        assert source.qos == 1
        assert source.clean_session is True
    
    def test_custom_config(self):
        """Test with custom configuration."""
        config = StreamConfig(batch_size=50, timeout=0.5)
        source = MQTTStreamDataSource(
            broker='mqtt.example.com',
            port=1884,
            topics=['sensors/#', 'status/#'],
            config=config,
            client_id='test_client',
            qos=2
        )
        
        assert source.broker == 'mqtt.example.com'
        assert source.port == 1884
        assert source.topics == ['sensors/#', 'status/#']
        assert source.client_id == 'test_client'
        assert source.qos == 2
        assert source.config.batch_size == 50
    
    def test_authentication(self):
        """Test with authentication."""
        source = MQTTStreamDataSource(
            broker='localhost',
            username='test_user',
            password='test_pass'
        )
        
        assert source.username == 'test_user'
        assert source.password == 'test_pass'
    
    def test_default_parser_text(self):
        """Test default parser with text data."""
        source = MQTTStreamDataSource(broker='localhost')
        
        topic = 'sensors/temperature'
        payload = b"25.5"
        timestamp = time.time()
        
        data_dict, timestamps = source._default_parser(topic, payload, timestamp)
        
        assert f"{topic}/value" in data_dict
        assert data_dict[f"{topic}/value"][0] == 25.5
        assert len(timestamps) == 1
    
    def test_default_parser_json(self):
        """Test default parser with JSON data."""
        source = MQTTStreamDataSource(broker='localhost')
        
        topic = 'sensors'
        payload = b'{"temperature": 25.5, "humidity": 60.0}'
        timestamp = time.time()
        
        data_dict, timestamps = source._default_parser(topic, payload, timestamp)
        
        assert f"{topic}/temperature" in data_dict
        assert f"{topic}/humidity" in data_dict
        assert data_dict[f"{topic}/temperature"][0] == 25.5
        assert data_dict[f"{topic}/humidity"][0] == 60.0
    
    def test_default_parser_json_array(self):
        """Test default parser with JSON array."""
        source = MQTTStreamDataSource(broker='localhost')
        
        topic = 'sensors/values'
        payload = b'[1.0, 2.0, 3.0, 4.0]'
        timestamp = time.time()
        
        data_dict, timestamps = source._default_parser(topic, payload, timestamp)
        
        assert len(data_dict) == 4
        assert f"{topic}/value_0" in data_dict
        assert data_dict[f"{topic}/value_0"][0] == 1.0
    
    def test_default_parser_comma_separated(self):
        """Test default parser with comma-separated values."""
        source = MQTTStreamDataSource(broker='localhost')
        
        topic = 'sensors/data'
        payload = b"1.0,2.0,3.0"
        timestamp = time.time()
        
        data_dict, timestamps = source._default_parser(topic, payload, timestamp)
        
        assert len(data_dict) == 3
        assert f"{topic}/value_0" in data_dict
    
    def test_default_parser_binary(self):
        """Test default parser with binary data."""
        source = MQTTStreamDataSource(broker='localhost')
        
        topic = 'sensors/binary'
        payload = b'\x00\x01\x02\x03'  # Non-UTF-8 data
        timestamp = time.time()
        
        data_dict, timestamps = source._default_parser(topic, payload, timestamp)
        
        assert f"{topic}/raw_data" in data_dict
    
    def test_statistics(self):
        """Test statistics tracking."""
        source = MQTTStreamDataSource(broker='localhost')
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
        assert 'subscribed_topics' in stats
        
        source.close()
    
    def test_reset(self):
        """Test resetting the source."""
        source = MQTTStreamDataSource(broker='localhost')
        source.open()
        
        # Simulate some data
        source._messages_received = 50
        source._batch_index = 5
        
        source.reset()
        
        assert source._messages_received == 0
        assert source._batch_index == 0
        
        source.close()


@pytest.mark.skipif(not MQTT_AVAILABLE, reason="paho-mqtt not installed")
class TestJSONMQTTDataSource:
    """Tests for JSONMQTTDataSource."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        source = JSONMQTTDataSource(broker='localhost', topics=['sensors/#'])
        
        assert source.broker == 'localhost'
        assert source.topics == ['sensors/#']
    
    def test_json_parser_dict(self):
        """Test JSON parser with dictionary."""
        source = JSONMQTTDataSource(broker='localhost')
        
        topic = 'sensors/temperature'
        payload = b'{"value": 25.5, "unit": "celsius"}'
        timestamp = time.time()
        
        data_dict, timestamps = source._parse_json_message(topic, payload, timestamp)
        
        assert f"{topic}/value" in data_dict
        assert f"{topic}/unit" in data_dict
        assert data_dict[f"{topic}/value"][0] == 25.5
        assert data_dict[f"{topic}/unit"][0] == "celsius"
    
    def test_json_parser_list(self):
        """Test JSON parser with list."""
        source = JSONMQTTDataSource(broker='localhost')
        
        topic = 'sensors/values'
        payload = b'[1.0, 2.0, 3.0]'
        timestamp = time.time()
        
        data_dict, timestamps = source._parse_json_message(topic, payload, timestamp)
        
        assert len(data_dict) == 3
        assert f"{topic}/value_0" in data_dict
        assert data_dict[f"{topic}/value_0"][0] == 1.0
    
    def test_json_parser_single_value(self):
        """Test JSON parser with single value."""
        source = JSONMQTTDataSource(broker='localhost')
        
        topic = 'sensors/temp'
        payload = b'25.5'
        timestamp = time.time()
        
        data_dict, timestamps = source._parse_json_message(topic, payload, timestamp)
        
        assert f"{topic}/value" in data_dict
        assert data_dict[f"{topic}/value"][0] == 25.5
    
    def test_json_parser_invalid_json(self):
        """Test JSON parser with invalid JSON."""
        source = JSONMQTTDataSource(broker='localhost')
        
        topic = 'sensors/invalid'
        payload = b'not valid json'
        timestamp = time.time()
        
        data_dict, timestamps = source._parse_json_message(topic, payload, timestamp)
        
        # Should fallback to raw data
        assert f"{topic}/raw" in data_dict
