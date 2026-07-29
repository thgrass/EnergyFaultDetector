"""MQTT stream data source for IoT and MQTT-based data acquisition.

This module provides MQTT-based data sources suitable for:
- IoT sensor networks
- MQTT-based SCADA systems
- Cloud-based data streams
- Any MQTT-compatible data source
"""

import json
import time
import logging
import threading
from typing import Optional, Dict, Any, Callable, List, Union
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    mqtt = None

from energy_fault_detector.data_sources.base import DataSource, DataBatch, StreamConfig

logger = logging.getLogger('energy_fault_detector.data_sources.mqtt_stream')


class MQTTStreamDataSource(DataSource):
    """Data source for receiving MQTT stream data.
    
    This class provides a flexible interface for receiving data over MQTT,
    including support for:
    - Multiple topic subscriptions
    - JSON and binary payloads
    - QoS levels
    - Custom message parsing
    - Last Will and Testament (LWT)
    
    Args:
        broker: MQTT broker address (default: 'localhost')
        port: MQTT broker port (default: 1883)
        topics: List of topics to subscribe to (default: ['#'])
        config: Stream configuration
        client_id: MQTT client ID (default: auto-generated)
        username: Username for authentication (optional)
        password: Password for authentication (optional)
        qos: Quality of Service level (0, 1, or 2, default: 1)
        clean_session: Whether to start with a clean session (default: True)
        message_parser: Custom function to parse MQTT messages (optional)
        
    Example:
        >>> # Basic MQTT stream
        >>> source = MQTTStreamDataSource(
        ...     broker='mqtt.example.com',
        ...     topics=['sensors/#']
        ... )
        >>> for batch in source:
        ...     print(f"Received {len(batch.data)} messages")
        
        >>> # With authentication
        >>> source = MQTTStreamDataSource(
        ...     broker='mqtt.example.com',
        ...     topics=['sensors/temperature'],
        ...     username='user',
        ...     password='pass'
        ... )
    """
    
    def __init__(
        self,
        broker: str = 'localhost',
        port: int = 1883,
        topics: List[str] = None,
        config: Optional[StreamConfig] = None,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        qos: int = 1,
        clean_session: bool = True,
        message_parser: Optional[Callable[[str, bytes, float], tuple[Dict[str, Any], np.ndarray]]] = None,
        **kwargs
    ):
        if not MQTT_AVAILABLE:
            raise ImportError(
                "paho-mqtt is required for MQTT support. "
                "Install it with: pip install paho-mqtt"
            )
        
        super().__init__(config=config, **kwargs)
        
        self.broker = broker
        self.port = port
        self.topics = topics or ['#']
        self.client_id = client_id or f"energy_fault_detector_{time.time()}"
        self.username = username
        self.password = password
        self.qos = qos
        self.clean_session = clean_session
        self.message_parser = message_parser or self._default_parser
        
        # MQTT client
        self._client: Optional[mqtt.Client] = None
        
        # State
        self._running = False
        self._connected = False
        self._message_queue: list[tuple[str, bytes, float]] = []
        self._queue_lock = threading.Lock()
        
        # Statistics
        self._messages_received = 0
        self._bytes_received = 0
        self._start_time: Optional[datetime] = None
        
        logger.info(f"MQTTStreamDataSource initialized for broker {broker}:{port}")
    
    def _default_parser(
        self,
        topic: str,
        payload: bytes,
        timestamp: float
    ) -> tuple[Dict[str, Any], np.ndarray]:
        """Default message parser for MQTT data.
        
        This parser handles:
        - JSON payloads
        - Text payloads
        - Binary payloads
        
        Args:
            topic: MQTT topic
            payload: Message payload bytes
            timestamp: Reception timestamp
            
        Returns:
            Tuple of (data_dict, timestamps_array)
        """
        data_dict = {}
        
        try:
            # Try to decode as text
            text = payload.decode('utf-8')
            
            # Try to parse as JSON
            try:
                json_data = json.loads(text)
                if isinstance(json_data, dict):
                    # Add topic to each key
                    for key, value in json_data.items():
                        data_dict[f"{topic}/{key}"] = [value]
                elif isinstance(json_data, list):
                    # List of values
                    for i, value in enumerate(json_data):
                        data_dict[f"{topic}/value_{i}"] = [value]
                else:
                    # Single value
                    data_dict[f"{topic}/value"] = [json_data]
                
                timestamps = np.array([timestamp] * len(data_dict))
                return data_dict, timestamps
            except json.JSONDecodeError:
                pass
            
            # Plain text - split by whitespace or commas
            if ',' in text:
                values = text.split(',')
            else:
                values = text.split()
            
            if len(values) > 1:
                try:
                    numeric_values = [float(v) for v in values]
                    data_dict = {f"{topic}/value_{i}": [v] for i, v in enumerate(numeric_values)}
                    timestamps = np.array([timestamp] * len(numeric_values))
                    return data_dict, timestamps
                except ValueError:
                    pass
            
            # Single text value
            data_dict[f"{topic}/message"] = [text]
            timestamps = np.array([timestamp])
            return data_dict, timestamps
            
        except UnicodeDecodeError:
            # Binary data
            data_dict[f"{topic}/raw_data"] = [payload.hex()]
            timestamps = np.array([timestamp])
            return data_dict, timestamps
    
    def _on_connect(self, client, userdata, flags, rc) -> None:
        """Callback for when the client connects to the broker."""
        if rc == 0:
            self._connected = True
            logger.info(f"Connected to MQTT broker {self.broker}:{self.port}")
            
            # Subscribe to topics
            for topic in self.topics:
                client.subscribe(topic, qos=self.qos)
                logger.info(f"Subscribed to topic: {topic} (QoS {self.qos})")
        else:
            self._connected = False
            logger.error(f"MQTT connection failed with result code {rc}")
    
    def _on_disconnect(self, client, userdata, rc) -> None:
        """Callback for when the client disconnects from the broker."""
        self._connected = False
        logger.info(f"Disconnected from MQTT broker (rc={rc})")
    
    def _on_message(self, client, userdata, msg) -> None:
        """Callback for when a message is received."""
        timestamp = time.time()
        
        with self._queue_lock:
            self._message_queue.append((msg.topic, msg.payload, timestamp))
            self._messages_received += 1
            self._bytes_received += len(msg.payload)
        
        logger.debug(f"Received message on topic {msg.topic} ({len(msg.payload)} bytes)")
    
    def open(self) -> 'MQTTStreamDataSource':
        """Connect to the MQTT broker and start receiving."""
        if self._is_open:
            return self
        
        # Create MQTT client
        self._client = mqtt.Client(
            client_id=self.client_id,
            clean_session=self.clean_session
        )
        
        # Set credentials if provided
        if self.username and self.password:
            self._client.username_pw_set(self.username, self.password)
        
        # Set callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        
        # Connect to broker
        try:
            self._client.connect(self.broker, self.port, keepalive=60)
            self._running = True
            self._start_time = datetime.now()
            
            # Start network loop in a thread
            self._client.loop_start()
            
            # Wait for connection
            time.sleep(0.5)
            
            if not self._connected:
                logger.warning("MQTT connection not established")
            
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
        
        self._is_open = True
        self._batch_index = 0
        
        return self
    
    def close(self) -> None:
        """Disconnect from the MQTT broker and stop receiving."""
        self._running = False
        
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from MQTT broker: {e}")
            self._client = None
        
        self._is_open = False
        self._connected = False
        logger.info("MQTT data source closed")
    
    def reset(self) -> None:
        """Reset the data source."""
        with self._queue_lock:
            self._message_queue.clear()
        self._batch_index = 0
        self._messages_received = 0
        self._bytes_received = 0
        self._start_time = datetime.now()
        logger.debug("MQTT data source reset")
    
    def __next__(self) -> DataBatch:
        """Get the next batch of data from received messages."""
        if not self._is_open:
            raise RuntimeError("Data source is not open. Call open() first.")
        
        if self._client is None:
            raise RuntimeError("MQTT client is not initialized.")
        
        # Collect messages for this batch
        batch_data = []
        batch_timestamps = []
        
        target_messages = self.config.batch_size
        start_time = time.time()
        
        while len(batch_data) < target_messages:
            # Check for timeout
            if time.time() - start_time > self.config.timeout:
                if not batch_data:
                    raise StopIteration("Timeout waiting for data")
                break
            
            # Get available messages
            with self._queue_lock:
                available_messages = len(self._message_queue)
                if available_messages > 0:
                    # Take up to remaining needed messages
                    take_count = min(available_messages, target_messages - len(batch_data))
                    messages = self._message_queue[:take_count]
                    self._message_queue = self._message_queue[take_count:]
                else:
                    messages = []
            
            if not messages:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue
            
            # Parse messages
            for topic, payload, message_timestamp in messages:
                try:
                    data_dict, timestamps = self.message_parser(topic, payload, message_timestamp)
                    batch_data.append(data_dict)
                    batch_timestamps.extend(timestamps)
                except Exception as e:
                    logger.error(f"Error parsing MQTT message from {topic}: {e}")
                    continue
        
        if not batch_data:
            raise StopIteration("No data available")
        
        # Convert to DataFrame
        df = pd.DataFrame(batch_data)
        
        # If all messages had the same timestamp, use a range
        if len(set(batch_timestamps)) == 1 and len(batch_timestamps) > 1:
            timestamps = np.linspace(batch_timestamps[0], batch_timestamps[0] + len(batch_timestamps) * 0.001, len(batch_timestamps))
        else:
            timestamps = np.array(batch_timestamps)
        
        # Create metadata
        metadata = {
            'source_type': 'mqtt',
            'broker': self.broker,
            'port': self.port,
            'topics': self.topics,
            'messages_in_batch': len(batch_data),
            'total_messages': self._messages_received,
            'total_bytes': self._bytes_received,
            'connected': self._connected,
        }
        
        batch = DataBatch(
            data=df,
            timestamps=timestamps,
            batch_index=self._batch_index,
            is_complete=False,  # MQTT streams are continuous
            metadata=metadata
        )
        
        self._batch_index += 1
        
        return batch
    
    @property
    def is_connected(self) -> bool:
        """Whether the MQTT connection is currently active."""
        return self._connected
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get statistics about the MQTT stream."""
        uptime = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        
        return {
            'messages_received': self._messages_received,
            'bytes_received': self._bytes_received,
            'uptime_seconds': uptime,
            'messages_per_second': self._messages_received / uptime if uptime > 0 else 0,
            'bytes_per_second': self._bytes_received / uptime if uptime > 0 else 0,
            'connected': self._connected,
            'subscribed_topics': self.topics,
        }
    
    def publish(self, topic: str, payload: Union[str, bytes], qos: int = 1) -> None:
        """Publish a message to a topic.
        
        Args:
            topic: Topic to publish to
            payload: Message payload (string or bytes)
            qos: Quality of Service level
        """
        if not self._is_open or self._client is None:
            raise RuntimeError("MQTT client is not connected")
        
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        
        result = self._client.publish(topic, payload, qos=qos)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error(f"Failed to publish message to {topic}: {mqtt.error_string(result.rc)}")
        else:
            logger.debug(f"Published message to {topic}")


class JSONMQTTDataSource(MQTTStreamDataSource):
    """MQTT data source specialized for JSON payloads.
    
    This class assumes all messages are JSON-formatted and provides
    convenient access to the parsed data.
    
    Args:
        broker: MQTT broker address
        topics: List of topics to subscribe to
        config: Stream configuration
        client_id: MQTT client ID
        username: Username for authentication
        password: Password for authentication
        qos: Quality of Service level
        
    Example:
        >>> source = JSONMQTTDataSource(
        ...     broker='mqtt.example.com',
        ...     topics=['sensors/#']
        ... )
        >>> for batch in source:
        ...     print(f"Received JSON data: {batch.data.columns}")
    """
    
    def __init__(
        self,
        broker: str = 'localhost',
        topics: List[str] = None,
        config: Optional[StreamConfig] = None,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        qos: int = 1,
        **kwargs
    ):
        def json_parser(topic: str, payload: bytes, timestamp: float) -> tuple[Dict[str, Any], np.ndarray]:
            return self._parse_json_message(topic, payload, timestamp)
        
        super().__init__(
            broker=broker,
            topics=topics,
            config=config,
            client_id=client_id,
            username=username,
            password=password,
            qos=qos,
            message_parser=json_parser,
            **kwargs
        )
    
    def _parse_json_message(
        self,
        topic: str,
        payload: bytes,
        timestamp: float
    ) -> tuple[Dict[str, Any], np.ndarray]:
        """Parse a JSON message.
        
        Args:
            topic: MQTT topic
            payload: Message payload bytes
            timestamp: Reception timestamp
            
        Returns:
            Tuple of (data_dict, timestamps_array)
        """
        try:
            text = payload.decode('utf-8')
            json_data = json.loads(text)
            
            if isinstance(json_data, dict):
                # Flatten the JSON structure
                data_dict = {}
                for key, value in json_data.items():
                    full_key = f"{topic}/{key}" if topic else key
                    if isinstance(value, (int, float)):
                        data_dict[full_key] = [value]
                    elif isinstance(value, str):
                        data_dict[full_key] = [value]
                    elif isinstance(value, bool):
                        data_dict[full_key] = [int(value)]
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, (int, float)):
                                data_dict[f"{full_key}_{i}"] = [item]
            elif isinstance(json_data, list):
                data_dict = {}
                for i, item in enumerate(json_data):
                    if isinstance(item, (int, float)):
                        data_dict[f"{topic}/value_{i}"] = [item]
            else:
                data_dict = {f"{topic}/value": [json_data]}
            
            timestamps = np.array([timestamp] * len(data_dict))
            return data_dict, timestamps
            
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning(f"Error parsing JSON from {topic}: {e}")
            return {f"{topic}/raw": [payload.hex()]}, np.array([timestamp])
