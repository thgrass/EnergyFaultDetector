# Network Stream Data Sources

This guide covers the network-based data sources for real-time streaming data acquisition.

## Overview

EnergyFaultDetector provides several network-based data sources for real-time data acquisition:

- **UDPStreamDataSource**: For UDP-based protocols (including phasor data)
- **PhasorUDPDataSource**: Specialized for IEEE C37.118 phasor measurement data
- **TCPStreamDataSource**: For reliable TCP-based data streams
- **LineDelimitedTCPDataSource**: For text-based TCP protocols with line delimiters
- **MQTTStreamDataSource**: For MQTT-based IoT and sensor networks
- **JSONMQTTDataSource**: Specialized for JSON payloads over MQTT

## UDP Stream Data Source

### Basic Usage

```python
from energy_fault_detector.data_sources import UDPStreamDataSource, StreamConfig

# Create UDP data source
source = UDPStreamDataSource(
    host='0.0.0.0',      # Listen on all interfaces
    port=5000,          # Port to listen on
    config=StreamConfig(batch_size=100, timeout=1.0)
)

# Process stream
for batch in source:
    print(f"Received batch {batch.batch_index} with {len(batch.data)} samples")
    print(f"Data: {batch.data.head()}")
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | '0.0.0.0' | Host address to bind to |
| `port` | int | 5000 | Port to listen on |
| `config` | StreamConfig | None | Streaming configuration |
| `packet_parser` | callable | None | Custom packet parser function |
| `buffer_size` | int | 65536 | Socket buffer size in bytes |
| `timeout` | float | 1.0 | Socket timeout in seconds |
| `multicast_group` | str | None | Multicast group address |
| `multicast_interface` | str | None | Network interface for multicast |

### Custom Packet Parsing

The UDP data source accepts a custom packet parser function:

```python
def my_udp_parser(packet_data: bytes, timestamp: float) -> tuple[dict, np.ndarray]:
    """Parse custom UDP packet format.
    
    Args:
        packet_data: Raw UDP packet bytes
        timestamp: Reception timestamp
        
    Returns:
        Tuple of (data_dict, timestamps_array)
    """
    # Example: Parse as 4 float32 values
    import struct
    values = struct.unpack('4f', packet_data)
    data_dict = {
        'sensor_1': [values[0]],
        'sensor_2': [values[1]],
        'sensor_3': [values[2]],
        'sensor_4': [values[3]],
    }
    timestamps = np.array([timestamp] * 4)
    return data_dict, timestamps

# Use custom parser
source = UDPStreamDataSource(
    port=5000,
    packet_parser=my_udp_parser
)
```

### Multicast Support

```python
# Join a multicast group
source = UDPStreamDataSource(
    host='0.0.0.0',
    port=5000,
    multicast_group='239.255.255.250',  # Multicast address
    multicast_interface='eth0'           # Network interface
)
```

### Statistics

```python
source.open()
# ... process data ...
stats = source.statistics
print(f"Packets received: {stats['packets_received']}")
print(f"Bytes received: {stats['bytes_received']}")
print(f"Packets/sec: {stats['packets_per_second']:.2f}")
```

## Phasor UDP Data Source

Specialized for IEEE C37.118 phasor measurement unit (PMU) data.

### Basic Usage

```python
from energy_fault_detector.data_sources import PhasorUDPDataSource, StreamConfig

# Create phasor data source
source = PhasorUDPDataSource(
    host='0.0.0.0',
    port=4712,          # Common PMU port
    config=StreamConfig(batch_size=50),
    n_phasors=3,        # Number of phasors
    n_analog=4,         # Number of analog values
    n_digital=2         # Number of digital status words
)

# Process phasor data
for batch in source:
    print(f"Phasor data: {batch.data.columns}")
    # Columns will include: phasor_0_real, phasor_0_imag, phasor_0_mag, phasor_0_phase, etc.
```

### Phasor Data Format

The `PhasorUDPDataSource` parses IEEE C37.118 messages and extracts:

- **Phasor values**: Real, imaginary, magnitude, and phase for each phasor
- **Analog values**: Additional analog measurements
- **Digital status**: Digital status words

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | '0.0.0.0' | Host address to bind to |
| `port` | int | 4712 | Port to listen on (4712 is common for PMUs) |
| `config` | StreamConfig | None | Streaming configuration |
| `n_phasors` | int | 3 | Number of phasors expected |
| `n_analog` | int | 0 | Number of analog values expected |
| `n_digital` | int | 0 | Number of digital status words expected |

### Example: Processing PMU Data

```python
from energy_fault_detector import StreamingFaultDetector
from energy_fault_detector.data_sources import PhasorUDPDataSource, StreamConfig
from energy_fault_detector.config import generate_quickstart_config

# Create detector
config = generate_quickstart_config()
detector = StreamingFaultDetector(config=config)

# Initialize with normal phasor data (from a file or previous recording)
detector.initialize(fit_data=normal_phasor_data)

# Create PMU data source
pmu_source = PhasorUDPDataSource(
    port=4712,
    n_phasors=3,
    config=StreamConfig(batch_size=100)
)

# Process real-time PMU data
result = detector.process_stream(pmu_source)

# Analyze results
print(f"Detected {result.total_anomalies} anomalies in PMU data")
```

## TCP Stream Data Source

### Basic Usage

#### Server Mode (Listening for Connections)

```python
from energy_fault_detector.data_sources import TCPStreamDataSource, StreamConfig

# Create TCP server
source = TCPStreamDataSource(
    host='0.0.0.0',
    port=5000,
    mode='server',
    config=StreamConfig(batch_size=100)
)

# Process incoming connections
for batch in source:
    print(f"Received {len(batch.data)} messages from client")
```

#### Client Mode (Connecting to Server)

```python
# Create TCP client
source = TCPStreamDataSource(
    host='server_ip',
    port=5000,
    mode='client',
    config=StreamConfig(batch_size=100)
)

# Process data from server
for batch in source:
    print(f"Received {len(batch.data)} messages from server")
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | '0.0.0.0' | Host address |
| `port` | int | 5000 | Port number |
| `config` | StreamConfig | None | Streaming configuration |
| `mode` | str | 'server' | 'server' or 'client' |
| `packet_parser` | callable | None | Custom message parser |
| `buffer_size` | int | 65536 | Socket buffer size |
| `timeout` | float | 1.0 | Socket timeout |
| `delimiter` | bytes | None | Message delimiter for text protocols |
| `message_size` | int | None | Fixed message size for binary protocols |

### Message Framing

The TCP data source supports two framing modes:

#### 1. Delimiter-Based (Text Protocols)

```python
# Use newline as delimiter
source = TCPStreamDataSource(
    port=5000,
    mode='server',
    delimiter=b'\n'  # Newline delimiter
)
```

#### 2. Fixed Size (Binary Protocols)

```python
# Fixed 64-byte messages
source = TCPStreamDataSource(
    port=5000,
    mode='server',
    message_size=64
)
```

### Custom Message Parsing

```python
def my_tcp_parser(message_data: bytes, timestamp: float) -> tuple[dict, np.ndarray]:
    """Parse custom TCP message format."""
    # Example: Parse CSV-like text
    text = message_data.decode('utf-8').strip()
    values = text.split(',')
    data_dict = {f'sensor_{i}': [float(v)] for i, v in enumerate(values)}
    timestamps = np.array([timestamp] * len(values))
    return data_dict, timestamps

source = TCPStreamDataSource(
    port=5000,
    mode='server',
    packet_parser=my_tcp_parser
)
```

### Connection Management

```python
source.open()
print(f"Connected: {source.is_connected}")

# ... process data ...

source.close()
print(f"Connected: {source.is_connected}")
```

## Line Delimited TCP Data Source

Specialized for text-based protocols where each line is a separate message.

### Basic Usage

```python
from energy_fault_detector.data_sources import LineDelimitedTCPDataSource, StreamConfig

# Create line-delimited TCP source
source = LineDelimitedTCPDataSource(
    host='0.0.0.0',
    port=5000,
    mode='server',
    delimiter=b'\n',  # Newline delimiter
    config=StreamConfig(batch_size=100)
)

# Process line-delimited data
for batch in source:
    print(f"Received {len(batch.data)} lines")
```

### Custom Delimiters

```python
# Use semicolon as delimiter
source = LineDelimitedTCPDataSource(
    port=5000,
    delimiter=b';'
)

# Use custom delimiter
source = LineDelimitedTCPDataSource(
    port=5000,
    delimiter=b'|END|'
)
```

## MQTT Stream Data Source

### Basic Usage

```python
from energy_fault_detector.data_sources import MQTTStreamDataSource, StreamConfig

# Create MQTT data source
source = MQTTStreamDataSource(
    broker='mqtt.example.com',
    port=1883,
    topics=['sensors/#', 'status/#'],
    config=StreamConfig(batch_size=50)
)

# Process MQTT messages
for batch in source:
    print(f"Received {len(batch.data)} MQTT messages")
    print(f"Topics: {batch.metadata.get('topics')}")
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `broker` | str | 'localhost' | MQTT broker address |
| `port` | int | 1883 | MQTT broker port |
| `topics` | list | ['#'] | List of topics to subscribe to |
| `config` | StreamConfig | None | Streaming configuration |
| `client_id` | str | auto-generated | MQTT client ID |
| `username` | str | None | Username for authentication |
| `password` | str | None | Password for authentication |
| `qos` | int | 1 | Quality of Service level (0, 1, or 2) |
| `clean_session` | bool | True | Start with clean session |
| `message_parser` | callable | None | Custom message parser |

### Authentication

```python
# With authentication
source = MQTTStreamDataSource(
    broker='mqtt.example.com',
    username='my_user',
    password='my_password',
    topics=['sensors/#']
)
```

### Quality of Service

```python
# QoS 0: At most once delivery
source = MQTTStreamDataSource(qos=0)

# QoS 1: At least once delivery (default)
source = MQTTStreamDataSource(qos=1)

# QoS 2: Exactly once delivery
source = MQTTStreamDataSource(qos=2)
```

### Multiple Topics

```python
# Subscribe to multiple topics
source = MQTTStreamDataSource(
    topics=[
        'sensors/temperature',
        'sensors/pressure',
        'sensors/vibration',
        'status/#'
    ]
)
```

### Custom Message Parsing

```python
def my_mqtt_parser(topic: str, payload: bytes, timestamp: float) -> tuple[dict, np.ndarray]:
    """Parse custom MQTT message format."""
    # Example: Parse binary sensor data
    import struct
    values = struct.unpack('4f', payload)  # 4 float32 values
    data_dict = {
        f"{topic}/sensor_1": [values[0]],
        f"{topic}/sensor_2": [values[1]],
        f"{topic}/sensor_3": [values[2]],
        f"{topic}/sensor_4": [values[3]],
    }
    timestamps = np.array([timestamp] * 4)
    return data_dict, timestamps

source = MQTTStreamDataSource(
    broker='localhost',
    topics=['sensors/binary'],
    message_parser=my_mqtt_parser
)
```

### Publishing Messages

The MQTT data source also supports publishing messages:

```python
source.open()

# Publish a message
source.publish(
    topic='commands/start',
    payload='{"command": "start"}',
    qos=1
)

# Publish binary data
source.publish(
    topic='data/binary',
    payload=b'\x00\x01\x02\x03',
    qos=1
)

source.close()
```

### Connection Status

```python
source.open()
print(f"Connected: {source.is_connected}")

# Wait for connection
import time
while not source.is_connected:
    time.sleep(0.1)

# Process data
for batch in source:
    if not source.is_connected:
        print("Connection lost!")
        break
    # Process batch
```

## JSON MQTT Data Source

Specialized for JSON payloads over MQTT.

### Basic Usage

```python
from energy_fault_detector.data_sources import JSONMQTTDataSource, StreamConfig

# Create JSON MQTT data source
source = JSONMQTTDataSource(
    broker='mqtt.example.com',
    topics=['sensors/#'],
    config=StreamConfig(batch_size=50)
)

# Process JSON data
for batch in source:
    print(f"Received JSON data: {batch.data.columns}")
```

### JSON Data Handling

The `JSONMQTTDataSource` automatically parses JSON payloads and flattens the structure:

- **Objects**: Keys become column names with topic prefix
- **Arrays**: Elements become separate columns
- **Primitive values**: Directly mapped to columns

### Example JSON Payloads

#### Simple Object
```json
{"temperature": 25.5, "humidity": 60.0}
```
Parsed to columns: `sensors/temperature`, `sensors/humidity`

#### Array of Values
```json
[1.0, 2.0, 3.0, 4.0]
```
Parsed to columns: `sensors/value_0`, `sensors/value_1`, `sensors/value_2`, `sensors/value_3`

#### Nested Object
```json
{"sensor": {"value": 25.5, "unit": "celsius"}}
```
Parsed to columns: `sensors/sensor_value`, `sensors/sensor_unit`

## Performance Considerations

### UDP
- **Pros**: Low latency, multicast support, good for high-frequency data
- **Cons**: No guaranteed delivery, no ordering, limited packet size
- **Use for**: Phasor data, high-frequency sensor data, multicast applications

### TCP
- **Pros**: Reliable delivery, ordered messages, larger message sizes
- **Cons**: Higher latency, connection-oriented, more overhead
- **Use for**: Reliable data streams, command/response protocols

### MQTT
- **Pros**: Lightweight, supports QoS, topic-based routing, good for many sensors
- **Cons**: Requires broker, additional protocol overhead
- **Use for**: IoT networks, many distributed sensors, cloud-based systems

## Error Handling

### UDP
```python
try:
    for batch in source:
        process(batch)
except socket.timeout:
    print("No data received within timeout")
except Exception as e:
    print(f"UDP error: {e}")
```

### TCP
```python
try:
    for batch in source:
        if not source.is_connected:
            print("Connection lost")
            break
        process(batch)
except ConnectionResetError:
    print("Connection reset by peer")
except Exception as e:
    print(f"TCP error: {e}")
```

### MQTT
```python
try:
    for batch in source:
        if not source.is_connected:
            print("Disconnected from broker")
            break
        process(batch)
except Exception as e:
    print(f"MQTT error: {e}")
```

## Security Considerations

### UDP/TCP
- Use firewalls to restrict access to specific IPs
- Consider using VPNs for remote connections
- For sensitive data, implement encryption at the application level

### MQTT
- Use authentication (`username` and `password`)
- Use TLS/SSL for encrypted connections (port 8883)
- Restrict topic subscriptions to only necessary topics
- Use client certificates for mutual authentication

## Example: Complete Streaming Pipeline with UDP

```python
from energy_fault_detector import StreamingFaultDetector
from energy_fault_detector.data_sources import UDPStreamDataSource, StreamConfig
from energy_fault_detector.config import generate_quickstart_config

# 1. Create configuration
config = generate_quickstart_config()

# 2. Create streaming detector
detector = StreamingFaultDetector(
    config=config,
    buffer_size=1000,
    online_learning=True,
    online_learning_interval=100
)

# 3. Initialize with training data
detector.initialize(fit_data=training_data)

# 4. Create UDP data source
udp_source = UDPStreamDataSource(
    host='0.0.0.0',
    port=5000,
    config=StreamConfig(batch_size=100, timeout=0.5)
)

# 5. Process stream
try:
    result = detector.process_stream(udp_source)
    
    # 6. Analyze results
    print(f"Total samples: {result.total_samples}")
    print(f"Anomalies detected: {result.total_anomalies}")
    print(f"Throughput: {result.throughput:.2f} samples/sec")
    
    # Get statistics
    stats = udp_source.statistics
    print(f"Packets received: {stats['packets_received']}")
    print(f"Packets/sec: {stats['packets_per_second']:.2f}")
    
except KeyboardInterrupt:
    print("Stream processing interrupted")
finally:
    detector.close()
    udp_source.close()
```

## Example: PMU Data Processing

```python
from energy_fault_detector import StreamingFaultDetector
from energy_fault_detector.data_sources import PhasorUDPDataSource, StreamConfig
from energy_fault_detector.config import generate_quickstart_config

# Create detector optimized for phasor data
config = generate_quickstart_config()
# Adjust config for phasor data characteristics

 detector = StreamingFaultDetector(config=config)

# Initialize with normal PMU data
pmu_normal_data = load_normal_pmu_data()  # From a file or database
detector.initialize(fit_data=pmu_normal_data)

# Create PMU data source
pmu_source = PhasorUDPDataSource(
    host='0.0.0.0',
    port=4712,          # Standard PMU port
    n_phasors=3,        # 3-phase system
    n_analog=4,         # 4 analog measurements
    config=StreamConfig(batch_size=50, timeout=0.1)
)

# Process real-time PMU data
result = detector.process_stream(pmu_source)

# Analyze phasor anomalies
anomalies = result.get_anomalies()
print(f"Detected {len(anomalies)} phasor anomalies")

# Check specific phasor channels
for col in anomalies.columns:
    if col.startswith('phasor_'):
        print(f"{col}: {anomalies[col].describe()}")
```

## Troubleshooting

### UDP Issues

**Problem**: No data received  
**Solutions**:
- Check firewall settings
- Verify port is not blocked
- Check network connectivity
- Use `netstat -anu` to verify socket is listening
- Try with `multicast_group` if using multicast

**Problem**: Packet loss  
**Solutions**:
- Increase buffer_size
- Check network quality
- Consider using TCP for reliable delivery

### TCP Issues

**Problem**: Connection refused  
**Solutions**:
- Check server is running
- Verify host and port
- Check firewall settings
- Try `telnet host port` to test connectivity

**Problem**: Connection reset  
**Solutions**:
- Check server is not closing connections
- Increase timeout
- Handle reconnection in your code

### MQTT Issues

**Problem**: Connection failed  
**Solutions**:
- Check broker is running
- Verify host and port
- Check authentication credentials
- Check broker logs for errors

**Problem**: No messages received  
**Solutions**:
- Verify topic subscriptions
- Check message QoS level
- Use MQTT client (like mosquitto_sub) to test
- Verify topic wildcards are correct

## API Reference

### UDPStreamDataSource

**Attributes:**
- `host`: Host address
- `port`: Port number
- `buffer_size`: Socket buffer size
- `timeout`: Socket timeout
- `multicast_group`: Multicast group address
- `multicast_interface`: Network interface for multicast

**Methods:**
- `open()`: Open socket and start receiving
- `close()`: Close socket
- `reset()`: Reset statistics and buffers
- `statistics`: Get stream statistics

**Properties:**
- `is_open`: Whether the source is open
- `batch_index`: Current batch index

### PhasorUDPDataSource

**Inherits from**: UDPStreamDataSource

**Additional Attributes:**
- `n_phasors`: Number of phasors
- `n_analog`: Number of analog values
- `n_digital`: Number of digital status words

### TCPStreamDataSource

**Attributes:**
- `host`: Host address
- `port`: Port number
- `mode`: 'server' or 'client'
- `buffer_size`: Socket buffer size
- `timeout`: Socket timeout
- `delimiter`: Message delimiter
- `message_size`: Fixed message size

**Methods:**
- `open()`: Open connection and start receiving
- `close()`: Close connection
- `reset()`: Reset statistics and buffers
- `statistics`: Get stream statistics

**Properties:**
- `is_open`: Whether the source is open
- `is_connected`: Whether the connection is active
- `batch_index`: Current batch index

### LineDelimitedTCPDataSource

**Inherits from**: TCPStreamDataSource

**Additional Attributes:**
- `delimiter`: Line delimiter

### MQTTStreamDataSource

**Attributes:**
- `broker`: MQTT broker address
- `port`: MQTT broker port
- `topics`: List of subscribed topics
- `client_id`: MQTT client ID
- `username`: Authentication username
- `password`: Authentication password
- `qos`: Quality of Service level
- `clean_session`: Clean session flag

**Methods:**
- `open()`: Connect to broker and start receiving
- `close()`: Disconnect from broker
- `reset()`: Reset statistics and buffers
- `publish(topic, payload, qos)`: Publish a message
- `statistics`: Get stream statistics

**Properties:**
- `is_open`: Whether the source is open
- `is_connected`: Whether connected to broker
- `batch_index`: Current batch index

### JSONMQTTDataSource

**Inherits from**: MQTTStreamDataSource

**Additional Methods:**
- `_parse_json_message(topic, payload, timestamp)`: Parse JSON messages
