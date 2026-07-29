# OpenPMU XML Datagram Support

EnergyFaultDetector now supports OpenPMU V2 XML datagrams as described in the [OpenPMU Documentation](https://github.com/OpenPMU/OpenPMUdocs/tree/master/XML_Datagrams).

This guide covers how to use OpenPMU XML datagrams with EnergyFaultDetector for real-time phasor data processing.

## Overview

OpenPMU V2 uses XML to markup data in streams of UDP datagrams. This provides a highly flexible, plug-and-play system for phasor measurement unit (PMU) data acquisition and processing.

EnergyFaultDetector supports both OpenPMU datagram types:

1. **XML Sampled Values (SV) Datagrams**: Raw sampled values from ADCs
2. **XML Phasor Values (PV) Datagrams**: Processed phasor values with magnitude, angle, frequency, and ROCOF

## OpenPMU Datagram Formats

### XML Sampled Values (SV) Datagram

Sampled Values datagrams contain raw waveform data from the ADC. Each datagram includes:

- **Metadata**: Format, date, time, frame number, sampling rate, number of samples, bits per sample, number of channels
- **Channel Information**: For each channel, includes name, type (V/I), phase, range, and payload
- **Payload**: Base64-encoded binary data containing the sampled values

Example:
```xml
<OpenPMU>
    <Format>Samples</Format>
    <Date>2021-02-28</Date>
    <Time>22:04:00.000</Time>
    <Frame>0</Frame>
    <Fs>12800</Fs>
    <n>128</n>
    <bits>16</bits>
    <Channels>3</Channels>
    <Channel_0>
        <Name>Belfast_Va</Name>
        <Type>V</Type>
        <Phase>a</Phase>
        <Range>275</Range>
        <Payload>JWMmESa8J2QoBCieKTEpuypAKrwrMSufLAYsZSy+LQ8t...</Payload>
    </Channel_0>
    <!-- More channels... -->
</OpenPMU>
```

### XML Phasor Values (PV) Datagram

Phasor Values datagrams contain processed phasor data. Each datagram includes:

- **Metadata**: Format, date, time, frame number, algorithm name, number of channels
- **Channel Information**: For each channel, includes name, type, phase, range, magnitude, angle, frequency, and ROCOF

Example:
```xml
<OpenPMU>
    <Format>Phasors</Format>
    <Date>2021-02-28</Date>
    <Time>22:04:00.460</Time>
    <Frame>23</Frame>
    <Algorithm>LSE V1.0 by Xiaodong Zhao</Algorithm>
    <Channels>3</Channels>
    <Channel_0>
        <Name>Belfast_Va</Name>
        <Type>V</Type>
        <Phase>a</Phase>
        <Range>275</Range>
        <Mag>240.00001</Mag>
        <Angle>0.403</Angle>
        <Freq>50.690</Freq>
        <ROCOF>0.001</ROCOF>
    </Channel_0>
    <!-- More channels... -->
</OpenPMU>
```

## Quick Start

### Basic Usage

```python
from energy_fault_detector import StreamingFaultDetector
from energy_fault_detector.data_sources import OpenPMUUDPDataSource, StreamConfig
from energy_fault_detector.config import generate_quickstart_config

# 1. Create configuration
config = generate_quickstart_config()

# 2. Create streaming detector
detector = StreamingFaultDetector(config=config)

# 3. Initialize with training data
detector.initialize(fit_data=training_data)

# 4. Create OpenPMU UDP data source
source = OpenPMUUDPDataSource(
    host='0.0.0.0',
    port=4713,  # Default OpenPMU port
    config=StreamConfig(batch_size=50, timeout=1.0),
    datagram_type='phasors'  # or 'samples' or 'auto'
)

# 5. Process the stream
result = detector.process_stream(source)

# 6. View results
print(f"Total samples: {result.total_samples}")
print(f"Anomalies detected: {result.total_anomalies}")
```

### Auto Detection

The `datagram_type` parameter can be set to `'auto'` to automatically detect the datagram type:

```python
source = OpenPMUUDPDataSource(
    port=4713,
    datagram_type='auto'  # Auto-detect Samples vs Phasors
)
```

## OpenPMUUDPDataSource

The main class for receiving OpenPMU XML datagrams over UDP.

### Initialization

```python
from energy_fault_detector.data_sources import OpenPMUUDPDataSource, StreamConfig

# Basic initialization
source = OpenPMUUDPDataSource(port=4713)

# With custom configuration
source = OpenPMUUDPDataSource(
    host='0.0.0.0',
    port=4713,
    config=StreamConfig(batch_size=100, timeout=0.5),
    datagram_type='phasors'
)

# For multicast
source = OpenPMUUDPDataSource(
    host='0.0.0.0',
    port=4713,
    multicast_group='239.255.255.250',
    multicast_interface='eth0'
)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | '0.0.0.0' | Host address to bind to |
| `port` | int | 4713 | Port to listen on (4713 is common for OpenPMU) |
| `config` | StreamConfig | None | Streaming configuration |
| `datagram_type` | str | 'auto' | Expected datagram type ('auto', 'samples', or 'phasors') |

### Methods

```python
# Open the source
source.open()

# Process batches
for batch in source:
    print(f"Received batch {batch.batch_index}")
    print(f"Data: {batch.data.head()}")
    print(f"Metadata: {batch.metadata}")

# Close the source
source.close()
```

### Properties

```python
# Check if open
print(f"Is open: {source.is_open}")

# Get statistics
stats = source.statistics
print(f"Packets received: {stats['packets_received']}")
print(f"Packets/sec: {stats['packets_per_second']:.2f}")
```

## Data Structure

### Phasor Values Data

When processing Phasor Values datagrams, each batch contains:

- **Magnitude**: Phasor magnitude for each channel
- **Angle**: Phasor angle in degrees for each channel
- **Frequency**: Estimated frequency for each channel
- **ROCOF**: Rate of change of frequency for each channel
- **Metadata**: Channel name, type, phase, range

Example data columns:
```
Belfast_Va_mag, Belfast_Va_angle, Belfast_Va_freq, Belfast_Va_rocof,
Belfast_Va_type, Belfast_Va_phase, Belfast_Va_range,
Belfast_Vb_mag, Belfast_Vb_angle, ...
```

### Sampled Values Data

When processing Sampled Values datagrams, each batch contains:

- **Samples**: Individual sampled values for each channel
- **Metadata**: Channel name, type, phase, range, sampling rate, etc.

Example data columns:
```
Belfast_Va_sample_0, Belfast_Va_sample_1, ..., Belfast_Va_sample_127,
Belfast_Vb_sample_0, ...
```

## OpenPMUParser

The parser class for OpenPMU XML datagrams.

### Usage

```python
from energy_fault_detector.data_sources import OpenPMUParser

# Parse XML string
xml_string = """<OpenPMU>...</OpenPMU>"""
datagram = OpenPMUParser.parse_xml(xml_string)

# Access datagram properties
print(f"Format: {datagram.format}")
print(f"Timestamp: {datagram.timestamp}")
print(f"Frame: {datagram.frame}")
print(f"Number of channels: {len(datagram.channels)}")

# Access channel data
for channel in datagram.channels:
    print(f"Channel: {channel.name}")
    print(f"  Type: {channel.channel_type}")
    print(f"  Phase: {channel.phase}")
    print(f"  Range: {channel.range}")
    if datagram.is_phasor_values:
        print(f"  Mag: {channel.mag}")
        print(f"  Angle: {channel.angle}")
        print(f"  Freq: {channel.freq}")
        print(f"  ROCOF: {channel.rocof}")
```

### Decoding Payloads

For Sampled Values datagrams, you can decode the Base64 payload:

```python
# Decode payload from a channel
samples = OpenPMUParser.decode_payload(
    channel.payload,
    n_samples=datagram.n or 128,
    bits=datagram.bits or 16,
    is_signed=True
)

# Scale samples by range
scaled_samples = samples / (2 ** bits - 1) * channel.range
```

## OpenPMUGenerator

A utility class for generating OpenPMU XML datagrams for testing.

### Usage

```python
from energy_fault_detector.data_sources import OpenPMUGenerator
from datetime import datetime

# Create generator
generator = OpenPMUGenerator(
    n_channels=3,
    fs=12800.0,
    n_samples=128
)

# Set custom channel information
generator.set_channel_info(
    names=['Belfast_Va', 'Belfast_Vb', 'Belfast_Vc'],
    types=['V', 'V', 'V'],
    phases=['a', 'b', 'c'],
    ranges=[275.0, 275.0, 275.0]
)

# Generate Sampled Values datagram
sv_xml = generator.generate_sampled_values_datagram()

# Generate Phasor Values datagram
pv_xml = generator.generate_phasor_values_datagram()

# Generate with custom timestamp
timestamp = datetime(2024, 1, 15, 10, 30, 45, 123456)
xml = generator.generate_phasor_values_datagram(timestamp)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_channels` | int | 3 | Number of channels |
| `fs` | float | 12800.0 | Sampling rate in Hz |
| `n_samples` | int | 128 | Number of samples per payload |

### Methods

```python
# Generate specific datagram type
xml = generator.generate_datagram('phasors')
xml = generator.generate_datagram('samples')

# Set custom channel info
generator.set_channel_info(
    names=['Va', 'Vb', 'Vc'],
    types=['V', 'V', 'V'],
    phases=['a', 'b', 'c'],
    ranges=[275.0, 275.0, 275.0]
)
```

## Complete Example: OpenPMU Phasor Fault Detection

```python
from energy_fault_detector import StreamingFaultDetector
from energy_fault_detector.data_sources import OpenPMUUDPDataSource, StreamConfig
from energy_fault_detector.config import generate_quickstart_config
import pandas as pd
import numpy as np

# 1. Create configuration
config = generate_quickstart_config()

# 2. Create streaming detector
detector = StreamingFaultDetector(
    config=config,
    buffer_size=2000,
    online_learning=True,
    online_learning_interval=100
)

# 3. Create training data (simulated normal phasor data)
n_samples = 1000
normal_data = pd.DataFrame({
    'Belfast_Va_mag': np.random.normal(240.0, 1.0, n_samples),
    'Belfast_Va_angle': np.random.normal(0.0, 0.1, n_samples),
    'Belfast_Vb_mag': np.random.normal(240.0, 1.0, n_samples),
    'Belfast_Vb_angle': np.random.normal(120.0, 0.1, n_samples),
    'Belfast_Vc_mag': np.random.normal(240.0, 1.0, n_samples),
    'Belfast_Vc_angle': np.random.normal(-120.0, 0.1, n_samples),
})

# 4. Initialize detector
detector.initialize(fit_data=normal_data)

# 5. Create OpenPMU data source
source = OpenPMUUDPDataSource(
    host='0.0.0.0',
    port=4713,
    config=StreamConfig(batch_size=50, timeout=0.5),
    datagram_type='phasors'
)

# 6. Process stream
try:
    result = detector.process_stream(source)
    
    # 7. Analyze results
    print(f"Total samples: {result.total_samples}")
    print(f"Anomalies detected: {result.total_anomalies}")
    print(f"Anomaly rate: {result.total_anomalies / result.total_samples * 100:.2f}%")
    
    # Get anomalies
    anomalies = result.get_anomalies()
    if len(anomalies) > 0:
        print(f"\nAnomaly details:")
        for col in anomalies.columns:
            if col.endswith('_mag'):
                print(f"  {col}: mean={anomalies[col].mean():.2f}, std={anomalies[col].std():.2f}")
    
    # Get statistics
    stats = source.statistics
    print(f"\nStream statistics:")
    print(f"  Packets received: {stats['packets_received']}")
    print(f"  Packets/sec: {stats['packets_per_second']:.2f}")
    
except KeyboardInterrupt:
    print("Stream processing interrupted")
finally:
    detector.close()
    source.close()
```

## Complete Example: OpenPMU Sampled Values Processing

```python
from energy_fault_detector import StreamingFaultDetector
from energy_fault_detector.data_sources import OpenPMUUDPDataSource, StreamConfig
from energy_fault_detector.config import generate_quickstart_config

# Create detector
config = generate_quickstart_config()
detector = StreamingFaultDetector(config=config)

# Initialize with normal sampled values data
detector.initialize(fit_data=normal_sv_data)

# Create OpenPMU data source for sampled values
source = OpenPMUUDPDataSource(
    host='0.0.0.0',
    port=4713,
    config=StreamConfig(batch_size=100, timeout=0.5),
    datagram_type='samples'
)

# Process stream
result = detector.process_stream(source)

# Analyze results
print(f"Total samples: {result.total_samples}")
print(f"Anomalies detected: {result.total_anomalies}")

# Note: Sampled Values processing may require more memory
# as each datagram contains many samples per channel
```

## Integration with Other Components

### Using with DataBuffer

```python
from energy_fault_detector.streaming import DataBuffer

# Create buffer
buffer = DataBuffer(max_size=10000)

# Process stream and add to buffer
for batch in source:
    buffer.add_dataframe(batch.data)
    
    # Check if buffer is full
    if buffer.is_full:
        # Process buffered data
        buffered_data = buffer.to_dataframe()
        # ... do something with buffered data
        buffer.clear()
```

### Using with SlidingWindowBuffer

```python
from energy_fault_detector.streaming import SlidingWindowBuffer

# Create window buffer for sequence models
window_buffer = SlidingWindowBuffer(
    window_size=10,  # 10 phasor values per window
    stride=1
)

# Process stream and create windows
for batch in source:
    # Add data to window buffer
    new_windows = window_buffer.add_dataframe(batch.data)
    
    # Process each new window
    for window in new_windows:
        # window shape: (window_size, n_features)
        # Use with LSTM/CNN models
        pass
```

## Error Handling

### Common Issues

**Problem**: No data received  
**Solutions**:
- Check that OpenPMU device is sending data to the correct IP and port
- Verify firewall settings
- Check that the datagram type matches what's being sent
- Use `netstat -anu` to verify socket is listening

**Problem**: Parse errors  
**Solutions**:
- Check that the XML format matches OpenPMU V2 specification
- Verify that all required tags are present
- Check for malformed Base64 payloads
- Enable debug logging for detailed error messages

**Problem**: Missing channels  
**Solutions**:
- Verify that the number of channels in the datagram matches the expected count
- Check that all Channel_N tags are present
- Ensure channel indices are sequential (Channel_0, Channel_1, etc.)

### Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create source with debug logging
source = OpenPMUUDPDataSource(port=4713)
source.open()

# Process with error handling
try:
    for batch in source:
        print(f"Received batch: {batch.batch_index}")
except Exception as e:
    print(f"Error: {e}")
finally:
    source.close()
```

## Performance Considerations

### Memory Usage

- **Sampled Values**: Each datagram can contain many samples per channel. For example, with 3 channels, 128 samples each, and 16-bit values, each datagram is approximately 1KB.
- **Phasor Values**: Each datagram contains one phasor value per channel, so memory usage is much lower.

### Processing Speed

- **Batch Size**: Larger batches reduce overhead but increase latency. Recommended: 50-200 datagrams per batch.
- **Buffer Size**: Larger buffers provide more context for sequence models but use more memory.
- **Online Learning**: Enabling online learning adds processing overhead.

### Network Configuration

- **Unicast**: Use for point-to-point connections
- **Multicast**: Use for one-to-many connections (e.g., multiple detectors receiving the same data)
- **Port**: 4713 is commonly used for OpenPMU, but any port can be used

## OpenPMU vs IEEE C37.118

### Key Differences

| Feature | OpenPMU XML | IEEE C37.118 |
|---------|-------------|---------------|
| **Format** | XML text | Binary |
| **Transport** | UDP | UDP/TCP |
| **Data Types** | Samples, Phasors | Phasors only |
| **Metadata** | Extensive in XML | In configuration frame |
| **Encoding** | Base64 for binary data | Binary |
| **Flexibility** | High (self-describing) | Standardized |
| **Size** | Larger (text-based) | Smaller (binary) |

### When to Use OpenPMU

- **Development and Testing**: OpenPMU's self-describing XML format makes it easy to work with
- **Interoperability**: XML can be easily parsed by many tools and languages
- **Flexibility**: Easy to extend with custom metadata
- **Debugging**: Human-readable format simplifies troubleshooting

### When to Use IEEE C37.118

- **Production Systems**: Binary format is more efficient for high-volume data
- **Standard Compliance**: Required for compatibility with existing PMU infrastructure
- **Bandwidth Constraints**: Binary format uses less network bandwidth

## API Reference

### OpenPMUUDPDataSource

**Inherits from**: UDPStreamDataSource

**Attributes:**
- `host`: Host address
- `port`: Port number
- `datagram_type`: Expected datagram type ('auto', 'samples', or 'phasors')

**Methods:**
- `open()`: Open the UDP socket and start receiving
- `close()`: Close the UDP socket
- `reset()`: Reset statistics and buffers
- `statistics`: Get stream statistics

**Properties:**
- `is_open`: Whether the source is open
- `batch_index`: Current batch index

### OpenPMUParser

**Methods:**
- `parse_xml(xml_string)`: Parse an OpenPMU XML datagram
- `decode_payload(payload, n_samples, bits, is_signed)`: Decode Base64 payload

### OpenPMUDatagram

**Attributes:**
- `format`: Datagram format ('Samples' or 'Phasors')
- `date`: Date string
- `time`: Time string
- `frame`: Frame number
- `channels`: List of OpenPMUChannel objects
- `fs`: Sampling rate (for Samples)
- `n`: Number of samples (for Samples)
- `bits`: Bits per sample (for Samples)
- `algorithm`: Algorithm name (for Phasors)

**Properties:**
- `timestamp`: Datetime object
- `is_sampled_values`: Whether this is a Samples datagram
- `is_phasor_values`: Whether this is a Phasors datagram

### OpenPMUChannel

**Attributes:**
- `name`: Channel name
- `channel_type`: Channel type ('V' or 'I')
- `phase`: Phase ('a', 'b', 'c', 'n', etc.)
- `range`: Full scale deflection
- `payload`: Base64-encoded payload (for Samples)
- `mag`: Magnitude (for Phasors)
- `angle`: Angle in degrees (for Phasors)
- `freq`: Frequency in Hz (for Phasors)
- `rocof`: Rate of change of frequency (for Phasors)

### OpenPMUGenerator

**Attributes:**
- `n_channels`: Number of channels
- `fs`: Sampling rate in Hz
- `n_samples`: Number of samples per payload
- `frame_counter`: Current frame number
- `channel_names`: Channel names
- `channel_types`: Channel types
- `channel_phases`: Channel phases
- `channel_ranges`: Channel ranges

**Methods:**
- `set_channel_info(names, types, phases, ranges)`: Set custom channel information
- `generate_sampled_values_datagram(timestamp)`: Generate SV datagram
- `generate_phasor_values_datagram(timestamp)`: Generate PV datagram
- `generate_datagram(datagram_type, timestamp)`: Generate datagram of specified type

## References

- [OpenPMU Documentation](https://github.com/OpenPMU/OpenPMUdocs/tree/master/XML_Datagrams)
- [OpenPMU V2 Paper](https://ieeexplore.ieee.org/document/8273986): "A modular phasor measurement unit design featuring open data exchange methods"
- [IEEE C37.118 Standard](https://standards.ieee.org/standard/118-2005.html): Standard for Synchrophasors for Power Systems
- [IEC 61850-90-5 Standard](https://www.iec.ch/): Communication networks and systems for power utility automation

## Troubleshooting

### XML Parsing Errors

**Error**: `ValueError: Failed to parse XML`
- **Cause**: Malformed XML or invalid OpenPMU format
- **Solution**: Check the XML structure matches the OpenPMU specification

**Error**: `ValueError: Unknown datagram format`
- **Cause**: Format tag is missing or not 'Samples' or 'Phasors'
- **Solution**: Ensure the Format tag is present and valid

### Payload Decoding Errors

**Error**: `struct.error: unpack requires a buffer of X bytes`
- **Cause**: Payload size doesn't match expected size based on n and bits
- **Solution**: Check that n and bits values match the actual payload size

**Error**: `ValueError: Unsupported bit depth`
- **Cause**: Bit depth is not 16, 32, or 64
- **Solution**: Use supported bit depths (16, 32, or 64)

### Network Errors

**Error**: `socket.error: [Errno 99] Cannot assign requested address`
- **Cause**: Invalid host or port
- **Solution**: Check host and port values

**Error**: `socket.timeout: timed out`
- **Cause**: No data received within timeout period
- **Solution**: Increase timeout or check data source

## Example: Custom OpenPMU Processing

```python
from energy_fault_detector.data_sources import OpenPMUUDPDataSource, OpenPMUParser

class CustomOpenPMUProcessor:
    def __init__(self):
        self.parser = OpenPMUParser()
        
    def process_datagram(self, xml_string):
        """Process a single OpenPMU datagram."""
        datagram = self.parser.parse_xml(xml_string)
        
        if datagram.is_phasor_values:
            return self.process_phasor_datagram(datagram)
        elif datagram.is_sampled_values:
            return self.process_sampled_datagram(datagram)
        else:
            return None
    
    def process_phasor_datagram(self, datagram):
        """Process a Phasor Values datagram."""
        result = {
            'timestamp': datagram.timestamp,
            'frame': datagram.frame,
            'algorithm': datagram.algorithm,
            'phasors': []
        }
        
        for channel in datagram.channels:
            phasor = {
                'name': channel.name,
                'type': channel.channel_type,
                'phase': channel.phase,
                'range': channel.range,
                'mag': channel.mag,
                'angle': channel.angle,
                'freq': channel.freq,
                'rocof': channel.rocof
            }
            result['phasors'].append(phasor)
        
        return result
    
    def process_sampled_datagram(self, datagram):
        """Process a Sampled Values datagram."""
        result = {
            'timestamp': datagram.timestamp,
            'frame': datagram.frame,
            'fs': datagram.fs,
            'n': datagram.n,
            'bits': datagram.bits,
            'channels': []
        }
        
        for channel in datagram.channels:
            # Decode payload
            samples = self.parser.decode_payload(
                channel.payload,
                datagram.n or 0,
                datagram.bits or 16,
                is_signed=True
            )
            
            # Scale samples
            if channel.range > 0:
                samples = samples / (2 ** (datagram.bits or 16) - 1) * channel.range
            
            channel_data = {
                'name': channel.name,
                'type': channel.channel_type,
                'phase': channel.phase,
                'range': channel.range,
                'samples': samples.tolist()
            }
            result['channels'].append(channel_data)
        
        return result

# Usage
processor = CustomOpenPMUProcessor()

# Process a datagram
xml = """<OpenPMU>...</OpenPMU>"""
result = processor.process_datagram(xml)
print(result)
```
