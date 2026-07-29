# Streaming Fault Detection Guide

EnergyFaultDetector now supports streaming data sources for real-time fault detection. This guide covers the new streaming capabilities, including data sources, configuration, and usage patterns.

## Overview

The streaming functionality allows you to:

- Process data from **real-time streams** (UDP, TCP, MQTT - coming in Phase 2)
- Simulate streaming with **CSV files** in batches
- Generate **synthetic data** for testing
- Perform **online learning** (optional) to adapt models to changing conditions
- Handle **sequence data** with sliding windows for LSTM/CNN models

## Architecture

The streaming system consists of several key components:

```
┌─────────────────────────────────────────────────────────────┐
│                    StreamingFaultDetector                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌───────────┐ │
│  │  Data Source     │    │   Data Buffer    │    │  Model    │ │
│  │ (CSV, UDP, etc.) │───▶│ (Sliding Window) │───▶│ Autoencoder│ │
│  └─────────────────┘    └─────────────────┘    └───────────┘ │
│           ▲                    ▲                    ▲          │
│           │                    │                    │          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌───────────┐ │
│  │  StreamConfig    │    │  BatchResult     │    │Threshold  │ │
│  └─────────────────┘    └─────────────────┘    │Selector   │ │
│                                          ┌─────────────────┐ │
│                                          │ StreamingResult  │ │
│                                          └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Streaming with CSV

```python
from energy_fault_detector import StreamingFaultDetector, Config
from energy_fault_detector.data_sources import CSVStreamDataSource, StreamConfig
from energy_fault_detector.config import generate_quickstart_config

# 1. Create configuration
config = generate_quickstart_config()

# 2. Create streaming detector
detector = StreamingFaultDetector(config=config)

# 3. Initialize with training data (or load existing model)
# For this example, we'll train on the first part of our CSV
import pandas as pd
df = pd.read_csv("my_data.csv")
detector.initialize(fit_data=df.iloc[:1000])  # Train on first 1000 samples

# 4. Create data source for streaming
source = CSVStreamDataSource(
    file_path="my_data.csv",
    config=StreamConfig(batch_size=100, delay_seconds=0.01)  # 100 samples per batch, 10ms delay
)

# 5. Process the stream
result = detector.process_stream(source)

# 6. View results
print(f"Total samples: {result.total_samples}")
print(f"Anomalies detected: {result.total_anomalies}")
print(f"Throughput: {result.throughput:.2f} samples/sec")

# Get anomalies as DataFrame
anomalies_df = result.get_anomalies()
```

### Synthetic Data for Testing

```python
from energy_fault_detector import StreamingFaultDetector
from energy_fault_detector.data_sources import SimulatedFaultDataSource, StreamConfig
from energy_fault_detector.config import generate_quickstart_config

# Create synthetic data source with faults
source = SimulatedFaultDataSource(
    n_samples=10000,
    n_features=5,
    config=StreamConfig(batch_size=100, delay_seconds=0.01),
    fault_rate=0.1,  # 10% chance of fault in each batch
    fault_magnitude=5.0
)

# Initialize detector with normal data
config = generate_quickstart_config()
detector = StreamingFaultDetector(config=config)

# Train on synthetic normal data
import pandas as pd
import numpy as np
normal_data = pd.DataFrame(np.random.randn(500, 5), columns=[f"sensor_{i}" for i in range(5)])
detector.initialize(fit_data=normal_data)

# Process the synthetic stream
result = detector.process_stream(source)

print(f"Detected {result.total_anomalies} anomalies in synthetic data")
```

## Data Sources

### CSVStreamDataSource

Reads CSV files in configurable batches to simulate streaming.

**Parameters:**
- `file_path`: Path to the CSV file
- `config`: `StreamConfig` object with batch settings
- `parse_dates`: Column(s) to parse as dates
- `index_col`: Column to use as index
- `timestamp_col`: Column containing timestamps
- `dtype`: Data types for columns

**Example:**
```python
from energy_fault_detector.data_sources import CSVStreamDataSource, StreamConfig

# Basic usage
source = CSVStreamDataSource(
    file_path="data.csv",
    config=StreamConfig(batch_size=100, delay_seconds=0.01)
)

# With timestamp parsing
source = CSVStreamDataSource(
    file_path="data.csv",
    config=StreamConfig(batch_size=50),
    parse_dates=["timestamp"],
    index_col="timestamp"
)

# Read all data at once
all_data = source.read_all()

# Iterate through batches
for batch in source:
    print(f"Batch {batch.batch_index}: {len(batch.data)} samples")
    print(f"Timestamps: {batch.timestamps}")
    print(f"Metadata: {batch.metadata}")
```

### SimulatedFaultDataSource

Generates synthetic time series data with configurable faults.

**Parameters:**
- `n_samples`: Total number of samples to generate (default: 10000)
- `n_features`: Number of features/sensors (default: 5)
- `config`: `StreamConfig` object
- `fault_rate`: Probability of a fault occurring in each batch (default: 0.1)
- `fault_magnitude`: Magnitude of faults as multiple of std (default: 3.0)
- `base_freq`: Base frequency for seasonal patterns in Hz (default: 0.01)
- `noise_level`: Standard deviation of Gaussian noise (default: 0.1)
- `feature_names`: Custom names for the features

**Example:**
```python
from energy_fault_detector.data_sources import SimulatedFaultDataSource, StreamConfig

source = SimulatedFaultDataSource(
    n_samples=5000,
    n_features=3,
    config=StreamConfig(batch_size=100, delay_seconds=0.05),
    fault_rate=0.2,
    fault_magnitude=4.0,
    feature_names=["temperature", "pressure", "vibration"]
)

# Use with context manager
with source:
    for batch in source:
        print(f"Generated batch {batch.batch_index}")
```

### SineWaveDataSource

Generates clean sine wave data for basic testing.

**Parameters:**
- `n_samples`: Total number of samples
- `n_features`: Number of sine waves to generate
- `config`: `StreamConfig` object
- `frequency`: Frequency of the sine wave in Hz (default: 1.0)
- `amplitude`: Amplitude of the sine wave (default: 1.0)

**Example:**
```python
from energy_fault_detector.data_sources import SineWaveDataSource, StreamConfig

source = SineWaveDataSource(
    n_samples=1000,
    n_features=2,
    config=StreamConfig(batch_size=50),
    frequency=2.0,
    amplitude=1.5
)
```

## StreamingFaultDetector

The main class for real-time fault detection on streaming data.

### Initialization

```python
from energy_fault_detector import StreamingFaultDetector, Config

# Basic initialization
detector = StreamingFaultDetector(config=config)

# With custom parameters
detector = StreamingFaultDetector(
    config=config,
    buffer_size=2000,           # Size of data buffer
    window_size=10,             # Window size for sequence models
    online_learning=True,       # Enable online learning
    online_learning_interval=50 # Update model every 50 batches
)
```

### Training and Initialization

```python
import pandas as pd

# Method 1: Train with initial data
detector.initialize(fit_data=training_data, normal_index=normal_mask)

# Method 2: Load existing model
detector.initialize()  # Will try to load from model_directory

# Method 3: Load explicitly
detector.load_models("/path/to/models")
```

### Processing Data

```python
# Process a complete stream
result = detector.process_stream(data_source)

# Process with limit
result = detector.process_stream(data_source, max_batches=100)

# Process DataFrame in batches
result = detector.process_dataframe(df, batch_size=100)

# Process single batch
batch_result = detector.process_batch(batch)
```

### Results

```python
# Access results
print(f"Total samples: {result.total_samples}")
print(f"Total anomalies: {result.total_anomalies}")
print(f"Anomaly rate: {result.summary()['anomaly_rate']:.2%}")
print(f"Throughput: {result.throughput:.2f} samples/sec")

# Get results as DataFrame
results_df = result.to_dataframe()
print(results_df.head())

# Get only anomalies
anomalies_df = result.get_anomalies()

# Access individual batch results
for batch_result in result.batch_results:
    print(f"Batch {batch_result.batch_index}: {batch_result.n_anomalies} anomalies")
```

### Model Management

```python
# Save models
detector.save_models("/path/to/save")

# Load models
detector.load_models("/path/to/models")

# Close detector (clean up resources)
detector.close()
```

## Configuration

### StreamConfig

Configures streaming behavior.

```python
from energy_fault_detector.data_sources import StreamConfig

config = StreamConfig(
    batch_size=100,           # Samples per batch
    delay_seconds=0.01,     # Delay between batches (for simulation)
    buffer_size=1000,        # Maximum buffer size
    timeout=30.0,            # Timeout for stream operations
    max_retries=3,           # Maximum retry attempts
    metadata={}             # Additional metadata
)
```

### StreamingConfig

Extended configuration for streaming fault detection.

```python
from energy_fault_detector.config import StreamingConfig

config = StreamingConfig(config_dict={
    'batch_size': 100,
    'delay_seconds': 0.01,
    'buffer_size': 1000,
    'window_size': 10,
    'online_learning': True,
    'online_learning_interval': 50,
    'data_source': {
        'type': 'csv',
        'file_path': 'data.csv'
    }
})
```

## Data Buffering

### DataBuffer

Simple FIFO buffer for storing recent data.

```python
from energy_fault_detector.streaming import DataBuffer

buffer = DataBuffer(max_size=1000)

# Add data
buffer.add(data_array)  # numpy array
buffer.add_dataframe(df)  # pandas DataFrame

# Get recent data
recent = buffer.get_recent(100)  # Last 100 samples

# Convert to DataFrame
df = buffer.to_dataframe(columns=['col1', 'col2'])

# Check status
print(f"Size: {buffer.size}, Is full: {buffer.is_full}")
```

### SlidingWindowBuffer

Buffer for sequence models that maintains sliding windows.

```python
from energy_fault_detector.streaming import SlidingWindowBuffer

buffer = SlidingWindowBuffer(
    window_size=10,      # Size of each window
    stride=1,            # Stride between windows
    max_windows=100,    # Maximum windows to store
    pad_value=0.0       # Padding value
)

# Add data
new_windows = buffer.add(data_array)

# Get windows
windows = buffer.windows  # List of window arrays

# Get as sequence dataset (for LSTM/CNN)
X, timestamps = buffer.get_sequence_dataset()
# X shape: (n_windows, window_size, n_features)

# Convert to DataFrame
window_df = buffer.get_window_dataframe()
```

## Online Learning

The `StreamingFaultDetector` supports optional online learning to adapt to changing data patterns.

```python
# Enable online learning
detector = StreamingFaultDetector(
    config=config,
    online_learning=True,
    online_learning_interval=50  # Update every 50 batches
)

# The detector will automatically update the model
# using buffered data at the specified interval

# Manual online learning
detector._perform_online_learning()  # Internal method
```

**Note:** Online learning is currently implemented as a simple refit with one epoch. For production use, you may want to implement more sophisticated online learning strategies.

## Error Handling

```python
try:
    result = detector.process_stream(source)
except Exception as e:
    print(f"Error during streaming: {e}")
    # Handle error

# Check for errors in individual batches
for batch_result in result.batch_results:
    if batch_result.metadata.get('error'):
        print(f"Error in batch {batch_result.batch_index}")
```

## Performance Optimization

### Batch Size
- Larger batches: Better throughput but higher latency
- Smaller batches: Lower latency but more overhead
- Recommended: 100-1000 samples per batch

### Buffer Size
- Larger buffers: More context for sequence models, better online learning
- Smaller buffers: Lower memory usage
- Recommended: 1000-10000 samples

### Window Size (for sequence models)
- Should match your model's expected sequence length
- Typical: 10-100 timesteps

## Use Cases

### 1. Real-time Monitoring
```python
# Continuous monitoring of a data stream
while True:
    batch = get_new_data()  # From your data acquisition system
    batch_result = detector.process_batch(batch)
    
    if batch_result.n_anomalies > 0:
        send_alert(batch_result)
    
    time.sleep(0.01)
```

### 2. Batch Processing with Streaming Interface
```python
# Process large historical datasets in batches
source = CSVStreamDataSource(
    file_path="large_dataset.csv",
    config=StreamConfig(batch_size=1000)
)

result = detector.process_stream(source)
```

### 3. Testing and Validation
```python
# Test with synthetic data
source = SimulatedFaultDataSource(
    n_samples=10000,
    fault_rate=0.05,
    config=StreamConfig(batch_size=100)
)

result = detector.process_stream(source)

# Validate results
assert result.total_anomalies > 0, "Should detect some anomalies"
print(f"Detection rate: {result.total_anomalies / (result.total_samples * 0.05):.2f}x")
```

### 4. Model Comparison
```python
# Compare different models on streaming data
models = [
    Config.from_file("model1.yaml"),
    Config.from_file("model2.yaml"),
]

for model_config in models:
    detector = StreamingFaultDetector(config=model_config)
    detector.initialize(fit_data=training_data)
    
    result = detector.process_stream(test_source)
    print(f"Model: {model_config.autoencoder.name}, Anomalies: {result.total_anomalies}")
```

## Troubleshooting

### Common Issues

**Problem:** "Data source is not open"
**Solution:** Call `source.open()` before iterating, or use context manager:
```python
with source:
    for batch in source:
        # process batch
```

**Problem:** "No valid windows found"
**Solution:** Check that your window_size is smaller than your data length:
```python
# window_size must be <= data length
buffer = SlidingWindowBuffer(window_size=min(10, len(data)))
```

**Problem:** Low throughput
**Solution:** Increase batch_size, reduce delay_seconds, or check for bottlenecks:
```python
# Monitor processing time
for batch_result in result.batch_results:
    print(f"Batch processing time: {batch_result.processing_time:.3f}s")
```

**Problem:** Memory issues
**Solution:** Reduce buffer_size, batch_size, or max_windows:
```python
detector = StreamingFaultDetector(
    buffer_size=1000,  # Reduce from default 10000
    window_size=10      # Reduce from default
)
```

## API Reference

### DataSource Classes

#### CSVStreamDataSource

**Attributes:**
- `file_path`: Path to CSV file
- `config`: StreamConfig object
- `total_rows`: Total rows in CSV
- `columns`: Column names
- `estimated_batches`: Estimated number of batches

**Methods:**
- `open()`: Open the data source
- `close()`: Close the data source
- `reset()`: Reset to beginning
- `read_all()`: Read all data as DataFrame
- `take(n)`: Take n batches

#### SimulatedFaultDataSource

**Attributes:**
- `n_samples`: Total samples
- `n_features`: Number of features
- `fault_rate`: Fault probability
- `fault_magnitude`: Fault size
- `remaining_samples`: Samples left to generate

**Methods:**
- `set_seed(seed)`: Set random seed for reproducibility

### StreamingFaultDetector

**Attributes:**
- `config`: Configuration
- `buffer_size`: Data buffer size
- `window_size`: Window size for sequences
- `online_learning`: Whether online learning is enabled
- `batch_count`: Number of batches processed
- `total_samples`: Total samples processed
- `result`: Current StreamingResult
- `fault_detector`: Underlying FaultDetector

**Methods:**
- `initialize(fit_data, normal_index)`: Initialize with training data
- `process_stream(data_source, max_batches)`: Process a data stream
- `process_dataframe(df, batch_size)`: Process DataFrame in batches
- `process_batch(batch)`: Process a single batch
- `save_models(path)`: Save models to disk
- `load_models(path)`: Load models from disk
- `close()`: Clean up resources

### StreamConfig

**Attributes:**
- `batch_size`: Samples per batch (default: 1000)
- `delay_seconds`: Delay between batches (default: 0.1)
- `buffer_size`: Maximum buffer size (default: 10)
- `timeout`: Operation timeout (default: 30.0)
- `max_retries`: Maximum retry attempts (default: 3)
- `metadata`: Additional metadata

### DataBatch

**Attributes:**
- `data`: DataFrame with batch data
- `timestamps`: Array of timestamps
- `batch_index`: Index of this batch
- `is_complete`: Whether this is the final batch
- `metadata`: Additional metadata

**Properties:**
- `shape`: Shape of the data (rows, columns)
- `len`: Number of samples in batch

### StreamingResult

**Attributes:**
- `batch_results`: List of BatchResult objects
- `total_samples`: Total samples processed
- `total_anomalies`: Total anomalies detected
- `start_time`: When processing started
- `end_time`: When processing ended
- `model_metadata`: Model information
- `stream_metadata`: Stream information

**Methods:**
- `add_batch_result(batch_result)`: Add a batch result
- `to_dataframe()`: Convert to DataFrame
- `get_anomalies()`: Get only anomalies as DataFrame
- `summary()`: Get summary statistics

**Properties:**
- `all_predictions`: All predictions concatenated
- `all_scores`: All scores concatenated
- `all_timestamps`: All timestamps concatenated
- `processing_times`: Processing times for all batches
- `avg_processing_time`: Average processing time
- `total_processing_time`: Total processing time
- `throughput`: Samples per second

### BatchResult

**Attributes:**
- `batch_index`: Index of the batch
- `timestamps`: Timestamps for predictions
- `predictions`: Anomaly predictions (boolean array)
- `scores`: Anomaly scores (float array)
- `reconstructed`: Reconstructed data (optional)
- `processing_time`: Processing time in seconds
- `metadata`: Additional metadata

**Properties:**
- `n_anomalies`: Number of anomalies in batch
- `anomaly_indices`: Indices of anomalies in batch
