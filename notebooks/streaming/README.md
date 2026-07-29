# Streaming Fault Detection Notebooks

This directory contains Jupyter notebooks demonstrating the streaming capabilities of EnergyFaultDetector.

## Available Notebooks

### 1. [01_Streaming_Quick_Start.ipynb](./01_Streaming_Quick_Start.ipynb)
**Level:** Beginner  
**Duration:** 15-20 minutes  
**Description:** Introduction to streaming fault detection with EnergyFaultDetector. Covers:
- Basic streaming architecture
- CSV file streaming
- Synthetic data generation
- Basic fault detection on streams
- Result visualization

**What you'll learn:**
- How to set up a streaming data source
- How to initialize and use StreamingFaultDetector
- How to process streams and interpret results
- Basic visualization of streaming results

### 2. [02_Advanced_Streaming.ipynb](./02_Advanced_Streaming.ipynb)
**Level:** Intermediate  
**Duration:** 25-30 minutes  
**Description:** Advanced streaming features and techniques. Covers:
- Data buffering and sliding windows
- Online learning with concept drift
- Performance analysis and optimization
- Custom data source implementation
- Advanced result analysis and visualization

**What you'll learn:**
- How to use DataBuffer and SlidingWindowBuffer
- How to enable and configure online learning
- How to analyze streaming performance
- How to create custom data sources
- Advanced visualization techniques

### 3. [03_Network_Streams.ipynb](./03_Network_Streams.ipynb)
**Level:** Intermediate/Advanced  
**Duration:** 20-25 minutes  
**Description:** Network-based streaming data sources. Covers:
- UDP stream data sources
- Phasor UDP data sources (IEEE C37.118 compatible)
- TCP stream data sources
- MQTT stream data sources
- Integration with StreamingFaultDetector

**What you'll learn:**
- How to use UDP for high-speed data acquisition
- How to process phasor measurement unit (PMU) data
- How to use TCP for reliable data streams
- How to use MQTT for IoT and sensor networks
- How to integrate network sources with fault detection

## Prerequisites

To run these notebooks, you need:

1. **Python 3.10-3.12**
2. **EnergyFaultDetector** installed:
   ```bash
   pip install energy-fault-detector
   ```
3. **Jupyter Notebook** or **JupyterLab**:
   ```bash
   pip install notebook
   ```
4. **Required dependencies**:
   ```bash
   pip install pandas numpy matplotlib
   ```
5. **Optional dependencies** (for network streams):
   ```bash
   pip install paho-mqtt
   ```

## Running the Notebooks

### Option 1: Using Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Then open the notebook in your browser
```

### Option 2: Using JupyterLab

```bash
# Start JupyterLab
jupyter lab

# Then open the notebook in your browser
```

### Option 3: Using Google Colab

1. Upload the notebook to Google Colab
2. Install EnergyFaultDetector:
   ```python
   !pip install energy-fault-detector
   ```
3. Install optional dependencies:
   ```python
   !pip install paho-mqtt
   ```
4. Run the cells

## Data Requirements

The notebooks use synthetic data by default, so no external data files are required. However, if you want to use your own data:

- **CSV files**: Should have a consistent structure with sensor readings
- **Network data**: Requires appropriate network infrastructure
- **MQTT**: Requires a running MQTT broker (e.g., Mosquitto)

## Learning Path

1. **Beginner**: Start with `01_Streaming_Quick_Start.ipynb`
2. **Intermediate**: Move to `02_Advanced_Streaming.ipynb`
3. **Network Streaming**: Try `03_Network_Streams.ipynb`
4. **Integration**: Connect to real data streams in your applications

## Network Stream Testing

For testing network streams without actual network infrastructure:

- **UDP/TCP**: The notebooks include simulator classes that create test data
- **MQTT**: Requires a running MQTT broker. You can use:
  ```bash
  # Using Docker
  docker run -d -p 1883:1883 eclipse-mosquitto
  
  # Or install Mosquitto locally
  sudo apt-get install mosquitto mosquitto-clients
  ```

## Additional Resources

- [Streaming Guide](../../docs/streaming/streaming_guide.md) - Comprehensive documentation
- [Network Streams Guide](../../docs/streaming/network_streams.md) - Network-specific documentation
- [API Reference](../../docs/streaming/streaming_guide.md#api-reference) - Detailed API documentation
- [Main Documentation](../../docs/) - Complete EnergyFaultDetector documentation

## Troubleshooting

**Issue: ModuleNotFoundError**  
Make sure EnergyFaultDetector is installed:
```bash
pip install energy-fault-detector
```

**Issue: Import errors for network modules**  
Install the required dependencies:
```bash
pip install paho-mqtt
```

**Issue: Connection errors**  
- For UDP/TCP: Check firewall settings and port availability
- For MQTT: Verify the broker is running and accessible

**Issue: Slow performance**  
Reduce batch_size, buffer_size, or use smaller datasets for testing.

## Feedback

If you have questions, find bugs, or have suggestions for improving these notebooks, please:

1. Open an issue on GitHub
2. Contact the development team
3. Contribute improvements via pull requests

Happy streaming! 🚀
