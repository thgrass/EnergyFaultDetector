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
3. Run the cells

## Data Requirements

The notebooks use synthetic data by default, so no external data files are required. However, if you want to use your own data:

- **CSV files**: Should have a consistent structure with sensor readings
- **Timestamp column**: Recommended for time-series analysis
- **Normal data**: Required for training the fault detection model

## Tips for Running Notebooks

1. **Start with the Quick Start notebook** if you're new to streaming
2. **Run cells sequentially** - some cells depend on previous ones
3. **Check memory usage** - streaming can use significant memory for large datasets
4. **Adjust parameters** - feel free to change batch sizes, window sizes, etc.
5. **Monitor performance** - use the performance analysis sections to optimize

## Customization

You can customize the notebooks by:

- **Changing data sources**: Replace synthetic data with your own CSV files
- **Adjusting configurations**: Modify StreamConfig and StreamingConfig parameters
- **Adding more analysis**: Extend the visualization and analysis sections
- **Integrating with your systems**: Connect to real data streams

## Troubleshooting

**Issue: ModuleNotFoundError**  
Make sure EnergyFaultDetector is installed:
```bash
pip install energy-fault-detector
```

**Issue: Import errors**  
Check that you're using Python 3.10-3.12 and all dependencies are installed.

**Issue: Slow performance**  
Reduce batch_size, buffer_size, or use smaller datasets for testing.

**Issue: Memory errors**  
Reduce buffer_size, window_size, or process data in smaller chunks.

## Learning Path

1. **Beginner**: Start with `01_Streaming_Quick_Start.ipynb`
2. **Intermediate**: Move to `02_Advanced_Streaming.ipynb`
3. **Advanced**: Create your own notebooks using the patterns from these examples
4. **Integration**: Connect to real data streams in your applications

## Additional Resources

- [Streaming Guide](../../docs/streaming/streaming_guide.md) - Comprehensive documentation
- [API Reference](../../docs/streaming/streaming_guide.md#api-reference) - Detailed API documentation
- [Main Documentation](../../docs/) - Complete EnergyFaultDetector documentation

## Feedback

If you have questions, find bugs, or have suggestions for improving these notebooks, please:

1. Open an issue on GitHub
2. Contact the development team
3. Contribute improvements via pull requests

Happy streaming! 🚀
