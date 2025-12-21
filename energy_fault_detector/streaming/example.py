"""
Example code for how to use the Energy Fault Detector on streamed data sources.
"""

from energy_fault_detector import StreamingFaultDetector
from energy_fault_detector.streaming.windowed_data_stream import WindowedDataStream
from energy_fault_detector.streaming.c37118_stream import (
    C37118TCPDataStream,
    C37118FileDataStream,
)

# ----- Train on recorded CSV data -----
train_stream = WindowedDataStream.from_csv(
    "your_training_file.csv", window_size=50, step_size=10, timestamp_col="timestamp"
)

detector = StreamingFaultDetector.from_config("path/to/config.yaml")
for window_df in train_stream:
    detector.update(window_df)  # or whatever your training API is

# ----- Evaluate on recorded PMU frames -----
file_rows = C37118FileDataStream("capture.bin", frames_per_chunk=1, include_raw=False)
windowed_file = WindowedDataStream(source=file_rows, window_size=50, step_size=1)
for window_df in windowed_file:
    result = detector.predict(window_df)
    print(result)

# ----- Evaluate on live PMU data -----
live_rows = C37118TCPDataStream("192.168.0.10", 4712, frames_per_chunk=1)
windowed_live = WindowedDataStream(source=live_rows, window_size=100, step_size=10)
for window_df in windowed_live:
    result = detector.predict(window_df)
    # handle result (alert, log, etc.)
