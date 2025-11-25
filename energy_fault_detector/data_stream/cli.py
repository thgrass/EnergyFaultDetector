"""CLI interface for streamed data sources."""

import argparse
from .ieee_c37 import IeeeC37StreamSource
from .pipeline import stream_to_csv_and_run_quick_fault_detector


def main():
    parser = argparse.ArgumentParser(
        description="Quick fault detection on IEEE synchrophasor streams."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcap", help="Path to PCAP file containing C37.118 traffic.")
    src.add_argument("--interface", help="Network interface for live C37.118 stream.")
    parser.add_argument(
        "--chunk-size", type=int, default=100, help="Number of frames per chunk."
    )
    # optionally forward some of the quick_fault_detector CLI options here
    args = parser.parse_args()

    live = args.interface is not None
    source_str = args.interface if live else args.pcap

    stream = IeeeC37StreamSource(
        source=source_str,
        chunk_size=args.chunk_size,
        live=live,
    )

    stream_to_csv_and_run_quick_fault_detector(
        stream=stream,
        quick_fd_kwargs={
            # The existing quick_fault_detection expects CSV path; we set it internally.
            # If EnergyFaultDetector requires specific column names, we can pass those here.
            "time_column_name": "timestamp",
            # other kwargs as needed...
        },
    )
