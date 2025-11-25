"""Pipeline to feed streamed data into the existing energy fault detector workflow."""

from __future__ import annotations
import tempfile
import os
from typing import Optional, Dict, Any

from .base import StreamDataSource

from energy_fault_detector.quick_fault_detection.quick_fault_detector import (
    quick_fault_detector,
)


def stream_to_csv_and_run_quick_fault_detector(
    stream: StreamDataSource,
    output_csv: Optional[str] = None,
    quick_fd_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convenience function:
    - materialise a streamed source into a CSV
    - run the existing quick_fault_detection pipeline on that CSV.

    quick_fd_kwargs are passed through to `quick_fault_detection`, e.g.
    {
        "csv_data_path": <will be filled>,
        "time_column_name": "timestamp",
        "status_data_column_name": None,
        ...
    }
    """
    if quick_fd_kwargs is None:
        quick_fd_kwargs = {}

    # choose CSV path
    tmp_file = None
    if output_csv is None:
        fd, tmp_path = tempfile.mkstemp(prefix="efd_stream_", suffix=".csv")
        os.close(fd)
        output_csv = tmp_path
        tmp_file = tmp_path

    # 1) stream → CSV
    stream.to_csv(output_csv)

    # 2) call existing quick fault detection pipeline
    quick_fd_kwargs = dict(quick_fd_kwargs)
    quick_fd_kwargs["csv_data_path"] = output_csv
    results = quick_fault_detector(**quick_fd_kwargs)

    # 3) clean up temp file
    if tmp_file is not None and os.path.exists(tmp_file):
        os.remove(tmp_file)

    return results
