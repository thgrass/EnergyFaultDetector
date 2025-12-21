"""Fault detector for streaming data.

This module introduces :class:`StreamingFaultDetector`, a subclass of
:class:`~energy_fault_detector.fault_detector.FaultDetector` that adds
support for processing streaming data sources.  Traditional use of the
Energy Fault Detector library assumes that the entire dataset is
available up front.  In many real‑world scenarios, such as
synchrophasor measurement applications, data arrive continuously and
must be evaluated in near real‑time.  The streaming fault detector
provides a generator interface for consuming a
:class:`~energy_fault_detector.streaming.data_stream.DataStream` and
yielding anomaly detection results for each chunk.
"""

from __future__ import annotations

from typing import Iterator, Optional
import pandas as pd

from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.core.fault_detection_result import FaultDetectionResult
from .data_stream import DataStream


class StreamingFaultDetector(FaultDetector):
    """Fault detector that supports live evaluation on streaming data.

    The :class:`StreamingFaultDetector` extends the existing
    :class:`~energy_fault_detector.fault_detector.FaultDetector` by
    providing a method to process data from a :class:`DataStream`.  For
    each window of data yielded by the stream, the underlying
    autoencoder, anomaly score and threshold selector are applied.

    Note that model initialization and training behave identically to
    the base class.  The streaming functionality only affects
    inference.
    """

    def live_predict(
        self,
        data_stream: DataStream,
        model_path: Optional[str] = None,
        root_cause_analysis: bool = False,
        track_losses: bool = False,
        track_bias: bool = False,
    ) -> Iterator[FaultDetectionResult]:
        """Run the fault detector on a streaming data source.

        This generator iterates over the provided ``data_stream`` and
        yields a :class:`FaultDetectionResult` for each window.  It
        leverages the existing :meth:`FaultDetector.predict` method
        internally, so pre‑ and post‑processing remain unchanged.

        Args:
            data_stream (DataStream): An iterable yielding
                :class:`pandas.DataFrame` windows of measurements.
            model_path (Optional[str], optional): Path to pre‑trained
                models.  If provided, models are loaded before
                processing begins.
            root_cause_analysis (bool, optional): Whether to run
                ARCANA for each window.  Defaults to ``False``.
            track_losses (bool, optional): If ``True``, record
                ARCANA losses for each window.  Defaults to ``False``.
            track_bias (bool, optional): If ``True``, record bias
                information during root‑cause analysis.  Defaults to
                ``False``.

        Yields:
            FaultDetectionResult: The anomaly detection results for
                each window.
        """
        # Optionally load models once at the beginning of streaming
        if model_path is not None:
            # Load the models into the current instance.  This will
            # overwrite any existing fitted models.
            self.load_models(model_path=model_path)

        for window in data_stream:
            # The underlying predict method requires a DataFrame with
            # timestamps as index and sensor values as columns.  This
            # streaming window satisfies this requirement.
            result = self.predict(
                sensor_data=window,
                model_path=None,  # models are already loaded or attached
                root_cause_analysis=root_cause_analysis,
                track_losses=track_losses,
                track_bias=track_bias,
            )
            yield result
