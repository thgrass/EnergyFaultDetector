"""UDP stream data source for real-time data acquisition.

This module provides UDP-based data sources suitable for:
- Phasor Measurement Unit (PMU) data
- IEEE C37.118 protocol
- Custom UDP-based sensor protocols
- Any UDP multicast/unicast data streams
"""

import socket
import struct
import time
import logging
import threading
from typing import Optional, Dict, Any, Callable, Union
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from energy_fault_detector.data_sources.base import DataSource, DataBatch, StreamConfig

logger = logging.getLogger('energy_fault_detector.data_sources.udp_stream')


class UDPStreamDataSource(DataSource):
    """Data source for receiving UDP stream data.
    
    This class provides a flexible interface for receiving data over UDP,
    including support for:
    - Unicast and multicast UDP
    - Custom packet parsing
    - Timestamp synchronization
    - Phasor data (IEEE C37.118 compatible)
    
    Args:
        host: Host address to bind to (default: '0.0.0.0')
        port: Port to listen on (default: 5000)
        config: Stream configuration
        packet_parser: Custom function to parse UDP packets (optional)
        buffer_size: Socket buffer size in bytes (default: 65536)
        timeout: Socket timeout in seconds (default: 1.0)
        multicast_group: Multicast group address for multicast (optional)
        multicast_interface: Network interface for multicast (optional)
        
    Example:
        >>> # Basic UDP stream
        >>> source = UDPStreamDataSource(host='0.0.0.0', port=5000)
        >>> for batch in source:
        ...     print(f"Received {len(batch.data)} samples")
        
        >>> # With custom parser
        >>> def my_parser(packet_data, timestamp):
        ...     # Parse your custom UDP packet format
        ...     return data_dict, timestamps
        >>> 
        >>> source = UDPStreamDataSource(
        ...     port=5000,
        ...     packet_parser=my_parser,
        ...     config=StreamConfig(batch_size=100)
        ... )
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 5000,
        config: Optional[StreamConfig] = None,
        packet_parser: Optional[Callable[[bytes, float], tuple[Dict[str, Any], np.ndarray]]] = None,
        buffer_size: int = 65536,
        timeout: float = 1.0,
        multicast_group: Optional[str] = None,
        multicast_interface: Optional[str] = None,
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        
        self.host = host
        self.port = port
        self.packet_parser = packet_parser or self._default_parser
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.multicast_group = multicast_group
        self.multicast_interface = multicast_interface
        
        # Socket and state
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._receive_thread: Optional[threading.Thread] = None
        self._packet_queue: list[tuple[bytes, float]] = []
        self._queue_lock = threading.Lock()
        
        # Statistics
        self._packets_received = 0
        self._bytes_received = 0
        self._start_time: Optional[datetime] = None
        
        logger.info(f"UDPStreamDataSource initialized on {host}:{port}")
    
    def _default_parser(self, packet_data: bytes, timestamp: float) -> tuple[Dict[str, Any], np.ndarray]:
        """Default packet parser for simple float64 arrays.
        
        This parser assumes packets contain a sequence of float64 values.
        Override with a custom parser for your specific protocol.
        
        Args:
            packet_data: Raw UDP packet bytes
            timestamp: Reception timestamp
            
        Returns:
            Tuple of (data_dict, timestamps_array)
        """
        try:
            # Try to unpack as float64 array
            n_floats = len(packet_data) // 8
            if n_floats > 0:
                values = struct.unpack(f'{n_floats}d', packet_data[:n_floats * 8])
                data_dict = {f'value_{i}': [v] for i, v in enumerate(values)}
                timestamps = np.array([timestamp] * n_floats)
                return data_dict, timestamps
            else:
                # Fallback: treat as single value
                data_dict = {'value': [float(packet_data.hex())]}
                timestamps = np.array([timestamp])
                return data_dict, timestamps
        except struct.error:
            # If unpacking fails, return raw bytes as string
            data_dict = {'raw_data': [packet_data.hex()]}
            timestamps = np.array([timestamp])
            return data_dict, timestamps
    
    def _create_socket(self) -> socket.socket:
        """Create and configure the UDP socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
        
        if self.multicast_group:
            # Join multicast group
            sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                socket.inet_aton(self.multicast_group) + socket.inet_aton(self.multicast_interface or self.host)
            )
        
        sock.settimeout(self.timeout)
        return sock
    
    def _receive_packets(self) -> None:
        """Thread function to receive UDP packets."""
        if self._socket is None:
            return
        
        logger.debug(f"Starting UDP receiver thread on {self.host}:{self.port}")
        
        while self._running:
            try:
                data, addr = self._socket.recvfrom(self.buffer_size)
                timestamp = time.time()
                
                with self._queue_lock:
                    self._packet_queue.append((data, timestamp))
                    self._packets_received += 1
                    self._bytes_received += len(data)
                
                logger.debug(f"Received {len(data)} bytes from {addr}")
                
            except socket.timeout:
                # Timeout is normal, just continue
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Error receiving UDP packet: {e}")
        
        logger.debug("UDP receiver thread stopped")
    
    def open(self) -> 'UDPStreamDataSource':
        """Open the UDP socket and start receiving."""
        if self._is_open:
            return self
        
        # Create socket
        self._socket = self._create_socket()
        
        try:
            self._socket.bind((self.host, self.port))
            logger.info(f"UDP socket bound to {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to bind UDP socket: {e}")
            raise
        
        # Start receive thread
        self._running = True
        self._start_time = datetime.now()
        self._receive_thread = threading.Thread(target=self._receive_packets, daemon=True)
        self._receive_thread.start()
        
        self._is_open = True
        self._batch_index = 0
        
        return self
    
    def close(self) -> None:
        """Close the UDP socket and stop receiving."""
        self._running = False
        
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
            self._receive_thread = None
        
        if self._socket:
            try:
                self._socket.close()
            except Exception as e:
                logger.error(f"Error closing UDP socket: {e}")
            self._socket = None
        
        self._is_open = False
        logger.info("UDP data source closed")
    
    def reset(self) -> None:
        """Reset the data source."""
        with self._queue_lock:
            self._packet_queue.clear()
        self._batch_index = 0
        self._packets_received = 0
        self._bytes_received = 0
        self._start_time = datetime.now()
        logger.debug("UDP data source reset")
    
    def __next__(self) -> DataBatch:
        """Get the next batch of data from received packets."""
        if not self._is_open:
            raise RuntimeError("Data source is not open. Call open() first.")
        
        if self._socket is None:
            raise RuntimeError("Socket is not initialized.")
        
        # Collect packets for this batch
        batch_data = []
        batch_timestamps = []
        
        target_packets = self.config.batch_size
        start_time = time.time()
        
        while len(batch_data) < target_packets:
            # Check for timeout
            if time.time() - start_time > self.config.timeout:
                if not batch_data:
                    raise StopIteration("Timeout waiting for data")
                break
            
            # Get available packets
            with self._queue_lock:
                available_packets = len(self._packet_queue)
                if available_packets > 0:
                    # Take up to remaining needed packets
                    take_count = min(available_packets, target_packets - len(batch_data))
                    packets = self._packet_queue[:take_count]
                    self._packet_queue = self._packet_queue[take_count:]
                else:
                    packets = []
            
            if not packets:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue
            
            # Parse packets
            for packet_data, packet_timestamp in packets:
                try:
                    data_dict, timestamps = self.packet_parser(packet_data, packet_timestamp)
                    batch_data.append(data_dict)
                    batch_timestamps.extend(timestamps)
                except Exception as e:
                    logger.error(f"Error parsing packet: {e}")
                    continue
        
        if not batch_data:
            raise StopIteration("No data available")
        
        # Convert to DataFrame
        df = pd.DataFrame(batch_data)
        
        # If all packets had the same timestamp, use a range
        if len(set(batch_timestamps)) == 1 and len(batch_timestamps) > 1:
            timestamps = np.linspace(batch_timestamps[0], batch_timestamps[0] + len(batch_timestamps) * 0.001, len(batch_timestamps))
        else:
            timestamps = np.array(batch_timestamps)
        
        # Create metadata
        metadata = {
            'source_type': 'udp',
            'host': self.host,
            'port': self.port,
            'packets_in_batch': len(batch_data),
            'total_packets': self._packets_received,
            'total_bytes': self._bytes_received,
        }
        
        batch = DataBatch(
            data=df,
            timestamps=timestamps,
            batch_index=self._batch_index,
            is_complete=False,  # UDP streams are continuous
            metadata=metadata
        )
        
        self._batch_index += 1
        
        return batch
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get statistics about the UDP stream."""
        uptime = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        
        return {
            'packets_received': self._packets_received,
            'bytes_received': self._bytes_received,
            'uptime_seconds': uptime,
            'packets_per_second': self._packets_received / uptime if uptime > 0 else 0,
            'bytes_per_second': self._bytes_received / uptime if uptime > 0 else 0,
        }


class PhasorUDPDataSource(UDPStreamDataSource):
    """Specialized UDP data source for phasor measurement data.
    
    This class provides parsing for IEEE C37.118 phasor data messages.
    It handles the common format used by PMUs (Phasor Measurement Units).
    
    Args:
        host: Host to bind to (default: '0.0.0.0')
        port: Port to listen on (default: 4712, common for PMUs)
        config: Stream configuration
        n_phasors: Number of phasors expected (default: 3)
        n_analog: Number of analog values expected (default: 0)
        n_digital: Number of digital status words expected (default: 0)
        
    Example:
        >>> # PMU data source
        >>> source = PhasorUDPDataSource(port=4712, n_phasors=3)
        >>> for batch in source:
        ...     print(f"Phasor data: {batch.data.columns}")
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 4712,
        config: Optional[StreamConfig] = None,
        n_phasors: int = 3,
        n_analog: int = 0,
        n_digital: int = 0,
        **kwargs
    ):
        # Create custom parser for phasor data
        def phasor_parser(packet_data: bytes, timestamp: float) -> tuple[Dict[str, Any], np.ndarray]:
            return self._parse_phasor_packet(packet_data, timestamp, n_phasors, n_analog, n_digital)
        
        super().__init__(
            host=host,
            port=port,
            config=config,
            packet_parser=phasor_parser,
            **kwargs
        )
        
        self.n_phasors = n_phasors
        self.n_analog = n_analog
        self.n_digital = n_digital
        
        logger.info(f"PhasorUDPDataSource initialized for {n_phasors} phasors, {n_analog} analog, {n_digital} digital")
    
    def _parse_phasor_packet(
        self,
        packet_data: bytes,
        timestamp: float,
        n_phasors: int,
        n_analog: int,
        n_digital: int
    ) -> tuple[Dict[str, Any], np.ndarray]:
        """Parse a phasor data packet.
        
        This is a simplified parser for demonstration. Real IEEE C37.118
        parsing would need to handle the full protocol specification.
        
        Args:
            packet_data: Raw packet bytes
            timestamp: Reception timestamp
            n_phasors: Number of phasors
            n_analog: Number of analog values
            n_digital: Number of digital status words
            
        Returns:
            Tuple of (data_dict, timestamps)
        """
        data_dict = {}
        
        try:
            # IEEE C37.118 uses a specific format
            # This is a simplified version for demonstration
            
            # Header: sync word (2 bytes), frame size (2 bytes), ID code (2 bytes)
            # For this example, we'll assume the data starts after a header
            header_size = 16  # Typical header size
            
            if len(packet_data) < header_size:
                # Too short, return empty
                return data_dict, np.array([timestamp])
            
            # Extract phasor data (assuming float32 values)
            data_start = header_size
            data_size = len(packet_data) - data_start
            
            # Each phasor has real and imaginary parts (2 floats)
            expected_size = (n_phasors * 2 + n_analog + n_digital) * 4  # float32 = 4 bytes
            
            if data_size >= expected_size:
                # Parse phasor values
                phasor_format = f'{n_phasors * 2}f'  # Real and imaginary for each phasor
                phasor_values = struct.unpack(phasor_format, packet_data[data_start:data_start + n_phasors * 8])
                
                for i in range(n_phasors):
                    real = phasor_values[i * 2]
                    imag = phasor_values[i * 2 + 1]
                    magnitude = np.sqrt(real**2 + imag**2)
                    phase = np.arctan2(imag, real) * 180 / np.pi  # Convert to degrees
                    
                    data_dict[f'phasor_{i}_real'] = [real]
                    data_dict[f'phasor_{i}_imag'] = [imag]
                    data_dict[f'phasor_{i}_mag'] = [magnitude]
                    data_dict[f'phasor_{i}_phase'] = [phase]
                
                # Parse analog values
                analog_start = data_start + n_phasors * 8
                if n_analog > 0 and data_size >= analog_start + n_analog * 4:
                    analog_format = f'{n_analog}f'
                    analog_values = struct.unpack(analog_format, packet_data[analog_start:analog_start + n_analog * 4])
                    for i, value in enumerate(analog_values):
                        data_dict[f'analog_{i}'] = [value]
                
                # Parse digital status
                digital_start = analog_start + n_analog * 4
                if n_digital > 0 and data_size >= digital_start + n_digital * 4:
                    digital_format = f'{n_digital}I'  # Unsigned int
                    digital_values = struct.unpack(digital_format, packet_data[digital_start:digital_start + n_digital * 4])
                    for i, value in enumerate(digital_values):
                        data_dict[f'digital_{i}'] = [value]
            
            timestamps = np.array([timestamp] * max(1, len(data_dict)))
            return data_dict, timestamps
            
        except struct.error as e:
            logger.warning(f"Error parsing phasor packet: {e}")
            # Return raw data
            data_dict['raw_data'] = [packet_data.hex()]
            return data_dict, np.array([timestamp])
