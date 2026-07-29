"""TCP stream data source for reliable data acquisition.

This module provides TCP-based data sources suitable for:
- Reliable point-to-point data streams
- Custom TCP-based sensor protocols
- Data streams requiring guaranteed delivery
- Any TCP-based data acquisition
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

logger = logging.getLogger('energy_fault_detector.data_sources.tcp_stream')


class TCPStreamDataSource(DataSource):
    """Data source for receiving TCP stream data.
    
    This class provides a flexible interface for receiving data over TCP,
    including support for:
    - TCP server mode (listening for connections)
    - TCP client mode (connecting to a server)
    - Custom message parsing
    - Message framing and delimiting
    
    Args:
        host: Host address (default: '0.0.0.0' for server, 'localhost' for client)
        port: Port number (default: 5000)
        config: Stream configuration
        mode: 'server' or 'client' (default: 'server')
        packet_parser: Custom function to parse TCP messages (optional)
        buffer_size: Socket buffer size in bytes (default: 65536)
        timeout: Socket timeout in seconds (default: 1.0)
        delimiter: Message delimiter for text-based protocols (optional)
        message_size: Fixed message size for binary protocols (optional)
        
    Example:
        >>> # TCP server
        >>> source = TCPStreamDataSource(host='0.0.0.0', port=5000, mode='server')
        >>> for batch in source:
        ...     print(f"Received {len(batch.data)} samples")
        
        >>> # TCP client
        >>> source = TCPStreamDataSource(host='server_ip', port=5000, mode='client')
        >>> for batch in source:
        ...     print(f"Received {len(batch.data)} samples")
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 5000,
        config: Optional[StreamConfig] = None,
        mode: str = 'server',
        packet_parser: Optional[Callable[[bytes, float], tuple[Dict[str, Any], np.ndarray]]] = None,
        buffer_size: int = 65536,
        timeout: float = 1.0,
        delimiter: Optional[bytes] = None,
        message_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(config=config, **kwargs)
        
        self.host = host
        self.port = port
        self.mode = mode.lower()
        self.packet_parser = packet_parser or self._default_parser
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.delimiter = delimiter
        self.message_size = message_size
        
        # Socket and state
        self._socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._running = False
        self._receive_thread: Optional[threading.Thread] = None
        self._message_buffer: bytes = b''
        self._packet_queue: list[tuple[bytes, float]] = []
        self._queue_lock = threading.Lock()
        
        # Statistics
        self._messages_received = 0
        self._bytes_received = 0
        self._start_time: Optional[datetime] = None
        self._connected = False
        
        if self.mode not in ['server', 'client']:
            raise ValueError("mode must be 'server' or 'client'")
            
        logger.info(f"TCPStreamDataSource initialized as {mode} on {host}:{port}")
    
    def _default_parser(self, message_data: bytes, timestamp: float) -> tuple[Dict[str, Any], np.ndarray]:
        """Default message parser for TCP data.
        
        This parser handles:
        - Text data (split by whitespace)
        - Binary float64 arrays
        - Raw bytes as hex strings
        
        Args:
            message_data: Raw message bytes
            timestamp: Reception timestamp
            
        Returns:
            Tuple of (data_dict, timestamps_array)
        """
        try:
            # Try to decode as text
            text = message_data.decode('utf-8').strip()
            
            # Try to parse as space-separated values
            values = text.split()
            if len(values) > 1:
                try:
                    numeric_values = [float(v) for v in values]
                    data_dict = {f'value_{i}': [v] for i, v in enumerate(numeric_values)}
                    timestamps = np.array([timestamp] * len(numeric_values))
                    return data_dict, timestamps
                except ValueError:
                    pass
            
            # Try to parse as comma-separated values
            if ',' in text:
                values = text.split(',')
                try:
                    numeric_values = [float(v) for v in values]
                    data_dict = {f'value_{i}': [v] for i, v in enumerate(numeric_values)}
                    timestamps = np.array([timestamp] * len(numeric_values))
                    return data_dict, timestamps
                except ValueError:
                    pass
            
            # Fallback: single value
            try:
                value = float(text)
                data_dict = {'value': [value]}
                timestamps = np.array([timestamp])
                return data_dict, timestamps
            except ValueError:
                # Return as string
                data_dict = {'message': [text]}
                timestamps = np.array([timestamp])
                return data_dict, timestamps
                
        except UnicodeDecodeError:
            # Binary data - try as float64 array
            try:
                n_floats = len(message_data) // 8
                if n_floats > 0:
                    values = struct.unpack(f'{n_floats}d', message_data[:n_floats * 8])
                    data_dict = {f'value_{i}': [v] for i, v in enumerate(values)}
                    timestamps = np.array([timestamp] * n_floats)
                    return data_dict, timestamps
            except struct.error:
                pass
            
            # Return raw bytes as hex
            data_dict = {'raw_data': [message_data.hex()]}
            timestamps = np.array([timestamp])
            return data_dict, timestamps
    
    def _create_server_socket(self) -> socket.socket:
        """Create and configure the TCP server socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
        sock.settimeout(self.timeout)
        return sock
    
    def _create_client_socket(self) -> socket.socket:
        """Create and configure the TCP client socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
        sock.settimeout(self.timeout)
        return sock
    
    def _receive_messages(self) -> None:
        """Thread function to receive TCP messages."""
        if self.mode == 'server' and self._socket is None:
            return
        if self.mode == 'client' and self._client_socket is None:
            return
        
        sock = self._client_socket if self.mode == 'client' else self._socket
        if sock is None:
            return
        
        logger.debug(f"Starting TCP receiver thread (mode={self.mode})")
        
        while self._running:
            try:
                if self.mode == 'server':
                    # Accept new connection
                    if self._client_socket is None:
                        try:
                            self._client_socket, addr = self._socket.accept()
                            self._client_socket.settimeout(self.timeout)
                            self._connected = True
                            logger.info(f"TCP server: accepted connection from {addr}")
                        except socket.timeout:
                            continue
                        except Exception as e:
                            logger.error(f"Error accepting TCP connection: {e}")
                            continue
                    
                    sock = self._client_socket
                
                # Receive data
                if self.message_size:
                    # Fixed size messages
                    data = sock.recv(self.message_size)
                    if not data:
                        # Connection closed
                        self._connected = False
                        if self.mode == 'server':
                            self._client_socket = None
                        continue
                    
                    timestamp = time.time()
                    with self._queue_lock:
                        self._packet_queue.append((data, timestamp))
                        self._messages_received += 1
                        self._bytes_received += len(data)
                else:
                    # Variable size messages with delimiter or until timeout
                    chunk = sock.recv(self.buffer_size)
                    if not chunk:
                        # Connection closed
                        self._connected = False
                        if self.mode == 'server':
                            self._client_socket = None
                        continue
                    
                    self._bytes_received += len(chunk)
                    
                    if self.delimiter:
                        # Accumulate until delimiter is found
                        self._message_buffer += chunk
                        
                        while self.delimiter in self._message_buffer:
                            # Split at first delimiter
                            parts = self._message_buffer.split(self.delimiter, 1)
                            message = parts[0]
                            self._message_buffer = parts[1] if len(parts) > 1 else b''
                            
                            timestamp = time.time()
                            with self._queue_lock:
                                self._packet_queue.append((message, timestamp))
                                self._messages_received += 1
                    else:
                        # No delimiter, treat each chunk as a message
                        timestamp = time.time()
                        with self._queue_lock:
                            self._packet_queue.append((chunk, timestamp))
                            self._messages_received += 1
                
            except socket.timeout:
                # Timeout is normal, just continue
                continue
            except ConnectionResetError:
                # Connection was reset
                self._connected = False
                if self.mode == 'server':
                    self._client_socket = None
                logger.warning("TCP connection reset by peer")
            except Exception as e:
                if self._running:
                    logger.error(f"Error receiving TCP message: {e}")
        
        logger.debug("TCP receiver thread stopped")
    
    def open(self) -> 'TCPStreamDataSource':
        """Open the TCP connection and start receiving."""
        if self._is_open:
            return self
        
        if self.mode == 'server':
            # Create server socket
            self._socket = self._create_server_socket()
            
            try:
                self._socket.bind((self.host, self.port))
                self._socket.listen(5)
                logger.info(f"TCP server listening on {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Failed to bind TCP server socket: {e}")
                raise
        else:
            # Client mode - connect to server
            self._client_socket = self._create_client_socket()
            
            try:
                self._client_socket.connect((self.host, self.port))
                self._connected = True
                logger.info(f"TCP client connected to {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Failed to connect TCP client: {e}")
                raise
        
        # Start receive thread
        self._running = True
        self._start_time = datetime.now()
        self._receive_thread = threading.Thread(target=self._receive_messages, daemon=True)
        self._receive_thread.start()
        
        self._is_open = True
        self._batch_index = 0
        
        return self
    
    def close(self) -> None:
        """Close the TCP connection and stop receiving."""
        self._running = False
        
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
            self._receive_thread = None
        
        if self._client_socket:
            try:
                self._client_socket.close()
            except Exception as e:
                logger.error(f"Error closing TCP client socket: {e}")
            self._client_socket = None
        
        if self._socket:
            try:
                self._socket.close()
            except Exception as e:
                logger.error(f"Error closing TCP server socket: {e}")
            self._socket = None
        
        self._is_open = False
        self._connected = False
        logger.info("TCP data source closed")
    
    def reset(self) -> None:
        """Reset the data source."""
        with self._queue_lock:
            self._packet_queue.clear()
        self._message_buffer = b''
        self._batch_index = 0
        self._messages_received = 0
        self._bytes_received = 0
        self._start_time = datetime.now()
        logger.debug("TCP data source reset")
    
    def __next__(self) -> DataBatch:
        """Get the next batch of data from received messages."""
        if not self._is_open:
            raise RuntimeError("Data source is not open. Call open() first.")
        
        # Collect messages for this batch
        batch_data = []
        batch_timestamps = []
        
        target_messages = self.config.batch_size
        start_time = time.time()
        
        while len(batch_data) < target_messages:
            # Check for timeout
            if time.time() - start_time > self.config.timeout:
                if not batch_data:
                    raise StopIteration("Timeout waiting for data")
                break
            
            # Get available messages
            with self._queue_lock:
                available_messages = len(self._packet_queue)
                if available_messages > 0:
                    # Take up to remaining needed messages
                    take_count = min(available_messages, target_messages - len(batch_data))
                    messages = self._packet_queue[:take_count]
                    self._packet_queue = self._packet_queue[take_count:]
                else:
                    messages = []
            
            if not messages:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue
            
            # Parse messages
            for message_data, message_timestamp in messages:
                try:
                    data_dict, timestamps = self.packet_parser(message_data, message_timestamp)
                    batch_data.append(data_dict)
                    batch_timestamps.extend(timestamps)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    continue
        
        if not batch_data:
            raise StopIteration("No data available")
        
        # Convert to DataFrame
        df = pd.DataFrame(batch_data)
        
        # If all messages had the same timestamp, use a range
        if len(set(batch_timestamps)) == 1 and len(batch_timestamps) > 1:
            timestamps = np.linspace(batch_timestamps[0], batch_timestamps[0] + len(batch_timestamps) * 0.001, len(batch_timestamps))
        else:
            timestamps = np.array(batch_timestamps)
        
        # Create metadata
        metadata = {
            'source_type': 'tcp',
            'host': self.host,
            'port': self.port,
            'mode': self.mode,
            'messages_in_batch': len(batch_data),
            'total_messages': self._messages_received,
            'total_bytes': self._bytes_received,
            'connected': self._connected,
        }
        
        batch = DataBatch(
            data=df,
            timestamps=timestamps,
            batch_index=self._batch_index,
            is_complete=False,  # TCP streams are continuous
            metadata=metadata
        )
        
        self._batch_index += 1
        
        return batch
    
    @property
    def is_connected(self) -> bool:
        """Whether the TCP connection is currently active."""
        return self._connected
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Get statistics about the TCP stream."""
        uptime = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        
        return {
            'messages_received': self._messages_received,
            'bytes_received': self._bytes_received,
            'uptime_seconds': uptime,
            'messages_per_second': self._messages_received / uptime if uptime > 0 else 0,
            'bytes_per_second': self._bytes_received / uptime if uptime > 0 else 0,
            'connected': self._connected,
        }


class LineDelimitedTCPDataSource(TCPStreamDataSource):
    """TCP data source for line-delimited text protocols.
    
    This is a specialized version for text-based protocols where each line
    is a separate message (e.g., many SCADA systems, custom text protocols).
    
    Args:
        host: Host address
        port: Port number
        config: Stream configuration
        mode: 'server' or 'client'
        delimiter: Line delimiter (default: b'\\n')
        
    Example:
        >>> # For a server sending line-delimited data
        >>> source = LineDelimitedTCPDataSource(port=5000, mode='server')
        >>> for batch in source:
        ...     print(f"Received {len(batch.data)} lines")
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 5000,
        config: Optional[StreamConfig] = None,
        mode: str = 'server',
        delimiter: bytes = b'\n',
        **kwargs
    ):
        super().__init__(
            host=host,
            port=port,
            config=config,
            mode=mode,
            delimiter=delimiter,
            **kwargs
        )
        
        logger.info(f"LineDelimitedTCPDataSource initialized with delimiter={delimiter!r}")
