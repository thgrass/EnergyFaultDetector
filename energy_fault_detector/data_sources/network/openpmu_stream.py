"""OpenPMU XML datagram stream data source.

This module provides data sources for OpenPMU V2 XML datagrams as described in:
https://github.com/OpenPMU/OpenPMUdocs/tree/master/XML_Datagrams

Supports both:
- XML Sampled Values (SV) datagrams
- XML Phasor Values (PV) datagrams
"""

import base64
import struct
import time
import logging
import threading
import socket
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from energy_fault_detector.data_sources.base import DataSource, DataBatch, StreamConfig
from energy_fault_detector.data_sources.network.udp_stream import UDPStreamDataSource

logger = logging.getLogger('energy_fault_detector.data_sources.openpmu_stream')


@dataclass
class OpenPMUChannel:
    """Represents a single channel in an OpenPMU datagram."""
    name: str
    channel_type: str  # 'V' for voltage, 'I' for current
    phase: str  # 'a', 'b', 'c', 'n', etc.
    range: float  # Full scale deflection
    
    # For Sampled Values
    payload: Optional[str] = None  # Base64 encoded payload
    
    # For Phasor Values
    mag: Optional[float] = None
    angle: Optional[float] = None  # in degrees
    freq: Optional[float] = None  # in Hz
    rocof: Optional[float] = None  # rate of change of frequency


@dataclass
class OpenPMUDatagram:
    """Represents a parsed OpenPMU XML datagram."""
    format: str  # 'Samples' or 'Phasors'
    date: str
    time: str
    frame: int
    channels: List[OpenPMUChannel]
    
    # Additional metadata for Sampled Values
    fs: Optional[float] = None  # Sampling rate in Hz
    n: Optional[int] = None  # Number of samples per payload
    bits: Optional[int] = None  # Bits per sample
    
    # Additional metadata for Phasor Values
    algorithm: Optional[str] = None
    
    @property
    def timestamp(self) -> datetime:
        """Get the timestamp as a datetime object."""
        return datetime.strptime(f"{self.date} {self.time}", "%Y-%m-%d %H:%M:%S.%f")
    
    @property
    def is_sampled_values(self) -> bool:
        """Whether this is a Sampled Values datagram."""
        return self.format.lower() == 'samples'
    
    @property
    def is_phasor_values(self) -> bool:
        """Whether this is a Phasor Values datagram."""
        return self.format.lower() == 'phasors'


class OpenPMUParser:
    """Parser for OpenPMU XML datagrams.
    
    This class parses XML datagrams in the OpenPMU V2 format and extracts
    the data into structured Python objects.
    """
    
    @staticmethod
    def parse_xml(xml_string: str) -> OpenPMUDatagram:
        """Parse an OpenPMU XML datagram string.
        
        Args:
            xml_string: XML string to parse
            
        Returns:
            OpenPMUDatagram object
            
        Raises:
            ValueError: If the XML is malformed or not a valid OpenPMU datagram
        """
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML: {e}")
        
        # Extract common metadata
        format_tag = root.find('Format')
        format_type = format_tag.text if format_tag is not None else 'Unknown'
        
        date_tag = root.find('Date')
        date = date_tag.text if date_tag is not None else '1970-01-01'
        
        time_tag = root.find('Time')
        time_str = time_tag.text if time_tag is not None else '00:00:00.000'
        
        frame_tag = root.find('Frame')
        frame = int(frame_tag.text) if frame_tag is not None else 0
        
        channels_tag = root.find('Channels')
        n_channels = int(channels_tag.text) if channels_tag is not None else 0
        
        # Parse channels
        channels = []
        for i in range(n_channels):
            channel_tag = root.find(f'Channel_{i}')
            if channel_tag is None:
                continue
            
            channel = OpenPMUParser._parse_channel(channel_tag, format_type)
            channels.append(channel)
        
        # Create datagram
        datagram = OpenPMUDatagram(
            format=format_type,
            date=date,
            time=time_str,
            frame=frame,
            channels=channels
        )
        
        # Parse format-specific metadata
        if format_type.lower() == 'samples':
            datagram = OpenPMUParser._parse_sampled_values_metadata(root, datagram)
        elif format_type.lower() == 'phasors':
            datagram = OpenPMUParser._parse_phasor_values_metadata(root, datagram)
        
        return datagram
    
    @staticmethod
    def _parse_channel(channel_tag: ET.Element, format_type: str) -> OpenPMUChannel:
        """Parse a single channel element."""
        name_tag = channel_tag.find('Name')
        name = name_tag.text if name_tag is not None else f"Channel_{len(channel_tag.getparent())}"
        
        type_tag = channel_tag.find('Type')
        channel_type = type_tag.text if type_tag is not None else 'V'
        
        phase_tag = channel_tag.find('Phase')
        phase = phase_tag.text if phase_tag is not None else 'a'
        
        range_tag = channel_tag.find('Range')
        range_val = float(range_tag.text) if range_tag is not None else 1.0
        
        channel = OpenPMUChannel(
            name=name,
            channel_type=channel_type,
            phase=phase,
            range=range_val
        )
        
        # Parse format-specific data
        if format_type.lower() == 'samples':
            payload_tag = channel_tag.find('Payload')
            if payload_tag is not None:
                channel.payload = payload_tag.text
        elif format_type.lower() == 'phasors':
            mag_tag = channel_tag.find('Mag')
            if mag_tag is not None:
                channel.mag = float(mag_tag.text)
            
            angle_tag = channel_tag.find('Angle')
            if angle_tag is not None:
                channel.angle = float(angle_tag.text)
            
            freq_tag = channel_tag.find('Freq')
            if freq_tag is not None:
                channel.freq = float(freq_tag.text)
            
            rocof_tag = channel_tag.find('ROCOF')
            if rocof_tag is not None:
                channel.rocof = float(rocof_tag.text)
        
        return channel
    
    @staticmethod
    def _parse_sampled_values_metadata(root: ET.Element, datagram: OpenPMUDatagram) -> OpenPMUDatagram:
        """Parse Sampled Values specific metadata."""
        fs_tag = root.find('Fs')
        if fs_tag is not None:
            datagram.fs = float(fs_tag.text)
        
        n_tag = root.find('n')
        if n_tag is not None:
            datagram.n = int(n_tag.text)
        
        bits_tag = root.find('bits')
        if bits_tag is not None:
            datagram.bits = int(bits_tag.text)
        
        return datagram
    
    @staticmethod
    def _parse_phasor_values_metadata(root: ET.Element, datagram: OpenPMUDatagram) -> OpenPMUDatagram:
        """Parse Phasor Values specific metadata."""
        algorithm_tag = root.find('Algorithm')
        if algorithm_tag is not None:
            datagram.algorithm = algorithm_tag.text
        
        return datagram
    
    @staticmethod
    def decode_payload(payload: str, n_samples: int, bits: int, is_signed: bool = True) -> np.ndarray:
        """Decode a Base64-encoded payload into a numpy array.
        
        Args:
            payload: Base64-encoded payload string
            n_samples: Number of samples in the payload
            bits: Number of bits per sample
            is_signed: Whether the values are signed
            
        Returns:
            Numpy array of decoded values
        """
        if not payload:
            return np.array([])
        
        try:
            # Decode Base64
            binary_data = base64.b64decode(payload)
            
            # Determine format string based on bits
            if bits == 16:
                if is_signed:
                    fmt = f'{n_samples}h'  # signed short
                else:
                    fmt = f'{n_samples}H'  # unsigned short
            elif bits == 32:
                if is_signed:
                    fmt = f'{n_samples}i'  # signed int
                else:
                    fmt = f'{n_samples}I'  # unsigned int
            elif bits == 64:
                if is_signed:
                    fmt = f'{n_samples}q'  # signed long long
                else:
                    fmt = f'{n_samples}Q'  # unsigned long long
            elif bits == 32:
                fmt = f'{n_samples}f'  # float
            elif bits == 64:
                fmt = f'{n_samples}d'  # double
            else:
                # Try to determine from data length
                bytes_per_sample = bits // 8
                n_samples_actual = len(binary_data) // bytes_per_sample
                if bytes_per_sample == 2:
                    fmt = f'{n_samples_actual}h'
                elif bytes_per_sample == 4:
                    fmt = f'{n_samples_actual}f'
                elif bytes_per_sample == 8:
                    fmt = f'{n_samples_actual}d'
                else:
                    raise ValueError(f"Unsupported bit depth: {bits}")
            
            # Unpack binary data
            values = np.array(struct.unpack(fmt, binary_data))
            
            return values
            
        except Exception as e:
            logger.warning(f"Error decoding payload: {e}")
            return np.array([])


class OpenPMUUDPDataSource(UDPStreamDataSource):
    """UDP data source for OpenPMU XML datagrams.
    
    This class receives OpenPMU V2 XML datagrams over UDP and parses them
    into structured data for fault detection.
    
    Supports both:
    - XML Sampled Values (SV) datagrams
    - XML Phasor Values (PV) datagrams
    
    Args:
        host: Host address to bind to (default: '0.0.0.0')
        port: Port to listen on (default: 4713, common for OpenPMU)
        config: Stream configuration
        datagram_type: Expected datagram type ('auto', 'samples', or 'phasors')
        
    Example:
        >>> # Create OpenPMU UDP source
        >>> source = OpenPMUUDPDataSource(port=4713)
        >>> for batch in source:
        ...     print(f"Received {len(batch.data)} samples")
        ...     print(f"Channels: {batch.data.columns}")
    """
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 4713,
        config: Optional[StreamConfig] = None,
        datagram_type: str = 'auto',
        **kwargs
    ):
        # Create custom parser for OpenPMU XML
        def openpmu_parser(packet_data: bytes, timestamp: float) -> tuple[Dict[str, Any], np.ndarray]:
            return self._parse_openpmu_packet(packet_data, timestamp, datagram_type)
        
        super().__init__(
            host=host,
            port=port,
            config=config,
            packet_parser=openpmu_parser,
            **kwargs
        )
        
        self.datagram_type = datagram_type.lower()
        self._parser = OpenPMUParser()
        
        logger.info(f"OpenPMUUDPDataSource initialized on {host}:{port}, type={datagram_type}")
    
    def _parse_openpmu_packet(
        self,
        packet_data: bytes,
        timestamp: float,
        expected_type: str
    ) -> tuple[Dict[str, Any], np.ndarray]:
        """Parse an OpenPMU XML packet.
        
        Args:
            packet_data: Raw packet bytes
            timestamp: Reception timestamp
            expected_type: Expected datagram type
            
        Returns:
            Tuple of (data_dict, timestamps_array)
        """
        try:
            # Decode packet as UTF-8 XML
            xml_string = packet_data.decode('utf-8')
            
            # Parse XML
            datagram = self._parser.parse_xml(xml_string)
            
            # Check type if specified
            if expected_type != 'auto':
                if expected_type == 'samples' and not datagram.is_sampled_values:
                    logger.warning(f"Expected Samples datagram but got {datagram.format}")
                    return {}, np.array([timestamp])
                elif expected_type == 'phasors' and not datagram.is_phasor_values:
                    logger.warning(f"Expected Phasors datagram but got {datagram.format}")
                    return {}, np.array([timestamp])
            
            # Convert to data dictionary
            if datagram.is_sampled_values:
                return self._datagram_to_sampled_values_dict(datagram, timestamp)
            elif datagram.is_phasor_values:
                return self._datagram_to_phasor_values_dict(datagram, timestamp)
            else:
                logger.warning(f"Unknown datagram format: {datagram.format}")
                return {}, np.array([timestamp])
                
        except Exception as e:
            logger.error(f"Error parsing OpenPMU packet: {e}")
            return {'parse_error': [str(e)]}, np.array([timestamp])
    
    def _datagram_to_sampled_values_dict(
        self,
        datagram: OpenPMUDatagram,
        timestamp: float
    ) -> tuple[Dict[str, Any], np.ndarray]:
        """Convert a Sampled Values datagram to a data dictionary.
        
        Args:
            datagram: Parsed OpenPMU datagram
            timestamp: Reception timestamp
            
        Returns:
            Tuple of (data_dict, timestamps_array)
        """
        data_dict = {}
        timestamps = []
        
        # Add metadata
        data_dict['format'] = [datagram.format]
        data_dict['date'] = [datagram.date]
        data_dict['time'] = [datagram.time]
        data_dict['frame'] = [datagram.frame]
        data_dict['fs'] = [datagram.fs or 0]
        data_dict['n'] = [datagram.n or 0]
        data_dict['bits'] = [datagram.bits or 0]
        data_dict['n_channels'] = [len(datagram.channels)]
        
        # Process each channel
        for i, channel in enumerate(datagram.channels):
            if channel.payload:
                # Decode payload
                try:
                    samples = self._parser.decode_payload(
                        channel.payload,
                        datagram.n or 0,
                        datagram.bits or 16,
                        is_signed=True
                    )
                    
                    # Scale samples by range
                    if channel.range > 0:
                        samples = samples / (2 ** (datagram.bits or 16) - 1) * channel.range
                    
                    # Add to data dictionary
                    for j, sample in enumerate(samples):
                        col_name = f"{channel.name}_sample_{j}" if len(datagram.channels) > 1 else f"sample_{j}"
                        if col_name not in data_dict:
                            data_dict[col_name] = []
                        data_dict[col_name].append(sample)
                        timestamps.append(timestamp + j * (1.0 / (datagram.fs or 1.0)))
                
                except Exception as e:
                    logger.warning(f"Error decoding channel {i} payload: {e}")
            
            # Add channel metadata
            data_dict[f"{channel.name}_type"] = [channel.channel_type]
            data_dict[f"{channel.name}_phase"] = [channel.phase]
            data_dict[f"{channel.name}_range"] = [channel.range]
        
        # Convert lists to arrays
        for key in data_dict:
            if isinstance(data_dict[key], list) and len(data_dict[key]) == 1:
                data_dict[key] = data_dict[key][0]
        
        return data_dict, np.array(timestamps)
    
    def _datagram_to_phasor_values_dict(
        self,
        datagram: OpenPMUDatagram,
        timestamp: float
    ) -> tuple[Dict[str, Any], np.ndarray]:
        """Convert a Phasor Values datagram to a data dictionary.
        
        Args:
            datagram: Parsed OpenPMU datagram
            timestamp: Reception timestamp
            
        Returns:
            Tuple of (data_dict, timestamps_array)
        """
        data_dict = {}
        timestamps = [timestamp] * len(datagram.channels)
        
        # Add metadata
        data_dict['format'] = [datagram.format]
        data_dict['date'] = [datagram.date]
        data_dict['time'] = [datagram.time]
        data_dict['frame'] = [datagram.frame]
        data_dict['algorithm'] = [datagram.algorithm or '']
        data_dict['n_channels'] = [len(datagram.channels)]
        
        # Process each channel
        for channel in datagram.channels:
            # Add phasor values
            if channel.mag is not None:
                data_dict[f"{channel.name}_mag"] = [channel.mag]
            if channel.angle is not None:
                data_dict[f"{channel.name}_angle"] = [channel.angle]
            if channel.freq is not None:
                data_dict[f"{channel.name}_freq"] = [channel.freq]
            if channel.rocof is not None:
                data_dict[f"{channel.name}_rocof"] = [channel.rocof]
            
            # Add channel metadata
            data_dict[f"{channel.name}_type"] = [channel.channel_type]
            data_dict[f"{channel.name}_phase"] = [channel.phase]
            data_dict[f"{channel.name}_range"] = [channel.range]
        
        # Convert single-element lists to scalars
        for key in data_dict:
            if isinstance(data_dict[key], list) and len(data_dict[key]) == 1:
                data_dict[key] = data_dict[key][0]
        
        return data_dict, np.array(timestamps)


class OpenPMUGenerator:
    """Generator for creating OpenPMU XML datagrams for testing.
    
    This class can generate both Sampled Values and Phasor Values datagrams
    in the OpenPMU V2 XML format.
    """
    
    def __init__(self, n_channels: int = 3, fs: float = 12800.0, n_samples: int = 128):
        """Initialize the generator.
        
        Args:
            n_channels: Number of channels
            fs: Sampling rate in Hz
            n_samples: Number of samples per payload
        """
        self.n_channels = n_channels
        self.fs = fs
        self.n_samples = n_samples
        self.frame_counter = 0
        self.time_counter = 0.0
        
        # Default channel names
        self.channel_names = [f"Channel_{i}" for i in range(n_channels)]
        self.channel_types = ['V'] * n_channels
        self.channel_phases = ['a', 'b', 'c'][:n_channels]
        self.channel_ranges = [275.0] * n_channels
    
    def set_channel_info(
        self,
        names: List[str],
        types: List[str],
        phases: List[str],
        ranges: List[float]
    ) -> None:
        """Set custom channel information.
        
        Args:
            names: Channel names
            types: Channel types ('V' or 'I')
            phases: Channel phases ('a', 'b', 'c', 'n')
            ranges: Channel ranges (full scale deflection)
        """
        self.channel_names = names
        self.channel_types = types
        self.channel_phases = phases
        self.channel_ranges = ranges
    
    def generate_sampled_values_datagram(self, timestamp: Optional[datetime] = None) -> str:
        """Generate a Sampled Values XML datagram.
        
        Args:
            timestamp: Optional timestamp for the datagram
            
        Returns:
            XML string
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Generate random sampled values
        import random
        random.seed(int(time.time() * 1000) % (2**32))
        
        # Build XML
        xml_parts = ['<OpenPMU>']
        xml_parts.append(f'<Format>Samples</Format>')
        xml_parts.append(f'<Date>{timestamp.strftime("%Y-%m-%d")}</Date>')
        xml_parts.append(f'<Time>{timestamp.strftime("%H:%M:%S.%f")}</Time>')
        xml_parts.append(f'<Frame>{self.frame_counter}</Frame>')
        xml_parts.append(f'<Fs>{self.fs}</Fs>')
        xml_parts.append(f'<n>{self.n_samples}</n>')
        xml_parts.append(f'<bits>16</bits>')
        xml_parts.append(f'<Channels>{self.n_channels}</Channels>')
        
        # Generate payloads for each channel
        for i in range(self.n_channels):
            xml_parts.append(f'<Channel_{i}>')
            xml_parts.append(f'<Name>{self.channel_names[i]}</Name>')
            xml_parts.append(f'<Type>{self.channel_types[i]}</Type>')
            xml_parts.append(f'<Phase>{self.channel_phases[i]}</Phase>')
            xml_parts.append(f'<Range>{self.channel_ranges[i]}</Range>')
            
            # Generate random samples
            samples = []
            for j in range(self.n_samples):
                # Generate sine wave with some noise
                angle = 2 * np.pi * j / self.n_samples * (i + 1)
                sample = np.sin(angle) * self.channel_ranges[i] * 0.8
                sample += random.gauss(0, self.channel_ranges[i] * 0.05)
                samples.append(int(sample))
            
            # Convert to bytes and Base64
            payload_bytes = struct.pack(f'{self.n_samples}h', *samples)
            payload_b64 = base64.b64encode(payload_bytes).decode('utf-8')
            xml_parts.append(f'<Payload>{payload_b64}</Payload>')
            
            xml_parts.append(f'</Channel_{i}>')
        
        xml_parts.append('</OpenPMU>')
        
        # Increment counters
        self.frame_counter = (self.frame_counter + 1) % 100
        self.time_counter += self.n_samples / self.fs
        
        return '\n'.join(xml_parts)
    
    def generate_phasor_values_datagram(self, timestamp: Optional[datetime] = None) -> str:
        """Generate a Phasor Values XML datagram.
        
        Args:
            timestamp: Optional timestamp for the datagram
            
        Returns:
            XML string
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Generate random phasor values
        import random
        random.seed(int(time.time() * 1000) % (2**32))
        
        # Build XML
        xml_parts = ['<OpenPMU>']
        xml_parts.append(f'<Format>Phasors</Format>')
        xml_parts.append(f'<Date>{timestamp.strftime("%Y-%m-%d")}</Date>')
        xml_parts.append(f'<Time>{timestamp.strftime("%H:%M:%S.%f")}</Time>')
        xml_parts.append(f'<Frame>{self.frame_counter}</Frame>')
        xml_parts.append(f'<Algorithm>Test Generator V1.0</Algorithm>')
        xml_parts.append(f'<Channels>{self.n_channels}</Channels>')
        
        # Generate phasor values for each channel
        for i in range(self.n_channels):
            xml_parts.append(f'<Channel_{i}>')
            xml_parts.append(f'<Name>{self.channel_names[i]}</Name>')
            xml_parts.append(f'<Type>{self.channel_types[i]}</Type>')
            xml_parts.append(f'<Phase>{self.channel_phases[i]}</Phase>')
            xml_parts.append(f'<Range>{self.channel_ranges[i]}</Range>')
            
            # Generate random phasor values
            mag = self.channel_ranges[i] * (0.8 + random.gauss(0, 0.1))
            angle = 2 * np.pi * i / self.n_channels * 120 + random.gauss(0, 0.1)
            freq = 50.0 + random.gauss(0, 0.1)
            rocof = random.gauss(0, 0.01)
            
            xml_parts.append(f'<Mag>{mag:.6f}</Mag>')
            xml_parts.append(f'<Angle>{np.degrees(angle):.6f}</Angle>')
            xml_parts.append(f'<Freq>{freq:.6f}</Freq>')
            xml_parts.append(f'<ROCOF>{rocof:.6f}</ROCOF>')
            
            xml_parts.append(f'</Channel_{i}>')
        
        xml_parts.append('</OpenPMU>')
        
        # Increment counters
        self.frame_counter = (self.frame_counter + 1) % 50
        self.time_counter += 1.0 / 50.0  # Assuming 50 Hz system
        
        return '\n'.join(xml_parts)
    
    def generate_datagram(self, datagram_type: str = 'phasors', timestamp: Optional[datetime] = None) -> str:
        """Generate a datagram of the specified type.
        
        Args:
            datagram_type: 'samples' or 'phasors'
            timestamp: Optional timestamp
            
        Returns:
            XML string
        """
        if datagram_type.lower() == 'samples':
            return self.generate_sampled_values_datagram(timestamp)
        else:
            return self.generate_phasor_values_datagram(timestamp)
