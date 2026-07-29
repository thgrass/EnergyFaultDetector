"""Tests for OpenPMU stream data sources."""

import struct
import time
import threading
import socket
import pytest
from datetime import datetime

import numpy as np
import pandas as pd

from energy_fault_detector.data_sources import (
    OpenPMUUDPDataSource,
    OpenPMUParser,
    OpenPMUDatagram,
    OpenPMUChannel,
    OpenPMUGenerator,
    StreamConfig
)


class TestOpenPMUParser:
    """Tests for OpenPMUParser."""
    
    def test_parse_sampled_values_xml(self):
        """Test parsing Sampled Values XML."""
        xml = """<OpenPMU>
    <Format>Samples</Format>
    <Date>2021-02-28</Date>
    <Time>22:04:00.000</Time>
    <Frame>0</Frame>
    <Fs>12800</Fs>
    <n>128</n>
    <bits>16</bits>
    <Channels>3</Channels>
    <Channel_0>
        <Name>Belfast_Va</Name>
        <Type>V</Type>
        <Phase>a</Phase>
        <Range>275</Range>
        <Payload>JWMmESa8J2QoBCieKTEpuypAKrwrMSufLAYsZSy+LQ8tWS2bLdcuCS40LlUucC6CLn4uai5PLiot/y3KLY4tSSz+LKssUivyK4srGyqoKispqSkfKI8n9ydZJrUmCiVXJJ8j4CMaIk8heyClH8ce4x36HQwcGRsgGiIZIRgbFxEWBBTyE90SxhGqEIwPaw5IDSQL/ArUCaoIfQdRBiME9QPGApYBaAA5/wr92/yt+4H6Vfks+AL23PW49JXzdfJX8T7wJ+8S7gPs9+vt6ujp6Ojs5/XnA+YX5S/kTeNv4pnhyOD94Djfet7D3hLdZ9zC3CXbjtr+2nXZ89l42QTYlg==</Payload>
    </Channel_0>
    <Channel_1>
        <Name>Belfast_Vb</Name>
        <Type>V</Type>
        <Phase>b</Phase>
        <Range>275</Range>
        <Payload>AA==</Payload>
    </Channel_1>
    <Channel_2>
        <Name>Belfast_Vc</Name>
        <Type>V</Type>
        <Phase>c</Phase>
        <Range>275</Range>
        <Payload>AA==</Payload>
    </Channel_2>
</OpenPMU>"""
        
        datagram = OpenPMUParser.parse_xml(xml)
        
        assert datagram.format == 'Samples'
        assert datagram.date == '2021-02-28'
        assert datagram.time == '22:04:00.000'
        assert datagram.frame == 0
        assert datagram.fs == 12800.0
        assert datagram.n == 128
        assert datagram.bits == 16
        assert len(datagram.channels) == 3
        
        # Check first channel
        assert datagram.channels[0].name == 'Belfast_Va'
        assert datagram.channels[0].channel_type == 'V'
        assert datagram.channels[0].phase == 'a'
        assert datagram.channels[0].range == 275.0
        assert datagram.channels[0].payload is not None
    
    def test_parse_phasor_values_xml(self):
        """Test parsing Phasor Values XML."""
        xml = """<OpenPMU>
    <Format>Phasors</Format>
    <Date>2021-02-28</Date>
    <Time>22:04:00.460</Time>
    <Frame>23</Frame>
    <Algorithm>LSE V1.0 by Xiaodong Zhao</Algorithm>
    <Channels>3</Channels>
    <Channel_0>
        <Name>Belfast_Va</Name>
        <Type>V</Type>
        <Phase>a</Phase>
        <Range>275</Range>
        <Mag>240.00001</Mag>
        <Angle>0.403</Angle>
        <Freq>50.690</Freq>
        <ROCOF>0.001</ROCOF>
    </Channel_0>
    <Channel_1>
        <Name>Belfast_Vb</Name>
        <Type>V</Type>
        <Phase>b</Phase>
        <Range>275</Range>
        <Mag>239.99988</Mag>
        <Angle>120.904</Angle>
        <Freq>50.692</Freq>
        <ROCOF>0.000</ROCOF>
    </Channel_1>
    <Channel_2>
        <Name>Belfast_Vc</Name>
        <Type>V</Type>
        <Phase>c</Phase>
        <Range>275</Range>
        <Mag>239.99532</Mag>
        <Angle>-120.749</Angle>
        <Freq>50.689</Freq>
        <ROCOF>-0.002</ROCOF>
    </Channel_2>
</OpenPMU>"""
        
        datagram = OpenPMUParser.parse_xml(xml)
        
        assert datagram.format == 'Phasors'
        assert datagram.date == '2021-02-28'
        assert datagram.time == '22:04:00.460'
        assert datagram.frame == 23
        assert datagram.algorithm == 'LSE V1.0 by Xiaodong Zhao'
        assert len(datagram.channels) == 3
        
        # Check phasor values
        assert datagram.channels[0].mag == pytest.approx(240.00001)
        assert datagram.channels[0].angle == pytest.approx(0.403)
        assert datagram.channels[0].freq == pytest.approx(50.690)
        assert datagram.channels[0].rocof == pytest.approx(0.001)
    
    def test_decode_payload(self):
        """Test payload decoding."""
        # Create test payload: 4 samples of 16-bit signed integers
        samples = [100, -200, 300, -400]
        payload_bytes = struct.pack('4h', *samples)
        import base64
        payload_b64 = base64.b64encode(payload_bytes).decode('utf-8')
        
        decoded = OpenPMUParser.decode_payload(payload_b64, 4, 16, is_signed=True)
        
        assert len(decoded) == 4
        assert decoded[0] == 100
        assert decoded[1] == -200
        assert decoded[2] == 300
        assert decoded[3] == -400
    
    def test_timestamp_parsing(self):
        """Test timestamp parsing."""
        xml = """<OpenPMU>
    <Format>Phasors</Format>
    <Date>2021-02-28</Date>
    <Time>22:04:00.460</Time>
    <Frame>0</Frame>
    <Channels>1</Channels>
    <Channel_0>
        <Name>Test</Name>
        <Type>V</Type>
        <Phase>a</Phase>
        <Range>1</Range>
        <Mag>1.0</Mag>
        <Angle>0.0</Angle>
        <Freq>50.0</Freq>
        <ROCOF>0.0</ROCOF>
    </Channel_0>
</OpenPMU>"""
        
        datagram = OpenPMUParser.parse_xml(xml)
        timestamp = datagram.timestamp
        
        assert timestamp.year == 2021
        assert timestamp.month == 2
        assert timestamp.day == 28
        assert timestamp.hour == 22
        assert timestamp.minute == 4
        assert timestamp.second == 0
        assert timestamp.microsecond == 460000
    
    def test_is_sampled_values(self):
        """Test is_sampled_values property."""
        xml_samples = """<OpenPMU><Format>Samples</Format><Channels>1</Channels><Channel_0><Name>Test</Name><Type>V</Type><Phase>a</Phase><Range>1</Range><Payload>AA==</Payload></Channel_0></OpenPMU>"""
        xml_phasors = """<OpenPMU><Format>Phasors</Format><Channels>1</Channels><Channel_0><Name>Test</Name><Type>V</Type><Phase>a</Phase><Range>1</Range><Mag>1.0</Mag><Angle>0.0</Angle></Channel_0></OpenPMU>"""
        
        datagram_samples = OpenPMUParser.parse_xml(xml_samples)
        datagram_phasors = OpenPMUParser.parse_xml(xml_phasors)
        
        assert datagram_samples.is_sampled_values
        assert not datagram_samples.is_phasor_values
        assert not datagram_phasors.is_sampled_values
        assert datagram_phasors.is_phasor_values
    
    def test_invalid_xml(self):
        """Test parsing invalid XML."""
        with pytest.raises(ValueError):
            OpenPMUParser.parse_xml("Not valid XML")


class TestOpenPMUGenerator:
    """Tests for OpenPMUGenerator."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        generator = OpenPMUGenerator(n_channels=3, fs=12800.0, n_samples=128)
        
        assert generator.n_channels == 3
        assert generator.fs == 12800.0
        assert generator.n_samples == 128
        assert generator.frame_counter == 0
    
    def test_set_channel_info(self):
        """Test setting custom channel info."""
        generator = OpenPMUGenerator(n_channels=2)
        
        generator.set_channel_info(
            names=['Va', 'Vb'],
            types=['V', 'V'],
            phases=['a', 'b'],
            ranges=[275.0, 275.0]
        )
        
        assert generator.channel_names == ['Va', 'Vb']
        assert generator.channel_types == ['V', 'V']
        assert generator.channel_phases == ['a', 'b']
        assert generator.channel_ranges == [275.0, 275.0]
    
    def test_generate_sampled_values_datagram(self):
        """Test generating Sampled Values datagram."""
        generator = OpenPMUGenerator(n_channels=1, fs=1000.0, n_samples=10)
        
        xml = generator.generate_sampled_values_datagram()
        
        assert '<OpenPMU>' in xml
        assert '<Format>Samples</Format>' in xml
        assert '<Channels>1</Channels>' in xml
        assert '<Channel_0>' in xml
        assert '<Name>Channel_0</Name>' in xml
        assert '<Fs>1000.0</Fs>' in xml
        assert '<n>10</n>' in xml
        assert '<Payload>' in xml
    
    def test_generate_phasor_values_datagram(self):
        """Test generating Phasor Values datagram."""
        generator = OpenPMUGenerator(n_channels=2)
        
        xml = generator.generate_phasor_values_datagram()
        
        assert '<OpenPMU>' in xml
        assert '<Format>Phasors</Format>' in xml
        assert '<Channels>2</Channels>' in xml
        assert '<Channel_0>' in xml
        assert '<Channel_1>' in xml
        assert '<Mag>' in xml
        assert '<Angle>' in xml
        assert '<Freq>' in xml
        assert '<ROCOF>' in xml
    
    def test_generate_datagram_auto(self):
        """Test generate_datagram with auto type."""
        generator = OpenPMUGenerator(n_channels=1)
        
        xml_samples = generator.generate_datagram('samples')
        xml_phasors = generator.generate_datagram('phasors')
        
        assert '<Format>Samples</Format>' in xml_samples
        assert '<Format>Phasors</Format>' in xml_phasors
    
    def test_frame_counter_increments(self):
        """Test that frame counter increments."""
        generator = OpenPMUGenerator(n_channels=1)
        
        xml1 = generator.generate_phasor_values_datagram()
        xml2 = generator.generate_phasor_values_datagram()
        
        assert '<Frame>0</Frame>' in xml1
        assert '<Frame>1</Frame>' in xml2
    
    def test_custom_timestamp(self):
        """Test with custom timestamp."""
        generator = OpenPMUGenerator(n_channels=1)
        timestamp = datetime(2024, 1, 15, 10, 30, 45, 123456)
        
        xml = generator.generate_phasor_values_datagram(timestamp)
        
        assert '<Date>2024-01-15</Date>' in xml
        assert '<Time>10:30:45.123456</Time>' in xml


class TestOpenPMUUDPDataSource:
    """Tests for OpenPMUUDPDataSource."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        source = OpenPMUUDPDataSource(host='localhost', port=4713)
        
        assert source.host == 'localhost'
        assert source.port == 4713
        assert source.datagram_type == 'auto'
    
    def test_phasor_type(self):
        """Test with phasor type."""
        source = OpenPMUUDPDataSource(port=4714, datagram_type='phasors')
        
        assert source.datagram_type == 'phasors'
    
    def test_samples_type(self):
        """Test with samples type."""
        source = OpenPMUUDPDataSource(port=4715, datagram_type='samples')
        
        assert source.datagram_type == 'samples'
    
    def test_parser_integration(self):
        """Test that parser is integrated."""
        source = OpenPMUUDPDataSource(port=4716)
        
        xml = """<OpenPMU>
    <Format>Phasors</Format>
    <Date>2021-02-28</Date>
    <Time>22:04:00.460</Time>
    <Frame>23</Frame>
    <Channels>1</Channels>
    <Channel_0>
        <Name>Test</Name>
        <Type>V</Type>
        <Phase>a</Phase>
        <Range>275</Range>
        <Mag>240.0</Mag>
        <Angle>0.403</Angle>
        <Freq>50.690</Freq>
        <ROCOF>0.001</ROCOF>
    </Channel_0>
</OpenPMU>"""
        
        data_dict, timestamps = source._parse_openpmu_packet(
            xml.encode('utf-8'),
            time.time(),
            'auto'
        )
        
        assert 'Test_mag' in data_dict
        assert data_dict['Test_mag'] == pytest.approx(240.0)
        assert 'Test_angle' in data_dict
        assert data_dict['Test_angle'] == pytest.approx(0.403)


class TestOpenPMUDatagram:
    """Tests for OpenPMUDatagram."""
    
    def test_timestamp_property(self):
        """Test timestamp property."""
        datagram = OpenPMUDatagram(
            format='Phasors',
            date='2024-01-15',
            time='10:30:45.123456',
            frame=0,
            channels=[]
        )
        
        timestamp = datagram.timestamp
        assert timestamp.year == 2024
        assert timestamp.month == 1
        assert timestamp.day == 15
        assert timestamp.hour == 10
        assert timestamp.minute == 30
        assert timestamp.second == 45
        assert timestamp.microsecond == 123456


class TestOpenPMUChannel:
    """Tests for OpenPMUChannel."""
    
    def test_basic_creation(self):
        """Test basic channel creation."""
        channel = OpenPMUChannel(
            name='Belfast_Va',
            channel_type='V',
            phase='a',
            range=275.0
        )
        
        assert channel.name == 'Belfast_Va'
        assert channel.channel_type == 'V'
        assert channel.phase == 'a'
        assert channel.range == 275.0
        assert channel.payload is None
        assert channel.mag is None
        assert channel.angle is None
