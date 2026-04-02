#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


class PSIGloveRequestType:
    """Predefined byte payloads for communicating with the glove hardware."""
    READ_JOINT_POSITION = bytes.fromhex("010300010015D5C5")
    READ_GLOVE_ID = bytes.fromhex("01030007001535C4")


@dataclass
class PSIGloveRequestMessage:
    """Data class wrapper for a request message sent to the glove."""
    bytes: bytes


@dataclass
class PSIGloveStatusMessage:
    """Data class representing the joint position states of the glove.
    
    Each attribute stores a list of joint values corresponding to a specific finger.
    """
    thumb: List[int]
    index: List[int]
    middle: List[int]
    ring: List[int]
    pinky: List[int]


class CommunicationInterface(ABC):
    """Abstract base class for a communication interface with the glove."""

    def __init__(self, auto_connect: bool):
        """Initialize the interface state and attempt connection if requested."""
        self.connected = False
        if auto_connect:
            self.connect()

    @abstractmethod
    def connect(self) -> bool:
        """Establish the connection. To be implemented by subclasses."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close the connection. To be implemented by subclasses."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Return the current connection state. To be implemented by subclasses."""
        pass

    @abstractmethod
    def send_and_receive(self, message: PSIGloveRequestMessage) -> bytes:
        """Send a message and return the received byte response. To be implemented by subclasses."""
        pass


class SerialInterface(CommunicationInterface):
    """Serial port implementation of the communication interface for the PSI glove."""

    def __init__(
        self,
        port: str,
        baudrate: int,
        timeout: float = 0.006,
        auto_connect: bool = False,
        mock: bool = False,
    ):
        """Initialize serial interface settings.
        
        Args:
            port: Path to the serial port device (e.g., /dev/ttyACM0).
            baudrate: Communication speed.
            timeout: Read timeout in seconds.
            auto_connect: Whether to immediately attempt to open the port.
            mock: If true, do not actually open port (useful for dry runs/testing).
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_controller = None
        self.mock = mock
        super().__init__(auto_connect)

    def connect(self) -> bool:
        """Open the serial port connection."""
        # Handle mock mode where hardware is not required
        if self.mock:
            self.connected = True
            return True
            
        try:
            import serial

            # Initialize serial port with specific hardware settings
            self.serial_controller = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )
            self.connected = True
            logger.info(f"Serial port opened successfully: {self.port}")
            return True
            
        except ImportError:
            self.connected = False
            logger.error(
                "Serial library not installed, please install pyserial: "
                "pip install pyserial"
            )
            return False
            
        except Exception as e:
            self.connected = False
            logger.error(f"Serial port open failed: {e}")
            return False

    def disconnect(self):
        """Safely close the serial connection if currently open."""
        if self.serial_controller and self.connected:
            self.serial_controller.close()
            self.connected = False
            logger.info("Serial connection disconnected")

    def is_connected(self) -> bool:
        """Check if serial connection is active."""
        return self.connected

    def _send_message(self, message: PSIGloveRequestMessage) -> bool:
        """Internal helper to write bytes to the serial port."""
        if not self.connected:
            logger.error("Serial port not connected")
            return False

        if self.mock:
            # In mock mode, only log what would be sent
            logger.debug(
                f"Send - Frame data: {[hex(x) for x in message.bytes]}"
            )
            return True

        try:
            # Write out the raw payload
            self.serial_controller.write(message.bytes)
            logger.debug(
                f"Send - Frame data: {[hex(x) for x in message.bytes]}"
            )
            return True
        except Exception as e:
            logger.error(f"Serial message send failed: {e}")
            return False

    def _receive_message(self) -> bytes:
        """Internal helper to read bytes from the serial port."""
        if not self.connected:
            return None
            
        try:
            # Read up to 64 bytes for the response payload
            response = self.serial_controller.read(64)
            return response
        except Exception as e:
            logger.error(f"Serial message receive failed: {e}")
            return None

    def send_and_receive(self, message: PSIGloveRequestMessage) -> bytes:
        """Send request and block to receive the response."""
        # Step 1: Send the message payload
        if not self._send_message(message):
            return {}

        if self.mock:
            # Mock mode short-circuits to empty response
            logger.debug("Mock mode, no need to receive data")
            return {}

        # Step 2: Read the resulting data buffer
        response = self._receive_message()
        return response
