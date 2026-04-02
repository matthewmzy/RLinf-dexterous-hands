#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0

import logging
import struct
from enum import IntEnum

from .interface import (CommunicationInterface, PSIGloveRequestMessage,
                        PSIGloveRequestType, PSIGloveStatusMessage)

# Define the types of joints present in each finger
class PSIGloveJointType(IntEnum):
    """Enumeration of finger joint types."""
    tip = 0       # Distal joint (fingertip)
    mid = 1       # Proximal interphalangeal joint
    back = 2      # Metacarpophalangeal joint (base back)
    side = 3      # Abduction/adduction joint (side-to-side spread)
    rotate = 4    # Thumb rotation joint (only applicable to thumb)


logger = logging.getLogger(__name__)


class PSIGloveController:
    """Controller class to handle data retrieval and parsing for a PSI Glove."""

    def __init__(self, communication_interface: CommunicationInterface):
        """Initialize the glove controller with a specific communication interface.
        
        Args:
            communication_interface: The underlying interface (e.g., SerialInterface) for sending/receiving data.
        """
        self.communication_interface = communication_interface
        self.last_status = None

    def connect(self) -> bool:
        """Establish connection via the communication interface."""
        return self.communication_interface.connect()

    def disconnect(self):
        """Close connection to the glove hardware."""
        return self.communication_interface.disconnect()

    def is_connected(self) -> bool:
        """Check if the communication interface is currently connected."""
        return self.communication_interface.is_connected()

    def read_joint_positions(self) -> PSIGloveStatusMessage:
        """Request and retrieve current joint positions from the glove.
        
        Returns:
            A PSIGloveStatusMessage containing joint angles for all fingers.
        """
        # Create a request message to read joint positions
        request_message = PSIGloveRequestMessage(
            bytes=PSIGloveRequestType.READ_JOINT_POSITION
        )
        # Send the request and receive the raw byte response
        response = self.communication_interface.send_and_receive(
            message=request_message
        )
        # Parse the raw bytes into a structured status message
        status = self._parse_response(response)
        return status

    def loop(self) -> PSIGloveStatusMessage:
        """Execute a single control loop iteration to read positions and update state.
        
        Returns:
            The parsed glove status message.
        """
        status = self.read_joint_positions()
        # Cache the latest status for reference if needed
        self.last_status = status
        return status

    def _parse_response(self, raw_bytes: bytes) -> PSIGloveStatusMessage:
        """Parse joint position response from raw bytes.

        Protocol format:
        - Response header: 01 03 2A (slave address + function code + data length)
        - Data: 42 bytes joint data (21 joints, 2 bytes each)
        - CRC checksum: 2 bytes

        Joint ordering:
        - Thumb: 5 joints [tip, middle, base, side, rotation]
        - Index finger: 4 joints [tip, middle, base, side]
        - Middle finger: 4 joints [tip, middle, base, side]
        - Ring finger: 4 joints [tip, middle, base, side]
        - Little finger: 4 joints [tip, middle, base, side]
        """
        # Unpack the first 45 bytes (3 byte header + 42 byte data payload)
        # Format string breakdown:
        # > : Big-endian byte order
        # B : Unsigned char (1 byte) for slave address
        # B : Unsigned char (1 byte) for function code
        # B : Unsigned char (1 byte) for data length
        # 21H : 21 Unsigned shorts (2 bytes each) for the joint positions
        raw = struct.unpack(">BBB21H", raw_bytes[:45])
        
        # Extract the 21 joint values from the unpacked tuple
        joint_positions = list(raw[3:24])

        # Slice the array into individual finger lists based on the joint ordering
        thumb_joints = joint_positions[0:5]
        index_joints = joint_positions[5:9]
        middle_joints = joint_positions[9:13]
        ring_joints = joint_positions[13:17]
        pinky_joints = joint_positions[17:21]

        logger.debug(
            f"Thumb joints: {thumb_joints}, Index joints: {index_joints}, "
            f"Middle joints: {middle_joints}, Ring joints: {ring_joints}, Pinky joints: {pinky_joints}"
        )

        # Construct and return the final structured status message
        status_message = PSIGloveStatusMessage(
            thumb=thumb_joints,
            index=index_joints,
            middle=middle_joints,
            ring=ring_joints,
            pinky=pinky_joints,
        )
        return status_message
