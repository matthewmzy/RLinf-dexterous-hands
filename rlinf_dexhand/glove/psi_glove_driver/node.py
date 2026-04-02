#!/usr/bin/env python3
# Copyright (c) 2025 PSI Robot Team
# Licensed under the Apache License, Version 2.0

import logging
import time
import yaml
from pathlib import Path
from typing import Optional
from collections import deque
import numpy as np

from .controller import PSIGloveController, PSIGloveJointType
from .filters import LowPassFilter
from .interface import PSIGloveStatusMessage, SerialInterface

logger = logging.getLogger(__name__)


def _default_config_path() -> str:
    """Return the path to the bundled default_config.yaml.
    
    This function locates the configuration file relative to this script's directory.
    """
    return str(Path(__file__).parent / "default_config.yaml")


class PSIGloveStandalone:
    """Standalone PSI Glove controller (no ROS dependency).
    
    This class orchestrates connection, data reading, filtering, and 
    mapping for both left and right PSI gloves.
    """

    def __init__(
        self,
        left_hand: Optional[PSIGloveController] = None,
        right_hand: Optional[PSIGloveController] = None,
        left_port: str = "/dev/ttyACM5",
        right_port: str = "/dev/ttyACM4",
        baudrate: int = 115200,
        frequency: int = 10,
        config_file: Optional[str] = None,
        auto_connect: bool = True,
    ):
        """Initialize standalone PSI Glove controller.

        Args:
            left_hand: Left-hand controller (optional; created automatically
                if not provided).
            right_hand: Right-hand controller (optional).
            left_port: Serial device path for the left hand.
            right_port: Serial device path for the right hand.
            baudrate: Serial baudrate.
            frequency: Reading frequency in Hz.
            config_file: Config filename. Defaults to bundled
                default_config.yaml.
            auto_connect: Whether to auto-connect.
        """
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.frequency = frequency

        # Load config file for calibration parameters
        if config_file is None:
            config_file = _default_config_path()
        config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # Parse YAML configuration
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        # Create controllers if not explicitly provided during initialization
        if left_hand is None and right_hand is None:
            # Setup left hand if port is valid
            if left_port:
                left_hand_interface = SerialInterface(
                    port=left_port,
                    baudrate=baudrate,
                    mock=False,
                    auto_connect=auto_connect,
                )
                self.left_hand = PSIGloveController(left_hand_interface)

            # Setup right hand if port is valid
            if right_port:
                right_hand_interface = SerialInterface(
                    port=right_port,
                    baudrate=baudrate,
                    mock=False,
                    auto_connect=auto_connect,
                )
                self.right_hand = PSIGloveController(right_hand_interface)
        else:
            self.left_hand = left_hand
            self.right_hand = right_hand

        # Initialize low-pass filters to prevent sudden joint state jumps
        self.hand_low_pass_filters = {
            "left": LowPassFilter(delta=0.1, num_joints=6),
            "right": LowPassFilter(delta=0.1, num_joints=6),
        }

        # Initialize position queues to apply moving average smoothing
        self.hand_joint_position_queues = {
            "left": deque(maxlen=10),
            "right": deque(maxlen=10),
        }

    @staticmethod
    def _minmax_linear_map(
        calibration_min: int,
        calibration_max: int,
        value: int,
        clip_source_min: float,
        clip_source_max: float,
        clip_target_min: float,
        clip_target_max: float,
    ) -> float:
        """Two-step linear mapping with clamping.

        1. Normalise raw sensor value to [0, 1] using calibration params.
        2. Remap via clip.source range to [0, 1].
        3. Clamp final output with clip.target bounds.
        
        Args:
            calibration_min: Minimum raw value observed during calibration.
            calibration_max: Maximum raw value observed during calibration.
            value: Current raw sensor reading.
            clip_source_min: Minimum mapped value to treat as logical zero.
            clip_source_max: Maximum mapped value to treat as logical one.
            clip_target_min: Hard minimum bound for final output.
            clip_target_max: Hard maximum bound for final output.
        """
        # Step 1: Factory calibration normalisation to [0, 1]
        if calibration_max == calibration_min:
            normalized_value = 0.0
        else:
            normalized_value = (value - calibration_min) / (calibration_max - calibration_min)

        # Step 2: Remap to [0, 1] via source range
        if clip_source_max == clip_source_min:
            remapped_value = 0.0
        else:
            remapped_value = (normalized_value - clip_source_min) / (clip_source_max - clip_source_min)

        # Step 3: Clamp with target bounds to ensure safety and stability
        final_value = np.clip(remapped_value, clip_target_min, clip_target_max)

        return float(final_value)

    def _process_status(
        self, status: PSIGloveStatusMessage, hand_type: str
    ) -> list:
        """Process status message with calibration and filtering.

        Args:
            status: Raw glove readings object.
            hand_type: String identifier ('left' or 'right') used for config lookup.

        Returns:
            Processed joint positions
            ``[thumb_side, thumb_back, index_back, middle_back, ring_back, pinky_back]``.
        """
        # Extract hand-specific configuration
        cfg = self.config[f"{hand_type}_glove"]["calibration"]

        # Map each joint from raw integer readings to a unified float representation
        positions = [
            self._minmax_linear_map(
                cfg["thumb"]["side"]["calibration"]["min"],
                cfg["thumb"]["side"]["calibration"]["max"],
                status.thumb[PSIGloveJointType.side],
                cfg["thumb"]["side"]["clip"]["source"]["min"],
                cfg["thumb"]["side"]["clip"]["source"]["max"],
                cfg["thumb"]["side"]["clip"]["target"]["min"],
                cfg["thumb"]["side"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                cfg["thumb"]["back"]["calibration"]["min"],
                cfg["thumb"]["back"]["calibration"]["max"],
                status.thumb[PSIGloveJointType.back],
                cfg["thumb"]["back"]["clip"]["source"]["min"],
                cfg["thumb"]["back"]["clip"]["source"]["max"],
                cfg["thumb"]["back"]["clip"]["target"]["min"],
                cfg["thumb"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                cfg["index"]["back"]["calibration"]["min"],
                cfg["index"]["back"]["calibration"]["max"],
                status.index[PSIGloveJointType.back],
                cfg["index"]["back"]["clip"]["source"]["min"],
                cfg["index"]["back"]["clip"]["source"]["max"],
                cfg["index"]["back"]["clip"]["target"]["min"],
                cfg["index"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                cfg["middle"]["back"]["calibration"]["min"],
                cfg["middle"]["back"]["calibration"]["max"],
                status.middle[PSIGloveJointType.back],
                cfg["middle"]["back"]["clip"]["source"]["min"],
                cfg["middle"]["back"]["clip"]["source"]["max"],
                cfg["middle"]["back"]["clip"]["target"]["min"],
                cfg["middle"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                cfg["ring"]["back"]["calibration"]["min"],
                cfg["ring"]["back"]["calibration"]["max"],
                status.ring[PSIGloveJointType.back],
                cfg["ring"]["back"]["clip"]["source"]["min"],
                cfg["ring"]["back"]["clip"]["source"]["max"],
                cfg["ring"]["back"]["clip"]["target"]["min"],
                cfg["ring"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                cfg["pinky"]["back"]["calibration"]["min"],
                cfg["pinky"]["back"]["calibration"]["max"],
                status.pinky[PSIGloveJointType.back],
                cfg["pinky"]["back"]["clip"]["source"]["min"],
                cfg["pinky"]["back"]["clip"]["source"]["max"],
                cfg["pinky"]["back"]["clip"]["target"]["min"],
                cfg["pinky"]["back"]["clip"]["target"]["max"],
            ),
        ]

        # Apply low-pass filter to restrict maximum step change
        positions = self.hand_low_pass_filters[hand_type].filter(positions)

        # Append to moving average queue and compute average for final smoothing
        self.hand_joint_position_queues[hand_type].append(positions)
        positions = np.mean(self.hand_joint_position_queues[hand_type], axis=0).tolist()

        return positions

    def loop(self):
        """Execute one loop iteration: read and process data.
        
        This queries hardware (if connected) and returns a dictionary
        mapping each active hand to its raw and processed status.
        """
        results = {}

        # Handle left glove reading
        if self.left_hand:
            try:
                left_status = self.left_hand.loop()
                if left_status:
                    left_positions = self._process_status(left_status, "left")
                    results["left"] = {
                        "raw": left_status,
                        "processed": left_positions
                    }
            except Exception as e:
                logger.warning(f"Error reading left glove: {type(e).__name__}: {e}")

        # Handle right glove reading
        if self.right_hand:
            try:
                right_status = self.right_hand.loop()
                if right_status:
                    right_positions = self._process_status(right_status, "right")
                    results["right"] = {
                        "raw": right_status,
                        "processed": right_positions
                    }
            except Exception as e:
                logger.warning(f"Error reading right glove: {type(e).__name__}: {e}")

        return results

    def get_hand_action(self):
        """Helper to get processed states with a forced loop interval wait."""
        interval = 1.0 / self.frequency
        start_time = time.perf_counter()
        
        # Read the current states
        results = self.loop()
        
        # Calculate time taken and sleep remaining time to respect set frequency
        elapsed = time.perf_counter() - start_time
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)
        return results

    def run(self, print_output: bool = True):
        """Run the main loop (useful for standalone debugging).

        Args:
            print_output: Whether to print output.
        """
        interval = 1.0 / self.frequency

        try:
            while True:
                start_time = time.perf_counter()

                # Continuously execute reading iteration
                results = self.loop()

                if print_output:
                    self._print_results(results)

                # Maintain the specified refresh frequency
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            # Ensure safe hardware disconnection on exit
            self.disconnect()

    def _print_results(self, results: dict):
        """Print processed results in a human-readable format."""
        joint_names = [
            "thumb_side",
            "thumb_back",
            "index_back",
            "middle_back",
            "ring_back",
            "pinky_back"
        ]

        # Iterate dynamically based on available data keys
        for hand_type in ["left", "right"]:
            if hand_type in results:
                data = results[hand_type]
                status = data["raw"]
                positions = data["processed"]

                # Display detailed raw sensor payload
                print(f"\n[{hand_type.upper()} glove raw sensor values]")
                print(f"Thumb:  [tip={status.thumb[0]:4d}, mid={status.thumb[1]:4d}, back={status.thumb[2]:4d}, side={status.thumb[3]:4d}, rotate={status.thumb[4]:4d}]")
                print(f"Index:  [tip={status.index[0]:4d}, mid={status.index[1]:4d}, back={status.index[2]:4d}, side={status.index[3]:4d}]")
                print(f"Middle: [tip={status.middle[0]:4d}, mid={status.middle[1]:4d}, back={status.middle[2]:4d}, side={status.middle[3]:4d}]")
                print(f"Ring:   [tip={status.ring[0]:4d}, mid={status.ring[1]:4d}, back={status.ring[2]:4d}, side={status.ring[3]:4d}]")
                print(f"Pinky:  [tip={status.pinky[0]:4d}, mid={status.pinky[1]:4d}, back={status.pinky[2]:4d}, side={status.pinky[3]:4d}]")

                # Display final mapped output values
                print(f"\n[{hand_type.upper()} glove mapped values (normalised)]")
                for name, pos in zip(joint_names, positions):
                    print(f"  {name}: {pos:.3f}")
                print("-" * 80)

    def disconnect(self):
        """Disconnect from controllers."""
        if self.left_hand:
            self.left_hand.disconnect()
        if self.right_hand:
            self.right_hand.disconnect()


def main():
    """Main entry point for standalone glove testing."""
    import argparse

    # Define arguments to override standalone script execution
    parser = argparse.ArgumentParser(description="PSI Glove Standalone Controller")
    parser.add_argument("--left-port", type=str, default="/dev/ttyACM0", help="Left hand port")
    parser.add_argument("--right-port", type=str, default="/dev/ttyACM1", help="Right hand port")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baudrate")
    parser.add_argument("--frequency", type=int, default=100, help="Reading frequency (Hz)")
    parser.add_argument("--config", type=str, default=None, help="Config file")
    parser.add_argument("--no-left", action="store_true", help="Disable left hand")
    parser.add_argument("--no-right", action="store_true", help="Disable right hand")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging level based on arguments
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create controller handling object
    controller = PSIGloveStandalone(
        left_port=None if args.no_left else args.left_port,
        right_port=None if args.no_right else args.right_port,
        baudrate=args.baudrate,
        frequency=args.frequency,
        config_file=args.config,
        auto_connect=True,
    )

    # Validate active connection states and raise errors for failing ports
    if controller.left_hand and not controller.left_hand.is_connected():
        logger.error(f"Failed to connect to left hand: {args.left_port}")
    if controller.right_hand and not controller.right_hand.is_connected():
        logger.error(f"Failed to connect to right hand: {args.right_port}")

    # Terminate early if connections are missing and required
    if (controller.left_hand and not controller.left_hand.is_connected()) or \
       (controller.right_hand and not controller.right_hand.is_connected()):
        logger.error("Connection failed. Exiting.")
        return

    logger.info(f"Connected. Reading at {args.frequency}Hz")
    logger.info("Press Ctrl+C to stop")

    # Run main debug monitoring loop
    controller.run(print_output=True)


if __name__ == "__main__":
    main()
