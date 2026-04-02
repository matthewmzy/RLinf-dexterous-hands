"""Shared signal-processing utilities for the glove driver."""

import numpy as np


class LowPassFilter:
    """Simple delta-clipping low-pass filter.

    Each call to :meth:`filter` limits the per-joint change to at most
    ``±delta``, preventing abrupt jumps in the output signal.

    Args:
        delta: Maximum allowed change per call.
        num_joints: Number of joints (unused, kept for compatibility).
    """

    def __init__(self, delta: float = 0.1, num_joints: int = 6):
        # Set the maximum allowed change between consecutive readings
        self.delta = delta
        # Store the number of joints, though not strictly required for logic
        self.num_joints = num_joints
        # Initialize the state to track previous filtered values
        self.filtered_values = None

    def filter(self, values):
        """Apply delta-clipping to *values* and return the filtered list."""
        if self.filtered_values is None:
            # First call, initialize filtered values to current input values
            self.filtered_values = np.array(values)
        else:
            # Calculate raw difference between current values and previous filtered values
            _delta = np.array(values) - self.filtered_values
            # Clip the difference to be within [-delta, delta] bounds
            _delta = np.clip(_delta, -self.delta, self.delta)
            # Update the filtered values using the clipped difference
            self.filtered_values = self.filtered_values + _delta
            
        # Convert the numpy array back to a standard python list
        return self.filtered_values.tolist()

    def reset(self):
        """Reset filter state to clear previous values."""
        self.filtered_values = None
