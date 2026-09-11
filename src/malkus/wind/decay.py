import numba
import numpy as np


@numba.njit
def sigmoid_decay(x: np.ndarray, midpoint: float, slope: float) -> np.ndarray:
    """
    Transform input by a sigmoid shape decay to return values in range [0, 1].

    Args:
        x: Input array to transform
        midpoint: Decay midpoint in x
        slope: Larger values decay faster
    """
    return 0.5 * (1 + np.tanh(slope * (midpoint - x)))