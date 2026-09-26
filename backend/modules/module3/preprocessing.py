"""Module 3: CNN Preprocessing & RMS Normalization.

Performs validated RMS normalization for Engine B (Raw-IQ 1D CNN):
    P = (1 / 2N) * sum(I_n^2 + Q_n^2)
    x_norm = x / sqrt(P + epsilon)

CRITICAL:
- Normalization is applied strictly to a detached copy.
- The original / raw IQ signal remains unnormalized and preserved.
- Shapes input into (128, 2) format with channel 0 = I, channel 1 = Q.
"""

from typing import Optional, Tuple
import numpy as np


def preprocess_for_cnn(
    iq_signal: np.ndarray,
    target_length: int = 128,
    epsilon: float = 1e-12
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Transform canonical complex64 IQ into RMS-normalized 2-channel array of shape (target_length, 2).

    Returns:
        (normalized_cnn_input, measured_rms_power)

    The resulting array has shape (target_length, 2):
        [:, 0] -> In-phase (I)
        [:, 1] -> Quadrature (Q)
    """
    if iq_signal is None or len(iq_signal) == 0:
        return None, None

    if not np.all(np.isfinite(iq_signal)):
        return None, None

    # Slice or pad cleanly to target_length (default 128) on a detached copy
    n = len(iq_signal)
    if n >= target_length:
        x_slice = np.copy(iq_signal[:target_length])
    else:
        # Zero-pad if shorter than target_length
        x_slice = np.zeros(target_length, dtype=np.complex64)
        x_slice[:n] = iq_signal

    i_arr = x_slice.real.astype(np.float32)
    q_arr = x_slice.imag.astype(np.float32)

    # Frame power: P = (1 / 2N) * sum(I^2 + Q^2)
    frame_power = float(np.mean(i_arr ** 2 + q_arr ** 2) / 2.0)

    # RMS normalization
    norm_factor = float(np.sqrt(frame_power + epsilon))
    i_norm = i_arr / norm_factor
    q_norm = q_arr / norm_factor

    # Form (128, 2) with column 0 = I, column 1 = Q
    cnn_input = np.column_stack((i_norm, q_norm)).astype(np.float32)

    return cnn_input, frame_power
