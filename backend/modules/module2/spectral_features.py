"""Module 2: Spectral Supporting Observations.

Features:
- zero_bin_fraction: Fraction of total spectral energy in the zero-frequency (DC) bin.
  Retained as supporting evidence; CFO-sensitive and NOT a universal pre-synchronization feature.
"""

from typing import Dict, Optional
import numpy as np


def extract_spectral_features(iq_signal: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute zero_bin_fraction from the signal's discrete Fourier transform.

    Does not modify the signal.
    """
    res: Dict[str, Optional[float]] = {
        "zero_bin_fraction": None
    }

    if iq_signal is None or len(iq_signal) == 0:
        return res

    if not np.all(np.isfinite(iq_signal)):
        return res

    fft_vals = np.fft.fft(iq_signal)
    power_spectrum = np.abs(fft_vals) ** 2
    total_power = float(np.sum(power_spectrum))

    if total_power > 1e-12:
        zero_bin_power = float(power_spectrum[0])
        res["zero_bin_fraction"] = float(np.clip(zero_bin_power / total_power, 0.0, 1.0))
    else:
        res["zero_bin_fraction"] = None

    return res
