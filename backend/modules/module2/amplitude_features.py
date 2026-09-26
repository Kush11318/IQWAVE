"""Module 2: Amplitude Feature Extraction.

Extracts envelope statistics without modifying the signal:
- amp_mean
- amp_std
- amp_cv (coefficient of variation: std / mean)
- amp_skewness
- amp_kurtosis

Strict numerical safety: returns None for undefined cases rather than
manufacturing arbitrary values.
"""

from typing import Any, Dict, Optional
import numpy as np
from scipy import stats


def extract_amplitude_features(iq_signal: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute retained amplitude features from the complex envelope |x[n]|.

    Does not modify the signal.
    """
    res: Dict[str, Optional[float]] = {
        "mean": None,
        "std": None,
        "cv": None,
        "skewness": None,
        "kurtosis": None
    }

    if iq_signal is None or len(iq_signal) == 0:
        return res

    # Non-finite guard
    if not np.all(np.isfinite(iq_signal)):
        return res

    amp = np.abs(iq_signal).astype(np.float64)
    n = len(amp)

    amp_mean = float(np.mean(amp))
    amp_std = float(np.std(amp))

    res["mean"] = amp_mean
    res["std"] = amp_std

    # Coefficient of variation
    if amp_mean > 0:
        res["cv"] = float(amp_std / amp_mean)
    else:
        res["cv"] = None

    # Higher order moments require meaningful envelope variation above float32 noise floor
    is_variable_envelope = (amp_std > 1e-5) and (amp_mean > 0 and (amp_std / amp_mean) > 1e-4)

    if is_variable_envelope and n >= 3:
        res["skewness"] = float(stats.skew(amp, bias=False))
    else:
        res["skewness"] = None

    if is_variable_envelope and n >= 4:
        # Excess kurtosis (Fisher definition: normal distribution = 0.0)
        res["kurtosis"] = float(stats.kurtosis(amp, fisher=True, bias=False))
    else:
        res["kurtosis"] = None

    return res
