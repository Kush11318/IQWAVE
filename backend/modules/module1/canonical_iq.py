"""Module 1A: Canonical IQ Representation & Non-Destructive Validation (LOCKED).

This module implements the locked Module 1A behavior:
- Canonical complex representation: x[n] = I[n] + j*Q[n] (complex64)
- Non-destructive validation: NO DC removal, NO normalization, NO filtering,
  NO denoising, NO resampling, NO synchronization, NO IQ correction, NO modulation classification.
- Basic diagnostics: mean, power, RMS, peak, sample count.
- Structured machine-readable validation results with standard failure codes.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


def to_canonical_iq(
    i_channel: Any,
    q_channel: Any
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Convert separate I and Q arrays to canonical complex representation:
        x[n] = I[n] + j * Q[n]

    Reference implementation:
        iq = I.astype(np.float32) + 1j * Q.astype(np.float32)
    Resulting dtype: np.complex64
    """
    i_arr = np.asarray(i_channel)
    q_arr = np.asarray(q_channel)

    if i_arr.shape != q_arr.shape:
        return None, {
            "status": "INVALID",
            "num_samples": 0,
            "nan_count": 0,
            "inf_count": 0,
            "power": None,
            "warnings": ["I_Q_LENGTH_MISMATCH"]
        }

    # Canonical conversion
    iq = i_arr.astype(np.float32) + 1j * q_arr.astype(np.float32)
    validation = validate_iq(iq)
    return iq, validation


def compute_diagnostics(iq: np.ndarray) -> Dict[str, Any]:
    """Calculate basic non-destructive signal diagnostics:
    - mean: (1/N) * sum(x[n])  (recorded as complex, NEVER subtracted as DC)
    - power: P = (1/N) * sum(|x[n]|^2)
    - RMS: sqrt(P)
    - peak: max(|x[n]|)
    """
    n = len(iq)
    if n == 0:
        return {
            "num_samples": 0,
            "mean_real": None,
            "mean_imag": None,
            "power": None,
            "rms": None,
            "peak": None
        }

    mean_val = np.mean(iq)
    abs_iq = np.abs(iq)
    power = float(np.mean(abs_iq ** 2))
    rms = float(np.sqrt(power))
    peak = float(np.max(abs_iq))

    return {
        "num_samples": n,
        "mean_real": float(np.real(mean_val)),
        "mean_imag": float(np.imag(mean_val)),
        "power": power,
        "rms": rms,
        "peak": peak
    }


def validate_iq(iq: Any) -> Dict[str, Any]:
    """Canonical validator logic as documented in Module 1:

    Checks:
    - empty input -> INVALID / EMPTY_SIGNAL
    - NaN / Inf -> INVALID / NON_FINITE_VALUES
    - zero signal power -> INVALID / ZERO_POWER
    """
    iq = np.asarray(iq)

    result: Dict[str, Any] = {
        "status": "VALID",
        "num_samples": len(iq),
        "nan_count": 0,
        "inf_count": 0,
        "power": None,
        "warnings": []
    }

    if len(iq) == 0:
        result["status"] = "INVALID"
        result["warnings"].append("EMPTY_SIGNAL")
        return result

    result["nan_count"] = int(np.isnan(iq).sum())
    result["inf_count"] = int(np.isinf(iq).sum())

    if result["nan_count"] > 0 or result["inf_count"] > 0:
        result["status"] = "INVALID"
        result["warnings"].append("NON_FINITE_VALUES")
        return result

    power = float(np.mean(np.abs(iq) ** 2))
    result["power"] = power

    if result["power"] == 0:
        result["status"] = "INVALID"
        result["warnings"].append("ZERO_POWER")
        return result

    # Attach descriptive diagnostics when valid
    diagnostics = compute_diagnostics(iq)
    result["diagnostics"] = diagnostics

    return result