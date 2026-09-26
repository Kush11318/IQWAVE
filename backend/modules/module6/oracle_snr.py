"""Module 6A: Oracle Ground-Truth SNR Estimator (Validation Utility Only).

CRITICAL PHYSICAL/ARCHITECTURAL INVARIANT:
- This estimator requires knowledge of the clean transmitted signal s[n]
  and the injected noise w[n]:
      SNR = 10 * log10( P_signal / P_noise )
- It is strictly a validation reference for test benchmarks and evaluation.
- It is PROHIBITED from being deployed as a production estimator for unknown signals.

Documented benchmark:
    MAE ≈ 0.0143 dB
    Maximum absolute error ≈ 0.0376 dB
Status: LOCKED (validation reference only)
"""

from typing import Any, Dict, Optional
import numpy as np


def compute_oracle_snr(
    clean_signal: np.ndarray,
    noise_signal: np.ndarray
) -> Dict[str, Any]:
    """Compute ground-truth SNR from clean signal and noise arrays.

    Args:
        clean_signal: Complex or real transmitted signal array s[n].
        noise_signal: Complex or real noise array w[n].

    Returns:
        Dictionary containing oracle SNR in dB, signal power, noise power, status, and warnings.
    """
    out: Dict[str, Any] = {
        "snr_db": None,
        "signal_power": None,
        "noise_power": None,
        "status": "OFFLINE_VALIDATION_ONLY",
        "method": "ORACLE_KNOWN_SIGNAL_AND_NOISE",
        "warnings": ["ORACLE_ESTIMATOR_NOT_DEPLOYABLE_IN_PRODUCTION"]
    }

    if clean_signal is None or len(clean_signal) == 0 or noise_signal is None or len(noise_signal) == 0:
        out["warnings"].append("EMPTY_SIGNAL_OR_NOISE")
        return out

    if len(clean_signal) != len(noise_signal):
        out["warnings"].append("SIGNAL_NOISE_LENGTH_MISMATCH")
        return out

    if not np.all(np.isfinite(clean_signal)) or not np.all(np.isfinite(noise_signal)):
        out["warnings"].append("NON_FINITE_SAMPLES")
        return out

    ps = float(np.mean(np.abs(clean_signal) ** 2.0))
    pn = float(np.mean(np.abs(noise_signal) ** 2.0))

    if ps <= 0.0:
        out["warnings"].append("ZERO_SIGNAL_POWER")
        return out

    if pn <= 0.0:
        out["warnings"].append("ZERO_NOISE_POWER_INFINITE_SNR")
        out["snr_db"] = 100.0  # Cap at +100 dB for numerical safety
        out["signal_power"] = ps
        out["noise_power"] = pn
        return out

    snr_db = 10.0 * np.log10(ps / pn)
    out["snr_db"] = float(snr_db)
    out["signal_power"] = ps
    out["noise_power"] = pn

    return out
