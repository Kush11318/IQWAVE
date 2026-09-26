"""Module 5E: Rational Resampling.

Handles non-integer samples-per-symbol by converting to an exact integer SPS
using polyphase rational resampling:
    y = resample_poly(x, up, down)

For example:
    Fs = 1 MHz, Rs = 240 kSym/s -> SPS = 4.1667
    target_sps = 5
    ratio = (5 * 240k) / 1M = 1.2 / 1.0 = 6 / 5
    resample_poly(x, 6, 5) -> new SPS = 5.0
"""

from fractions import Fraction
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy import signal


def find_resampling_ratio(
    current_sps: float,
    target_sps: Optional[int] = None,
    max_denominator: int = 100
) -> Tuple[int, int, int]:
    """Calculate rational factors (up, down) to reach an integer SPS.

    Returns:
        (up, down, target_integer_sps)
    """
    if target_sps is None:
        target_sps = int(np.ceil(current_sps))
        if target_sps < 2:
            target_sps = 2

    # Ratio Fs' / Fs = target_sps / current_sps
    ratio = target_sps / current_sps
    frac = Fraction(ratio).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator, target_sps


def rational_resample(
    iq_signal: np.ndarray,
    samples_per_symbol: float,
    target_integer_sps: Optional[int] = None
) -> Dict[str, Any]:
    """Resample waveform to achieve integer samples per symbol.

    Args:
        iq_signal: Complex canonical IQ samples.
        samples_per_symbol: Input SPS (may be non-integer).
        target_integer_sps: Desired integer SPS.

    Returns:
        Dictionary containing resampled signal, new SPS, up/down factors, and status.
    """
    out: Dict[str, Any] = {
        "resampled_signal": iq_signal,
        "original_sps": samples_per_symbol,
        "new_sps": samples_per_symbol,
        "up": 1,
        "down": 1,
        "applied": False,
        "method": "NONE",
        "warnings": []
    }

    if iq_signal is None or len(iq_signal) == 0:
        out["warnings"].append("EMPTY_SIGNAL")
        return out

    if not np.all(np.isfinite(iq_signal)):
        out["warnings"].append("NON_FINITE_SAMPLES")
        return out

    # Check if already integer
    if abs(samples_per_symbol - round(samples_per_symbol)) < 0.005:
        out["new_sps"] = float(round(samples_per_symbol))
        out["method"] = "IDENTITY_ALREADY_INTEGER_SPS"
        return out

    up, down, target_sps = find_resampling_ratio(
        current_sps=samples_per_symbol,
        target_sps=target_integer_sps
    )

    # Perform polyphase rational resampling
    resampled = signal.resample_poly(iq_signal, up=up, down=down)
    effective_sps = float(samples_per_symbol * (up / down))

    out["resampled_signal"] = resampled
    out["new_sps"] = effective_sps
    out["up"] = up
    out["down"] = down
    out["applied"] = True
    out["method"] = f"RATIONAL_RESAMPLE_POLY (up={up}, down={down}, target_sps={target_sps})"

    return out
