"""Module 5B: Carrier-Phase Synchronization.

Removes unknown constant carrier phase rotation:
    x_corrected[n] = x[n] * exp(-j * phi_hat)

Methods:
- M-th power phase estimator:
    - BPSK: M=2, phi_hat = 0.5 * angle(sum(x^2))
    - QPSK: M=4, phi_hat = 0.25 * wrap(angle(sum(x^4)) - pi) [for standard diagonal QPSK]
    - 8PSK: M=8
    - QAM16: Weighted 4th-power phase estimate
- Exposes phase estimation quality, confidence, and low-SNR degradation flags.
"""

from typing import Any, Dict, Optional
import numpy as np


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle in radians to [-pi, pi]."""
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def synchronize_phase(
    iq_signal: np.ndarray,
    modulation: str,
    constellation_diagonal: bool = True
) -> Dict[str, Any]:
    """Estimate and correct constant carrier phase offset.

    Args:
        iq_signal: Complex canonical IQ samples.
        modulation: Modulation string.
        constellation_diagonal: True for diagonal QPSK/8PSK (e.g. (+-1+-j)/sqrt(2)).

    Returns:
        Dictionary containing corrected signal, phase estimate, quality, and warnings.
    """
    out: Dict[str, Any] = {
        "corrected_signal": iq_signal,
        "phase_estimate_rad": 0.0,
        "phase_estimate_deg": 0.0,
        "phase_applied_rad": 0.0,
        "quality": "UNKNOWN",
        "method": "NONE",
        "warnings": []
    }

    if iq_signal is None or len(iq_signal) == 0:
        out["warnings"].append("EMPTY_SIGNAL")
        out["quality"] = "LOW"
        return out

    if not np.all(np.isfinite(iq_signal)):
        out["warnings"].append("NON_FINITE_SAMPLES")
        out["quality"] = "LOW"
        return out

    mod_upper = (modulation or "").upper()

    if mod_upper in ["GFSK", "CPFSK"]:
        out["method"] = "NOT_APPLICABLE_TO_FSK"
        out["quality"] = "NOT_APPLICABLE_TO_FSK"
        out["corrected_signal"] = iq_signal
        return out

    # Determine order M and nominal phase offset of s^M
    if mod_upper == "BPSK":
        m_power = 2
        nominal_phase = 0.0
    elif mod_upper == "QPSK":
        m_power = 4
        # For diagonal QPSK (exp(j(pi/4 + k*pi/2))), s^4 = exp(j*pi) = -1
        # For axis-aligned QPSK (exp(j*k*pi/2)), s^4 = +1
        nominal_phase = np.pi if constellation_diagonal else 0.0
    elif mod_upper == "8PSK":
        m_power = 8
        nominal_phase = np.pi if constellation_diagonal else 0.0
    elif mod_upper in ["QAM16", "QAM64"]:
        m_power = 4
        nominal_phase = np.pi
    else:
        out["method"] = "UNKNOWN_MODULATION"
        out["quality"] = "LOW"
        out["warnings"].append("UNSUPPORTED_MODULATION_FOR_PHASE_SYNC")
        return out

    out["method"] = f"MTH_POWER_PHASE (M={m_power})"

    # Calculate M-th power sum
    if mod_upper in ["QAM16", "QAM64"]:
        w = np.abs(iq_signal).astype(np.float64) ** 4.0
        y = w * (iq_signal.astype(np.complex128) ** float(m_power))
    else:
        y = iq_signal.astype(np.complex128) ** float(m_power)

    sum_y = np.sum(y)
    mag_sum = np.abs(sum_y)

    if mag_sum < 1e-12:
        out["quality"] = "LOW"
        out["warnings"].append("ZERO_MAGNITUDE_PHASE_SUM")
        return out

    # Angle of M-th power
    angle_m = np.angle(sum_y)

    # Remove nominal constellation offset
    diff_angle = wrap_to_pi(angle_m - nominal_phase)

    # Divide by M to get estimated phase modulo 2*pi/M
    phi_hat = diff_angle / float(m_power)

    # Wrap to symmetry interval [-pi/M, pi/M]
    sym_half = np.pi / float(m_power)
    phi_wrapped = ((phi_hat + sym_half) % (2.0 * sym_half)) - sym_half

    # Confidence check based on vector strength
    n = len(iq_signal)
    mean_mag = mag_sum / (float(n) * (np.mean(np.abs(iq_signal) ** m_power) + 1e-12))
    if mean_mag > 0.4:
        out["quality"] = "HIGH"
    elif mean_mag > 0.15:
        out["quality"] = "MEDIUM"
    else:
        out["quality"] = "LOW"
        out["warnings"].append("LOW_PHASE_COHERENCE_OR_LOW_SNR")

    # Apply phase correction
    corrected = iq_signal * np.exp(-1j * phi_wrapped)

    out["corrected_signal"] = corrected
    out["phase_estimate_rad"] = float(phi_wrapped)
    out["phase_estimate_deg"] = float(np.degrees(phi_wrapped))
    out["phase_applied_rad"] = float(phi_wrapped)

    return out
