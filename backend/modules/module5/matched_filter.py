"""Module 5D: Root-Raised-Cosine (RRC) Matched Filtering.

Implements matched filtering for PSK/QAM pulse-shaped signals:
    y[n] = x[n] * h_rrc[n]

- Roll-off beta is taken from configuration, metadata, or explicit input.
- FSK bypasses RRC filtering.
- Filter is unit-energy normalized: sum(|h|^2) = 1.
"""

from typing import Any, Dict, Optional
import numpy as np
from scipy import signal


def design_rrc_filter(
    sps: int,
    beta: float = 0.35,
    span: int = 8
) -> np.ndarray:
    """Generate Root-Raised-Cosine (RRC) filter impulse response.

    Args:
        sps: Samples per symbol.
        beta: Roll-off factor in [0, 1].
        span: Filter span in symbols (total length = 2 * span * sps + 1).

    Returns:
        Unit-energy normalized filter tap array.
    """
    n_taps = 2 * span * sps + 1
    t = np.arange(-(n_taps // 2), (n_taps // 2) + 1, dtype=np.float64)
    h = np.zeros(n_taps, dtype=np.float64)

    # Singularity points where 1 - (4*beta*t/sps)^2 == 0
    t_singular = sps / (4.0 * beta) if beta > 0 else 1e9

    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[i] = 1.0 - beta + (4.0 * beta / np.pi)
        elif abs(abs(ti) - t_singular) < 1e-6:
            val = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)) +
                (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
            h[i] = val
        else:
            ts = ti / sps
            denom = np.pi * ts * (1.0 - (4.0 * beta * ts) ** 2.0)
            numer = np.sin(np.pi * ts * (1.0 - beta)) + 4.0 * beta * ts * np.cos(np.pi * ts * (1.0 + beta))
            h[i] = numer / denom

    # Normalize to unit energy
    energy = np.sqrt(np.sum(h ** 2.0))
    if energy > 0:
        h = h / energy
    return h


def apply_matched_filter(
    iq_signal: np.ndarray,
    modulation: str,
    samples_per_symbol: float,
    beta: Optional[float] = None
) -> Dict[str, Any]:
    """Apply RRC matched filter to IQ signal for PSK/QAM modulations.

    Args:
        iq_signal: Complex canonical IQ samples.
        modulation: Modulation class string.
        samples_per_symbol: Samples per symbol.
        beta: RRC roll-off factor (if known from metadata/config).

    Returns:
        Dictionary containing filtered signal, filter parameters, and warnings.
    """
    out: Dict[str, Any] = {
        "filtered_signal": iq_signal,
        "applied": False,
        "beta": beta,
        "sps": samples_per_symbol,
        "method": "NONE",
        "warnings": []
    }

    if iq_signal is None or len(iq_signal) == 0:
        out["warnings"].append("EMPTY_SIGNAL")
        return out

    if not np.all(np.isfinite(iq_signal)):
        out["warnings"].append("NON_FINITE_SAMPLES")
        return out

    mod_upper = (modulation or "").upper()

    # FSK does not use RRC matched filtering
    if mod_upper in ["GFSK", "CPFSK"]:
        out["method"] = "BYPASS_FSK"
        out["applied"] = False
        out["filtered_signal"] = iq_signal
        return out

    sps_int = max(2, int(round(samples_per_symbol)))
    # beta=0.35 is strictly an implementation baseline/default, NOT a universal production value.
    # The source validated beta=0.35 in controlled experiments alongside sweeps across [0.2, 0.35, 0.5, 1.0].
    active_beta = beta if beta is not None else 0.35

    if beta is None:
        out["warnings"].append("BETA_NOT_IN_METADATA_DEFAULT_0.35_APPLIED_NOT_UNIVERSAL")

    h_rrc = design_rrc_filter(sps=sps_int, beta=active_beta, span=8)

    # Convolution with zero-phase delay compensation (same length)
    filtered = signal.convolve(iq_signal, h_rrc, mode="same")

    out["filtered_signal"] = filtered
    out["applied"] = True
    out["beta"] = active_beta
    out["sps"] = sps_int
    out["method"] = f"RRC_MATCHED_FILTER (beta={active_beta}, sps={sps_int})"

    return out
