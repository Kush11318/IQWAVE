"""Module 5C: Symbol Timing Recovery (Gardner TED).

Estimates and corrects fractional symbol timing offset using the Gardner
Timing Error Detector (TED):
    e[k] = Re{ (x[n_k] - x[n_{k-1}]) * conj(x[n_{k - SPS/2}]) }

The timing error detector S-curve has a stable zero-crossing at the optimum
symbol sampling instant (eye opening). The fractional delay is recovered
via Gardner zero-crossing search and corrected using ideal bandlimited
fractional delay interpolation.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Apply ideal bandlimited fractional delay using frequency-domain phase ramp.

    A positive delay shifts the signal to the right: y(t) = x(t - delay).
    """
    if abs(delay_samples) < 1e-6:
        return x.copy()
    n = len(x)
    freqs = np.fft.fftfreq(n)
    phase_ramp = np.exp(-1j * 2.0 * np.pi * freqs * delay_samples)
    return np.fft.ifft(np.fft.fft(x) * phase_ramp)


def compute_gardner_error(
    x: np.ndarray,
    sps: int,
    start_idx: int = 0
) -> float:
    """Compute average Gardner timing error for a given symbol alignment.

    Args:
        x: Sampled signal at integer SPS (at least SPS >= 2).
        sps: Integer samples per symbol.
        start_idx: Integer sampling offset in [0, sps).

    Returns:
        Mean Gardner timing error value.
    """
    half_sps = sps // 2
    # Ensure indices are valid
    num_symbols = (len(x) - start_idx - sps) // sps
    if num_symbols < 10:
        return 0.0

    k_indices = np.arange(1, num_symbols) * sps + start_idx
    x_k = x[k_indices]
    x_km1 = x[k_indices - sps]
    x_mid = x[k_indices - half_sps]

    errors = np.real((x_k - x_km1) * np.conj(x_mid))
    return float(np.mean(errors))


def synchronize_timing_gardner(
    iq_signal: np.ndarray,
    samples_per_symbol: float,
    search_range_samples: Optional[float] = None
) -> Dict[str, Any]:
    """Recover fractional symbol timing offset using Gardner TED.

    Args:
        iq_signal: Complex canonical IQ samples.
        samples_per_symbol: Nominal samples per symbol.
        search_range_samples: Maximum fractional delay search range.

    Returns:
        Dictionary containing corrected signal, estimated timing offset, quality, and warnings.
    """
    out: Dict[str, Any] = {
        "corrected_signal": iq_signal,
        "timing_offset_samples": 0.0,
        "optimal_sample_phase": 0,
        "quality": "UNKNOWN",
        "method": "GARDNER_TED",
        "warnings": []
    }

    if iq_signal is None or len(iq_signal) < 64:
        out["warnings"].append("INSUFFICIENT_SAMPLES")
        out["quality"] = "LOW"
        return out

    if not np.all(np.isfinite(iq_signal)):
        out["warnings"].append("NON_FINITE_SAMPLES")
        out["quality"] = "LOW"
        return out

    sps = int(round(samples_per_symbol))
    if sps < 2:
        out["warnings"].append("SPS_TOO_LOW_FOR_GARDNER")
        out["quality"] = "LOW"
        return out

    max_range = search_range_samples if search_range_samples is not None else (sps / 2.0)

    # Candidate fractional shifts tau in [-max_range, max_range]
    tau_grid = np.linspace(-max_range, max_range, 161)
    errors = np.zeros(len(tau_grid))

    for idx, tau in enumerate(tau_grid):
        shifted = fractional_delay(iq_signal, -tau)
        errors[idx] = compute_gardner_error(shifted, sps=sps, start_idx=0)

    # Find zero crossing with positive slope (de/dtau > 0)
    zero_crossings = []
    for i in range(len(errors) - 1):
        if errors[i] <= 0.0 and errors[i + 1] >= 0.0:
            denom = errors[i + 1] - errors[i]
            if abs(denom) > 1e-12:
                frac = -errors[i] / denom
                zc = tau_grid[i] + frac * (tau_grid[i + 1] - tau_grid[i])
                zero_crossings.append(zc)

    if zero_crossings:
        # Pick principal zero-crossing closest to 0
        best_tau = float(min(zero_crossings, key=abs))
        out["quality"] = "HIGH"
    else:
        # Fallback to minimum absolute error
        min_idx = int(np.argmin(np.abs(errors)))
        best_tau = float(tau_grid[min_idx])
        out["quality"] = "MEDIUM"
        out["warnings"].append("NO_EXACT_ZERO_CROSSING_FOUND")

    # Correct signal by applying -best_tau
    corrected = fractional_delay(iq_signal, -best_tau)

    out["corrected_signal"] = corrected
    out["timing_offset_samples"] = best_tau
    out["optimal_sample_phase"] = 0

    return out
