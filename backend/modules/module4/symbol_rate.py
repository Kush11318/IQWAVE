"""Module 4A: Cyclic-Correlation Symbol-Rate & SPS Estimator.

Implements the validated cyclic-correlation symbol-rate estimation
(Ciblat et al., IEEE Trans. Info. Theory, 2002).

Digital signals pulse-shaped by Nyquist filters (e.g., RRC) exhibit cyclostationary
structure at cyclic frequency alpha = Rs. This implementation computes the cyclic
autocorrelation / nonlinear envelope spectrum to extract the symbol rate without
relying on fragile ordinary autocorrelation.

Physical Limitation Invariant:
- If sampling frequency (Fs) is available from metadata, outputs absolute symbol rate
  Rs (Hz) and samples per symbol (SPS = Fs / Rs).
- If Fs is unavailable, outputs normalized symbol rate (cycles/sample), and sets
  SPS = None with flag "METADATA_REQUIRED". Absolute Fs is NEVER invented.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


def estimate_symbol_rate(
    iq_signal: np.ndarray,
    sample_rate: Optional[float] = None
) -> Dict[str, Any]:
    """Estimate symbol rate Rs and SPS via cyclic correlation / envelope spectrum.

    Parameters:
        iq_signal: 1D complex64 NumPy array
        sample_rate: Optional physical sampling frequency Fs in Hz
    """
    res: Dict[str, Any] = {
        "symbol_rate": None,
        "symbol_rate_normalized": None,
        "samples_per_symbol": None,
        "confidence": "UNKNOWN",
        "method": "CYCLIC_CORRELATION_CIBLAT",
        "flags": []
    }

    if iq_signal is None or len(iq_signal) < 32:
        res["flags"].append("INSUFFICIENT_SAMPLES")
        res["confidence"] = "LOW"
        return res

    if not np.all(np.isfinite(iq_signal)):
        res["flags"].append("NON_FINITE_SAMPLES")
        res["confidence"] = "LOW"
        return res

    n = len(iq_signal)

    # 1. Cyclostationary envelope series: z[n] = |x[n]|^2
    # Zero-mean nonlinear transformation to expose cyclic spectral line at alpha = Rs
    z = np.abs(iq_signal.astype(np.complex64)) ** 2
    z = z - np.mean(z)

    # 2. FFT of envelope
    n_fft = 2 ** int(np.ceil(np.log2(n * 2)))
    fft_z = np.abs(np.fft.fft(z, n=n_fft))
    freqs = np.fft.fftfreq(n_fft, d=1.0)  # Normalized frequencies in [-0.5, 0.5]

    # Search over positive normalized frequencies [0.02, 0.48]
    # Avoiding DC (< 0.02) and Nyquist edge (> 0.48)
    pos_mask = (freqs > 0.02) & (freqs < 0.48)
    if not np.any(pos_mask):
        res["flags"].append("NO_VALID_SEARCH_BAND")
        return res

    search_freqs = freqs[pos_mask]
    search_spec = fft_z[pos_mask]

    peak_idx = int(np.argmax(search_spec))
    norm_rs = float(search_freqs[peak_idx])
    res["symbol_rate_normalized"] = norm_rs

    # Peak-to-average ratio of the cyclic line as a confidence indicator
    mean_spec = np.mean(search_spec)
    peak_val = search_spec[peak_idx]
    prominence = peak_val / (mean_spec + 1e-12)

    if prominence > 4.0 and n >= 500:
        res["confidence"] = "HIGH"
    elif prominence > 2.0 and n >= 128:
        res["confidence"] = "MEDIUM"
    else:
        res["confidence"] = "LOW"
        res["flags"].append("LOW_CYCLIC_PROMINENCE")

    # 3. Absolute conversion if Fs is known from metadata
    if sample_rate is not None and sample_rate > 0:
        abs_rs = norm_rs * sample_rate
        res["symbol_rate"] = float(abs_rs)
        sps = float(sample_rate / abs_rs)
        res["samples_per_symbol"] = sps
    else:
        res["symbol_rate"] = None
        res["samples_per_symbol"] = None
        res["flags"].append("METADATA_REQUIRED_FOR_ABSOLUTE_RATES")

    return res
