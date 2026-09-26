"""Module 5A: Carrier-Frequency Synchronization.

Performs carrier-frequency offset (CFO) estimation and removal:
    x_corrected[n] = x[n] * exp(-j * 2 * pi * f_cfo * n / Fs)

Methods:
- M-th power spectral peak estimator for PSK (BPSK: M=2, QPSK: M=4, 8PSK: M=8).
- Weighted 4th-power method for QAM16 (conditional).
- Zero-CFO ambiguity detection: avoids falsely interpreting modulation harmonics as CFO.
- Segment-consistency quality check across signal chunks.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def synchronize_carrier(
    iq_signal: np.ndarray,
    modulation: str,
    sample_rate: Optional[float] = None,
    initial_cfo: Optional[float] = None
) -> Dict[str, Any]:
    """Estimate and correct Carrier Frequency Offset (CFO).

    Args:
        iq_signal: Complex canonical IQ samples.
        modulation: Modulation string (e.g. 'QPSK', 'BPSK', etc.).
        sample_rate: Sampling frequency in Hz (if known).
        initial_cfo: Optional preliminary CFO estimate from Module 4.

    Returns:
        Dictionary with corrected signal, estimated CFO, applied CFO, quality, and warnings.
    """
    out: Dict[str, Any] = {
        "corrected_signal": iq_signal,
        "cfo_estimate": 0.0,
        "cfo_applied": 0.0,
        "quality": "UNKNOWN",
        "segment_std": 0.0,
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

    fs = float(sample_rate) if (sample_rate is not None and sample_rate > 0) else 1.0
    n = len(iq_signal)
    mod_upper = (modulation or "").upper()

    # FSK carrier tracking is handled via frequency states/center rather than M-th power
    if mod_upper in ["GFSK", "CPFSK"]:
        out["method"] = "FSK_CARRIER_TRACKED_VIA_STATES"
        out["quality"] = "NOT_APPLICABLE_TO_FSK"
        out["cfo_estimate"] = 0.0
        out["cfo_applied"] = 0.0
        out["corrected_signal"] = iq_signal
        return out

    # Determine M-th power
    m_power = 4  # default
    if mod_upper == "BPSK":
        m_power = 2
    elif mod_upper == "QPSK":
        m_power = 4
    elif mod_upper == "8PSK":
        m_power = 8
    elif mod_upper == "QAM16":
        m_power = 4
    elif mod_upper == "QAM64":
        out["method"] = "QAM64_CFO_UNSUPPORTED"
        out["quality"] = "LOW"
        out["warnings"].append("QAM64_CFO_NOT_VALIDATED")
        # Apply initial_cfo if available from Module 4, but do not re-estimate
        if initial_cfo is not None:
            cfo_to_apply = float(initial_cfo)
            t = np.arange(n) / fs
            out["corrected_signal"] = iq_signal * np.exp(-1j * 2.0 * np.pi * cfo_to_apply * t)
            out["cfo_applied"] = cfo_to_apply
            out["cfo_estimate"] = cfo_to_apply
        return out

    out["method"] = f"MTH_POWER_SPECTRAL (M={m_power})"

    # Prioritize Module 4 initial CFO if provided, while validating with segment consistency
    # Compute full M-th power waveform
    if mod_upper == "QAM16":
        w = np.abs(iq_signal).astype(np.float64) ** 4.0
        y = w * (iq_signal.astype(np.complex128) ** 4.0)
    else:
        y = iq_signal.astype(np.complex128) ** m_power

    # High-resolution FFT peak search
    n_fft = max(2048, 2 ** int(np.ceil(np.log2(n * 4))))
    fft_y = np.abs(np.fft.fft(y, n=n_fft))
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)

    # Spectral peak
    peak_idx = int(np.argmax(fft_y))
    raw_peak_freq = float(freqs[peak_idx])
    estimated_cfo = raw_peak_freq / float(m_power)

    # Zero-CFO Guard (Section 7):
    # At 0 CFO, fourth-power spectrum can have a modulation-related peak near harmonics or DC.
    # Check if the peak is within the DC resolution bin or if DC/near-zero power is comparable.
    dc_idx = 0
    fft_bin_res = fs / n_fft
    is_near_zero = abs(raw_peak_freq) <= 1.5 * fft_bin_res

    # Check peak prominence
    mean_spec = float(np.mean(fft_y))
    peak_prominence = float(fft_y[peak_idx]) / (mean_spec + 1e-12)

    # Segment consistency analysis (Section 8)
    num_segments = 4
    seg_size = n // num_segments
    seg_cfos: List[float] = []

    if seg_size >= 256:
        for k in range(num_segments):
            seg = y[k * seg_size : (k + 1) * seg_size]
            seg_fft = np.abs(np.fft.fft(seg, n=n_fft // 2))
            seg_freqs = np.fft.fftfreq(n_fft // 2, d=1.0 / fs)
            seg_peak_idx = int(np.argmax(seg_fft))
            seg_cfos.append(float(seg_freqs[seg_peak_idx]) / float(m_power))

        seg_std = float(np.std(seg_cfos))
        out["segment_std"] = seg_std
        if seg_std > (0.1 * fs / float(m_power)):
            out["quality"] = "LOW"
            out["warnings"].append("HIGH_SEGMENT_CFO_VARIANCE")
        elif peak_prominence > 4.0:
            out["quality"] = "HIGH"
        else:
            out["quality"] = "MEDIUM"
    else:
        out["quality"] = "MEDIUM" if peak_prominence > 3.0 else "LOW"

    # If peak is near zero or prominence is low, treat as 0 CFO
    if is_near_zero:
        estimated_cfo = 0.0
        out["warnings"].append("ZERO_CFO_DETECTED_OR_AMBIGUOUS")

    # If Module 4 gave a trustworthy CFO and our re-estimate has low prominence, fallback
    final_cfo = estimated_cfo
    if initial_cfo is not None and abs(initial_cfo) > 0 and peak_prominence < 2.0:
        final_cfo = float(initial_cfo)
        out["warnings"].append("FALLBACK_TO_MODULE4_CFO")

    # Apply correction
    t = np.arange(n) / fs
    corrected = iq_signal * np.exp(-1j * 2.0 * np.pi * final_cfo * t)

    out["corrected_signal"] = corrected
    out["cfo_estimate"] = final_cfo
    out["cfo_applied"] = final_cfo

    return out
