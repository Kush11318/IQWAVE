"""Module 4C: Carrier-Frequency Offset (CFO) Estimators.

Implements:
1. PSK CFO: M-th-power spectral peak estimator
   - BPSK: y[n] = x[n]^2  -> CFO = f_peak / 2
   - QPSK: y[n] = x[n]^4  -> CFO = f_peak / 4
   - 8PSK: y[n] = x[n]^8  -> CFO = f_peak / 8
   Status: LOCKED

2. QAM16 CFO: Weighted full-rate fourth-power CFO estimator
   - y[n] = (|x[n]|^4) * (x[n]^4) -> CFO = f_peak / 4
   - Emphasizes corner constellation points where 4th power is coherent.
   Status: CONDITIONAL (validated for tested QAM16; NOT universal).
   - QAM64: NOT_VALIDATED / UNSUPPORTED.

Physical Limitation Invariant:
- If Fs is available from metadata, outputs CFO in Hz.
- If Fs is unavailable, outputs normalized CFO in cycles/sample with flag "METADATA_REQUIRED".
"""

from typing import Any, Dict, Optional
import numpy as np


def estimate_cfo(
    iq_signal: np.ndarray,
    modulation_type: Optional[str] = None,
    sample_rate: Optional[float] = None
) -> Dict[str, Any]:
    """Estimate Carrier Frequency Offset based on modulation family."""
    mod = (modulation_type or "").upper()
    fs_val = sample_rate if (sample_rate is not None and sample_rate > 0) else 1.0

    res: Dict[str, Any] = {
        "cfo": None,
        "cfo_normalized": None,
        "method": "UNKNOWN",
        "confidence": "UNKNOWN",
        "status": "NOT_APPLICABLE",
        "flags": []
    }

    if iq_signal is None or len(iq_signal) < 64:
        res["flags"].append("INSUFFICIENT_SAMPLES")
        res["confidence"] = "LOW"
        return res

    if not np.all(np.isfinite(iq_signal)):
        res["flags"].append("NON_FINITE_SAMPLES")
        res["confidence"] = "LOW"
        return res

    n = len(iq_signal)

    # 1. PSK Family (BPSK, QPSK, 8PSK)
    if mod in ["BPSK", "QPSK", "8PSK"]:
        m_power = {"BPSK": 2, "QPSK": 4, "8PSK": 8}[mod]
        res["method"] = f"PSK_MTH_POWER_SPECTRAL (M={m_power})"
        res["status"] = "LOCKED"

        # Raise to M-th power
        y = (iq_signal.astype(np.complex128)) ** m_power

        # High-resolution FFT peak search
        n_fft = 2 ** int(np.ceil(np.log2(n * 4)))
        fft_y = np.abs(np.fft.fft(y, n=n_fft))
        freqs = np.fft.fftfreq(n_fft, d=1.0 / fs_val)

        peak_idx = int(np.argmax(fft_y))
        peak_freq = float(freqs[peak_idx])

        # Estimated CFO
        est_cfo = peak_freq / float(m_power)

        # Prominence check
        mean_spec = np.mean(fft_y)
        prominence = fft_y[peak_idx] / (mean_spec + 1e-12)
        if prominence > 4.0:
            res["confidence"] = "HIGH"
        elif prominence > 2.0:
            res["confidence"] = "MEDIUM"
        else:
            res["confidence"] = "LOW"
            res["flags"].append("LOW_SPECTRAL_PEAK_PROMINENCE")

        if sample_rate is not None and sample_rate > 0:
            res["cfo"] = est_cfo
        else:
            res["cfo_normalized"] = est_cfo
            res["flags"].append("METADATA_REQUIRED_FOR_ABSOLUTE_HERTZ")

        return res

    # 2. QAM16 (Conditional Weighted Fourth Power)
    elif mod == "QAM16":
        res["method"] = "WEIGHTED_FULL_RATE_FOURTH_POWER"
        res["status"] = "CONDITIONAL"
        res["flags"].append("QAM_CFO_IS_CONDITIONAL")

        amp = np.abs(iq_signal).astype(np.float64)
        w = amp ** 4.0  # Weight corner constellation points
        y = w * ((iq_signal.astype(np.complex128)) ** 4.0)

        n_fft = 2 ** int(np.ceil(np.log2(n * 4)))
        fft_y = np.abs(np.fft.fft(y, n=n_fft))
        freqs = np.fft.fftfreq(n_fft, d=1.0 / fs_val)

        peak_idx = int(np.argmax(fft_y))
        peak_freq = float(freqs[peak_idx])
        est_cfo = peak_freq / 4.0

        mean_spec = np.mean(fft_y)
        prominence = fft_y[peak_idx] / (mean_spec + 1e-12)
        res["confidence"] = "MEDIUM" if prominence > 2.5 else "LOW"

        if sample_rate is not None and sample_rate > 0:
            res["cfo"] = est_cfo
        else:
            res["cfo_normalized"] = est_cfo
            res["flags"].append("METADATA_REQUIRED_FOR_ABSOLUTE_HERTZ")

        return res

    # 3. QAM64 (Unsupported / Not Validated for CFO)
    elif mod == "QAM64":
        res["method"] = "QAM64_CFO_UNSUPPORTED"
        res["status"] = "NOT_VALIDATED"
        res["flags"].append("QAM64_CFO_NOT_VALIDATED")
        res["confidence"] = "UNKNOWN"
        return res

    # 4. FSK (CFO is handled through frequency states/center)
    elif mod in ["GFSK", "CPFSK"]:
        res["method"] = "FSK_CARRIER_TRACKED_VIA_STATES"
        res["status"] = "NOT_APPLICABLE_TO_FSK"
        res["flags"].append("FSK_USES_FREQUENCY_STATES")
        res["confidence"] = "UNKNOWN"
        return res

    else:
        res["method"] = "UNKNOWN_MODULATION_BRANCH"
        res["status"] = "UNKNOWN"
        res["flags"].append("UNSPECIFIED_MODULATION_TYPE")
        return res
