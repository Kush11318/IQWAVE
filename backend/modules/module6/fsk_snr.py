"""Module 6J: FSK SNR Estimation & Frequency-Residual Benchmarking.

Primary Estimator:
- Fourth-moment NDA power estimator is LOCKED for tested AWGN CPFSK and GFSK conditions:
    CPFSK MAE ≈ 0.102029 dB
    GFSK MAE ≈ 0.099736 dB

Rejected Estimator:
- FSK Frequency-Residual Estimator:
    Attempts to estimate SNR from instantaneous-frequency variance across FSK states.
    Severely fails to track physical SNR:
        CPFSK MAE ≈ 3.107 dB
        GFSK MAE ≈ 6.379 dB
    Status: REJECTED as a primary SNR estimator (retained as supporting feature only).
"""

from typing import Any, Dict, Optional
import numpy as np

from .fourth_moment import estimate_fourth_moment_snr


def compute_fsk_frequency_residual_benchmark(
    iq_signal: np.ndarray,
    sps: int = 4,
    sample_rate: float = 1000000.0,
    fdev_hz: float = 100000.0
) -> Dict[str, Any]:
    """Compute rejected FSK frequency-residual baseline for benchmark verification.

    Status: REJECTED as a primary SNR estimator.
    """
    out: Dict[str, Any] = {
        "snr_db": None,
        "frequency_variance": None,
        "status": "REJECTED_AS_PRIMARY",
        "method": "FSK_FREQUENCY_RESIDUAL_BASELINE",
        "documented_mae_db": 6.379,
        "reason": "HIGH BIAS: Frequency residual cannot track physical SNR (MAE ≈ 3.107 dB CPFSK, 6.379 dB GFSK).",
        "warnings": ["FREQUENCY_RESIDUAL_CANNOT_TRACK_PHYSICAL_SNR"]
    }

    if iq_signal is None or len(iq_signal) < 2:
        out["warnings"].append("EMPTY_OR_SHORT_SIGNAL")
        return out

    # Instantaneous frequency
    diff = iq_signal[1:] * np.conj(iq_signal[:-1])
    inst_freq = (sample_rate / (2.0 * np.pi)) * np.angle(diff)

    # Variance around nominal deviation levels
    freq_err = np.abs(inst_freq) - fdev_hz
    var_err = float(np.var(freq_err))
    out["frequency_variance"] = var_err

    if var_err > 0.0:
        snr_est = 10.0 * np.log10((fdev_hz ** 2.0) / (var_err + 1e-12))
        out["snr_db"] = float(snr_est)
    else:
        out["snr_db"] = 50.0

    return out


def estimate_fsk_snr(
    iq_signal: np.ndarray,
    modulation: str,
    sps: int = 4,
    sample_rate: Optional[float] = None,
    fdev_hz: Optional[float] = None
) -> Dict[str, Any]:
    """Primary FSK SNR estimation using validated Fourth-Moment estimator.

    Status: LOCKED for tested AWGN conditions.
    """
    mod_upper = (modulation or "").upper()
    fs = float(sample_rate) if (sample_rate is not None and sample_rate > 0) else 1000000.0
    fdev = float(fdev_hz) if (fdev_hz is not None and fdev_hz > 0) else 100000.0

    # 1. Primary: Fourth-moment estimator
    fm_res = estimate_fourth_moment_snr(iq_signal, modulation=mod_upper)

    # 2. Supporting: Frequency residual feature (for quality/diagnostics only)
    freq_res = compute_fsk_frequency_residual_benchmark(
        iq_signal,
        sps=sps,
        sample_rate=fs,
        fdev_hz=fdev
    )

    out: Dict[str, Any] = {
        "snr_db": fm_res.get("snr_db"),
        "primary_method": "FOURTH_MOMENT_NDA",
        "primary_estimator": "fourth_moment",
        "status": "LOCKED_FOR_TESTED_AWGN",
        "fourth_moment_output": fm_res,
        "frequency_residual_feature": freq_res,
        "frequency_residual": freq_res,
        "warnings": fm_res.get("warnings", [])
    }

    return out
