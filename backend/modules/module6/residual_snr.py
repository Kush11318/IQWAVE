"""Module 6D & 6E: Decision-Directed Residual SNR Estimators and EVM.

Given received symbol-rate samples r_k and detected constellation symbols s_hat_k:
    e_k = r_k - s_hat_k

1. Ordinary Residual (6D):
    P_e = E[|e_k|^2]
    P_s = E[|s_hat_k|^2]
    SNR = 10 * log10( P_s / P_e )
    Status: SUPPORTING (degrades heavily at low SNR due to decision errors).

2. Robust Residual (6E):
    P_n_robust ≈ median(|e_k|^2) / ln(2)
    SNR_robust = 10 * log10( P_s / P_n_robust )
    Status: CONDITIONAL / SUPPORTING (not universally superior).

3. Error Vector Magnitude (EVM):
    EVM = sqrt( E[|r_k - s_hat_k|^2] ) / sqrt( E[|s_hat_k|^2] )
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


def compute_evm(
    received_symbols: np.ndarray,
    detected_symbols: np.ndarray
) -> Optional[float]:
    """Calculate Error Vector Magnitude (EVM) as a unitless ratio."""
    if len(received_symbols) == 0 or len(detected_symbols) == 0:
        return None
    e = received_symbols - detected_symbols
    num = np.sqrt(np.mean(np.abs(e) ** 2.0))
    den = np.sqrt(np.mean(np.abs(detected_symbols) ** 2.0))
    if den <= 0.0:
        return None
    return float(num / den)


def estimate_residual_snr(
    received_symbols: np.ndarray,
    detected_symbols: np.ndarray,
    modulation: Optional[str] = None
) -> Dict[str, Any]:
    """Compute ordinary residual SNR, robust median residual SNR, and EVM.

    Args:
        received_symbols: Synchronized symbol-rate samples r_k.
        detected_symbols: Decided constellation points s_hat_k.
        modulation: Modulation string.

    Returns:
        Dictionary containing ordinary SNR, robust SNR, EVM, and quality flags.
    """
    out: Dict[str, Any] = {
        "ordinary_snr_db": None,
        "robust_snr_db": None,
        "evm": None,
        "error_power": None,
        "signal_power": None,
        "num_symbols": 0,
        "warnings": []
    }

    if received_symbols is None or detected_symbols is None or len(received_symbols) == 0:
        out["warnings"].append("EMPTY_SYMBOLS")
        return out

    if len(received_symbols) != len(detected_symbols):
        out["warnings"].append("SYMBOL_LENGTH_MISMATCH")
        return out

    if not np.all(np.isfinite(received_symbols)) or not np.all(np.isfinite(detected_symbols)):
        out["warnings"].append("NON_FINITE_SAMPLES")
        return out

    r = np.asarray(received_symbols, dtype=np.complex128)
    s_hat = np.asarray(detected_symbols, dtype=np.complex128)
    n = len(r)
    out["num_symbols"] = n

    # For QAM: apply blind RMS normalization to align constellation lattices
    mod_upper = (modulation or "").upper()
    if mod_upper in ["QAM16", "QAM64"]:
        rms_r = np.sqrt(np.mean(np.abs(r) ** 2.0)) + 1e-12
        rms_s = np.sqrt(np.mean(np.abs(s_hat) ** 2.0)) + 1e-12
        r = r / rms_r
        s_hat = s_hat / rms_s

    e = r - s_hat
    p_signal = float(np.mean(np.abs(s_hat) ** 2.0))
    out["signal_power"] = p_signal

    if p_signal <= 0.0:
        out["warnings"].append("ZERO_SIGNAL_POWER")
        return out

    # 1. Ordinary Decision-Directed Residual (6D)
    p_error = float(np.mean(np.abs(e) ** 2.0))
    out["error_power"] = p_error

    if p_error > 0.0:
        ord_snr = 10.0 * np.log10(p_signal / p_error)
        out["ordinary_snr_db"] = float(ord_snr)
    else:
        out["ordinary_snr_db"] = 50.0  # Clean perfect match

    # 2. Robust Median-Based Residual (6E)
    # Under complex circular Gaussian noise, |e|^2 is exponentially distributed with mean sigma^2 = median / ln(2)
    squared_mag = np.abs(e) ** 2.0
    median_sq = float(np.median(squared_mag))
    p_noise_robust = median_sq / np.log(2.0)

    if p_noise_robust > 0.0:
        rob_snr = 10.0 * np.log10(p_signal / p_noise_robust)
        out["robust_snr_db"] = float(rob_snr)
    else:
        out["robust_snr_db"] = 50.0

    # 3. EVM
    evm_val = compute_evm(r, s_hat)
    out["evm"] = evm_val

    # Nested structured dictionaries
    out["ordinary_residual"] = {
        "snr_db": out["ordinary_snr_db"],
        "status": "SUPPORTING"
    }
    out["robust_residual"] = {
        "snr_db": out["robust_snr_db"],
        "status": "CONDITIONAL" if mod_upper in ["QAM16", "QAM64"] else "SUPPORTING"
    }

    # Warning on low-SNR decision contamination
    primary_est = out["robust_snr_db"] if out["robust_snr_db"] is not None else out["ordinary_snr_db"]
    if primary_est is not None and primary_est < 5.0:
        out["warnings"].append("DECISION_DIRECTED_BIAS_AT_LOW_SNR")

    return out
