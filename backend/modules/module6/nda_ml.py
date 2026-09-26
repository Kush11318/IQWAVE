"""Module 6C & 6C.1: Non-Data-Aided Maximum-Likelihood (NDA-ML) SNR Estimator.

Based on:
    Sun, Gong & Lu (2020), "Non-Data-Aided ML SNR Estimation for BPSK Signals in AWGN".

Scope:
- Strictly BPSK in AWGN conditions.
- Profile log-likelihood optimization over candidate SNR.
- High accuracy (+20 to -5 dB); unstable below -5 dB.
- Incorporates observation-length sensitivity.

Status: CONDITIONAL (Restricted to BPSK AWGN; do not generalize to other modulations).
"""

from typing import Any, Dict, Optional
import numpy as np
from scipy import optimize


def estimate_bpsk_nda_ml(
    real_samples: np.ndarray,
    snr_search_range_db: tuple = (-25.0, 30.0)
) -> Dict[str, Any]:
    """Estimate SNR for BPSK using 1D profile Non-Data-Aided Maximum-Likelihood.

    Args:
        real_samples: Received real-axis symbol or signal samples y[n] = Re(r[n]).
        snr_search_range_db: Bounds (min_db, max_db) for search.

    Returns:
        Dictionary containing estimated SNR in dB, status, observation length, quality, and warnings.
    """
    out: Dict[str, Any] = {
        "snr_db": None,
        "status": "CONDITIONAL",
        "method": "BPSK_NDA_ML_SUN_GONG_LU_2020",
        "observation_length": 0,
        "quality": "UNKNOWN",
        "warnings": []
    }

    if real_samples is None:
        out["warnings"].append("EMPTY_SIGNAL")
        out["quality"] = "LOW"
        return out

    arr = np.asarray(real_samples)
    if arr.size == 0 or arr.dtype == object:
        out["warnings"].append("EMPTY_SIGNAL")
        out["quality"] = "LOW"
        return out

    if not np.all(np.isfinite(arr)):
        out["warnings"].append("NON_FINITE_SAMPLES")
        out["quality"] = "LOW"
        return out

    y = np.real(arr).astype(np.float64)
    n = len(y)
    out["observation_length"] = n

    if n < 64:
        out["warnings"].append("INSUFFICIENT_OBSERVATION_LENGTH")
        out["quality"] = "LOW"
        return out

    p2 = float(np.mean(y ** 2.0))
    if p2 <= 0.0:
        out["warnings"].append("ZERO_SIGNAL_POWER")
        out["quality"] = "LOW"
        return out

    # 1. Coarse 1D profile log-likelihood grid search
    grid_db = np.linspace(snr_search_range_db[0], snr_search_range_db[1], 111)
    best_ll = -1e20
    best_snr_db = 0.0

    for s_db in grid_db:
        rho = 10.0 ** (s_db / 10.0)
        sigma2 = p2 / (1.0 + rho)
        a = np.sqrt(p2 * rho / (1.0 + rho))

        # Numerically stable log cosh: |u| + log1p(exp(-2|u|)) - log(2)
        u = (a * y) / (sigma2 + 1e-12)
        log_cosh = np.abs(u) + np.log1p(np.exp(-2.0 * np.abs(u))) - np.log(2.0)

        ll = -0.5 * n * np.log(2.0 * np.pi * sigma2) - 0.5 * n * (1.0 + 2.0 * rho) + np.sum(log_cosh)
        if ll > best_ll:
            best_ll = ll
            best_snr_db = float(s_db)

    # 2. Fine continuous optimization around peak grid point
    def neg_log_likelihood(cand_db: float) -> float:
        rho_c = 10.0 ** (cand_db / 10.0)
        sigma2_c = p2 / (1.0 + rho_c)
        a_c = np.sqrt(p2 * rho_c / (1.0 + rho_c))
        u_c = (a_c * y) / (sigma2_c + 1e-12)
        log_cosh_c = np.abs(u_c) + np.log1p(np.exp(-2.0 * np.abs(u_c))) - np.log(2.0)
        return float(0.5 * n * np.log(2.0 * np.pi * sigma2_c) + 0.5 * n * (1.0 + 2.0 * rho_c) - np.sum(log_cosh_c))

    fine_res = optimize.minimize_scalar(
        neg_log_likelihood,
        bounds=(best_snr_db - 1.5, best_snr_db + 1.5),
        method="bounded"
    )

    final_snr_db = float(fine_res.x)
    out["snr_db"] = final_snr_db

    # Quality & confidence handling based on SNR regime and observation length
    # Note: Low-SNR instability below -5 dB is experimentally validated in Module 6(2).txt.
    # The N < 25000 condition is an UNVALIDATED_ENGINEERING_HEURISTIC (source notes non-monotonic error vs N).
    if final_snr_db < -5.0:
        out["quality"] = "LOW"
        out["warnings"].append("LOW_SNR_BPSK_NDA_ML_INSTABILITY_BELOW_MINUS_5DB_DEGRADED")
    elif n < 25000:
        out["quality"] = "MEDIUM"
        out["warnings"].append("UNVALIDATED_ENGINEERING_HEURISTIC: SHORT_OBSERVATION_LENGTH_PENALTY")
    else:
        out["quality"] = "HIGH"

    return out
