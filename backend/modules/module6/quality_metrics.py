"""Module 6F, 6G & 6H.5: Multi-Estimator Disagreement & Quality Metrics.

Computes:
1. Estimator Disagreement Metrics:
    - range = max(estimates) - min(estimates)
    - std = standard deviation of candidate estimates
    - mad = median absolute deviation
    Experimental correlation in source: r ≈ 0.4165 to 0.617 with estimation error.

2. Error Vector Magnitude (EVM):
    Correlation in source: r ≈ 0.563 with SNR estimation error.

3. Bit Error Rate (BER):
    Only calculated when ground-truth reference bits are explicitly provided.
    Correlation in source: r ≈ 0.520 with error.
    Strictly UNAVAILABLE if unreferenced.

CRITICAL INVARIANTS & SCIENTIFIC BOUNDARIES:
- Disagreement and quality metrics are treated strictly as ENGINEERING QUALITY FEATURES.
- They are NEVER converted into a calibrated probability or Bayesian posterior.
- Numerical decision boundaries (e.g. 2.0 dB or 4.0 dB disagreement, EVM > 0.35, N < 5000)
  are UNVALIDATED_ENGINEERING_HEURISTIC implementations and are NOT experimentally validated
  production standards.
- Synchronization quality is preserved strictly as qualitative status (HIGH / MEDIUM / LOW).
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np


def compute_ber(recovered_bits: Optional[Any], reference_bits: Optional[Any]) -> Optional[float]:
    """Compute Bit Error Rate (BER) when ground-truth reference bits are available.

    Args:
        recovered_bits: Demodulated bit sequence.
        reference_bits: Known ground truth bit sequence.

    Returns:
        Bit error rate (float in [0, 1]) or None if reference is missing.
    """
    if recovered_bits is None or reference_bits is None:
        return None

    r_arr = np.asarray(recovered_bits, dtype=int).ravel()
    ref_arr = np.asarray(reference_bits, dtype=int).ravel()
    min_len = min(len(r_arr), len(ref_arr))
    if min_len == 0:
        return None

    errors = np.sum(r_arr[:min_len] != ref_arr[:min_len])
    return float(errors / min_len)


def compute_estimator_disagreement(estimates: List[float]) -> Dict[str, Optional[float]]:
    """Calculate disagreement metrics (range, std, MAD) across valid candidate estimates."""
    valid = [float(e) for e in estimates if e is not None and np.isfinite(e)]
    if len(valid) == 0:
        return {
            "range": None,
            "range_db": None,
            "std": None,
            "std_db": None,
            "mad": None,
            "mad_db": None,
            "num_candidates": 0
        }
    if len(valid) == 1:
        return {
            "range": 0.0,
            "range_db": 0.0,
            "std": 0.0,
            "std_db": 0.0,
            "mad": 0.0,
            "mad_db": 0.0,
            "num_candidates": 1
        }

    arr = np.array(valid, dtype=np.float64)
    rng = float(np.max(arr) - np.min(arr))
    std = float(np.std(arr))
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))

    return {
        "range": rng,
        "range_db": rng,
        "std": std,
        "std_db": std,
        "mad": mad,
        "mad_db": mad,
        "num_candidates": len(valid)
    }


def assess_snr_quality(
    snr_db: Optional[float],
    disagreement: Optional[Dict[str, Optional[float]]] = None,
    evm: Optional[float] = None,
    ber: Optional[float] = None,
    synchronization_quality: Optional[Union[str, Dict[str, Any]]] = None,
    observation_length: int = 0,
    estimator_status: str = "LOCKED",
    sync_quality: Optional[Union[str, Dict[str, Any]]] = None
) -> str:
    """Derive engineering quality assessment without fabricating probabilities.

    Returns:
        One of: 'HIGH', 'MEDIUM', 'LOW', 'CONDITIONAL', 'UNAVAILABLE'.
    """
    if snr_db is None or not np.isfinite(snr_db):
        return "UNAVAILABLE"

    if estimator_status in ["CONDITIONAL", "NOT_VALIDATED", "IMPLEMENTED_COMPONENTS_FUSION_RULE_UNSPECIFIED"]:
        return "CONDITIONAL"

    # Normalize qualitative sync quality input (HIGH / MEDIUM / LOW / UNKNOWN)
    sync_val = synchronization_quality if synchronization_quality is not None else sync_quality
    sync_is_low = False
    if isinstance(sync_val, dict):
        overall = sync_val.get("overall") or sync_val.get("overall_sync_quality") or "UNKNOWN"
        if str(overall).upper() == "LOW":
            sync_is_low = True
    elif isinstance(sync_val, str):
        if sync_val.upper() == "LOW":
            sync_is_low = True

    # UNVALIDATED_ENGINEERING_HEURISTIC thresholds:
    # Source identifies disagreement, EVM, and low SNR as error indicators,
    # but does NOT validate specific universal numerical decision boundaries.
    is_low_snr = snr_db < 5.0  # UNVALIDATED_ENGINEERING_HEURISTIC
    short_obs = observation_length > 0 and observation_length < 5000  # UNVALIDATED_ENGINEERING_HEURISTIC

    disag_range = None
    if disagreement:
        disag_range = disagreement.get("range_db") or disagreement.get("range")

    high_disag = disag_range is not None and disag_range > 4.0  # UNVALIDATED_ENGINEERING_HEURISTIC
    med_disag = disag_range is not None and disag_range > 2.0   # UNVALIDATED_ENGINEERING_HEURISTIC

    high_evm = evm is not None and evm > 0.35  # UNVALIDATED_ENGINEERING_HEURISTIC

    # Qualitative engineering logic
    if is_low_snr or sync_is_low or high_disag or high_evm:
        return "LOW"
    elif med_disag or short_obs:
        return "MEDIUM"
    else:
        return "HIGH"
