"""Module 7D: Reliability Evaluation & Diagnostics.

Implements:
1. Magnitude Binning:
   Divides |soft| into diagnostic intervals:
     - [0, 1)
     - [1, 2)
     - [2, 4)
     - [4, 8)
     - > 8
   Counts, percentages, and (if ground truth bits are supplied) empirical BER per bin.
   NOTE: Reliability bins are diagnostic verification metrics, NOT confidence thresholds.

2. Per-Bit Reliability:
   Computes mean |soft| per bit index for multi-bit constellations (8PSK, QAM16, QAM64)
   to expose unequal bit protection (coarse vs fine bit positions).

3. Hard Decision Agreement:
   Verifies sign convention: soft > 0 -> 1, soft < 0 -> 0 against upstream hard decisions.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def compute_reliability_summary(
    soft_bits: np.ndarray,
    ground_truth_bits: Optional[np.ndarray] = None,
    per_bit_matrix: Optional[np.ndarray] = None,
    reference_hard_bits: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compute diagnostic reliability summary and binning for soft bit values.

    Args:
        soft_bits: 1D array of signed soft bit values.
        ground_truth_bits: Optional ground-truth transmitted bits for diagnostic BER.
        per_bit_matrix: Optional [N, bits_per_sym] array for per-bit position stats.
        reference_hard_bits: Optional upstream Module 5 hard bits to verify agreement.

    Returns:
        Dictionary containing magnitude bins, per-bit stats, and hard decisions.
    """
    if len(soft_bits) == 0:
        return {
            "total_bits": 0,
            "mean_abs_soft": 0.0,
            "magnitude_bins": {},
            "per_bit_mean_abs": [],
            "hard_decision_agreement": None,
            "notes": "Empty input"
        }

    abs_soft = np.abs(soft_bits)
    total_bits = len(soft_bits)

    # Derived hard decisions: positive -> 1, negative -> 0
    derived_hard_bits = (soft_bits > 0.0).astype(np.uint8)

    # Hard decision agreement against reference
    hard_agreement = None
    if reference_hard_bits is not None and len(reference_hard_bits) > 0:
        comp_len = min(len(derived_hard_bits), len(reference_hard_bits))
        match_count = int(np.sum(derived_hard_bits[:comp_len] == reference_hard_bits[:comp_len]))
        hard_agreement = {
            "compared_bits": comp_len,
            "matching_bits": match_count,
            "agreement_rate": float(match_count / comp_len) if comp_len > 0 else 0.0
        }

    # Bins definition: [0, 1), [1, 2), [2, 4), [4, 8), > 8
    bins = [
        ("bin_0_to_1", 0.0, 1.0),
        ("bin_1_to_2", 1.0, 2.0),
        ("bin_2_to_4", 2.0, 4.0),
        ("bin_4_to_8", 4.0, 8.0),
        ("bin_gt_8", 8.0, float("inf")),
    ]

    bin_results = {}
    gt = np.asarray(ground_truth_bits, dtype=np.uint8) if ground_truth_bits is not None else None

    for bin_name, low, high in bins:
        if high == float("inf"):
            mask = abs_soft >= low
        else:
            mask = (abs_soft >= low) & (abs_soft < high)

        count = int(np.sum(mask))
        pct = float((count / total_bits) * 100.0)

        bin_info: Dict[str, Any] = {
            "count": count,
            "percentage": pct,
        }

        # If ground truth is available, compute empirical BER in this bin
        if gt is not None and len(gt) > 0:
            comp_len = min(len(gt), len(derived_hard_bits))
            mask_valid = mask[:comp_len]
            bin_count_gt = int(np.sum(mask_valid))
            if bin_count_gt > 0:
                errors = int(np.sum(derived_hard_bits[:comp_len][mask_valid] != gt[:comp_len][mask_valid]))
                bin_info["empirical_ber"] = float(errors / bin_count_gt)
                bin_info["errors"] = errors
            else:
                bin_info["empirical_ber"] = None
                bin_info["errors"] = 0

        bin_results[bin_name] = bin_info

    # Per-bit position statistics if matrix provided
    per_bit_stats = []
    if per_bit_matrix is not None and len(per_bit_matrix) > 0:
        n_cols = per_bit_matrix.shape[1]
        for col in range(n_cols):
            col_vals = np.abs(per_bit_matrix[:, col])
            per_bit_stats.append({
                "bit_index": col,
                "mean_abs_soft": float(np.mean(col_vals)),
                "std_abs_soft": float(np.std(col_vals)),
            })

    return {
        "total_bits": total_bits,
        "mean_abs_soft": float(np.mean(abs_soft)),
        "magnitude_bins": bin_results,
        "per_bit_stats": per_bit_stats,
        "hard_decision_agreement": hard_agreement,
        "status_note": "Diagnostic reliability verification metrics (NOT confidence thresholds)."
    }
