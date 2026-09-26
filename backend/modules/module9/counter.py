"""Module 9E.4: Counter Localization & Candidate Evidence.

Scientific Status: CONDITIONAL (9E.4).
Source: # Module 9 — Full Experimental Report, Section 13 (9E.4) and Section 14 (9E.4.3).

Method:
Frame-to-Frame Integer Step Consistency.
For candidate window starting at bit column j with bit-width W in {4, 8, 16}:
    Extract integer value for each frame m in [0, M-1]:
        c_m = sum_{b=0}^{W-1} x_{m, j+b} * 2^{W - 1 - b}  (MSB-oriented)
    Compute consecutive transition steps:
        Delta_m = (c_{m+1} - c_m) mod 2^W
    Consistency metric:
        consistency = (1 / (M-1)) * sum_{m=0}^{M-2} I(Delta_m == 1)

Validated Conditions & Behavior (Section 13):
    - True counter region in controlled test: 16–23 bits.
    - 0% BER: consistency = 1.000 (100% detection)
    - 1% BER: consistency = 0.851 (100% detection)
    - 5% BER: consistency = 0.463 (97.33% detection)
    - 10% BER: consistency = 0.228 (70.33% detection, substantially degraded)
    - 20% BER: consistency = 0.160 (10.67% detection, unusable)

CRITICAL SCIENTIFIC CONSTRAINTS:
1. Counter evidence MUST be reported as COUNTER_CANDIDATE, NEVER "CONFIRMED COUNTER".
2. Do NOT hard-code 16–23 bits as the only counter location; search candidate windows.
3. In noisy frames, false counter candidates can emerge in random payload.
4. Counter Transition Signature (9E.4.3) was experimentally REJECTED (failed even at 0% BER).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def detect_counter_candidates(
    frames: np.ndarray,
    candidate_widths: Optional[List[int]] = None,
    min_consistency: float = 0.20,
    max_candidates_to_report: int = 5
) -> Dict[str, Any]:
    """Search for candidate counter fields using frame-to-frame step consistency (9E.4).

    Args:
        frames: 2D binary numpy array of shape (M, P).
        candidate_widths: List of integer bit-widths to test. Default: [4, 8, 16].
        min_consistency: Consistency threshold to consider a window as candidate.
        max_candidates_to_report: Max number of top candidates to report.

    Returns:
        Structured dictionary containing candidate counter regions, consistency scores,
        and scientific diagnostics.
    """
    if not isinstance(frames, np.ndarray) or frames.ndim != 2:
        return {
            "status": "INVALID_INPUT",
            "detected": False,
            "candidates": [],
            "best_candidate": None,
            "scientific_status": "CONDITIONAL (9E.4: Counter Candidate Localization)",
            "limitations": "Input must be 2D numpy array of shape (M, P)."
        }

    M, P = frames.shape
    if M < 3:
        return {
            "status": "INSUFFICIENT_FRAMES",
            "detected": False,
            "candidates": [],
            "best_candidate": None,
            "scientific_status": "CONDITIONAL (9E.4: Counter Candidate Localization)",
            "limitations": "Requires at least 3 frames to evaluate transition consistency."
        }

    widths = candidate_widths if candidate_widths is not None else [4, 8, 16]
    tested_candidates: List[Dict[str, Any]] = []

    # Powers of 2 for fast integer conversion
    for W in widths:
        if W > P or W > 32:
            continue

        powers = 2 ** np.arange(W - 1, -1, -1, dtype=np.int64)
        mod_val = 2 ** W

        for j in range(0, P - W + 1):
            window_bits = frames[:, j : j + W]  # Shape (M, W)
            # Compute integer values for each frame
            values = np.dot(window_bits, powers)  # Shape (M,)

            # Consecutive differences mod 2^W
            diffs = (values[1:] - values[:-1]) % mod_val

            # Fraction of transitions with step == +1
            step1_matches = int(np.sum(diffs == 1))
            consistency = float(step1_matches / (M - 1))

            if consistency >= min_consistency:
                tested_candidates.append({
                    "start_bit": int(j),
                    "end_bit": int(j + W - 1),
                    "bit_width": int(W),
                    "consistency": float(consistency),
                    "step_one_count": int(step1_matches),
                    "total_transitions": int(M - 1),
                    "candidate_type": "COUNTER_CANDIDATE",
                    "orientation": "MSB_FIRST"
                })

    # Sort descending by consistency
    tested_candidates.sort(key=lambda c: c["consistency"], reverse=True)

    # Filter overlapping redundant candidates: prioritize higher consistency
    non_overlapping: List[Dict[str, Any]] = []
    claimed_bits = set()
    for cand in tested_candidates:
        cand_bits = set(range(cand["start_bit"], cand["end_bit"] + 1))
        # If overlap is small (< 50%), keep it
        overlap = cand_bits.intersection(claimed_bits)
        if len(overlap) < (cand["bit_width"] // 2):
            non_overlapping.append(cand)
            claimed_bits.update(cand_bits)
            if len(non_overlapping) >= max_candidates_to_report:
                break

    best_cand = non_overlapping[0] if non_overlapping else None
    detected = (best_cand is not None)

    return {
        "status": "SUCCESS",
        "detected": detected,
        "best_candidate": best_cand,
        "candidates": non_overlapping,
        "scientific_status": "CONDITIONAL (9E.4: Counter Candidate Localization)",
        "safeguard_note": (
            "Reported strictly as COUNTER_CANDIDATE, NOT confirmed counter. "
            "Useful through ~5% BER; degrades significantly at 10-20% BER. "
            "Counter Transition Signature (9E.4.3) was experimentally rejected."
        )
    }
