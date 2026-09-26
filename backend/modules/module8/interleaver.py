"""Module 8D: Structured Row-Column Interleaver Width Estimation and Deinterleaving.

Implements:
1. Structured Row-Column Interleaver:
   - Data is organized into a matrix of shape (R, W).
   - Writing occurs by rows; transmission occurs by columns.
   - Deinterleaving: received sequence is reshaped into (W, R) and transposed.

2. Candidate Width Search:
   - Evaluates candidate widths W in {2, 3, ..., W_max}.
   - For each candidate W, performs row-column deinterleaving and evaluates
     code syndrome consistency (e.g., Hamming or BCH zero-syndrome fraction).
   - Selected width maximizes parity consistency.

3. Rejection Boundaries:
   - Arbitrary permutation recovery is EXPLICITLY REJECTED (underdetermined due to code symmetries).
   - Unknown general interleaver phase is EXPLICITLY REJECTED (parameterization ambiguity).

Status: LOCKED (restricted to structured row-column model).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .hamming import compute_hamming_syndromes
from .bch import compute_bch_syndromes


def deinterleave_row_column(
    stream: np.ndarray,
    width: int
) -> np.ndarray:
    """Deinterleave a row-column interleaved stream of width W.

    Data was written by rows into (R, W) and read by columns.
    Deinterleaving writes into (W, R) and reads by rows (or transposes).

    Args:
        stream: 1D array of bits or soft values.
        width: Interleaver matrix width W.

    Returns:
        Deinterleaved 1D array of length (R * W).
    """
    w = int(width)
    if w <= 1 or len(stream) < w:
        return np.copy(stream)

    r_depth = len(stream) // w
    valid_len = r_depth * w
    trimmed = stream[:valid_len]

    # In transmission, column j of length r_depth was transmitted sequentially
    # Reshaping into (w, r_depth) and transposing recovers the original (r_depth, w) rows
    mat = trimmed.reshape(w, r_depth)
    return mat.T.flatten()


def interleave_row_column(
    stream: np.ndarray,
    width: int
) -> np.ndarray:
    """Interleave a stream by writing into (R, W) by rows and reading by columns.

    Args:
        stream: 1D array of bits or soft values.
        width: Interleaver matrix width W.

    Returns:
        Interleaved 1D array.
    """
    w = int(width)
    if w <= 1 or len(stream) < w:
        return np.copy(stream)

    r_depth = len(stream) // w
    valid_len = r_depth * w
    trimmed = stream[:valid_len]
    mat = trimmed.reshape(r_depth, w)
    return mat.T.flatten()


def search_interleaver_width(
    bits: np.ndarray,
    fec_family: str = "HAMMING",
    candidate_widths: Optional[List[int]] = None,
    min_syndrome_threshold: float = 0.5
) -> Dict[str, Any]:
    """Search for candidate row-column interleaver width W using syndrome consistency.

    Args:
        bits: 1D array of received hard bits.
        fec_family: 'HAMMING' (block length 7) or 'BCH' (block length 15).
        candidate_widths: Optional list of widths to search. Defaults to [2..16].
        min_syndrome_threshold: Threshold above random baseline to declare interleaving.

    Returns:
        Dictionary containing identified width, confidence, and per-candidate scores.
    """
    widths = candidate_widths or list(range(2, 17))
    mod_family = str(fec_family).upper()

    scores: Dict[int, float] = {}
    best_w: Optional[int] = None
    best_score = -1.0

    # Also evaluate un-deinterleaved baseline
    if mod_family == "BCH":
        _, baseline_score = compute_bch_syndromes(bits)
        random_baseline = 1.0 / 256.0
    else:
        _, baseline_score = compute_hamming_syndromes(bits)
        random_baseline = 0.125

    for w in widths:
        if len(bits) < w * 2:
            continue
        deint_bits = deinterleave_row_column(bits, width=w)
        if mod_family == "BCH":
            _, score = compute_bch_syndromes(deint_bits)
        else:
            _, score = compute_hamming_syndromes(deint_bits)

        scores[w] = float(score)
        if score > best_score:
            best_score = score
            best_w = w

    # Determine if interleaving is present:
    # If the un-deinterleaved baseline is already high (e.g. > 0.6), signal is likely not interleaved.
    # If best_score is substantially higher than baseline_score and exceeds threshold:
    is_interleaved = False
    if best_score > baseline_score + 0.2 and best_score >= min_syndrome_threshold:
        is_interleaved = True

    return {
        "is_interleaved": is_interleaved,
        "interleaver_type": "ROW_COLUMN" if is_interleaved else "NONE",
        "estimated_width": best_w if is_interleaved else None,
        "best_syndrome_score": float(best_score),
        "baseline_uninterleaved_score": float(baseline_score),
        "random_baseline": float(random_baseline),
        "candidate_scores": scores,
        "rejected_approaches": [
            "Arbitrary permutation recovery is REJECTED (underdetermined).",
            "General unknown interleaver phase recovery is REJECTED (parameterization ambiguity)."
        ]
    }
