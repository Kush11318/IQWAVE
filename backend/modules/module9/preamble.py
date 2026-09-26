"""Module 9A: Known Preamble Detection.

Scientific Status: LOCKED / VALIDATED
Source: # Module 9 — Full Experimental Report, Section 3 (9A).

Method:
Sliding Hamming-distance comparison across the bitstream.
For candidate bit position k:
    D(k) = sum_{i=0}^{L-1} p_i XOR x_{k+i}

The candidate position k* with minimum Hamming distance is selected.
Validated conditions:
    BER: 0%, 1%, 5%, 10%, 20%.
    Performance is strong through 10% BER, but degrades substantially at 20% BER.

Explicitly Rejected Components (Sections 4.1, 4.2, 4.3):
- Unknown preamble-length recovery: REJECTED (harmonic ambiguity).
- Periodicity-based preamble detection: REJECTED (harmonic candidate ambiguity).
- Harmonic suppression / autocorrelation: REJECTED (favors false short periods).
- Repeated complete-block discovery: REJECTED / ABANDONED (unstable under noise).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Standard 16-bit known preamble validated in controlled experiments
DEFAULT_PREAMBLE_16 = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1], dtype=np.uint8)


def detect_known_preamble(
    bits: Union[np.ndarray, List[int]],
    preamble: Optional[Union[np.ndarray, List[int]]] = None,
    max_positions_to_report: int = 10
) -> Dict[str, Any]:
    """Detect candidate positions of a known preamble sequence via sliding Hamming distance.

    Args:
        bits: 1D array of received binary bits.
        preamble: 1D array of known preamble bits. Defaults to the source 16-bit sequence.
        max_positions_to_report: Number of top candidate positions to include in report.

    Returns:
        Structured dictionary containing detection results, best position,
        Hamming distance, normalized correlation, and scientific status.
    """
    rx = np.asarray(bits, dtype=np.uint8).flatten()
    p = np.asarray(preamble if preamble is not None else DEFAULT_PREAMBLE_16, dtype=np.uint8).flatten()

    L = len(p)
    N = len(rx)

    if L == 0 or N < L:
        return {
            "status": "INSUFFICIENT_DATA",
            "preamble_length": L,
            "stream_length": N,
            "detected": False,
            "best_position": None,
            "hamming_distance": None,
            "normalized_correlation": None,
            "bit_error_rate_estimate": None,
            "candidate_positions": [],
            "scientific_status": "LOCKED (9A: Known Preamble Detection)"
        }

    # Sliding Hamming distance
    n_candidates = N - L + 1
    distances = np.zeros(n_candidates, dtype=np.int32)

    for k in range(n_candidates):
        distances[k] = int(np.sum(p ^ rx[k : k + L]))

    best_idx = int(np.argmin(distances))
    min_dist = int(distances[best_idx])
    norm_corr = float(1.0 - (min_dist / L))
    ber_est = float(min_dist / L)

    # Top candidate positions sorted by distance
    sorted_indices = np.argsort(distances)
    top_candidates = []
    for idx in sorted_indices[:max_positions_to_report]:
        top_candidates.append({
            "position": int(idx),
            "hamming_distance": int(distances[idx]),
            "normalized_correlation": float(1.0 - (distances[idx] / L)),
            "ber_estimate": float(distances[idx] / L)
        })

    # Detection is supported if min distance is within typical channel tolerance (BER <= 0.25)
    is_detected = (ber_est <= 0.25)

    return {
        "status": "SUCCESS",
        "preamble_length": L,
        "stream_length": N,
        "detected": is_detected,
        "best_position": best_idx,
        "hamming_distance": min_dist,
        "normalized_correlation": norm_corr,
        "bit_error_rate_estimate": ber_est,
        "candidate_positions": top_candidates,
        "scientific_status": "LOCKED (9A: Known Preamble Detection)",
        "limitations": "Reliable through 10% BER; degrades substantially at 20% BER under AWGN bit flips."
    }
