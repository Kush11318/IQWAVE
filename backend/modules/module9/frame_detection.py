"""Module 9C: Repeated Frame Detection & Period Estimation.

Scientific Status: LOCKED through 10% BER under tested conditions (9C.1).
Source: # Module 9 — Full Experimental Report, Section 5 (9C) and Section 6 (9C.1).

Method:
Consistent Repeated Positions via Longest Arithmetic Chain.
Given candidate positions where preamble or repeated structure matches:
Evaluates candidate frame periods P in a search bound (default: 100–180 bits, derived from experiment).
For each candidate period P, finds the longest arithmetic chain of consistent positions:
    k_0, k_0 + P, k_0 + 2P, ..., k_0 + (M-1)*P

The period P* with the maximum chain length is selected.

Validated Conditions:
    BER: 0%, 1%, 5%, 10% (robust: 100% accuracy through 5%, 99.67% at 10%).
    Degradation: at 20% BER, period accuracy drops to 43.67% and mean chain length to 5.02.

Implementation Note:
    The 100–180 bit period search range is an implementation search bound derived
    from the controlled experiment. It is NOT a universal scientifically validated range.

Explicitly Rejected Methods (Section 5):
    - Median position differences period estimator: REJECTED / FAILED under noise
      (e.g., true period=144, initial estimate was 59).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .preamble import DEFAULT_PREAMBLE_16


def detect_frame_period(
    bits: Union[np.ndarray, List[int]],
    preamble: Optional[Union[np.ndarray, List[int]]] = None,
    candidate_periods: Optional[Tuple[int, int]] = (100, 180),
    max_ber_threshold: float = 0.25
) -> Dict[str, Any]:
    """Estimate frame period using the validated longest arithmetic chain method (9C.1).

    Args:
        bits: 1D array of binary bits.
        preamble: Known preamble bit pattern. Defaults to standard 16-bit sequence.
        candidate_periods: Tuple of (min_period, max_period) to search. Defaults to (100, 180).
        max_ber_threshold: Maximum tolerated BER for a candidate preamble match at candidate step.

    Returns:
        Structured dictionary containing estimated period, chain length, candidate positions,
        and scientific diagnostics.
    """
    rx = np.asarray(bits, dtype=np.uint8).flatten()
    p = np.asarray(preamble if preamble is not None else DEFAULT_PREAMBLE_16, dtype=np.uint8).flatten()

    L = len(p)
    N = len(rx)

    if candidate_periods is None:
        p_min, p_max = 100, 180
    else:
        p_min, p_max = candidate_periods

    if N < p_min + L or L == 0:
        return {
            "status": "INSUFFICIENT_DATA",
            "frame_period": None,
            "chain_length": 0,
            "detected": False,
            "search_bounds": [p_min, p_max],
            "chain_positions": [],
            "scientific_status": "LOCKED through 10% BER (9C.1: Arithmetic Chain)",
            "limitations": "Insufficient bitstream length for search bound."
        }

    # 1. Compute Hamming distances at all candidate positions
    n_candidates = N - L + 1
    distances = np.zeros(n_candidates, dtype=np.int32)
    for k in range(n_candidates):
        distances[k] = int(np.sum(p ^ rx[k : k + L]))

    max_dist = int(np.floor(L * max_ber_threshold))

    # Candidate positions where distance <= max_dist, or local minima
    cand_set = set()
    for k in range(n_candidates):
        if distances[k] <= max_dist:
            cand_set.add(k)

    # In case cand_set is too sparse (high noise), include top local minima
    if len(cand_set) < 3:
        sorted_k = np.argsort(distances)
        for k in sorted_k[:max(10, min(50, n_candidates))]:
            cand_set.add(int(k))

    sorted_cand_positions = sorted(list(cand_set))

    # 2. Search for the longest arithmetic chain for each P in [p_min, p_max]
    best_period: Optional[int] = None
    best_chain: List[int] = []
    max_chain_len = 0

    actual_p_max = min(p_max, N - L)

    for P in range(p_min, actual_p_max + 1):
        # Check arithmetic chains starting at each candidate position
        for start_pos in sorted_cand_positions:
            current_chain = [start_pos]
            current_pos = start_pos + P
            while current_pos + L <= N:
                # Check if current_pos has a matching preamble
                dist = int(np.sum(p ^ rx[current_pos : current_pos + L]))
                if dist <= max_dist:
                    current_chain.append(current_pos)
                    current_pos += P
                else:
                    # Allow one candidate position tolerance check within +/- 1 bit jitter if needed
                    # But the experimental method uses strict arithmetic grid k + m*P
                    break

            if len(current_chain) > max_chain_len:
                max_chain_len = len(current_chain)
                best_chain = current_chain
                best_period = P

    # Period detection confidence requires at least 2 consecutive frames in chain
    detected = (best_period is not None and max_chain_len >= 2)

    return {
        "status": "SUCCESS" if detected else "NO_REPEATED_PERIOD_FOUND",
        "frame_period": best_period,
        "chain_length": max_chain_len,
        "detected": detected,
        "search_bounds": [p_min, p_max],
        "search_bounds_note": (
            "The 100-180 bit range is an implementation search bound derived from the "
            "controlled experiment, not a universal scientifically validated range."
        ),
        "chain_positions": best_chain,
        "scientific_status": "LOCKED through 10% BER (9C.1: Arithmetic Chain)",
        "limitations": "Degrades substantially at 20% BER (accuracy ~43.67%)."
    }
