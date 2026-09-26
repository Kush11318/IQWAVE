"""Module 10C: LDPC Code Analysis, Syndrome Statistics & Candidate Ranking.

Scientific Status: 🔒 LOCKED (Controlled Validation Model).
Source: # Module 10 — FEC, CRC & Interleaver Analysis, Section 8.

Controlled Parity-Check Matrix (6 x 12):
    H = [
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    ]
Characteristics:
    - 6 parity checks, 12 variable bits, k = 6 info bits (64 valid codewords).
    - Zero-syndrome probability vs. BER:
      0% -> 1.000; 1% -> 0.8908; 5% -> 0.5324; 10% -> 0.2828; 20% -> 0.0780; 30% -> 0.0250.
    - Identification: Aggregating multiple observations substantially improves candidate ranking.
      1% BER -> few observations; 5% BER -> tens; 10% BER -> ~50; 20% BER -> hundreds.

CRITICAL SCIENTIFIC CONSTRAINTS:
1. This is a small controlled validation model, NOT a universal LDPC solver.
2. Universal blind LDPC reconstruction from noisy data is EXPLICITLY REJECTED.
3. Syndrome weight != bit error count.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


LDPC_CONTROLLED_H = np.array([
    [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
], dtype=np.uint8)


def _generate_valid_ldpc_codewords() -> np.ndarray:
    """Find all 64 valid codewords for the controlled 6x12 LDPC matrix."""
    all_vectors = np.array([
        [(i >> (11 - b)) & 1 for b in range(12)] for i in range(4096)
    ], dtype=np.uint8)
    syndromes = np.dot(all_vectors, LDPC_CONTROLLED_H.T) % 2
    valid_mask = np.all(syndromes == 0, axis=1)
    return all_vectors[valid_mask]


LDPC_VALID_CODEWORDS_64 = _generate_valid_ldpc_codewords()


def evaluate_ldpc_syndrome(
    bits: Union[np.ndarray, List[int]],
    H: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Evaluate empirical zero-syndrome probability and mean syndrome weight over 12-bit blocks.

    Args:
        bits: 1D array of binary bits.
        H: Parity check matrix (default: controlled 6x12 matrix).

    Returns:
        Dictionary with zero-syndrome fraction, mean syndrome weight, block count, and status.
    """
    h_mat = H if H is not None else LDPC_CONTROLLED_H
    m_checks, n_vars = h_mat.shape

    rx = np.asarray(bits, dtype=np.uint8).flatten()
    n_blocks = len(rx) // n_vars

    if n_blocks == 0:
        return {
            "status": "INSUFFICIENT_DATA",
            "zero_syndrome_fraction": 0.0,
            "mean_syndrome_weight": float(m_checks),
            "blocks_evaluated": 0,
            "scientific_status": "LOCKED (Controlled Validation Model)"
        }

    blocks = rx[: n_blocks * n_vars].reshape(n_blocks, n_vars)
    syndromes = np.dot(blocks, h_mat.T) % 2

    zero_mask = np.all(syndromes == 0, axis=1)
    zero_count = int(np.sum(zero_mask))
    p0 = float(zero_count / n_blocks)

    weights = np.sum(syndromes, axis=1)
    mean_weight = float(np.mean(weights))

    return {
        "status": "SUCCESS",
        "zero_syndrome_fraction": p0,
        "mean_syndrome_weight": mean_weight,
        "blocks_evaluated": int(n_blocks),
        "zero_syndrome_count": int(zero_count),
        "random_zero_baseline": float(2.0 ** (-m_checks)),
        "scientific_status": "LOCKED (Controlled Validation Model, Section 8)",
        "safeguards": "Controlled 6x12 validation model; universal blind LDPC recovery is rejected."
    }


def rank_ldpc_candidates(
    bits: Union[np.ndarray, List[int]],
    candidate_matrices: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Rank candidate LDPC parity-check matrices via aggregate syndrome consistency (10C.8.2).

    Args:
        bits: 1D array of binary bits.
        candidate_matrices: List of candidate dicts with 'name' and 'matrix'.

    Returns:
        Structured dictionary with ranked candidates and identification confidence.
    """
    cands = candidate_matrices
    if cands is None:
        # Generate competing candidate matrices by perturbing rows
        cands = [
            {"name": "LDPC_CONTROLLED_6x12", "matrix": LDPC_CONTROLLED_H}
        ]
        rng = np.random.RandomState(42)
        for i in range(1, 4):
            H_alt = LDPC_CONTROLLED_H.copy()
            # Flip 2 random bits in parity matrix to create competing invalid candidate
            r1, c1 = rng.randint(0, 6), rng.randint(0, 12)
            r2, c2 = rng.randint(0, 6), rng.randint(0, 12)
            H_alt[r1, c1] ^= 1
            H_alt[r2, c2] ^= 1
            cands.append({"name": f"LDPC_PERMUTED_{i}", "matrix": H_alt})

    results = []
    for cand in cands:
        H_mat = np.asarray(cand["matrix"], dtype=np.uint8)
        ev = evaluate_ldpc_syndrome(bits, H=H_mat)
        results.append({
            "candidate": cand["name"],
            "zero_syndrome_fraction": ev["zero_syndrome_fraction"],
            "mean_syndrome_weight": ev["mean_syndrome_weight"],
            "blocks_evaluated": ev["blocks_evaluated"]
        })

    # Sort descending by zero_syndrome_fraction (or ascending by mean_syndrome_weight)
    results.sort(key=lambda x: (x["zero_syndrome_fraction"], -x["mean_syndrome_weight"]), reverse=True)
    best = results[0] if results else None

    return {
        "status": "SUCCESS",
        "best_candidate": best["candidate"] if best else None,
        "candidates": results,
        "scientific_status": "LOCKED for candidate-pool ranking (Section 8.2)",
        "limitation_note": (
            "Observation aggregation required at higher BER (few at 1%, tens at 5%, ~50 at 10%, "
            "hundreds at 20%). Scalability limited to candidate pools."
        )
    }
