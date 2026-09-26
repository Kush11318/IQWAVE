"""Module 10D: Structured Row-Column Interleaver & Blind Width Identification.

Scientific Status: 🔒 LOCKED for structured row-column model (Section 10D, 11).
Source: # Module 10 — FEC, CRC & Interleaver Analysis, Section 10D & Section 11.

Method:
- Forward Interleaver (R rows, C columns):
    Write row-wise across C columns, read column-wise down R rows.
    interleaved = matrix.T.reshape(-1)
- Inverse Deinterleaver:
    recovered = interleaved.reshape(cols, rows).T.reshape(-1)
- Blind Width Identification:
    True width: 20
    Candidate widths: {5, 10, 20, 25, 50}
    Evaluates downstream Hamming(7,4) syndrome consistency after deinterleaving.
    At 10% BER: width 20 score = 0.51, false candidates = 0.83–0.85 (Rank 1).

CRITICAL SCIENTIFIC CONSTRAINT:
    Arbitrary permutation interleaver recovery is EXPLICITLY REJECTED.
    The source validates candidate width search for the structured row-column model only.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .hamming import evaluate_hamming_evidence


def interleave_row_column(bits: Union[np.ndarray, List[int]], width: int) -> np.ndarray:
    """Interleave binary bits using structured row-column matrix transpose.

    Args:
        bits: 1D array of binary bits.
        width: Number of columns C.

    Returns:
        1D interleaved binary array.
    """
    b = np.asarray(bits, dtype=np.uint8).flatten()
    C = int(width)
    if C <= 0:
        return b

    R = len(b) // C
    if R <= 0:
        return b

    matrix = b[: R * C].reshape(R, C)
    # Read column-wise: transpose and flatten
    interleaved = matrix.T.reshape(-1)

    # Append any trailing remainder bits
    if len(b) > R * C:
        interleaved = np.concatenate([interleaved, b[R * C :]])

    return interleaved.astype(np.uint8)


def deinterleave_row_column(bits: Union[np.ndarray, List[int]], width: int) -> np.ndarray:
    """Deinterleave binary bits using exact inverse row-column matrix transpose.

    Args:
        bits: 1D array of interleaved bits.
        width: Interleaver width C (columns).

    Returns:
        1D deinterleaved binary array.
    """
    b = np.asarray(bits, dtype=np.uint8).flatten()
    C = int(width)
    if C <= 0:
        return b

    R = len(b) // C
    if R <= 0:
        return b

    # Interleaved data has C columns transposed into R rows of length C,
    # so reshaped as (C, R), then transposed back to (R, C) and flattened
    reshaped = b[: C * R].reshape(C, R)
    recovered = reshaped.T.reshape(-1)

    if len(b) > C * R:
        recovered = np.concatenate([recovered, b[C * R :]])

    return recovered.astype(np.uint8)


VALIDATED_INTERLEAVER_CANDIDATE_WIDTHS: List[int] = [5, 10, 20, 25, 50]


def identify_interleaver_width(
    rx_bits: Union[np.ndarray, List[int]],
    candidate_widths: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Identify interleaver width using downstream Hamming(7,4) syndrome consistency (10D.11).

    The scientifically validated candidate set is strictly {5, 10, 20, 25, 50}.
    Width 1 represents an identity (no-interleaving) baseline and is never ranked as an interleaver candidate.

    Args:
        rx_bits: 1D array of received bits.
        candidate_widths: List of candidate widths to evaluate (default: [5, 10, 20, 25, 50]).

    Returns:
        Structured dictionary containing candidate width rankings, scores, and status.
    """
    raw_cands = candidate_widths if candidate_widths is not None else VALIDATED_INTERLEAVER_CANDIDATE_WIDTHS
    # Filter scientific candidates (w > 1); width=1 cannot be a scientific interleaver candidate
    cands = [int(w) for w in raw_cands if int(w) > 1]
    rx = np.asarray(rx_bits, dtype=np.uint8).flatten()

    identity_baseline: Optional[Dict[str, Any]] = None
    if candidate_widths is not None and 1 in [int(w) for w in candidate_widths]:
        ev1 = evaluate_hamming_evidence(rx)
        identity_baseline = {
            "description": "IDENTITY / NO-INTERLEAVING BASELINE",
            "width": 1,
            "non_zero_syndrome_score": float(1.0 - ev1["zero_syndrome_fraction"]),
            "zero_syndrome_fraction": float(ev1["zero_syndrome_fraction"]),
            "normalized_lift": float(ev1["normalized_lift"]),
            "blocks_evaluated": int(ev1["blocks_evaluated"]),
            "is_scientific_candidate": False
        }

    results: List[Dict[str, Any]] = []

    for w in cands:
        deint = deinterleave_row_column(rx, width=w)
        # Evaluate downstream Hamming syndrome consistency
        ev = evaluate_hamming_evidence(deint)
        # Score is fraction of blocks with NON-ZERO syndrome (lower is better, matching source report)
        non_zero_syn_rate = 1.0 - ev["zero_syndrome_fraction"]
        lift = ev["normalized_lift"]

        results.append({
            "width": int(w),
            "non_zero_syndrome_score": float(non_zero_syn_rate),
            "zero_syndrome_fraction": float(ev["zero_syndrome_fraction"]),
            "normalized_lift": float(lift),
            "blocks_evaluated": int(ev["blocks_evaluated"])
        })

    # Sort ascending by non_zero_syndrome_score (descending by zero_syndrome_fraction)
    results.sort(key=lambda x: x["non_zero_syndrome_score"])

    best = results[0] if results else None
    second = results[1] if len(results) > 1 else None
    margin = float(second["non_zero_syndrome_score"] - best["non_zero_syndrome_score"]) if (best and second) else 0.0

    ret: Dict[str, Any] = {
        "status": "SUCCESS",
        "best_width": best["width"] if best else None,
        "best_score": best["non_zero_syndrome_score"] if best else 1.0,
        "margin": margin,
        "candidates": results,
        "scientific_status": "LOCKED for tested structured row-column model (Section 11)",
        "safeguards": (
            "Arbitrary permutation interleaver recovery is REJECTED. "
            "Scientifically validated candidate set is strictly {5, 10, 20, 25, 50}. "
            "Width 1 is an identity / no-interleaving baseline and does not participate in candidate ranking."
        )
    }

    if identity_baseline is not None:
        ret["identity_baseline"] = identity_baseline

    return ret


def deinterleave_convolutional(
    bits: Union[np.ndarray, List[int]],
    branches: int = 4,
    delay_step: int = 2
) -> np.ndarray:
    """Convolutional (Forney/Ramsey) de-interleaver (Task iii).
    
    Branch i (0 <= i < branches) applies a FIFO delay of (branches - 1 - i) * delay_step symbols.
    """
    b = np.asarray(bits, dtype=np.uint8).flatten()
    M = max(1, int(branches))
    D = max(1, int(delay_step))
    out = np.zeros(len(b), dtype=np.uint8)

    delays = [(M - 1 - i) * D for i in range(M)]
    fifos = [np.zeros(d, dtype=np.uint8) if d > 0 else None for d in delays]

    for idx, bit in enumerate(b):
        branch = idx % M
        d = delays[branch]
        if d == 0:
            out[idx] = bit
        else:
            q = fifos[branch]
            out[idx] = q[0]
            fifos[branch] = np.roll(q, -1)
            fifos[branch][-1] = bit

    return out


def deinterleave_diagonal(
    bits: Union[np.ndarray, List[int]],
    rows: int = 4,
    cols: int = 8
) -> np.ndarray:
    """Diagonal matrix de-interleaver (Task iii).
    
    Reads along diagonal paths (row + col) % cols.
    """
    b = np.asarray(bits, dtype=np.uint8).flatten()
    R = max(1, int(rows))
    C = max(1, int(cols))
    block_size = R * C
    n_blocks = len(b) // block_size
    if n_blocks == 0:
        return b

    out = np.zeros(len(b), dtype=np.uint8)
    for blk in range(n_blocks):
        sub = b[blk * block_size : (blk + 1) * block_size].reshape(R, C)
        rec = np.zeros((R, C), dtype=np.uint8)
        for r in range(R):
            for c in range(C):
                rec[r, c] = sub[r, (c + r) % C]
        out[blk * block_size : (blk + 1) * block_size] = rec.flatten()

    if len(b) > n_blocks * block_size:
        out[n_blocks * block_size :] = b[n_blocks * block_size :]

    return out


def deinterleave_pseudorandom(
    bits: Union[np.ndarray, List[int]],
    seed: int = 42,
    block_size: int = 64
) -> np.ndarray:
    """Pseudo-random permutation de-interleaver (Task iii).
    
    Applies inverse pseudo-random permutation generated via reproducible PRNG seed.
    """
    b = np.asarray(bits, dtype=np.uint8).flatten()
    B = max(2, int(block_size))
    n_blocks = len(b) // B
    if n_blocks == 0:
        return b

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(B)
    inv_perm = np.zeros(B, dtype=int)
    inv_perm[perm] = np.arange(B)

    out = np.zeros(len(b), dtype=np.uint8)
    for blk in range(n_blocks):
        sub = b[blk * B : (blk + 1) * B]
        out[blk * B : (blk + 1) * B] = sub[inv_perm]

    if len(b) > n_blocks * B:
        out[n_blocks * B :] = b[n_blocks * B :]

    return out


def evaluate_all_interleaver_types(
    rx_bits: Union[np.ndarray, List[int]]
) -> Dict[str, Any]:
    """Evaluate Block, Convolutional, Diagonal, and Pseudo-Random de-interleavers (Task iii)."""
    rx = np.asarray(rx_bits, dtype=np.uint8).flatten()

    # 1. Block (Row-Column)
    block_res = identify_interleaver_width(rx)

    # 2. Convolutional
    conv_bits = deinterleave_convolutional(rx, branches=4, delay_step=2)
    ev_conv = evaluate_hamming_evidence(conv_bits)

    # 3. Diagonal
    diag_bits = deinterleave_diagonal(rx, rows=4, cols=8)
    ev_diag = evaluate_hamming_evidence(diag_bits)

    # 4. Pseudo-random
    prn_bits = deinterleave_pseudorandom(rx, seed=42, block_size=64)
    ev_prn = evaluate_hamming_evidence(prn_bits)

    types = [
        {"type": "BLOCK_ROW_COLUMN", "config": f"Width {block_res.get('best_width', 20)}", "score": float(block_res.get("best_score", 0.51))},
        {"type": "CONVOLUTIONAL", "config": "Branches=4, Delay=2", "score": float(1.0 - ev_conv["zero_syndrome_fraction"])},
        {"type": "DIAGONAL", "config": "4x8 Matrix Grid", "score": float(1.0 - ev_diag["zero_syndrome_fraction"])},
        {"type": "PSEUDO_RANDOM", "config": "Block 64, PRN Seed 42", "score": float(1.0 - ev_prn["zero_syndrome_fraction"])}
    ]
    types.sort(key=lambda x: x["score"])

    return {
        "status": "SUCCESS",
        "top_interleaver_type": types[0]["type"],
        "top_configuration": types[0]["config"],
        "evaluated_interleavers": types,
        "block_analysis": block_res
    }

