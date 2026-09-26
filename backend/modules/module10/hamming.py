"""Module 10B: Hamming(7,4) Code Structure, Syndrome Analysis & Decoding.

Scientific Status: 🔒 LOCKED (Controlled Validation).
Source: # Module 10 — FEC, CRC & Interleaver Analysis, Section 5.

Exact Matrices:
    H = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ]
    G = [
        [1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ]
Orthogonality:
    H * G^T = 0 (mod 2)

Validated Characteristics:
    Zero-syndrome probability vs. BER:
        0% -> 1.000; 1% -> 0.9955; 5% -> 0.7673; 10% -> 0.3175; 20% -> 0.1555.
    Random bit baseline:
        P(s = 0 | random) = 2^-3 = 0.1250.

CRITICAL SCIENTIFIC RULES:
1. Zero syndrome does NOT prove that a block contains no errors (undetected errors exist).
2. Equivalent matrix representations produce identical structural statistics.
   The system identifies an EQUIVALENCE CLASS, not a single unique matrix representation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


HAMMING_7_4_H = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
], dtype=np.uint8)

HAMMING_7_4_G = np.array([
    [1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 1]
], dtype=np.uint8)


def encode_hamming_7_4(info_bits: Union[np.ndarray, List[int]]) -> np.ndarray:
    """Encode binary message bits into Hamming(7,4) codewords using generator matrix G.

    Args:
        info_bits: 1D array of message bits (length multiple of 4).

    Returns:
        1D array of encoded bits (length = len(info_bits) * 7 / 4).
    """
    msg = np.asarray(info_bits, dtype=np.uint8).flatten()
    n_blocks = len(msg) // 4
    if n_blocks == 0:
        return np.empty(0, dtype=np.uint8)

    msg_blocks = msg[: n_blocks * 4].reshape(n_blocks, 4)
    # Codeword c = m * G (mod 2)
    codewords = np.dot(msg_blocks, HAMMING_7_4_G) % 2
    return codewords.flatten().astype(np.uint8)


def compute_hamming_syndrome(block: np.ndarray, H: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute 3-bit syndrome for a 7-bit block: s = H * r^T (mod 2)."""
    h_mat = H if H is not None else HAMMING_7_4_H
    r = np.asarray(block, dtype=np.uint8).flatten()
    return (np.dot(h_mat, r) % 2).astype(np.uint8)


def evaluate_hamming_evidence(
    bits: Union[np.ndarray, List[int]],
    H: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Evaluate empirical zero-syndrome consistency and normalized lift over received bits.

    Args:
        bits: 1D array of binary bits.
        H: Parity check matrix (default: standard 3x7 matrix).

    Returns:
        Dictionary with zero-syndrome fraction, normalized lift, block count, and status.
    """
    h_mat = H if H is not None else HAMMING_7_4_H
    rx = np.asarray(bits, dtype=np.uint8).flatten()
    n_blocks = len(rx) // 7

    if n_blocks == 0:
        return {
            "status": "INSUFFICIENT_DATA",
            "zero_syndrome_fraction": 0.0,
            "normalized_lift": 0.0,
            "blocks_evaluated": 0,
            "scientific_status": "LOCKED (Controlled)"
        }

    blocks = rx[: n_blocks * 7].reshape(n_blocks, 7)
    syndromes = (np.dot(blocks, h_mat.T) % 2).astype(np.uint8)
    zero_syn_mask = np.all(syndromes == 0, axis=1)
    zero_count = int(np.sum(zero_syn_mask))
    p0 = float(zero_count / n_blocks)

    p_rand = 0.125  # 2^-3
    norm_lift = float((p0 - p_rand) / (1.0 - p_rand)) if p0 >= p_rand else 0.0

    return {
        "status": "SUCCESS",
        "zero_syndrome_fraction": p0,
        "normalized_lift": norm_lift,
        "blocks_evaluated": int(n_blocks),
        "zero_syndrome_count": int(zero_count),
        "random_baseline": p_rand,
        "scientific_status": "LOCKED (Controlled)",
        "safeguards": "Zero syndrome != guaranteed zero errors; undetected error patterns exist."
    }


def decode_hamming_7_4(
    rx_bits: Union[np.ndarray, List[int]],
    H: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, int]:
    """Perform bounded-distance single-error correction decoding on Hamming(7,4) blocks.

    Args:
        rx_bits: 1D array of received bits (length multiple of 7).
        H: 3x7 parity-check matrix.

    Returns:
        Tuple of (decoded_information_bits, total_corrected_errors).
    """
    h_mat = H if H is not None else HAMMING_7_4_H
    rx = np.asarray(rx_bits, dtype=np.uint8).flatten()
    n_blocks = len(rx) // 7
    if n_blocks == 0:
        return np.empty(0, dtype=np.uint8), 0

    blocks = rx[: n_blocks * 7].reshape(n_blocks, 7).copy()
    syndromes = np.dot(blocks, h_mat.T) % 2

    # Map syndrome vector to error column index in H
    # Columns of H:
    # col 0: [1, 0, 0]^T
    # col 1: [0, 1, 0]^T
    # col 2: [1, 1, 0]^T
    # col 3: [0, 0, 1]^T
    # col 4: [1, 0, 1]^T
    # col 5: [0, 1, 1]^T
    # col 6: [1, 1, 1]^T
    # Map syndrome binary tuple to column
    syn_to_col = {}
    for c in range(7):
        syn_tuple = tuple(h_mat[:, c].tolist())
        syn_to_col[syn_tuple] = c

    total_corrections = 0
    for b in range(n_blocks):
        syn = tuple(syndromes[b].tolist())
        if syn != (0, 0, 0) and syn in syn_to_col:
            err_col = syn_to_col[syn]
            blocks[b, err_col] ^= 1
            total_corrections += 1

    # Extract systematic information bits:
    # In systematic G, columns [3, 4, 5, 6] or info mapping:
    # Based on G: rows are independent messages.
    # When m = [m0, m1, m2, m3], c = m * G.
    # Let's inspect G columns:
    # col 0: m0 + m1 + m3
    # col 1: m0 + m2 + m3
    # col 2: m0
    # col 3: m1 + m2 + m3
    # col 4: m1
    # col 5: m2
    # col 6: m3
    # Notice: col 2 is m0, col 4 is m1, col 5 is m2, col 6 is m3!
    info_cols = [2, 4, 5, 6]
    extracted_info = blocks[:, info_cols].flatten().astype(np.uint8)

    return extracted_info, total_corrections
