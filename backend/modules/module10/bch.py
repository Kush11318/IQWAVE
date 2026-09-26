"""Module 10B: BCH(15,7) Code Structure, Generator Polynomial & Decoding.

Scientific Status: 🔒 LOCKED (Controlled Characterization).
Source: # Module 10 — FEC, CRC & Interleaver Analysis, Section 6.

Parameters:
    n = 15, k = 7, t = 2 (corrects up to 2 bit errors)
    g(x) = x^8 + x^7 + x^6 + x^4 + 1
    Polynomial representation: [1, 1, 1, 0, 1, 0, 0, 0, 1]
    Message count: 2^7 = 128 codewords.
    Random zero-syndrome baseline: 1 / 256 ≈ 0.0039.

Validated Characteristics:
    - Clear syndrome lift separation between BCH and Hamming streams.
    - Decoded BER improvement at moderate/high SNR (e.g. 5 dB, 10 dB).
    - Miscorrection / degradation at low SNR (<= 0 dB): decoded BER can exceed raw BER.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


BCH_15_7_G = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1], dtype=np.uint8)


def _build_bch_15_7_codebook() -> Tuple[np.ndarray, np.ndarray]:
    """Generate all 128 valid BCH(15,7) codewords by polynomial multiplication."""
    n_msgs = 128
    messages = np.zeros((n_msgs, 7), dtype=np.uint8)
    codewords = np.zeros((n_msgs, 15), dtype=np.uint8)

    poly = BCH_15_7_G
    deg_g = len(poly) - 1  # 8

    for m in range(n_msgs):
        msg_bits = np.array([(m >> (6 - b)) & 1 for b in range(7)], dtype=np.uint8)
        messages[m] = msg_bits

        # c(x) = m(x) * g(x) mod 2
        # Convolution over GF(2)
        prod = np.convolve(msg_bits, poly) % 2
        # Pad to 15 bits
        c = np.zeros(15, dtype=np.uint8)
        c[: len(prod)] = prod
        codewords[m] = c

    return messages, codewords


BCH_MESSAGES_128, BCH_CODEWORDS_128 = _build_bch_15_7_codebook()


def encode_bch_15_7(info_bits: Union[np.ndarray, List[int]]) -> np.ndarray:
    """Encode binary message bits into BCH(15,7) codewords.

    Args:
        info_bits: 1D array of message bits (length multiple of 7).

    Returns:
        1D array of encoded bits (length = len(info_bits) * 15 / 7).
    """
    msg = np.asarray(info_bits, dtype=np.uint8).flatten()
    n_blocks = len(msg) // 7
    if n_blocks == 0:
        return np.empty(0, dtype=np.uint8)

    encoded_blocks = []
    poly = BCH_15_7_G
    for b in range(n_blocks):
        m_blk = msg[b * 7 : (b + 1) * 7]
        prod = np.convolve(m_blk, poly) % 2
        c = np.zeros(15, dtype=np.uint8)
        c[: len(prod)] = prod
        encoded_blocks.append(c)

    return np.concatenate(encoded_blocks).astype(np.uint8)


def compute_bch_remainder(block_15: np.ndarray) -> np.ndarray:
    """Compute polynomial remainder of 15-bit block divided by g(x) over GF(2)."""
    r = np.asarray(block_15, dtype=np.uint8).flatten()
    poly = BCH_15_7_G
    deg_g = len(poly) - 1  # 8

    rem = r.copy()
    for i in range(len(rem) - deg_g):
        if rem[i] == 1:
            rem[i : i + len(poly)] ^= poly

    return rem[-deg_g:]


def evaluate_bch_evidence(bits: Union[np.ndarray, List[int]]) -> Dict[str, Any]:
    """Evaluate empirical zero-syndrome / remainder consistency and normalized lift.

    Args:
        bits: 1D array of binary bits.

    Returns:
        Dictionary containing zero-remainder fraction, normalized lift, and status.
    """
    rx = np.asarray(bits, dtype=np.uint8).flatten()
    n_blocks = len(rx) // 15

    if n_blocks == 0:
        return {
            "status": "INSUFFICIENT_DATA",
            "zero_syndrome_fraction": 0.0,
            "normalized_lift": 0.0,
            "blocks_evaluated": 0,
            "scientific_status": "LOCKED (Controlled)"
        }

    zero_count = 0
    for b in range(n_blocks):
        blk = rx[b * 15 : (b + 1) * 15]
        rem = compute_bch_remainder(blk)
        if np.all(rem == 0):
            zero_count += 1

    p0 = float(zero_count / n_blocks)
    p_rand = 1.0 / 256.0  # 2^-8

    norm_lift = float((p0 - p_rand) / (1.0 - p_rand)) if p0 >= p_rand else 0.0

    return {
        "status": "SUCCESS",
        "zero_syndrome_fraction": p0,
        "normalized_lift": norm_lift,
        "blocks_evaluated": int(n_blocks),
        "zero_syndrome_count": int(zero_count),
        "random_baseline": p_rand,
        "scientific_status": "LOCKED (Controlled)",
        "safeguards": "High channel error rates cause decoder miscorrection (decoded BER > raw BER)."
    }


def decode_bch_15_7(rx_bits: Union[np.ndarray, List[int]]) -> Tuple[np.ndarray, int]:
    """Decode BCH(15,7) blocks via minimum Hamming distance codebook search (t=2 capability).

    Args:
        rx_bits: 1D array of received bits (length multiple of 15).

    Returns:
        Tuple of (decoded_information_bits, total_corrected_errors).
    """
    rx = np.asarray(rx_bits, dtype=np.uint8).flatten()
    n_blocks = len(rx) // 15
    if n_blocks == 0:
        return np.empty(0, dtype=np.uint8), 0

    decoded_msgs = []
    total_corrections = 0

    for b in range(n_blocks):
        blk = rx[b * 15 : (b + 1) * 15]
        # Distances to all 128 valid codewords
        dists = np.sum(BCH_CODEWORDS_128 ^ blk, axis=1)
        best_idx = int(np.argmin(dists))
        min_dist = int(dists[best_idx])

        total_corrections += min_dist
        decoded_msgs.append(BCH_MESSAGES_128[best_idx])

    return np.concatenate(decoded_msgs).astype(np.uint8), total_corrections
