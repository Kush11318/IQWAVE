"""Module 8A: Hamming (7,4) Code Identification, Alignment, and Decoding.

Implements:
1. Parity-Check Matrix H and Generator Matrix G:
       H = [[1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]]
       G = [[1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, 1]]
   Validation: H @ G.T = 0 (mod 2).
   Systematic basis columns: columns [2, 4, 5, 6] in G are standard unit vectors.
   Information bits m = [c_2, c_4, c_5, c_6].

2. Syndrome Consistency & Normalized Lift:
   - Syndrome: s = H @ r.T (mod 2).
   - Valid blocks produce s = [0, 0, 0]^T.
   - Random baseline: P_rand = 2^(-3) = 0.125.
   - Normalized Lift: Lift_Hamming = P(zero syndrome) / P_rand.

3. Alignment Search:
   - Evaluates cyclic offsets tau in {0, 1, ..., 6}.
   - Selected offset maximizes zero-syndrome fraction.

4. Decoders:
   - Hard Syndrome Decoder: Single-bit correction via syndrome lookup table.
     CRITICAL: At low SNR (<= 5 dB), multi-bit errors cause miscorrection
     where decoded BER exceeds raw BER (experimentally documented in Section 16).
   - Soft ML Decoder: Evaluates soft correlation score across all 16 codewords:
       c_hat = argmax_{c in C} sum_{j=0}^6 s(c_j) * Lambda_j
     where s(1) = +1, s(0) = -1.

Status: LOCKED.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# Exact Parity-Check Matrix H from Section 2
HAMMING_H = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
], dtype=np.uint8)

# Exact Generator Matrix G from Section 2
HAMMING_G = np.array([
    [1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 1]
], dtype=np.uint8)

# Precomputed all 16 messages and codewords
HAMMING_MESSAGES = np.array([
    [(i >> b) & 1 for b in range(4)] for i in range(16)
], dtype=np.uint8)
HAMMING_CODEWORDS = (HAMMING_MESSAGES @ HAMMING_G) % 2  # shape (16, 7)

# Information bit extraction indices
INFO_BIT_INDICES = [2, 4, 5, 6]

# Syndrome to error bit index mapping (1-based syndrome bit pattern to column index 0..6)
SYNDROME_TO_ERROR_BIT: Dict[Tuple[int, int, int], int] = {}
for col in range(7):
    syn_key = (int(HAMMING_H[0, col]), int(HAMMING_H[1, col]), int(HAMMING_H[2, col]))
    SYNDROME_TO_ERROR_BIT[syn_key] = col


def encode_hamming_7_4(messages: np.ndarray) -> np.ndarray:
    """Encode an array of 4-bit messages into 7-bit Hamming codewords.

    Args:
        messages: 1D or 2D binary array of messages (length must be multiple of 4).

    Returns:
        Flattened 1D binary array of codewords.
    """
    m = np.asarray(messages, dtype=np.uint8).flatten()
    n_blocks = len(m) // 4
    if n_blocks == 0:
        return np.array([], dtype=np.uint8)
    m_blocks = m[:n_blocks * 4].reshape(n_blocks, 4)
    codewords = (m_blocks @ HAMMING_G) % 2
    return codewords.flatten()


def compute_hamming_syndromes(bits: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute syndromes and zero-syndrome fraction for contiguous 7-bit blocks.

    Args:
        bits: 1D array of hard bits.

    Returns:
        Tuple of (syndromes array [N, 3], zero_syndrome_fraction).
    """
    n_blocks = len(bits) // 7
    if n_blocks == 0:
        return np.zeros((0, 3), dtype=np.uint8), 0.0

    blocks = bits[:n_blocks * 7].reshape(n_blocks, 7)
    syndromes = (blocks @ HAMMING_H.T) % 2
    zero_syndrome_fraction = float(np.mean(np.all(syndromes == 0, axis=1)))
    return syndromes, zero_syndrome_fraction


def find_hamming_alignment(bits: np.ndarray) -> Tuple[int, float, Dict[int, float]]:
    """Search for optimal codeword alignment tau in {0, 1, ..., 6}.

    Args:
        bits: 1D array of received bits.

    Returns:
        Tuple of (best_offset, best_zero_fraction, dict_of_all_offsets).
    """
    best_offset = 0
    best_frac = -1.0
    all_offsets: Dict[int, float] = {}

    for offset in range(7):
        sub_bits = bits[offset:]
        _, frac = compute_hamming_syndromes(sub_bits)
        all_offsets[offset] = frac
        if frac > best_frac:
            best_frac = frac
            best_offset = offset

    return best_offset, best_frac, all_offsets


def compute_hamming_evidence(
    bits: np.ndarray,
    alignment_offset: Optional[int] = None
) -> Dict[str, Any]:
    """Compute normalized Hamming evidence (lift) and syndrome statistics.

    Args:
        bits: 1D array of hard bits.
        alignment_offset: Optional known/forced offset. If None, alignment is searched.

    Returns:
        Dictionary containing zero_fraction, normalized_lift, and alignment.
    """
    if alignment_offset is None:
        best_offset, zero_frac, all_offsets = find_hamming_alignment(bits)
    else:
        best_offset = alignment_offset
        _, zero_frac = compute_hamming_syndromes(bits[best_offset:])
        all_offsets = {best_offset: zero_frac}

    random_baseline = 0.125  # 2^(-3)
    raw_lift = float(zero_frac / random_baseline) if random_baseline > 0 else 0.0
    normalized_lift = float(max(0.0, (zero_frac - random_baseline) / (1.0 - random_baseline)))

    return {
        "family": "HAMMING",
        "code": "HAMMING_7_4",
        "n": 7,
        "k": 4,
        "alignment_offset": best_offset,
        "zero_syndrome_fraction": zero_frac,
        "random_baseline": random_baseline,
        "raw_lift": raw_lift,
        "normalized_lift": normalized_lift,
        "all_offsets": all_offsets
    }


def decode_hamming_hard(bits: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Perform standard Hamming (7,4) hard syndrome decoding.

    Corrects single-bit errors per block.
    NOTE: Multi-bit errors can cause miscorrections at low SNR (as documented in Section 16).

    Args:
        bits: 1D array of received hard bits (must be aligned to block boundary).

    Returns:
        Tuple of (decoded_information_bits, decoding_metadata).
    """
    n_blocks = len(bits) // 7
    if n_blocks == 0:
        return np.array([], dtype=np.uint8), {"blocks": 0, "corrected_errors": 0}

    blocks = np.copy(bits[:n_blocks * 7].reshape(n_blocks, 7))
    syndromes = (blocks @ HAMMING_H.T) % 2

    corrected_count = 0
    miscorrection_warning = False

    for b_idx in range(n_blocks):
        s = (int(syndromes[b_idx, 0]), int(syndromes[b_idx, 1]), int(syndromes[b_idx, 2]))
        if s != (0, 0, 0):
            err_col = SYNDROME_TO_ERROR_BIT.get(s)
            if err_col is not None:
                blocks[b_idx, err_col] ^= 1
                corrected_count += 1

    # Extract systematic information bits [c_2, c_4, c_5, c_6]
    info_bits = blocks[:, INFO_BIT_INDICES].flatten()

    return info_bits, {
        "blocks": n_blocks,
        "corrected_errors": corrected_count,
        "correction_rate": float(corrected_count / n_blocks) if n_blocks > 0 else 0.0,
        "decoder_type": "HAMMING_HARD"
    }


def decode_hamming_soft_ml(soft_bits: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Perform soft Maximum-Likelihood (ML) decoding over all 16 candidate codewords.

    Evaluates:
        c_hat = argmax_{c in C} sum_{j=0}^6 s(c_j) * Lambda_j
    where s(1) = +1, s(0) = -1.

    Args:
        soft_bits: 1D array of signed soft values / LLRs from Module 7.

    Returns:
        Tuple of (decoded_information_bits, decoding_metadata).
    """
    n_blocks = len(soft_bits) // 7
    if n_blocks == 0:
        return np.array([], dtype=np.uint8), {"blocks": 0, "decoder_type": "HAMMING_SOFT_ML"}

    soft_matrix = soft_bits[:n_blocks * 7].reshape(n_blocks, 7)

    # Polar mapping for all 16 codewords: 1 -> +1.0, 0 -> -1.0
    cw_polar = np.where(HAMMING_CODEWORDS == 1, 1.0, -1.0)  # shape (16, 7)

    # Correlation scores: shape (n_blocks, 16)
    scores = soft_matrix @ cw_polar.T

    best_cw_indices = np.argmax(scores, axis=1)
    best_codewords = HAMMING_CODEWORDS[best_cw_indices]

    # Extract information bits
    info_bits = best_codewords[:, INFO_BIT_INDICES].flatten()

    return info_bits, {
        "blocks": n_blocks,
        "mean_ml_score": float(np.mean(np.max(scores, axis=1))),
        "decoder_type": "HAMMING_SOFT_ML"
    }
