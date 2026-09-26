"""Module 7B: QAM Soft Bit / LLR Generation (QAM16, QAM64).

Implements:
1. QAM16:
   - Constellation: 16-QAM square grid {-3, -1, +1, +3} / sqrt(10).
   - Gray mapping matching Module 5:
     b0, b1 for I axis: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
     b2, b3 for Q axis: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
   - Exact constellation likelihood via logsumexp over bit partitions:
       LLR_k = log( sum_{s in S_{k,1}} exp(-|r-s|^2 / N0) )
             - log( sum_{s in S_{k,0}} exp(-|r-s|^2 / N0) )
   - Preserves unequal bit reliability: coarse bits (|LLR| ~ 100 at 20dB),
     fine bits (|LLR| ~ 36 at 20dB).
   - Status: EXACT_QAM16_LLR_LOCKED.
   - Max-Log retained as conditional optimization candidate.

2. QAM64:
   - Constellation: 64-QAM square grid {-7, -5, -3, -1, +1, +3, +5, +7} / sqrt(42).
   - Gray mapping matching Module 5: 3 bits per axis (coarse, middle, fine).
   - Exact constellation likelihood via logsumexp over bit partitions.
   - Preserves unequal bit reliability: coarse (|LLR| ~ 71 at 20dB),
     middle (|LLR| ~ 17 at 20dB), fine (|LLR| ~ 7 at 20dB).
   - Status: EXACT_QAM64_LLR_LOCKED.
   - Max-Log retained as conditional optimization candidate (diverges from exact at lower SNR).
"""

from typing import Tuple
import numpy as np
from scipy.special import logsumexp


def get_qam16_constellation() -> Tuple[np.ndarray, np.ndarray]:
    """Return 16-QAM complex constellation points and bit mapping [16, 4]."""
    levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64) / np.sqrt(10.0)
    # Gray mapping: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
    gray_map = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)

    pts = []
    bit_labels = []
    for i_idx, r_val in enumerate(levels):
        for q_idx, i_val in enumerate(levels):
            pts.append(r_val + 1j * i_val)
            bits = np.concatenate([gray_map[i_idx], gray_map[q_idx]])
            bit_labels.append(bits)

    return np.array(pts, dtype=np.complex128), np.array(bit_labels, dtype=np.uint8)


def compute_qam16_exact_llr(
    received_symbols: np.ndarray,
    n0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact constellation likelihood LLR for 16-QAM.

    Args:
        received_symbols: Array of synchronized complex symbol samples.
        n0: Total noise spectral density / variance N0 > 0.

    Returns:
        Tuple of (flattened_llrs, per_bit_llrs_matrix [N, 4]).
    """
    r = np.asarray(received_symbols, dtype=np.complex128)
    if len(r) == 0:
        return np.array([], dtype=np.float64), np.zeros((0, 4), dtype=np.float64)

    inv_n0 = 1.0 / float(n0)
    pts, bit_labels = get_qam16_constellation()  # pts: (16,), bit_labels: (16, 4)

    # Distances squared: shape (N, 16)
    dist_sq = np.abs(r[:, None] - pts[None, :]) ** 2.0
    exp_terms = -dist_sq * inv_n0

    llr_cols = []
    for bit_idx in range(4):
        mask_1 = np.where(bit_labels[:, bit_idx] == 1)[0]
        mask_0 = np.where(bit_labels[:, bit_idx] == 0)[0]
        col_llr = logsumexp(exp_terms[:, mask_1], axis=1) - logsumexp(exp_terms[:, mask_0], axis=1)
        llr_cols.append(col_llr)

    llr_matrix = np.column_stack(llr_cols)
    return llr_matrix.flatten(), llr_matrix


def compute_qam16_maxlog_llr(
    received_symbols: np.ndarray,
    n0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Max-Log LLR approximation for 16-QAM (candidate only)."""
    r = np.asarray(received_symbols, dtype=np.complex128)
    if len(r) == 0:
        return np.array([], dtype=np.float64), np.zeros((0, 4), dtype=np.float64)

    inv_n0 = 1.0 / float(n0)
    pts, bit_labels = get_qam16_constellation()

    dist_sq = np.abs(r[:, None] - pts[None, :]) ** 2.0
    exp_terms = -dist_sq * inv_n0

    llr_cols = []
    for bit_idx in range(4):
        mask_1 = np.where(bit_labels[:, bit_idx] == 1)[0]
        mask_0 = np.where(bit_labels[:, bit_idx] == 0)[0]
        col_llr = np.max(exp_terms[:, mask_1], axis=1) - np.max(exp_terms[:, mask_0], axis=1)
        llr_cols.append(col_llr)

    llr_matrix = np.column_stack(llr_cols)
    return llr_matrix.flatten(), llr_matrix


def get_qam64_constellation() -> Tuple[np.ndarray, np.ndarray]:
    """Return 64-QAM complex constellation points and bit mapping [64, 6]."""
    levels = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float64) / np.sqrt(42.0)
    # Gray mapping matching Module 5
    gray_map = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
        [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]
    ], dtype=np.uint8)

    pts = []
    bit_labels = []
    for i_idx, r_val in enumerate(levels):
        for q_idx, i_val in enumerate(levels):
            pts.append(r_val + 1j * i_val)
            bits = np.concatenate([gray_map[i_idx], gray_map[q_idx]])
            bit_labels.append(bits)

    return np.array(pts, dtype=np.complex128), np.array(bit_labels, dtype=np.uint8)


def compute_qam64_exact_llr(
    received_symbols: np.ndarray,
    n0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact constellation likelihood LLR for 64-QAM.

    Args:
        received_symbols: Array of synchronized complex symbol samples.
        n0: Total noise spectral density / variance N0 > 0.

    Returns:
        Tuple of (flattened_llrs, per_bit_llrs_matrix [N, 6]).
    """
    r = np.asarray(received_symbols, dtype=np.complex128)
    if len(r) == 0:
        return np.array([], dtype=np.float64), np.zeros((0, 6), dtype=np.float64)

    inv_n0 = 1.0 / float(n0)
    pts, bit_labels = get_qam64_constellation()

    dist_sq = np.abs(r[:, None] - pts[None, :]) ** 2.0
    exp_terms = -dist_sq * inv_n0

    llr_cols = []
    for bit_idx in range(6):
        mask_1 = np.where(bit_labels[:, bit_idx] == 1)[0]
        mask_0 = np.where(bit_labels[:, bit_idx] == 0)[0]
        col_llr = logsumexp(exp_terms[:, mask_1], axis=1) - logsumexp(exp_terms[:, mask_0], axis=1)
        llr_cols.append(col_llr)

    llr_matrix = np.column_stack(llr_cols)
    return llr_matrix.flatten(), llr_matrix


def compute_qam64_maxlog_llr(
    received_symbols: np.ndarray,
    n0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Max-Log LLR approximation for 64-QAM (candidate only)."""
    r = np.asarray(received_symbols, dtype=np.complex128)
    if len(r) == 0:
        return np.array([], dtype=np.float64), np.zeros((0, 6), dtype=np.float64)

    inv_n0 = 1.0 / float(n0)
    pts, bit_labels = get_qam64_constellation()

    dist_sq = np.abs(r[:, None] - pts[None, :]) ** 2.0
    exp_terms = -dist_sq * inv_n0

    llr_cols = []
    for bit_idx in range(6):
        mask_1 = np.where(bit_labels[:, bit_idx] == 1)[0]
        mask_0 = np.where(bit_labels[:, bit_idx] == 0)[0]
        col_llr = np.max(exp_terms[:, mask_1], axis=1) - np.max(exp_terms[:, mask_0], axis=1)
        llr_cols.append(col_llr)

    llr_matrix = np.column_stack(llr_cols)
    return llr_matrix.flatten(), llr_matrix
