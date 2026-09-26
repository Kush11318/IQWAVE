"""Module 7A: PSK Soft Bit / LLR Generation (BPSK, QPSK, 8PSK).

Implements:
1. BPSK:
   - Analytical LLR derived from AWGN likelihood model: LLR = 4 * A * Re(r) / N0
   - Exact likelihood formulation (numerically identical with diff <= 1e-13).
   - Sign convention: Re(r) >= 0 -> bit 1 (positive LLR), Re(r) < 0 -> bit 0 (negative LLR).
   - Status: LOCKED.

2. QPSK:
   - Constellation: 00 -> (+I,+Q), 01 -> (+I,-Q), 10 -> (-I,+Q), 11 -> (-I,-Q).
   - Corrected analytical equations:
       LLR_0 = -K * Re(r)
       LLR_1 = -K * Im(r)
     where K = 2 * sqrt(2) / N0.
   - Numerically equivalent to exact constellation likelihood (diff <= 1e-13).
   - Status: LOCKED.

3. 8PSK:
   - Natural binary mapping: 000, 001, 010, 011, 100, 101, 110, 111.
   - Exact constellation likelihood evaluated via logsumexp over bit partitions.
   - Preserves unequal bit reliability: Bit 0 (coarse), Bit 1, Bit 2 (fine).
   - Max-Log approximation available as optional candidate only.
   - Status: EXACT_8PSK_LLR_LOCKED.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.special import logsumexp


def compute_bpsk_analytical_llr(
    received_symbols: np.ndarray,
    n0: float,
    amplitude: float = 1.0
) -> np.ndarray:
    """Compute analytical LLR for BPSK under AWGN model.

    Args:
        received_symbols: Synchronized complex or real symbol samples r_k.
        n0: Total noise spectral density / variance N0 > 0.
        amplitude: Nominal symbol amplitude A.

    Returns:
        Array of signed LLR values (positive -> 1, negative -> 0).
    """
    r_real = np.real(received_symbols).astype(np.float64)
    # LLR = 4 * A * Re(r) / N0
    return (4.0 * amplitude * r_real) / float(n0)


def compute_bpsk_exact_llr(
    received_symbols: np.ndarray,
    n0: float,
    amplitude: float = 1.0
) -> np.ndarray:
    """Compute exact constellation likelihood LLR for BPSK.

    b=1 corresponds to s=+A, b=0 corresponds to s=-A.
    """
    r_real = np.real(received_symbols).astype(np.float64)
    # d0^2 = (r - (-A))^2 = (r + A)^2
    # d1^2 = (r - A)^2
    # LLR = (-d1^2 - (-d0^2)) / N0 = (d0^2 - d1^2) / N0 = 4 A r / N0
    d0_sq = (r_real + amplitude) ** 2.0
    d1_sq = (r_real - amplitude) ** 2.0
    return (d0_sq - d1_sq) / float(n0)


def compute_qpsk_analytical_llr(
    received_symbols: np.ndarray,
    n0: float
) -> np.ndarray:
    """Compute corrected analytical LLR for QPSK with source constellation:

    00 -> (+1/sqrt(2), +1/sqrt(2))
    01 -> (+1/sqrt(2), -1/sqrt(2))
    10 -> (-1/sqrt(2), +1/sqrt(2))
    11 -> (-1/sqrt(2), -1/sqrt(2))

    Returns:
        Flattened array of signed LLRs interleaving [LLR_0, LLR_1].
    """
    r = np.asarray(received_symbols, dtype=np.complex128)
    k = (2.0 * np.sqrt(2.0)) / float(n0)
    llr_0 = -k * np.real(r)
    llr_1 = -k * np.imag(r)
    return np.column_stack((llr_0, llr_1)).flatten()


def compute_qpsk_exact_llr(
    received_symbols: np.ndarray,
    n0: float
) -> np.ndarray:
    """Compute exact constellation likelihood LLR for QPSK."""
    r = np.asarray(received_symbols, dtype=np.complex128)
    inv_n0 = 1.0 / float(n0)

    # Constellation definitions
    pts = np.array([
        (1.0 + 1j) / np.sqrt(2.0),   # 00
        (1.0 - 1j) / np.sqrt(2.0),   # 01
        (-1.0 + 1j) / np.sqrt(2.0),  # 10
        (-1.0 - 1j) / np.sqrt(2.0)   # 11
    ], dtype=np.complex128)

    # Distances squared: shape (N, 4)
    dist_sq = np.abs(r[:, None] - pts[None, :]) ** 2.0
    exp_terms = -dist_sq * inv_n0

    # Bit 0: b0=1 is indices [2, 3]; b0=0 is indices [0, 1]
    llr_0 = logsumexp(exp_terms[:, [2, 3]], axis=1) - logsumexp(exp_terms[:, [0, 1]], axis=1)

    # Bit 1: b1=1 is indices [1, 3]; b1=0 is indices [0, 2]
    llr_1 = logsumexp(exp_terms[:, [1, 3]], axis=1) - logsumexp(exp_terms[:, [0, 2]], axis=1)

    return np.column_stack((llr_0, llr_1)).flatten()


def compute_8psk_exact_llr(
    received_symbols: np.ndarray,
    n0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact constellation likelihood LLR for 8PSK with natural binary mapping:

    000 -> 0*pi/4, 001 -> 1*pi/4, 010 -> 2*pi/4, 011 -> 3*pi/4,
    100 -> 4*pi/4, 101 -> 5*pi/4, 110 -> 6*pi/4, 111 -> 7*pi/4.

    Returns:
        Tuple of (flattened_llrs, per_bit_llrs_matrix [N, 3]).
    """
    r = np.asarray(received_symbols, dtype=np.complex128)
    inv_n0 = 1.0 / float(n0)

    k_indices = np.arange(8)
    pts = np.exp(1j * k_indices * (np.pi / 4.0))

    # Bit masks for natural binary
    # Bit 0 (MSB): k in [4, 5, 6, 7]
    s0_1 = [4, 5, 6, 7]
    s0_0 = [0, 1, 2, 3]

    # Bit 1 (Middle): k in [2, 3, 6, 7]
    s1_1 = [2, 3, 6, 7]
    s1_0 = [0, 1, 4, 5]

    # Bit 2 (LSB): k in [1, 3, 5, 7]
    s2_1 = [1, 3, 5, 7]
    s2_0 = [0, 2, 4, 6]

    dist_sq = np.abs(r[:, None] - pts[None, :]) ** 2.0
    exp_terms = -dist_sq * inv_n0

    llr_0 = logsumexp(exp_terms[:, s0_1], axis=1) - logsumexp(exp_terms[:, s0_0], axis=1)
    llr_1 = logsumexp(exp_terms[:, s1_1], axis=1) - logsumexp(exp_terms[:, s1_0], axis=1)
    llr_2 = logsumexp(exp_terms[:, s2_1], axis=1) - logsumexp(exp_terms[:, s2_0], axis=1)

    llr_matrix = np.column_stack((llr_0, llr_1, llr_2))
    return llr_matrix.flatten(), llr_matrix


def compute_8psk_maxlog_llr(
    received_symbols: np.ndarray,
    n0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optional Max-Log LLR approximation for 8PSK."""
    r = np.asarray(received_symbols, dtype=np.complex128)
    inv_n0 = 1.0 / float(n0)

    k_indices = np.arange(8)
    pts = np.exp(1j * k_indices * (np.pi / 4.0))

    s0_1 = [4, 5, 6, 7]
    s0_0 = [0, 1, 2, 3]

    s1_1 = [2, 3, 6, 7]
    s1_0 = [0, 1, 4, 5]

    s2_1 = [1, 3, 5, 7]
    s2_0 = [0, 2, 4, 6]

    dist_sq = np.abs(r[:, None] - pts[None, :]) ** 2.0
    exp_terms = -dist_sq * inv_n0

    llr_0 = np.max(exp_terms[:, s0_1], axis=1) - np.max(exp_terms[:, s0_0], axis=1)
    llr_1 = np.max(exp_terms[:, s1_1], axis=1) - np.max(exp_terms[:, s1_0], axis=1)
    llr_2 = np.max(exp_terms[:, s2_1], axis=1) - np.max(exp_terms[:, s2_0], axis=1)

    llr_matrix = np.column_stack((llr_0, llr_1, llr_2))
    return llr_matrix.flatten(), llr_matrix
