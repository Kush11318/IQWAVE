"""Module 5F: Modulation-Specific Demodulators.

Implements hard-decision digital symbol and bit recovery for supported classes:
- BPSK: Sign decision on real axis (real < 0 -> 0, real >= 0 -> 1)
- QPSK: Quadrant decisions (I >= 0 -> 1, Q >= 0 -> 1)
- 8PSK: 8-phase decision with blind RMS normalization
- QAM16: Nearest-constellation decision with blind RMS normalization
- QAM64: Nearest-constellation decision with blind RMS normalization
- CPFSK: Instantaneous frequency with 0-Hz threshold (f < 0 -> 0, f >= 0 -> 1)
- GFSK: Direct Gaussian smoothed instantaneous frequency with sign decision
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import signal


def gaussian_kernel(length: int, std: float) -> np.ndarray:
    """Direct Gaussian filter implementation (independent of scipy.signal.gaussian)."""
    if length <= 1:
        return np.ones(1, dtype=np.float64)
    n = np.arange(length, dtype=np.float64) - (float(length) - 1.0) / 2.0
    sig2 = 2.0 * (std ** 2.0)
    w = np.exp(-n ** 2.0 / (sig2 + 1e-12))
    return w / np.sum(w)


def demodulate_bpsk(symbol_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Demodulate BPSK symbols to hard bits."""
    real_parts = np.real(symbol_samples)
    bits = (real_parts >= 0.0).astype(np.uint8)
    decided_symbols = np.where(bits == 1, 1.0 + 0j, -1.0 + 0j)
    return decided_symbols, bits


def demodulate_qpsk(symbol_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Demodulate QPSK symbols to hard bits using quadrant decisions."""
    b0 = (np.real(symbol_samples) >= 0.0).astype(np.uint8)
    b1 = (np.imag(symbol_samples) >= 0.0).astype(np.uint8)
    bits = np.column_stack((b0, b1)).flatten()
    s_real = np.where(b0 == 1, 1.0, -1.0) / np.sqrt(2.0)
    s_imag = np.where(b1 == 1, 1.0, -1.0) / np.sqrt(2.0)
    decided_symbols = s_real + 1j * s_imag
    return decided_symbols, bits


def demodulate_8psk(symbol_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Demodulate 8PSK symbols with blind RMS normalization and phase slicing."""
    rms = np.sqrt(np.mean(np.abs(symbol_samples) ** 2.0)) + 1e-12
    norm_samples = symbol_samples / rms

    angles = np.angle(norm_samples)  # [-pi, pi]
    angles_pos = (angles + 2.0 * np.pi) % (2.0 * np.pi)

    # 8-state constellation bins centered at k * pi/4
    step = np.pi / 4.0
    k_indices = np.round(angles_pos / step).astype(int) % 8

    # Gray/binary bit mapping for 3 bits
    b0 = ((k_indices >> 2) & 1).astype(np.uint8)
    b1 = ((k_indices >> 1) & 1).astype(np.uint8)
    b2 = (k_indices & 1).astype(np.uint8)
    bits = np.column_stack((b0, b1, b2)).flatten()

    decided_symbols = np.exp(1j * k_indices * step)
    return decided_symbols, bits


def demodulate_qam16(symbol_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Demodulate 16-QAM with blind RMS normalization and nearest constellation slicing."""
    rms = np.sqrt(np.mean(np.abs(symbol_samples) ** 2.0)) + 1e-12
    norm_samples = symbol_samples / rms

    # Standard 16-QAM grid: {-3, -1, +1, +3} / sqrt(10)
    levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)

    # Slicing real and imaginary components
    r = np.real(norm_samples)
    i = np.imag(norm_samples)

    # Closest level index in [0, 1, 2, 3]
    idx_r = np.argmin(np.abs(r[:, None] - levels[None, :]), axis=1)
    idx_i = np.argmin(np.abs(i[:, None] - levels[None, :]), axis=1)

    decided_r = levels[idx_r]
    decided_i = levels[idx_i]
    decided_symbols = decided_r + 1j * decided_i

    # 2 bits per axis (Gray mapped: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3)
    gray_map = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
    bits_r = gray_map[idx_r]
    bits_i = gray_map[idx_i]
    bits = np.column_stack((bits_r[:, 0], bits_r[:, 1], bits_i[:, 0], bits_i[:, 1])).flatten()

    return decided_symbols, bits


def demodulate_qam64(symbol_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Demodulate 64-QAM with blind RMS normalization and nearest constellation slicing."""
    rms = np.sqrt(np.mean(np.abs(symbol_samples) ** 2.0)) + 1e-12
    norm_samples = symbol_samples / rms

    # Standard 64-QAM grid: {-7, -5, -3, -1, +1, +3, +5, +7} / sqrt(42)
    levels = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]) / np.sqrt(42.0)

    r = np.real(norm_samples)
    i = np.imag(norm_samples)

    idx_r = np.argmin(np.abs(r[:, None] - levels[None, :]), axis=1)
    idx_i = np.argmin(np.abs(i[:, None] - levels[None, :]), axis=1)

    decided_r = levels[idx_r]
    decided_i = levels[idx_i]
    decided_symbols = decided_r + 1j * decided_i

    # 3 bits per axis (Gray mapped)
    gray_map = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
        [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]
    ], dtype=np.uint8)

    bits_r = gray_map[idx_r]
    bits_i = gray_map[idx_i]
    bits = np.column_stack((
        bits_r[:, 0], bits_r[:, 1], bits_r[:, 2],
        bits_i[:, 0], bits_i[:, 1], bits_i[:, 2]
    )).flatten()

    return decided_symbols, bits


def demodulate_cpfsk(
    iq_signal: np.ndarray,
    sps: int,
    sample_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Demodulate CPFSK via instantaneous frequency and 0-Hz threshold."""
    if len(iq_signal) < 2:
        return np.array([]), np.array([], dtype=np.uint8)

    # 1. Phase difference -> instantaneous frequency
    diff = iq_signal[1:] * np.conj(iq_signal[:-1])
    inst_phase = np.angle(diff)
    inst_freq = (sample_rate / (2.0 * np.pi)) * inst_phase

    # 2. Smoothing filter (moving average of length sps // 2)
    win_len = max(3, sps // 2)
    smoothed_freq = signal.convolve(inst_freq, np.ones(win_len) / win_len, mode="same")

    # 3. Symbol-center sampling
    half_sps = sps // 2
    sample_indices = np.arange(half_sps, len(smoothed_freq), sps)
    sampled_freq = smoothed_freq[sample_indices]

    # 4. 0-Hz threshold decision
    bits = (sampled_freq >= 0.0).astype(np.uint8)
    decided_symbols = np.where(bits == 1, 1.0 + 0j, -1.0 + 0j)

    return decided_symbols, bits


def demodulate_gfsk(
    iq_signal: np.ndarray,
    sps: int,
    sample_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Demodulate GFSK via direct Gaussian smoothed instantaneous frequency and sign."""
    if len(iq_signal) < 2:
        return np.array([]), np.array([], dtype=np.uint8)

    # 1. Phase difference
    diff = iq_signal[1:] * np.conj(iq_signal[:-1])
    inst_phase = np.angle(diff)
    inst_freq = (sample_rate / (2.0 * np.pi)) * inst_phase

    # 2. Direct Gaussian smoothing (BT ~ 0.5)
    kernel_len = max(5, sps)
    std = float(sps) / 2.5
    g_win = gaussian_kernel(kernel_len, std=std)
    smoothed_freq = signal.convolve(inst_freq, g_win, mode="same")

    # 3. Symbol-center sampling
    half_sps = sps // 2
    sample_indices = np.arange(half_sps, len(smoothed_freq), sps)
    sampled_freq = smoothed_freq[sample_indices]

    # 4. Frequency sign decision
    bits = (sampled_freq >= 0.0).astype(np.uint8)
    decided_symbols = np.where(bits == 1, 1.0 + 0j, -1.0 + 0j)

    return decided_symbols, bits


def demodulate_signal(
    signal_waveform: np.ndarray,
    modulation: str,
    sps: int,
    sample_rate: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Unified entry point for modulation-specific symbol and bit recovery."""
    mod_upper = (modulation or "").upper()
    fs = float(sample_rate) if (sample_rate is not None and sample_rate > 0) else 1000000.0

    if mod_upper == "CPFSK":
        return demodulate_cpfsk(signal_waveform, sps=sps, sample_rate=fs)
    elif mod_upper == "GFSK":
        return demodulate_gfsk(signal_waveform, sps=sps, sample_rate=fs)

    # For PSK/QAM: extract symbol samples at optimal spacing
    # Signal waveform here is already timing-aligned matched-filtered waveform
    if sps > 1:
        # Sample once per symbol
        symbol_samples = signal_waveform[::sps]
    else:
        symbol_samples = signal_waveform

    if mod_upper == "BPSK":
        return demodulate_bpsk(symbol_samples)
    elif mod_upper == "QPSK":
        return demodulate_qpsk(symbol_samples)
    elif mod_upper == "8PSK":
        return demodulate_8psk(symbol_samples)
    elif mod_upper == "QAM16":
        return demodulate_qam16(symbol_samples)
    elif mod_upper == "QAM64":
        return demodulate_qam64(symbol_samples)
    else:
        raise ValueError(f"Unsupported modulation class: {modulation}")
