"""Module 7C: FSK Soft Bit Generation (CPFSK, GFSK).

Implements:
1. CPFSK:
   - Primary: Real waveform-correlation soft metric comparing received symbol waveform
     against candidate bit-1 and bit-0 hypotheses:
       Lambda_k = Re{<r_k, s_1>} - Re{<r_k, s_0>}
   - Positive -> bit 1, Negative -> bit 0.
   - Status: LOCKED (controlled-model validation; NOT mathematically calibrated LLR).
   - Metric type: SOFT_METRIC.
   - Mathematical status: NOT_CALIBRATED_LLR.
   - Calibrated LLR: False.
   - Strictly NO 2/N0 scaling applied.
   - Strictly NO magnitude operation (|...|) used.

2. GFSK:
   - Primary: Real waveform-correlation soft metric against Gaussian-filtered (BT=0.5)
     candidate waveform templates s_1 and s_0:
       Lambda_k = Re{<r_k, s_1>} - Re{<r_k, s_0>}
   - Status: LOCKED (controlled-model validation).
   - Metric type: SOFT_METRIC.
   - Mathematical status: NOT_CALIBRATED_LLR.
   - Calibrated LLR: False.
   - Strictly NO 2/N0 scaling applied.
   - Strictly NO magnitude operation (|...|) used.

3. Rejected Baseline Comparators (retained for diagnostic verification):
   - Instantaneous frequency soft metric (all-sample average)
   - Symbol-center instantaneous frequency metric
   - Two-state frequency difference metric
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy import signal


def generate_cpfsk_templates(
    sps: int,
    sample_rate: float,
    h: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate candidate symbol waveform templates s_1 and s_0 for CPFSK.

    Args:
        sps: Samples per symbol.
        sample_rate: Sampling frequency in Hz.
        h: Modulation index (default 0.5).

    Returns:
        Tuple of (s1_template, s0_template) each of length sps.
    """
    rs = float(sample_rate) / float(sps)
    delta_f = (float(h) * rs) / 2.0
    t = np.arange(sps, dtype=np.float64) / float(sample_rate)

    s1 = np.exp(1j * 2.0 * np.pi * delta_f * t)
    s0 = np.exp(-1j * 2.0 * np.pi * delta_f * t)

    # Normalize templates to unit energy
    s1 /= np.linalg.norm(s1) + 1e-12
    s0 /= np.linalg.norm(s0) + 1e-12

    return s1, s0


def generate_gfsk_templates(
    sps: int,
    sample_rate: float,
    h: float = 0.5,
    bt: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate candidate symbol waveform templates s_1 and s_0 for GFSK with Gaussian pulse (BT=0.5).

    Args:
        sps: Samples per symbol.
        sample_rate: Sampling frequency in Hz.
        h: Modulation index (default 0.5).
        bt: Bandwidth-time product (default 0.5).

    Returns:
        Tuple of (s1_template, s0_template) each of length sps.
    """
    rs = float(sample_rate) / float(sps)
    delta_f = (float(h) * rs) / 2.0

    # Gaussian frequency pulse shape over 3 symbol span
    span_symbols = 3
    t_pulse = np.linspace(-span_symbols / 2.0, span_symbols / 2.0, span_symbols * sps)
    sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * float(bt))
    g = np.exp(-0.5 * (t_pulse / sigma) ** 2.0)
    g /= np.sum(g)

    # Unit pulse frequency deviation
    f_pulse_1 = np.zeros(span_symbols * sps)
    center_idx = span_symbols // 2
    f_pulse_1[center_idx * sps : (center_idx + 1) * sps] = delta_f
    f_shaped_1 = signal.convolve(f_pulse_1, g, mode="same")

    f_pulse_0 = -f_pulse_1
    f_shaped_0 = signal.convolve(f_pulse_0, g, mode="same")

    # Extract central symbol window
    f_sym_1 = f_shaped_1[center_idx * sps : (center_idx + 1) * sps]
    f_sym_0 = f_shaped_0[center_idx * sps : (center_idx + 1) * sps]

    phase_1 = 2.0 * np.pi * np.cumsum(f_sym_1) / float(sample_rate)
    phase_0 = 2.0 * np.pi * np.cumsum(f_sym_0) / float(sample_rate)

    s1 = np.exp(1j * phase_1)
    s0 = np.exp(1j * phase_0)

    s1 /= np.linalg.norm(s1) + 1e-12
    s0 /= np.linalg.norm(s0) + 1e-12

    return s1, s0


def compute_fsk_waveform_correlation_soft(
    iq_signal: np.ndarray,
    sps: int,
    sample_rate: float,
    modulation: str = "CPFSK",
    h: float = 0.5,
    bt: float = 0.5,
    phase_reference: bool = True
) -> np.ndarray:
    """Compute primary waveform-correlation soft metric for CPFSK / GFSK.

    Formula (Approved Blueprint & Source):
        Lambda_k = Re{<r_k, s_1>} - Re{<r_k, s_0>}

    Sign convention:
        Lambda_k > 0 -> bit 1
        Lambda_k < 0 -> bit 0

    NO magnitude operation |...| is used.
    NO 2/N0 scaling is applied (source specifies uncalibrated soft metric).

    Args:
        iq_signal: Complex baseband IQ waveform samples.
        sps: Samples per symbol.
        sample_rate: Sampling frequency in Hz.
        modulation: 'CPFSK' or 'GFSK'.
        h: Modulation index (default 0.5).
        bt: Bandwidth-time product (default 0.5 for GFSK).
        phase_reference: When True, accounts for the continuous-phase state at the
            symbol boundary by referencing candidate templates to the starting phase
            angle(r_k[0]).

    Returns:
        1D array of soft values (length = N_symbols).
    """
    mod_upper = str(modulation).upper()
    if mod_upper == "GFSK":
        s1, s0 = generate_gfsk_templates(sps=sps, sample_rate=sample_rate, h=h, bt=bt)
    else:
        s1, s0 = generate_cpfsk_templates(sps=sps, sample_rate=sample_rate, h=h)

    n_samples = len(iq_signal)
    n_symbols = n_samples // sps
    if n_symbols == 0:
        return np.array([], dtype=np.float64)

    # Truncate to integer symbols and reshape (n_symbols, sps)
    iq_trimmed = iq_signal[: n_symbols * sps].reshape(n_symbols, sps)

    if phase_reference:
        # Reference the received symbol to its initial boundary sample phase
        # s_k = s * exp(j * angle(r_k[0]))  <=>  r_k_aligned = r_k * exp(-j * angle(r_k[0]))
        init_phase = np.angle(iq_trimmed[:, 0])
        r_eval = iq_trimmed * np.exp(-1j * init_phase)[:, None]
    else:
        r_eval = iq_trimmed

    # Real correlation against candidate hypotheses: Re{<r_k, s1>} - Re{<r_k, s0>}
    c1 = np.real(np.sum(r_eval * np.conj(s1)[None, :], axis=1))
    c0 = np.real(np.sum(r_eval * np.conj(s0)[None, :], axis=1))

    # Raw real correlation difference without 2/N0 scaling
    soft_metric = c1 - c0
    return soft_metric.astype(np.float64)


# ---------------------------------------------------------------------------
# Rejected Baseline Comparators (Retained for diagnostics only)
# ---------------------------------------------------------------------------

def compute_rejected_instantaneous_freq_metric(
    iq_signal: np.ndarray,
    sps: int,
    sample_rate: float
) -> np.ndarray:
    """Rejected baseline: All-sample instantaneous frequency average per symbol."""
    n_symbols = len(iq_signal) // sps
    if n_symbols == 0 or len(iq_signal) < 2:
        return np.array([], dtype=np.float64)

    diff = iq_signal[1:] * np.conj(iq_signal[:-1])
    inst_freq = (sample_rate / (2.0 * np.pi)) * np.angle(diff)

    # Pad 1 sample to match length
    inst_freq = np.pad(inst_freq, (0, 1), mode="edge")
    freq_syms = inst_freq[: n_symbols * sps].reshape(n_symbols, sps)
    return np.mean(freq_syms, axis=1)


def compute_rejected_symbol_center_freq_metric(
    iq_signal: np.ndarray,
    sps: int,
    sample_rate: float
) -> np.ndarray:
    """Rejected baseline: Symbol-center instantaneous frequency sample."""
    n_symbols = len(iq_signal) // sps
    if n_symbols == 0 or len(iq_signal) < 2:
        return np.array([], dtype=np.float64)

    diff = iq_signal[1:] * np.conj(iq_signal[:-1])
    inst_freq = (sample_rate / (2.0 * np.pi)) * np.angle(diff)
    inst_freq = np.pad(inst_freq, (0, 1), mode="edge")

    half_sps = sps // 2
    indices = np.arange(half_sps, n_symbols * sps, sps)
    return inst_freq[indices]
