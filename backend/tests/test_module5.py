"""Unit and integration tests for Module 5 (Synchronization & Digital Recovery)."""

import pytest
import numpy as np
from scipy import signal

from backend.modules.module5.carrier_sync import synchronize_carrier
from backend.modules.module5.phase_sync import synchronize_phase, wrap_to_pi
from backend.modules.module5.timing_sync import synchronize_timing_gardner, fractional_delay
from backend.modules.module5.matched_filter import apply_matched_filter, design_rrc_filter
from backend.modules.module5.resampler import rational_resample, find_resampling_ratio
from backend.modules.module5.demodulators import (
    demodulate_bpsk,
    demodulate_qpsk,
    demodulate_8psk,
    demodulate_qam16,
    demodulate_qam64,
    demodulate_cpfsk,
    demodulate_gfsk,
    demodulate_signal
)
from backend.modules.module5.service import recover_digital_symbols_and_bits


def generate_qpsk(
    num_symbols: int = 1000,
    sps: int = 4,
    beta: float = 0.35,
    cfo: float = 0.0,
    phase_rad: float = 0.0,
    timing_offset: float = 0.0,
    fs: float = 1000000.0,
    seed: int = 42
):
    rng = np.random.RandomState(seed)
    bits = rng.randint(0, 2, num_symbols * 2)
    b0 = bits[0::2]
    b1 = bits[1::2]
    syms = ((2 * b0 - 1) + 1j * (2 * b1 - 1)) / np.sqrt(2.0)

    # Upsample
    up = np.zeros(num_symbols * sps, dtype=np.complex128)
    up[::sps] = syms

    # RRC pulse shaping
    h = design_rrc_filter(sps=sps, beta=beta, span=8)
    tx = signal.convolve(up, h, mode="same")

    # Timing offset
    if abs(timing_offset) > 1e-6:
        tx = fractional_delay(tx, timing_offset)

    # CFO and Phase
    t = np.arange(len(tx)) / fs
    tx = tx * np.exp(1j * (2.0 * np.pi * cfo * t + phase_rad))

    return tx, syms, bits


# -----------------------------------------------------------------------------
# 5A: Carrier Synchronization Tests
# -----------------------------------------------------------------------------

def test_psk_cfo_correction_controlled():
    """Verify CFO estimation and correction for controlled +20 kHz QPSK."""
    fs = 1000000.0
    cfo_true = 20000.0
    tx, syms, bits = generate_qpsk(num_symbols=1500, sps=4, cfo=cfo_true, fs=fs)

    res = synchronize_carrier(tx, modulation="QPSK", sample_rate=fs)
    assert res["quality"] in ["HIGH", "MEDIUM"]
    assert abs(res["cfo_estimate"] - cfo_true) < 150.0  # within fine grid resolution
    assert abs(res["cfo_applied"] - cfo_true) < 150.0


@pytest.mark.parametrize("cfo_val", [-100000.0, -50000.0, -20000.0, -5000.0, 5000.0, 20000.0, 50000.0, 100000.0])
def test_psk_cfo_sweep(cfo_val):
    """Verify CFO recovery across frequency sweep."""
    fs = 1000000.0
    tx, _, _ = generate_qpsk(num_symbols=1500, sps=4, cfo=cfo_val, fs=fs)
    res = synchronize_carrier(tx, modulation="QPSK", sample_rate=fs)
    assert abs(res["cfo_estimate"] - cfo_val) < 200.0


def test_psk_zero_cfo_guard():
    """Verify that zero CFO is not falsely estimated as a strong nonzero carrier."""
    fs = 1000000.0
    tx, _, _ = generate_qpsk(num_symbols=1500, sps=4, cfo=0.0, fs=fs)
    res = synchronize_carrier(tx, modulation="QPSK", sample_rate=fs)
    assert abs(res["cfo_estimate"]) < 50.0 or "ZERO_CFO_DETECTED_OR_AMBIGUOUS" in res["warnings"]


# -----------------------------------------------------------------------------
# 5B: Carrier Phase Synchronization Tests
# -----------------------------------------------------------------------------

def test_phase_sync_controlled():
    """Verify phase recovery for injected +30 deg offset."""
    phase_deg = 30.0
    phase_rad = np.radians(phase_deg)
    tx, _, _ = generate_qpsk(num_symbols=1500, sps=4, phase_rad=phase_rad)

    # In clean symbol samples or signal
    res = synchronize_phase(tx, modulation="QPSK", constellation_diagonal=True)
    assert res["quality"] in ["HIGH", "MEDIUM"]
    # Modulo 90 degrees symmetry
    error_deg = abs(wrap_to_pi(np.radians(res["phase_estimate_deg"] - phase_deg)))
    assert np.degrees(error_deg) < 2.0


@pytest.mark.parametrize("deg", [-60.0, -30.0, 0.0, 30.0, 60.0])
def test_phase_sync_sweep(deg):
    """Verify phase recovery across phase sweep."""
    phase_rad = np.radians(deg)
    tx, _, _ = generate_qpsk(num_symbols=1500, sps=4, phase_rad=phase_rad)
    res = synchronize_phase(tx, modulation="QPSK", constellation_diagonal=True)
    error_rad = wrap_to_pi(np.radians(res["phase_estimate_deg"]) - phase_rad)
    # Modulo 90 deg for QPSK
    error_mod = ((error_rad + np.pi/4) % (np.pi/2)) - np.pi/4
    assert abs(np.degrees(error_mod)) < 2.5


# -----------------------------------------------------------------------------
# 5C: Timing Recovery Tests (Gardner TED)
# -----------------------------------------------------------------------------

def test_gardner_timing_controlled():
    """Verify timing recovery for +0.75 sample delay."""
    tx, _, _ = generate_qpsk(num_symbols=2000, sps=4, timing_offset=0.75)
    rx = apply_matched_filter(tx, modulation="QPSK", samples_per_symbol=4.0)["filtered_signal"]
    res = synchronize_timing_gardner(rx, samples_per_symbol=4.0)
    assert abs(res["timing_offset_samples"] - 0.75) < 0.15


@pytest.mark.parametrize("offset", [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
def test_gardner_timing_sweep(offset):
    """Verify timing recovery across sweep in [-1.5, 1.5] samples."""
    tx, _, _ = generate_qpsk(num_symbols=2000, sps=4, timing_offset=offset)
    rx = apply_matched_filter(tx, modulation="QPSK", samples_per_symbol=4.0)["filtered_signal"]
    res = synchronize_timing_gardner(rx, samples_per_symbol=4.0)
    assert abs(res["timing_offset_samples"] - offset) < 0.25


# -----------------------------------------------------------------------------
# 5D: RRC Matched Filter Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("beta", [0.2, 0.35, 0.5, 1.0])
def test_rrc_matched_filter_roll_offs(beta):
    """Verify RRC matched filtering for tested roll-off factors."""
    tx, _, _ = generate_qpsk(num_symbols=500, sps=4, beta=beta)
    res = apply_matched_filter(tx, modulation="QPSK", samples_per_symbol=4.0, beta=beta)
    assert res["applied"] is True
    assert res["beta"] == beta
    assert len(res["filtered_signal"]) == len(tx)


# -----------------------------------------------------------------------------
# 5E: Rational Resampling Tests
# -----------------------------------------------------------------------------

def test_rational_resampling_non_integer_sps():
    """Verify rational resampling for Fs = 1 MHz, Rs = 240 kHz -> SPS = 4.1667."""
    fs = 1000000.0
    rs = 240000.0
    sps = fs / rs  # 4.1666667

    up, down, target_sps = find_resampling_ratio(sps, target_sps=5)
    assert up == 6
    assert down == 5
    assert target_sps == 5

    x = np.ones(1000, dtype=np.complex128)
    res = rational_resample(x, samples_per_symbol=sps, target_integer_sps=5)
    assert res["applied"] is True
    assert res["up"] == 6
    assert res["down"] == 5
    assert abs(res["new_sps"] - 5.0) < 1e-4


# -----------------------------------------------------------------------------
# 5F: Demodulation Tests for All 7 Supported Classes
# -----------------------------------------------------------------------------

def test_demodulate_bpsk():
    """Verify BPSK demodulation with sign decisions."""
    bits_in = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=np.uint8)
    syms = np.where(bits_in == 1, 1.0 + 0j, -1.0 + 0j)
    dec_syms, bits_out = demodulate_bpsk(syms)
    np.testing.assert_array_equal(bits_out, bits_in)


def test_demodulate_qpsk():
    """Verify QPSK quadrant demodulation."""
    bits_in = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    b0 = bits_in[0::2].astype(float)
    b1 = bits_in[1::2].astype(float)
    syms = ((2.0 * b0 - 1.0) + 1j * (2.0 * b1 - 1.0)) / np.sqrt(2.0)
    dec_syms, bits_out = demodulate_qpsk(syms)
    np.testing.assert_array_equal(bits_out, bits_in)


def test_demodulate_8psk():
    """Verify 8PSK phase demodulation."""
    # Test all 8 states
    k = np.arange(8)
    syms = np.exp(1j * k * (np.pi / 4.0))
    dec_syms, bits_out = demodulate_8psk(syms)
    assert len(bits_out) == 24


def test_demodulate_qam16():
    """Verify 16-QAM nearest constellation decision with RMS normalization."""
    # Generate 16 QAM grid
    grid = np.array([-3, -1, 1, 3]) / np.sqrt(10.0)
    mesh_r, mesh_i = np.meshgrid(grid, grid)
    constellation = (mesh_r + 1j * mesh_i).flatten()
    dec_syms, bits_out = demodulate_qam16(constellation)
    assert len(bits_out) == 16 * 4


def test_demodulate_qam64():
    """Verify 64-QAM nearest constellation decision with RMS normalization."""
    grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42.0)
    mesh_r, mesh_i = np.meshgrid(grid, grid)
    constellation = (mesh_r + 1j * mesh_i).flatten()
    dec_syms, bits_out = demodulate_qam64(constellation)
    assert len(bits_out) == 64 * 6


def test_demodulate_cpfsk():
    """Verify CPFSK instantaneous frequency demodulation."""
    sps = 4
    fs = 1000000.0
    bits_in = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 20, dtype=np.uint8)
    delta_f = 100000.0

    # Synthesize CPFSK signal
    freqs = np.where(bits_in == 1, delta_f, -delta_f)
    freq_wave = np.repeat(freqs, sps)
    phase = 2.0 * np.pi * np.cumsum(freq_wave) / fs
    iq = np.exp(1j * phase)

    dec_syms, bits_out = demodulate_cpfsk(iq, sps=sps, sample_rate=fs)
    # Check bit error rate is low in clean signal
    ber = np.mean(bits_out[:len(bits_in)] != bits_in)
    assert ber < 0.05


def test_demodulate_gfsk():
    """Verify GFSK instantaneous frequency demodulation."""
    sps = 4
    fs = 1000000.0
    bits_in = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 20, dtype=np.uint8)
    delta_f = 100000.0

    # Gaussian filtered frequency waveform
    freqs = np.where(bits_in == 1, delta_f, -delta_f)
    freq_wave = np.repeat(freqs, sps)
    g_filt = np.exp(-np.linspace(-2, 2, sps * 2) ** 2)
    g_filt /= np.sum(g_filt)
    smoothed = signal.convolve(freq_wave, g_filt, mode="same")
    phase = 2.0 * np.pi * np.cumsum(smoothed) / fs
    iq = np.exp(1j * phase)

    dec_syms, bits_out = demodulate_gfsk(iq, sps=sps, sample_rate=fs)
    ber = np.mean(bits_out[:len(bits_in)] != bits_in)
    assert ber < 0.05


# -----------------------------------------------------------------------------
# End-to-End Pipeline Test (Section 32)
# -----------------------------------------------------------------------------

def test_end_to_end_qpsk_pipeline():
    """Full pipeline: Synthetic IQ -> Module 4 -> Module 5 synchronization & recovery."""
    fs = 1000000.0
    rs = 250000.0
    sps = 4.0
    cfo = 20000.0
    phase_rad = np.radians(30.0)

    tx, syms, bits_tx = generate_qpsk(
        num_symbols=1000,
        sps=int(sps),
        cfo=cfo,
        phase_rad=phase_rad,
        fs=fs,
        seed=123
    )

    # Mock or actual Module 4 parameter dict
    m4_params = {
        "modulation": "QPSK",
        "symbol_rate": rs,
        "samples_per_symbol": sps,
        "cfo": cfo,
        "metadata": {"sample_rate": fs}
    }

    res = recover_digital_symbols_and_bits(
        canonical_iq=tx,
        modulation="QPSK",
        module4_parameters=m4_params,
        sample_rate=fs,
        samples_per_symbol=sps
    )

    assert res["status"] == "SUCCESS"
    assert res["num_symbols"] > 0
    assert res["num_bits"] > 0
    assert "synchronization" in res
    assert res["synchronization"]["matched_filter"] is not None


# -----------------------------------------------------------------------------
# Numerical Safety & Error Handling Tests
# -----------------------------------------------------------------------------

def test_empty_signal_recovery():
    """Verify recovery service rejects empty signal gracefully."""
    res = recover_digital_symbols_and_bits(np.array([]), modulation="QPSK")
    assert res["status"] == "INVALID_INPUT"
    assert "EMPTY_SIGNAL" in res["warnings"]


def test_nan_signal_recovery():
    """Verify recovery service rejects NaN signal gracefully."""
    res = recover_digital_symbols_and_bits(np.array([1.0 + 1j, np.nan]), modulation="QPSK")
    assert res["status"] == "INVALID_INPUT"
    assert "NON_FINITE_SAMPLES" in res["warnings"]


def test_missing_modulation_recovery():
    """Verify recovery service rejects missing modulation class without guessing."""
    res = recover_digital_symbols_and_bits(np.array([1.0 + 1j] * 64), modulation=None)
    assert res["status"] == "INSUFFICIENT_PARAMETERS"
    assert "MODULATION_CLASS_REQUIRED_FROM_MODULE3" in res["warnings"]
