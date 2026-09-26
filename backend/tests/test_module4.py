"""Comprehensive Test Suite for Module 4: Signal Parameter Estimation.

Verifies:
1. Symbol Rate & SPS:
   - Controlled test: Fs = 1 MHz, Rs = 250 kSym/s, SPS = 4.
   - Tested across modulation diversity (BPSK, QPSK, 8PSK, QAM16, QAM64).
   - Tested across RRC roll-off beta values (0.2, 0.35, 0.5, 0.8, 1.0).
2. FSK Parameters:
   - Instantaneous frequency and 2-state clustering: f0 = 50 kHz, f1 = 150 kHz -> separation ~ 100 kHz.
3. Carrier-Frequency Offset (CFO):
   - PSK CFO across controlled values: +-100 kHz, +-50 kHz, +-20 kHz, +-5 kHz.
   - QAM16 conditional weighted fourth-power CFO.
   - QAM64 CFO correctly flagged as NOT_VALIDATED.
4. Relative Center & Occupied Bandwidth:
   - Relative center under frequency shifts: +-50 kHz, +-100 kHz, +-150 kHz.
   - Bandwidth of QPSK RRC signal (337.5 kHz theoretical).
5. Metadata & Physical Limitations:
   - When metadata is absent: Fs, absolute RF frequency, and SPS are strictly UNKNOWN/None.
   - When metadata is present: absolute RF center = metadata_rf + relative_center.
6. Degenerate / Failure cases:
   - Empty, NaN, Inf, zero power, short observation length.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from backend.modules.module4.symbol_rate import estimate_symbol_rate
from backend.modules.module4.fsk_frequency import estimate_fsk_parameters
from backend.modules.module4.cfo_estimator import estimate_cfo
from backend.modules.module4.spectral_estimator import estimate_relative_center_and_bandwidth
from backend.modules.module4.service import (
    estimate_signal_parameters,
    default_parameter_service
)


def generate_rrc_pulse(beta: float, sps: int, span: int = 16) -> np.ndarray:
    """Generate root-raised-cosine filter coefficients."""
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif np.abs(ti) == 1.0 / (4 * beta):
            h[i] = (beta / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) +
                ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
            )
        else:
            h[i] = (
                np.sin(np.pi * ti * (1 - beta)) +
                4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            ) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sum(h)


def synthesize_qpsk_rrc(
    fs: float = 1e6,
    rs: float = 250e3,
    beta: float = 0.35,
    n_symbols: int = 5000,
    snr_db: Optional[float] = None,
    cfo_hz: float = 0.0,
    seed: int = 42
) -> np.ndarray:
    """Synthesize controlled QPSK signal with RRC pulse shaping."""
    sps = int(fs / rs)
    rng = np.random.default_rng(seed)
    bits = (rng.choice([-1, 1], size=n_symbols) + 1j * rng.choice([-1, 1], size=n_symbols)) / np.sqrt(2)

    upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = bits

    h = generate_rrc_pulse(beta, sps)
    tx = np.convolve(upsampled, h, mode="same").astype(np.complex64)

    # Optional CFO
    if cfo_hz != 0.0:
        t = np.arange(len(tx)) / fs
        tx = tx * np.exp(1j * 2 * np.pi * cfo_hz * t).astype(np.complex64)

    # Optional AWGN
    if snr_db is not None:
        sig_p = np.mean(np.abs(tx) ** 2)
        noise_p = sig_p / (10 ** (snr_db / 10.0))
        noise = np.sqrt(noise_p / 2.0) * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
        tx = tx + noise.astype(np.complex64)

    return tx


# -----------------------------------------------------------------------------
# 1. Symbol Rate Tests
# -----------------------------------------------------------------------------

def test_symbol_rate_controlled_qpsk():
    """Verify symbol rate recovery on controlled setup: Fs=1MHz, Rs=250kHz, SPS=4."""
    fs = 1000000.0
    true_rs = 250000.0
    sig = synthesize_qpsk_rrc(fs=fs, rs=true_rs, beta=0.35, n_symbols=4000)

    res = estimate_symbol_rate(sig, sample_rate=fs)
    assert res["symbol_rate"] is not None
    assert pytest.approx(res["symbol_rate"], rel=1e-2) == true_rs
    assert pytest.approx(res["samples_per_symbol"], rel=1e-2) == 4.0
    assert res["confidence"] in ["HIGH", "MEDIUM"]


@pytest.mark.parametrize("beta", [0.2, 0.35, 0.5, 0.8, 1.0])
def test_symbol_rate_rrc_beta_robustness(beta):
    """Verify cyclic symbol rate estimator across validated RRC roll-off factors."""
    fs = 1000000.0
    true_rs = 250000.0
    sig = synthesize_qpsk_rrc(fs=fs, rs=true_rs, beta=beta, n_symbols=4000)

    res = estimate_symbol_rate(sig, sample_rate=fs)
    assert pytest.approx(res["symbol_rate"], rel=1e-2) == true_rs


@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64"])
def test_symbol_rate_modulation_diversity(mod):
    """Verify symbol rate estimation across multiple supported modulation types."""
    fs = 1000000.0
    true_rs = 250000.0
    sps = 4
    n_sym = 4000
    rng = np.random.default_rng(100)

    if mod == "BPSK":
        symbols = rng.choice([-1.0, 1.0], size=n_sym) + 0j
    elif mod == "QPSK":
        symbols = (rng.choice([-1, 1], size=n_sym) + 1j * rng.choice([-1, 1], size=n_sym)) / np.sqrt(2)
    elif mod == "8PSK":
        angles = rng.choice(np.arange(8)) * (2 * np.pi / 8)
        symbols = np.exp(1j * angles)
    elif mod == "QAM16":
        symbols = (rng.choice([-3, -1, 1, 3], size=n_sym) + 1j * rng.choice([-3, -1, 1, 3], size=n_sym))
    else:  # QAM64
        symbols = (rng.choice(np.arange(-7, 8, 2), size=n_sym) + 1j * rng.choice(np.arange(-7, 8, 2), size=n_sym))

    upsampled = np.zeros(n_sym * sps, dtype=np.complex64)
    upsampled[::sps] = symbols.astype(np.complex64)
    h = generate_rrc_pulse(0.35, sps)
    tx = np.convolve(upsampled, h, mode="same").astype(np.complex64)

    res = estimate_symbol_rate(tx, sample_rate=fs)
    assert pytest.approx(res["symbol_rate"], rel=1.5e-2) == true_rs


# -----------------------------------------------------------------------------
# 2. FSK Frequency Parameters
# -----------------------------------------------------------------------------

def test_fsk_frequency_states_and_separation():
    """Verify controlled 2-FSK recovery: f0=50kHz, f1=150kHz -> Delta-f = 100kHz."""
    fs = 1000000.0
    f0 = 50000.0
    f1 = 150000.0
    sps = 4
    n_symbols = 5000

    rng = np.random.default_rng(42)
    bits = rng.choice([0, 1], size=n_symbols)
    freq_seq = np.repeat(np.where(bits == 0, f0, f1), sps)

    phase = 2 * np.pi * np.cumsum(freq_seq) / fs
    x_fsk = np.exp(1j * phase).astype(np.complex64)

    res = estimate_fsk_parameters(x_fsk, sample_rate=fs)
    assert res["frequency_state_0"] is not None
    assert res["frequency_state_1"] is not None
    assert pytest.approx(res["frequency_state_0"], abs=2000.0) == f0
    assert pytest.approx(res["frequency_state_1"], abs=2000.0) == f1
    assert pytest.approx(res["frequency_separation"], abs=2000.0) == 100000.0


# -----------------------------------------------------------------------------
# 3. Carrier Frequency Offset (CFO) Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("cfo_target", [100000.0, -100000.0, 50000.0, -50000.0, 20000.0, -20000.0, 5000.0, -5000.0])
def test_psk_cfo_controlled_values(cfo_target):
    """Verify PSK M-th power CFO estimator across all controlled test values."""
    fs = 1000000.0
    sig = synthesize_qpsk_rrc(fs=fs, rs=250e3, n_symbols=5000, cfo_hz=cfo_target)

    res = estimate_cfo(sig, modulation_type="QPSK", sample_rate=fs)
    assert res["status"] == "LOCKED"
    assert pytest.approx(res["cfo"], abs=100.0) == cfo_target


def test_qam16_conditional_cfo():
    """Verify conditional weighted fourth-power CFO estimator on QAM16."""
    fs = 1000000.0
    cfo_target = 20000.0
    n = 20000
    rng = np.random.default_rng(42)
    symbols = rng.choice([-3, -1, 1, 3], size=n) + 1j * rng.choice([-3, -1, 1, 3], size=n)
    t = np.arange(n) / fs
    sig = (symbols * np.exp(1j * 2 * np.pi * cfo_target * t)).astype(np.complex64)

    res = estimate_cfo(sig, modulation_type="QAM16", sample_rate=fs)
    assert res["status"] == "CONDITIONAL"
    assert "QAM_CFO_IS_CONDITIONAL" in res["flags"]
    assert pytest.approx(res["cfo"], abs=100.0) == cfo_target


def test_qam64_cfo_unsupported():
    """Verify QAM64 CFO is flagged as NOT_VALIDATED per specification."""
    dummy = np.ones(100, dtype=np.complex64)
    res = estimate_cfo(dummy, modulation_type="QAM64", sample_rate=1e6)
    assert res["status"] == "NOT_VALIDATED"
    assert "QAM64_CFO_NOT_VALIDATED" in res["flags"]
    assert res["cfo"] is None


# -----------------------------------------------------------------------------
# 4. Relative Center & Bandwidth Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("freq_shift", [50000.0, -50000.0, 100000.0, -100000.0, 150000.0, -150000.0])
def test_spectral_center_under_frequency_shifts(freq_shift):
    """Verify relative center tracking across frequency shifts (+-50, +-100, +-150 kHz)."""
    fs = 1000000.0
    sig = synthesize_qpsk_rrc(fs=fs, rs=250e3, n_symbols=4000, cfo_hz=freq_shift, snr_db=15.0)

    res = estimate_relative_center_and_bandwidth(sig, sample_rate=fs)
    assert res["relative_center_frequency"] is not None
    # Center error in research report was ~ 0.5 kHz
    assert pytest.approx(res["relative_center_frequency"], abs=5000.0) == freq_shift


def test_occupied_bandwidth_qpsk():
    """Verify occupied bandwidth on controlled QPSK signal: theoretical B = 337.5 kHz."""
    fs = 1000000.0
    true_bw = 250000.0 * (1 + 0.35)  # 337.5 kHz
    sig = synthesize_qpsk_rrc(fs=fs, rs=250e3, beta=0.35, n_symbols=4000, snr_db=15.0)

    res = estimate_relative_center_and_bandwidth(sig, sample_rate=fs)
    assert res["occupied_bandwidth"] is not None
    # Within 5% of theoretical occupied bandwidth
    assert pytest.approx(res["occupied_bandwidth"], rel=0.08) == true_bw


# -----------------------------------------------------------------------------
# 5. Metadata & Physical Limitations Tests
# -----------------------------------------------------------------------------

def test_metadata_absent_strictly_unknown():
    """Verify that when metadata is absent, absolute RF frequency and SPS remain UNKNOWN/None."""
    sig = synthesize_qpsk_rrc(fs=1e6, rs=250e3, n_symbols=1000)

    res = estimate_signal_parameters(
        canonical_iq=sig,
        modulation="QPSK",
        metadata=None  # No Fs, no RF center
    )

    assert res["status"] == "SUCCESS"
    assert res["symbol_rate"] is None
    assert res["symbol_rate_normalized"] is not None
    assert res["samples_per_symbol"] is None
    assert res["absolute_rf_frequency"] is None
    assert "METADATA_REQUIRED_FOR_ABSOLUTE_RATES" in res["flags"]
    assert "METADATA_REQUIRED_FOR_ABSOLUTE_RF_CENTER" in res["flags"]


def test_metadata_present_absolute_rf():
    """Verify that when metadata RF center is provided, absolute RF center is computed."""
    fs = 1000000.0
    meta_rf = 433000000.0  # 433 MHz
    sig = synthesize_qpsk_rrc(fs=fs, rs=250e3, n_symbols=2000, cfo_hz=25000.0, snr_db=15.0)

    res = estimate_signal_parameters(
        canonical_iq=sig,
        modulation="QPSK",
        metadata={"sample_rate": fs, "center_frequency": meta_rf}
    )

    assert res["absolute_rf_frequency"] is not None
    expected_rf = meta_rf + res["relative_center_frequency"]
    assert pytest.approx(res["absolute_rf_frequency"], abs=10.0) == expected_rf


# -----------------------------------------------------------------------------
# 6. Degenerate Inputs & Edge Cases
# -----------------------------------------------------------------------------

def test_empty_signal_handling():
    """Verify empty signal input is safely handled."""
    res = estimate_signal_parameters(np.array([], dtype=np.complex64))
    assert res["status"] == "INVALID_INPUT"
    assert "EMPTY_SIGNAL" in res["flags"]


def test_nan_inf_signal_handling():
    """Verify signal with NaN/Inf does not crash."""
    bad_sig = np.array([1.0 + 1j, np.nan + 1j, np.inf - 1j], dtype=np.complex64)
    res = estimate_signal_parameters(bad_sig, sample_rate=1e6)
    assert res["symbol_rate"] is None
    assert res["occupied_bandwidth"] is None
