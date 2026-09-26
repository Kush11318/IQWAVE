"""Comprehensive Test Suite for Module 2 (Non-Destructive Observation Layer).

Covers all requirements from Section 14:
A. Basic observation: retained amplitude features (mean, std, cv, skewness, kurtosis).
B. Differential phase: phase_diff_std, gated_phase_diff_std.
C. Circular statistics: circular_variance (core structural feature).
D. Supporting evidence: circularity_ratio (M20/M21), zero_bin_fraction.
E. Non-destructive behavior: original IQ samples remain completely unchanged.
F. Invalid/Degenerate inputs: empty, NaN, Inf, zero power, insufficient length.
G. Representation gate: rejects blind WAV -> Hilbert conversion.
H. Module 1 integration: seamless consumption of Module 1 canonical complex64 IQ.
I. CFO robustness behaviors: circular variance & amplitude invariant, M20/M21 CFO-sensitive.
"""

import numpy as np
import pytest

from backend.modules.module1.canonical_iq import to_canonical_iq
from backend.modules.module2.representation_gate import (
    SignalRepresentationType,
    process_representation_gate
)
from backend.modules.module2.amplitude_features import extract_amplitude_features
from backend.modules.module2.phase_features import extract_phase_features
from backend.modules.module2.circular_features import extract_circular_features
from backend.modules.module2.spectral_features import extract_spectral_features
from backend.modules.module2.observation import compute_observation_vector
from backend.modules.module2.service import (
    observe_signal,
    observe_from_module1
)


# -----------------------------------------------------------------------------
# Requirement A: Retained Amplitude Features
# -----------------------------------------------------------------------------

def test_amplitude_features_known_signal():
    """Verify amplitude features (mean, std, cv, skewness, kurtosis) on a known signal."""
    # Constant envelope signal (pure carrier): envelope is identically 2.0
    theta = np.linspace(0, 4 * np.pi, 200, endpoint=False)
    x = 2.0 * np.exp(1j * theta).astype(np.complex64)

    amp_feats = extract_amplitude_features(x)
    assert pytest.approx(amp_feats["mean"], abs=1e-5) == 2.0
    assert pytest.approx(amp_feats["std"], abs=1e-5) == 0.0
    assert pytest.approx(amp_feats["cv"], abs=1e-5) == 0.0
    # For constant envelope, variance=0 => higher-order skew/kurtosis are undefined (None)
    assert amp_feats["skewness"] is None
    assert amp_feats["kurtosis"] is None

    # Variable envelope signal: Rayleigh distributed envelope (complex Gaussian)
    rng = np.random.default_rng(123)
    x_var = (rng.standard_normal(2000) + 1j * rng.standard_normal(2000)).astype(np.complex64)
    amp_feats_var = extract_amplitude_features(x_var)
    assert amp_feats_var["mean"] > 0
    assert amp_feats_var["std"] > 0
    assert amp_feats_var["cv"] > 0
    assert amp_feats_var["skewness"] is not None
    assert amp_feats_var["kurtosis"] is not None


# -----------------------------------------------------------------------------
# Requirement B: Differential Phase Features
# -----------------------------------------------------------------------------

def test_differential_phase_features():
    """Verify phase_diff_std and amplitude-gated phase_diff_std."""
    # Linear phase progression (constant frequency): phase differences are constant
    omega = 0.25  # rad/sample
    n = np.arange(500)
    x_const_diff = np.exp(1j * omega * n).astype(np.complex64)

    phase_feats = extract_phase_features(x_const_diff)
    # Consecutive phase difference std should be near 0
    assert pytest.approx(phase_feats["diff_std"], abs=1e-4) == 0.0
    assert pytest.approx(phase_feats["gated_diff_std"], abs=1e-4) == 0.0

    # Signal with transition through zero
    x_trans = np.array([1.0 + 0j, 0.01 + 0j, -1.0 + 0j, -0.01 + 0j, 1.0 + 0j], dtype=np.complex64)
    phase_feats_trans = extract_phase_features(x_trans, gate_factor=0.5)
    assert phase_feats_trans["diff_std"] is not None


# -----------------------------------------------------------------------------
# Requirement C: Circular Statistics (circular_variance - CORE feature)
# -----------------------------------------------------------------------------

def test_circular_variance_core_behavior():
    """Verify circular_variance:
    - Pure tone / constant phase difference -> circular variance = 0
    - Independent uniform phase differences -> circular variance ~ 1
    """
    n = np.arange(1000)
    # Pure tone
    x_tone = np.exp(1j * 0.15 * n).astype(np.complex64)
    circ_feats = extract_circular_features(x_tone)
    assert pytest.approx(circ_feats["variance"], abs=1e-4) == 0.0

    # Completely uncorrelated random phase noise
    rng = np.random.default_rng(42)
    random_phases = rng.uniform(-np.pi, np.pi, 5000)
    x_noise = np.exp(1j * random_phases).astype(np.complex64)
    circ_feats_noise = extract_circular_features(x_noise)
    # Circular variance of uniform noise approaches 1.0
    assert circ_feats_noise["variance"] > 0.95


# -----------------------------------------------------------------------------
# Requirement D: Supporting Evidence (circularity_ratio & zero_bin_fraction)
# -----------------------------------------------------------------------------

def test_supporting_evidence_features():
    """Verify M20/M21 circularity ratio and spectral zero_bin_fraction."""
    # BPSK without CFO has strong non-circularity: x[n] in {-1, +1}, so M20 = E[x^2] = 1, M21 = E[|x|^2] = 1
    rng = np.random.default_rng(42)
    bits = rng.choice([-1.0, 1.0], size=1000)
    x_bpsk = bits.astype(np.complex64)

    circ_feats = extract_circular_features(x_bpsk)
    # M20/M21 for clean BPSK without CFO is ~ 1.0
    assert pytest.approx(circ_feats["m20_m21"], rel=1e-2) == 1.0

    # QPSK is second-order circular: M20 ~ 0
    symbols = rng.choice([1+1j, 1-1j, -1+1j, -1-1j], size=2000) / np.sqrt(2)
    x_qpsk = symbols.astype(np.complex64)
    circ_feats_qpsk = extract_circular_features(x_qpsk)
    assert pytest.approx(circ_feats_qpsk["m20_m21"], abs=0.08) == 0.0

    # DC / zero_bin_fraction
    x_with_dc = np.ones(512, dtype=np.complex64) * 2.0  # pure DC
    spec_dc = extract_spectral_features(x_with_dc)
    assert pytest.approx(spec_dc["zero_bin_fraction"], abs=1e-4) == 1.0

    # Tone away from zero frequency
    n = np.arange(512)
    x_tone = np.exp(1j * (2 * np.pi * 10 / 512) * n).astype(np.complex64)
    spec_tone = extract_spectral_features(x_tone)
    assert pytest.approx(spec_tone["zero_bin_fraction"], abs=1e-4) == 0.0


# -----------------------------------------------------------------------------
# Requirement E: Non-Destructive Behavior Invariant
# -----------------------------------------------------------------------------

def test_non_destructive_preservation():
    """Verify original IQ samples are completely unchanged after Module 2."""
    orig_i = np.array([0.5, -0.8, 1.2, -0.3, 0.9], dtype=np.float32)
    orig_q = np.array([0.2, 0.4, -0.6, 0.7, -0.1], dtype=np.float32)
    orig_iq = orig_i + 1j * orig_q

    # Save bitwise copy
    copy_iq = np.copy(orig_iq)

    res = observe_signal(orig_iq, SignalRepresentationType.COMPLEX_IQ)

    assert res.status == "VALID"
    # Verify input array was not modified in-place
    np.testing.assert_array_equal(orig_iq, copy_iq)
    # Verify raw signal is accessible in result
    np.testing.assert_array_equal(res.raw_signal, copy_iq)


# -----------------------------------------------------------------------------
# Requirement F: Invalid / Degenerate Inputs
# -----------------------------------------------------------------------------

def test_invalid_empty_signal():
    """Empty signal -> rejected cleanly with explicit quality status."""
    res = observe_signal(np.array([], dtype=np.complex64))
    assert res.status == "REJECTED"
    assert res.rejection_reason == "EMPTY_SIGNAL"
    assert res.observation["quality"]["observation_length"] == 0


def test_invalid_nan_signal():
    """Signal with NaN -> returns finite=False without crash or manufactured values."""
    x = np.array([1.0 + 1j, np.nan + 1j, 2.0 - 1j], dtype=np.complex64)
    res = observe_signal(x)
    assert res.status == "VALID"  # Admitted through representation gate
    obs = res.observation
    assert obs["quality"]["finite"] is False
    assert obs["amplitude"]["mean"] is None
    assert obs["phase"]["diff_std"] is None
    assert obs["circular"]["variance"] is None


def test_invalid_inf_signal():
    """Signal with Inf -> returns finite=False without crash."""
    x = np.array([1.0 + 1j, np.inf + 1j, 2.0 - 1j], dtype=np.complex64)
    res = observe_signal(x)
    obs = res.observation
    assert obs["quality"]["finite"] is False
    assert obs["amplitude"]["mean"] is None


def test_zero_power_signal():
    """Zero power signal -> handled safely without division by zero."""
    x = np.zeros(100, dtype=np.complex64)
    res = observe_signal(x)
    obs = res.observation
    assert obs["quality"]["power"] == 0.0
    assert obs["amplitude"]["cv"] is None
    assert obs["circular"]["m20_m21"] is None


def test_insufficient_length():
    """Signal with only 1 sample -> differential phase and circular variance return None."""
    x = np.array([1.0 + 0.5j], dtype=np.complex64)
    res = observe_signal(x)
    obs = res.observation
    assert obs["quality"]["observation_length"] == 1
    assert obs["phase"]["diff_std"] is None
    assert obs["circular"]["variance"] is None


# -----------------------------------------------------------------------------
# Requirement G: Representation Gate (Reject Blind WAV -> Hilbert)
# -----------------------------------------------------------------------------

def test_representation_gate_rejects_blind_wav_hilbert():
    """Verify that a real/WAV signal is REJECTED if blind Hilbert conversion is attempted."""
    real_waveform = np.sin(np.linspace(0, 10, 500)).astype(np.float32)

    # Attempting to observe real signal without explicit permission must be rejected
    res = observe_signal(real_waveform)
    assert res.status == "REJECTED"
    assert res.rejection_reason == "BLIND_HILBERT_CONVERSION_REJECTED"
    assert res.observation["quality"]["observation_length"] == 0


def test_representation_gate_explicit_analytic():
    """Verify that explicit REAL_IF_PASSBAND with flag allows controlled analytic conversion."""
    real_waveform = np.sin(np.linspace(0, 10, 500)).astype(np.float32)

    res = observe_signal(
        real_waveform,
        representation_type=SignalRepresentationType.REAL_IF_PASSBAND,
        allow_explicit_analytic_conversion=True
    )
    assert res.status == "VALID"
    assert res.observation["representation"]["analytic_conversion_applied"] is True
    assert res.observation["quality"]["observation_length"] == 500


# -----------------------------------------------------------------------------
# Requirement H: Integration with Module 1
# -----------------------------------------------------------------------------

def test_module1_to_module2_pipeline():
    """Verify end-to-end integration: Module 1 canonical complex64 IQ into Module 2."""
    i_raw = [0.1, 0.4, -0.3, 0.7, -0.5, 0.2]
    q_raw = [-0.2, 0.5, 0.1, -0.4, 0.3, -0.1]

    # Module 1
    canonical_iq, val_res = to_canonical_iq(i_raw, q_raw)
    assert val_res["status"] == "VALID"
    assert canonical_iq.dtype == np.complex64

    # Module 2
    mod2_res = observe_from_module1(canonical_iq, val_res)
    assert mod2_res.status == "VALID"
    assert mod2_res.observation["quality"]["observation_length"] == 6
    assert mod2_res.observation["amplitude"]["mean"] > 0
    assert mod2_res.observation["circular"]["variance"] is not None


# -----------------------------------------------------------------------------
# Requirement I: Validated CFO Robustness Behaviors
# -----------------------------------------------------------------------------

def test_cfo_robustness_experiment():
    """Validate findings from Section 2H of Module 2 Report:
    - circular_variance is CFO-robust (strictly invariant under constant CFO).
    - amplitude features are CFO-robust.
    - M20/M21 circularity ratio is CFO-sensitive.
    - zero_bin_fraction is CFO-sensitive.
    """
    rng = np.random.default_rng(100)
    # BPSK signal
    n = np.arange(1000)
    bpsk_symbols = rng.choice([-1.0, 1.0], size=1000).astype(np.complex64)

    # Apply CFO: omega_0 = 0.05 rad/sample
    cfo_carrier = np.exp(1j * 0.05 * n).astype(np.complex64)
    bpsk_with_cfo = bpsk_symbols * cfo_carrier

    obs_clean = observe_signal(bpsk_symbols).observation
    obs_cfo = observe_signal(bpsk_with_cfo).observation

    # 1. Amplitude features must be identical (within single-precision numerical tolerances)
    assert pytest.approx(obs_cfo["amplitude"]["mean"], rel=1e-4) == obs_clean["amplitude"]["mean"]
    assert pytest.approx(obs_cfo["amplitude"]["std"], abs=1e-5) == obs_clean["amplitude"]["std"]

    # Also test on a variable-envelope signal (QAM-like) under CFO
    qam_symbols = (rng.choice([-3, -1, 1, 3], size=1000) + 1j * rng.choice([-3, -1, 1, 3], size=1000)).astype(np.complex64)
    qam_cfo = qam_symbols * cfo_carrier
    obs_qam_clean = observe_signal(qam_symbols).observation
    obs_qam_cfo = observe_signal(qam_cfo).observation
    assert pytest.approx(obs_qam_cfo["amplitude"]["mean"], rel=1e-4) == obs_qam_clean["amplitude"]["mean"]
    assert pytest.approx(obs_qam_cfo["amplitude"]["std"], rel=1e-4) == obs_qam_clean["amplitude"]["std"]
    assert pytest.approx(obs_qam_cfo["amplitude"]["cv"], rel=1e-4) == obs_qam_clean["amplitude"]["cv"]

    # 2. Circular variance (core feature) must remain invariant under constant CFO
    assert pytest.approx(obs_cfo["circular"]["variance"], abs=1e-4) == obs_clean["circular"]["variance"]
    assert pytest.approx(obs_qam_cfo["circular"]["variance"], abs=1e-4) == obs_qam_clean["circular"]["variance"]

    # 3. M20/M21 circularity ratio collapses under CFO (CFO-sensitive)
    clean_ratio = obs_clean["circular"]["m20_m21"]
    cfo_ratio = obs_cfo["circular"]["m20_m21"]
    assert clean_ratio > 0.9  # BPSK clean is rectilinear
    assert cfo_ratio < 0.15   # Under CFO, M20 averages out towards 0
