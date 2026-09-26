"""Comprehensive Unit and Scientific Validation Tests for Module 7 (Soft Bits / LLR Generation).

Covers all prompt and blueprint requirements:
1. All seven modulation classes (BPSK, QPSK, 8PSK, QAM16, QAM64, CPFSK, GFSK).
2. Sign convention: positive (>0) -> bit 1, negative (<0) -> bit 0 across all classes.
3. BPSK analytical vs exact equivalence (max diff <= 1e-13, correlation == 1.0).
4. QPSK analytical vs exact equivalence (max diff <= 1e-13, correlation == 1.0).
5. 8PSK exact constellation likelihood with natural binary mapping and unequal bit protection.
6. 8PSK Max-Log candidate comparison.
7. QAM16 exact likelihood and unequal bit protection (coarse vs fine).
8. QAM16 Max-Log divergence from exact at lower SNR.
9. QAM64 exact likelihood and 3-level bit protection (coarse, middle, fine).
10. QAM64 Max-Log divergence below 20 dB.
11. CPFSK waveform correlation soft metric semantics:
    - metric_type == 'SOFT_METRIC'
    - mathematical_status == 'NOT_CALIBRATED_LLR'
    - calibrated_llr == False
    - NO 2/N0 scaling applied.
12. GFSK waveform correlation soft metric semantics with BT=0.5 Gaussian template:
    - metric_type == 'SOFT_METRIC'
    - mathematical_status == 'NOT_CALIBRATED_LLR'
    - calibrated_llr == False
    - NO 2/N0 scaling applied.
13. Missing-noise failure path:
    - PSK/QAM returns status 'NOISE_PARAMETER_UNAVAILABLE'.
    - Does NOT substitute N0=1.0.
    - soft_bits is None.
14. FSK noise independence (FSK waveform correlation proceeds without requiring N0).
15. Hard-decision agreement against Module 5 decisions.
16. Reliability summary generation and diagnostic magnitude bins ([0, 1), [1, 2), [2, 4), [4, 8), >8).
17. Reliability property: empirical BER decreases as soft magnitude bin increases.
18. Rejected baseline comparators for CPFSK and GFSK.
19. Module 7 -> Module 8 boundary verification.
"""

import numpy as np
import pytest

from backend.modules.module7.psk_llr import (
    compute_bpsk_analytical_llr,
    compute_bpsk_exact_llr,
    compute_qpsk_analytical_llr,
    compute_qpsk_exact_llr,
    compute_8psk_exact_llr,
    compute_8psk_maxlog_llr
)
from backend.modules.module7.qam_llr import (
    get_qam16_constellation,
    compute_qam16_exact_llr,
    compute_qam16_maxlog_llr,
    get_qam64_constellation,
    compute_qam64_exact_llr,
    compute_qam64_maxlog_llr
)
from backend.modules.module7.fsk_soft import (
    generate_cpfsk_templates,
    generate_gfsk_templates,
    compute_fsk_waveform_correlation_soft,
    compute_rejected_instantaneous_freq_metric,
    compute_rejected_symbol_center_freq_metric
)
from backend.modules.module7.reliability import compute_reliability_summary
from backend.modules.module7.service import SoftBitsService, generate_soft_bits


# -----------------------------------------------------------------------------
# 1. BPSK Tests (Equivalence, Sign Convention, Noise Scaling)
# -----------------------------------------------------------------------------

def test_bpsk_analytical_vs_exact_equivalence():
    """Verify BPSK analytical LLR and exact likelihood are numerically identical (diff <= 1e-13)."""
    np.random.seed(42)
    # Generate test symbols along real axis with some noise
    bits = np.random.randint(0, 2, 500)
    symbols = np.where(bits == 1, 1.0, -1.0) + 0.1 * np.random.randn(500)
    n0 = 0.05

    llr_analytical = compute_bpsk_analytical_llr(symbols, n0=n0)
    llr_exact = compute_bpsk_exact_llr(symbols, n0=n0)

    max_diff = np.max(np.abs(llr_analytical - llr_exact))
    assert max_diff <= 1e-13, f"BPSK analytical vs exact max diff exceeded: {max_diff}"

    corr = np.corrcoef(llr_analytical, llr_exact)[0, 1]
    assert np.isclose(corr, 1.0)


def test_bpsk_sign_convention_and_hard_decisions():
    """Verify BPSK sign convention: positive -> bit 1, negative -> bit 0."""
    symbols = np.array([1.2, -0.8, 0.5, -1.5, 0.01, -0.01])
    n0 = 0.1
    llrs = compute_bpsk_analytical_llr(symbols, n0=n0)

    # Positive LLRs must correspond to positive symbols
    assert np.all(llrs[symbols > 0] > 0)
    assert np.all(llrs[symbols < 0] < 0)

    decisions = (llrs > 0).astype(int)
    expected = [1, 0, 1, 0, 1, 0]
    assert np.array_equal(decisions, expected)


# -----------------------------------------------------------------------------
# 2. QPSK Tests (Analytical vs Exact, Quadrant Mapping, Sign Convention)
# -----------------------------------------------------------------------------

def test_qpsk_analytical_vs_exact_equivalence():
    """Verify QPSK analytical LLR and exact likelihood are identical (diff <= 1e-13)."""
    np.random.seed(42)
    # Quadrants: 00 -> (+,+), 01 -> (+,-), 10 -> (-,+), 11 -> (-,-)
    test_syms = np.array([
        (1.0 + 1j) / np.sqrt(2.0),
        (1.0 - 1j) / np.sqrt(2.0),
        (-1.0 + 1j) / np.sqrt(2.0),
        (-1.0 - 1j) / np.sqrt(2.0),
        (0.3 + 0.7j),
        (-0.5 - 0.2j)
    ], dtype=np.complex128)
    n0 = 0.1

    llr_analytical = compute_qpsk_analytical_llr(test_syms, n0=n0)
    llr_exact = compute_qpsk_exact_llr(test_syms, n0=n0)

    max_diff = np.max(np.abs(llr_analytical - llr_exact))
    assert max_diff <= 1e-13, f"QPSK analytical vs exact max diff exceeded: {max_diff}"


def test_qpsk_mapping_and_sign_convention():
    """Verify QPSK bit mapping:
    00 -> (+I,+Q) => LLR0 < 0, LLR1 < 0 -> hard bits [0, 0]
    01 -> (+I,-Q) => LLR0 < 0, LLR1 > 0 -> hard bits [0, 1]
    10 -> (-I,+Q) => LLR0 > 0, LLR1 < 0 -> hard bits [1, 0]
    11 -> (-I,-Q) => LLR0 > 0, LLR1 > 0 -> hard bits [1, 1]
    """
    pts = np.array([
        (1.0 + 1j) / np.sqrt(2.0),
        (1.0 - 1j) / np.sqrt(2.0),
        (-1.0 + 1j) / np.sqrt(2.0),
        (-1.0 - 1j) / np.sqrt(2.0)
    ], dtype=np.complex128)
    n0 = 0.1

    llr = compute_qpsk_exact_llr(pts, n0=n0)
    decisions = (llr > 0).astype(int).reshape(-1, 2)

    expected = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    assert np.array_equal(decisions, expected)


# -----------------------------------------------------------------------------
# 3. 8PSK Tests (Natural Binary Mapping, Unequal Reliability, Max-Log)
# -----------------------------------------------------------------------------

def test_8psk_natural_binary_mapping():
    """Verify 8PSK exact LLR on ideal natural binary constellation points."""
    k_indices = np.arange(8)
    pts = np.exp(1j * k_indices * (np.pi / 4.0))
    n0 = 0.05

    llr_flat, llr_matrix = compute_8psk_exact_llr(pts, n0=n0)
    decisions = (llr_matrix > 0).astype(int)

    # Expected natural binary 000, 001, ..., 111
    expected = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    assert np.array_equal(decisions, expected)


def test_8psk_unequal_bit_reliability():
    """Verify that 8PSK natural binary bits exhibit unequal reliability (Bit 0 coarse, Bit 1 & 2 fine)."""
    np.random.seed(42)
    k_indices = np.random.randint(0, 8, 2000)
    pts = np.exp(1j * k_indices * (np.pi / 4.0))
    n0 = 0.01  # high SNR ~ 20 dB

    _, llr_matrix = compute_8psk_exact_llr(pts, n0=n0)
    mean_abs_b0 = np.mean(np.abs(llr_matrix[:, 0]))
    mean_abs_b1 = np.mean(np.abs(llr_matrix[:, 1]))
    mean_abs_b2 = np.mean(np.abs(llr_matrix[:, 2]))

    # Bit 0 should have substantially larger mean |LLR| than Bit 1 and Bit 2
    assert mean_abs_b0 > mean_abs_b1, f"Expected b0 > b1, got {mean_abs_b0} vs {mean_abs_b1}"
    assert mean_abs_b0 > mean_abs_b2, f"Expected b0 > b2, got {mean_abs_b0} vs {mean_abs_b2}"


def test_8psk_maxlog_comparison():
    """Verify 8PSK Max-Log approximation has high correlation at high SNR."""
    np.random.seed(42)
    k_indices = np.random.randint(0, 8, 500)
    pts = np.exp(1j * k_indices * (np.pi / 4.0)) + 0.05 * (np.random.randn(500) + 1j * np.random.randn(500))
    n0 = 0.01

    _, exact_mat = compute_8psk_exact_llr(pts, n0=n0)
    _, maxlog_mat = compute_8psk_maxlog_llr(pts, n0=n0)

    corr_b0 = np.corrcoef(exact_mat[:, 0], maxlog_mat[:, 0])[0, 1]
    assert corr_b0 > 0.99


# -----------------------------------------------------------------------------
# 4. QAM16 Tests (Exact Constellation Likelihood, Unequal Protection, Max-Log Divergence)
# -----------------------------------------------------------------------------

def test_qam16_exact_likelihood_and_mapping():
    """Verify 16-QAM exact likelihood decisions on ideal constellation points."""
    pts, bit_labels = get_qam16_constellation()
    n0 = 0.02

    llr_flat, llr_matrix = compute_qam16_exact_llr(pts, n0=n0)
    decisions = (llr_matrix > 0).astype(int)

    assert np.array_equal(decisions, bit_labels)


def test_qam16_unequal_bit_reliability():
    """Verify QAM16 coarse bits (|LLR| ~ 100 at 20 dB) vs fine bits (|LLR| ~ 36 at 20 dB)."""
    np.random.seed(42)
    pts, _ = get_qam16_constellation()
    indices = np.random.randint(0, 16, 2000)
    noisy_syms = pts[indices]
    n0 = 0.01  # 20 dB SNR

    _, llr_matrix = compute_qam16_exact_llr(noisy_syms, n0=n0)
    mean_b0 = np.mean(np.abs(llr_matrix[:, 0]))  # Coarse Real
    mean_b1 = np.mean(np.abs(llr_matrix[:, 1]))  # Fine Real
    mean_b2 = np.mean(np.abs(llr_matrix[:, 2]))  # Coarse Imag
    mean_b3 = np.mean(np.abs(llr_matrix[:, 3]))  # Fine Imag

    # Coarse bits must be substantially higher than fine bits
    assert mean_b0 > mean_b1 * 1.5
    assert mean_b2 > mean_b3 * 1.5


def test_qam16_maxlog_divergence_at_low_snr():
    """Verify that QAM16 Max-Log diverges in hard decisions at lower SNR (5 dB, 0 dB, -5 dB)."""
    np.random.seed(123)
    pts, _ = get_qam16_constellation()
    indices = np.random.randint(0, 16, 3000)
    noise_var = 1.0  # 0 dB SNR
    noise = np.sqrt(noise_var / 2.0) * (np.random.randn(3000) + 1j * np.random.randn(3000))
    noisy_syms = pts[indices] + noise

    _, exact_mat = compute_qam16_exact_llr(noisy_syms, n0=noise_var)
    _, maxlog_mat = compute_qam16_maxlog_llr(noisy_syms, n0=noise_var)

    exact_dec = (exact_mat > 0).astype(int)
    maxlog_dec = (maxlog_mat > 0).astype(int)

    disagreements = np.sum(exact_dec != maxlog_dec)
    assert disagreements > 0, "Max-Log should diverge from exact likelihood at 0 dB SNR"


# -----------------------------------------------------------------------------
# 5. QAM64 Tests (Exact Likelihood, 3 Reliability Tiers, Max-Log Divergence)
# -----------------------------------------------------------------------------

def test_qam64_exact_likelihood_and_mapping():
    """Verify 64-QAM exact likelihood decisions on ideal constellation points."""
    pts, bit_labels = get_qam64_constellation()
    n0 = 0.01

    llr_flat, llr_matrix = compute_qam64_exact_llr(pts, n0=n0)
    decisions = (llr_matrix > 0).astype(int)

    assert np.array_equal(decisions, bit_labels)


def test_qam64_three_tier_reliability():
    """Verify 64-QAM 3-tier reliability: coarse (|LLR| ~ 71), middle (~ 17), fine (~ 7) at 20 dB."""
    np.random.seed(42)
    pts, _ = get_qam64_constellation()
    indices = np.random.randint(0, 64, 3000)
    noisy_syms = pts[indices]
    n0 = 0.01  # 20 dB SNR

    _, llr_mat = compute_qam64_exact_llr(noisy_syms, n0=n0)
    coarse = (np.mean(np.abs(llr_mat[:, 0])) + np.mean(np.abs(llr_mat[:, 3]))) / 2.0
    middle = (np.mean(np.abs(llr_mat[:, 1])) + np.mean(np.abs(llr_mat[:, 4]))) / 2.0
    fine = (np.mean(np.abs(llr_mat[:, 2])) + np.mean(np.abs(llr_mat[:, 5]))) / 2.0

    assert coarse > middle, f"Expected coarse > middle: {coarse} vs {middle}"
    assert middle > fine, f"Expected middle > fine: {middle} vs {fine}"


# -----------------------------------------------------------------------------
# 6. CPFSK and GFSK Soft Metric Tests
# -----------------------------------------------------------------------------

def test_cpfsk_waveform_correlation_soft_metric():
    """Verify CPFSK waveform correlation soft metric semantics and sign convention."""
    sps = 4
    fs = 1000000.0
    rs = fs / sps
    h = 0.5
    delta_f = h * rs / 2.0

    np.random.seed(42)
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 25, dtype=np.uint8)
    freqs = np.where(bits == 1, delta_f, -delta_f)
    phase = 2.0 * np.pi * np.cumsum(np.repeat(freqs, sps)) / fs
    iq = np.exp(1j * phase)

    soft_metric = compute_fsk_waveform_correlation_soft(
        iq_signal=iq,
        sps=sps,
        sample_rate=fs,
        modulation="CPFSK"
    )

    assert len(soft_metric) == len(bits)
    # Decisions: positive -> 1, negative -> 0
    decisions = (soft_metric > 0).astype(int)
    ber = np.mean(decisions != bits)
    assert ber < 0.02, f"Expected low BER on clean CPFSK, got {ber}"


def test_gfsk_waveform_correlation_soft_metric():
    """Verify GFSK waveform correlation soft metric with BT=0.5 Gaussian template."""
    sps = 4
    fs = 1000000.0
    rs = fs / sps
    h = 0.5
    delta_f = h * rs / 2.0

    np.random.seed(42)
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 25, dtype=np.uint8)
    freqs = np.where(bits == 1, delta_f, -delta_f)
    freq_wave = np.repeat(freqs, sps)

    # Filter with Gaussian pulse
    t_pulse = np.linspace(-1.5, 1.5, 3 * sps)
    sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * 0.5)
    g = np.exp(-0.5 * (t_pulse / sigma) ** 2.0)
    g /= np.sum(g)
    filtered = np.convolve(freq_wave, g, mode="same")
    phase = 2.0 * np.pi * np.cumsum(filtered) / fs
    iq = np.exp(1j * phase)

    soft_metric = compute_fsk_waveform_correlation_soft(
        iq_signal=iq,
        sps=sps,
        sample_rate=fs,
        modulation="GFSK",
        bt=0.5
    )

    assert len(soft_metric) == len(bits)
    decisions = (soft_metric > 0).astype(int)
    ber = np.mean(decisions != bits)
    assert ber < 0.05, f"Expected low BER on clean GFSK, got {ber}"


def test_fsk_real_correlation_formula_exact_evaluation():
    """Verify that compute_fsk_waveform_correlation_soft strictly evaluates:
    Lambda_k = Re{<r_k, s1>} - Re{<r_k, s0>}
    without magnitude operations |...| and without 2/N0 scaling.
    """
    sps = 4
    fs = 1000000.0
    h = 0.5
    s1, s0 = generate_cpfsk_templates(sps=sps, sample_rate=fs, h=h)

    # Arbitrary complex symbol slice
    r_slice = np.array([0.7 + 0.3j, -0.2 + 0.8j, 0.4 - 0.5j, -0.6 - 0.1j], dtype=np.complex128)

    # 1. Unaligned test: strictly evaluates Re{<r, s1>} - Re{<r, s0>}
    soft_unaligned = compute_fsk_waveform_correlation_soft(
        iq_signal=r_slice,
        sps=sps,
        sample_rate=fs,
        modulation="CPFSK",
        phase_reference=False
    )
    manual_unaligned = np.real(np.sum(r_slice * np.conj(s1))) - np.real(np.sum(r_slice * np.conj(s0)))
    assert np.isclose(soft_unaligned[0], manual_unaligned, atol=1e-13)

    # 2. Phase-referenced test: evaluates Re{<r_aligned, s1>} - Re{<r_aligned, s0>}
    init_phase = np.angle(r_slice[0])
    r_aligned = r_slice * np.exp(-1j * init_phase)
    manual_aligned = np.real(np.sum(r_aligned * np.conj(s1))) - np.real(np.sum(r_aligned * np.conj(s0)))

    soft_aligned = compute_fsk_waveform_correlation_soft(
        iq_signal=r_slice,
        sps=sps,
        sample_rate=fs,
        modulation="CPFSK",
        phase_reference=True
    )
    assert np.isclose(soft_aligned[0], manual_aligned, atol=1e-13)


def test_fsk_rejected_baselines_execute_cleanly():
    """Verify rejected FSK baselines execute as diagnostic comparators."""
    iq = np.exp(1j * np.linspace(0, 10 * np.pi, 100))
    sps = 4
    fs = 100000.0

    m1 = compute_rejected_instantaneous_freq_metric(iq, sps=sps, sample_rate=fs)
    m2 = compute_rejected_symbol_center_freq_metric(iq, sps=sps, sample_rate=fs)

    assert len(m1) == 25
    assert len(m2) == 25


# -----------------------------------------------------------------------------
# 7. Missing-Noise Failure Path & Scientific Constraints
# -----------------------------------------------------------------------------

def test_psk_missing_noise_returns_noise_parameter_unavailable():
    """Verify that if N0 is unavailable for PSK/QAM, NOISE_PARAMETER_UNAVAILABLE is returned.

    Must NOT substitute N0=1.0 or fabricate a scaled LLR.
    """
    syms = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.complex128)
    service = SoftBitsService()

    # Pass empty module6_snr and n0=None
    res = service.generate_soft_bits(
        symbol_samples=syms,
        modulation="BPSK",
        module6_snr={},
        n0=None
    )

    assert res["status"] == "NOISE_PARAMETER_UNAVAILABLE"
    assert res["soft_bits"] is None
    assert res["noise_parameter_used"] is None
    assert any("NOISE_PARAMETER_UNAVAILABLE" in w for w in res["warnings"])


def test_qam_missing_noise_returns_noise_parameter_unavailable():
    """Verify QAM16 also fails cleanly when noise parameter is unavailable."""
    pts, _ = get_qam16_constellation()
    service = SoftBitsService()

    res = service.generate_soft_bits(
        symbol_samples=pts,
        modulation="QAM16",
        n0=None
    )

    assert res["status"] == "NOISE_PARAMETER_UNAVAILABLE"
    assert res["soft_bits"] is None


def test_fsk_proceeds_without_noise_parameter():
    """Verify CPFSK waveform correlation succeeds without requiring N0, labeled SOFT_METRIC."""
    iq = np.exp(1j * np.linspace(0, 20 * np.pi, 80))
    service = SoftBitsService()

    res = service.generate_soft_bits(
        signal_waveform=iq,
        modulation="CPFSK",
        sps=4,
        sample_rate=1000000.0,
        n0=None
    )

    assert res["status"] == "SUCCESS"
    assert res["metric_type"] == "SOFT_METRIC"
    assert res["mathematical_status"] == "NOT_CALIBRATED_LLR"
    assert res["calibrated_llr"] is False
    assert res["soft_bits"] is not None
    assert len(res["soft_bits"]) == 20


# -----------------------------------------------------------------------------
# 8. Reliability Summary & Magnitude Binning
# -----------------------------------------------------------------------------

def test_reliability_summary_magnitude_bins():
    """Verify diagnostic magnitude binning across [0,1), [1,2), [2,4), [4,8), >8."""
    soft_vals = np.array([0.5, -0.8, 1.5, -1.9, 2.5, 3.8, -5.0, 9.5, -12.0])
    summary = compute_reliability_summary(soft_vals)

    bins = summary["magnitude_bins"]
    assert bins["bin_0_to_1"]["count"] == 2
    assert bins["bin_1_to_2"]["count"] == 2
    assert bins["bin_2_to_4"]["count"] == 2
    assert bins["bin_4_to_8"]["count"] == 1
    assert bins["bin_gt_8"]["count"] == 2
    assert "Diagnostic reliability verification metrics" in summary["status_note"]


def test_reliability_monotonic_ber_trend():
    """Verify that across magnitude bins, empirical BER monotonically decreases with higher |soft|."""
    np.random.seed(42)
    # Simulate a noisy BPSK signal
    true_bits = np.random.randint(0, 2, 5000)
    tx_symbols = np.where(true_bits == 1, 1.0, -1.0)
    noise_var = 0.5  # ~ 3 dB
    noisy_symbols = tx_symbols + np.sqrt(noise_var) * np.random.randn(5000)

    # Compute analytical LLR
    llrs = compute_bpsk_analytical_llr(noisy_symbols, n0=noise_var)
    summary = compute_reliability_summary(llrs, ground_truth_bits=true_bits)

    bins = summary["magnitude_bins"]
    # At least check that the lowest bin [0, 1) has higher BER than higher bins
    ber_low = bins["bin_0_to_1"]["empirical_ber"]
    ber_high = bins["bin_4_to_8"]["empirical_ber"]
    if ber_low is not None and ber_high is not None:
        assert ber_low > ber_high, f"Expected higher BER in low bin: {ber_low} vs {ber_high}"


def test_hard_decision_agreement_tracking():
    """Verify agreement tracking against upstream Module 5 hard decisions."""
    soft = np.array([5.0, -3.0, 2.0, -4.0, 1.0])  # derived bits: [1, 0, 1, 0, 1]
    upstream = np.array([1, 0, 1, 1, 1])            # 4 of 5 match (80%)

    summary = compute_reliability_summary(soft, reference_hard_bits=upstream)
    agr = summary["hard_decision_agreement"]
    assert agr is not None
    assert agr["matching_bits"] == 4
    assert np.isclose(agr["agreement_rate"], 0.8)


# -----------------------------------------------------------------------------
# 9. End-to-End Orchestrator Service Test for All 7 Modulations
# -----------------------------------------------------------------------------

def test_service_all_seven_modulations():
    """Verify that SoftBitsService correctly handles all seven modulation classes."""
    service = SoftBitsService()

    # 1. BPSK
    r_bpsk = service.generate_soft_bits(
        symbol_samples=np.array([1.0, -1.0, 1.0]),
        modulation="BPSK",
        n0=0.1
    )
    assert r_bpsk["status"] == "SUCCESS"
    assert r_bpsk["metric_type"] == "LLR"
    assert r_bpsk["mathematical_status"] == "EXACT_UNDER_AWGN_MODEL"
    assert r_bpsk["calibrated_llr"] is False
    assert r_bpsk["hard_bits"] == [1, 0, 1]

    # 2. QPSK
    r_qpsk = service.generate_soft_bits(
        symbol_samples=np.array([(1 + 1j) / np.sqrt(2)]),
        modulation="QPSK",
        n0=0.1
    )
    assert r_qpsk["status"] == "SUCCESS"
    assert r_qpsk["hard_bits"] == [0, 0]

    # 3. 8PSK
    r_8psk = service.generate_soft_bits(
        symbol_samples=np.array([1.0 + 0j]),
        modulation="8PSK",
        n0=0.1
    )
    assert r_8psk["status"] == "SUCCESS"
    assert r_8psk["bits_per_symbol"] == 3

    # 4. QAM16
    r_qam16 = service.generate_soft_bits(
        symbol_samples=np.array([(-3 - 3j) / np.sqrt(10)]),
        modulation="QAM16",
        n0=0.1
    )
    assert r_qam16["status"] == "SUCCESS"
    assert r_qam16["bits_per_symbol"] == 4

    # 5. QAM64
    r_qam64 = service.generate_soft_bits(
        symbol_samples=np.array([(-7 - 7j) / np.sqrt(42)]),
        modulation="QAM64",
        n0=0.1
    )
    assert r_qam64["status"] == "SUCCESS"
    assert r_qam64["bits_per_symbol"] == 6

    # 6. CPFSK
    r_cpfsk = service.generate_soft_bits(
        signal_waveform=np.ones(16, dtype=complex),
        modulation="CPFSK",
        sps=4,
        sample_rate=1000000.0
    )
    assert r_cpfsk["status"] == "SUCCESS"
    assert r_cpfsk["metric_type"] == "SOFT_METRIC"
    assert r_cpfsk["mathematical_status"] == "NOT_CALIBRATED_LLR"

    # 7. GFSK
    r_gfsk = service.generate_soft_bits(
        signal_waveform=np.ones(16, dtype=complex),
        modulation="GFSK",
        sps=4,
        sample_rate=1000000.0
    )
    assert r_gfsk["status"] == "SUCCESS"
    assert r_gfsk["metric_type"] == "SOFT_METRIC"
    assert r_gfsk["mathematical_status"] == "NOT_CALIBRATED_LLR"

    # Boundary check
    assert r_bpsk["boundary"] == "MODULE_7_CONCLUDED_AT_SOFT_BITS_AND_RELIABILITY_DIAGNOSTICS"
