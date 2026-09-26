"""Comprehensive Unit and Scientific Validation Tests for Module 8 (Blind FEC & Interleaver).

Covers all prompt and Phase 0 blueprint requirements:
1. Hamming (7,4):
   - Parity matrix H @ G.T = 0 (mod 2) mathematical validation.
   - Zero syndrome fraction on valid codewords.
   - Random block baseline zero syndrome rate (P ≈ 0.125).
   - Alignment search recovering true cyclic offset.
   - Hard syndrome decoding with single-bit error correction.
   - CRITICAL: Documented low-SNR hard-decoder miscorrection verification (Section 16).
   - 16-codeword soft ML decoding outperforming hard decoding.
2. BCH (15,7):
   - Generator polynomial g(x) = x^8 + x^7 + x^6 + x^4 + 1.
   - Codebook generation: 128 codewords, minimum Hamming distance = 5.
   - Polynomial remainder zero-syndrome fraction.
   - Random block baseline remainder rate (P ≈ 1/256 ≈ 0.0039).
   - Codeword alignment search over 15 cyclic offsets.
   - Hard minimum distance decoding over 128 codewords.
   - Soft ML correlation decoding over 128 codewords.
3. Family Separation & Ranking (No scalar fusion score):
   - BCH stream: BCH lift >> Hamming lift.
   - Hamming stream: Hamming lift >> BCH lift.
   - Separate evidence metric reporting.
4. Convolutional (K=3, r=1/2, [111, 101]):
   - Encoding and trellis transitions.
   - Hard Viterbi decoding.
   - Soft Viterbi decoding with LLR inputs.
   - Viterbi reconstruction mismatch: true candidate [111, 101] << false candidates.
5. Structured Row-Column Interleaver:
   - Candidate width search: true width W identified via post-deinterleaving syndrome check.
   - Row-column interleaving and deinterleaving roundtrip.
   - Arbitrary permutation and general phase rejection documentation.
6. End-to-End Orchestrator Service:
   - Full pipeline execution on Hamming stream.
   - Full pipeline execution on BCH stream.
   - Full pipeline execution on Convolutional stream.
   - Uncoded signal detection (NO_FEC_DETECTED).
   - Insufficient input error handling.
   - BER diagnostics calculation when ground truth information bits are supplied.
   - Module 8 boundary verification.
"""

import numpy as np
import pytest

from backend.modules.module8.hamming import (
    HAMMING_H,
    HAMMING_G,
    HAMMING_CODEWORDS,
    encode_hamming_7_4,
    compute_hamming_syndromes,
    find_hamming_alignment,
    compute_hamming_evidence,
    decode_hamming_hard,
    decode_hamming_soft_ml
)
from backend.modules.module8.bch import (
    BCH_G_POLY,
    BCH_CODEWORDS,
    encode_bch_15_7,
    compute_bch_syndromes,
    find_bch_alignment,
    compute_bch_evidence,
    decode_bch_hard,
    decode_bch_soft_ml
)
from backend.modules.module8.convolutional import (
    CONV_G1,
    CONV_G2,
    conv_encode_k3_r12,
    viterbi_decode_hard,
    viterbi_decode_soft,
    compute_convolutional_evidence
)
from backend.modules.module8.interleaver import (
    interleave_row_column,
    deinterleave_row_column,
    search_interleaver_width
)
from backend.modules.module8.service import (
    FecRecoveryService,
    process_fec_recovery
)


# -----------------------------------------------------------------------------
# 1. Hamming (7,4) Tests
# -----------------------------------------------------------------------------

def test_hamming_matrix_orthogonality():
    """Verify H @ G.T = 0 (mod 2) from source Section 2."""
    ortho = (HAMMING_H @ HAMMING_G.T) % 2
    assert np.all(ortho == 0), "H @ G.T must be identically zero over GF(2)"


def test_hamming_syndrome_valid_and_random():
    """Verify valid codewords have zero syndrome and random blocks match 2^(-3) = 0.125."""
    np.random.seed(42)
    n_blocks = 200
    msg = np.random.randint(0, 2, n_blocks * 4)
    coded = encode_hamming_7_4(msg)

    # Valid codewords
    _, zero_frac = compute_hamming_syndromes(coded)
    assert zero_frac == 1.0, f"Valid codewords must have zero_syndrome_fraction 1.0, got {zero_frac}"

    # Random blocks
    rand_bits = np.random.randint(0, 2, 10000 * 7)
    _, rand_frac = compute_hamming_syndromes(rand_bits)
    # Expected ≈ 0.125 (measured in source ≈ 0.1256)
    assert np.isclose(rand_frac, 0.125, atol=0.015), f"Random zero syndrome should be ~0.125, got {rand_frac}"


def test_hamming_alignment_search():
    """Verify alignment search recovers true cyclic shift offset."""
    np.random.seed(42)
    msg = np.random.randint(0, 2, 100 * 4)
    coded = encode_hamming_7_4(msg)

    # Inject an offset of 3 bits at the beginning
    offset_true = 3
    shifted_bits = np.concatenate([np.random.randint(0, 2, offset_true), coded])

    best_offset, best_frac, all_offsets = find_hamming_alignment(shifted_bits)
    assert best_offset == offset_true, f"Expected offset {offset_true}, got {best_offset}"
    assert best_frac > 0.95


def test_hamming_hard_decoding_single_bit_error():
    """Verify hard decoder corrects single bit errors per block."""
    np.random.seed(42)
    msg = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)  # 2 blocks of 4 bits
    coded = encode_hamming_7_4(msg)

    # Introduce 1 bit error in block 0 (index 2) and 1 error in block 1 (index 10)
    corrupted = np.copy(coded)
    corrupted[2] ^= 1
    corrupted[10] ^= 1

    decoded, meta = decode_hamming_hard(corrupted)
    assert np.array_equal(decoded, msg), "Hard decoder must correct single-bit errors"
    assert meta["corrected_errors"] == 2


def test_hamming_hard_decoder_miscorrection_at_low_snr():
    """Verify documented Section 16 miscorrection: multi-bit errors cause decoded BER > raw BER."""
    np.random.seed(123)
    n_blocks = 2000
    msg = np.random.randint(0, 2, n_blocks * 4, dtype=np.uint8)
    coded = encode_hamming_7_4(msg)

    # Channel noise corresponding to Section 16 -5 dB condition (raw BER ≈ 0.285)
    noise = (np.random.rand(len(coded)) < 0.285).astype(np.uint8)
    rx = coded ^ noise
    raw_ber = float(np.mean(noise))

    decoded, meta = decode_hamming_hard(rx)
    decoded_ber = float(np.mean(decoded != msg))

    # In Hamming (7,4), multi-bit errors cause the syndrome to point to an incorrect bit,
    # leading to miscorrection where decoded BER exceeds raw BER (Section 16).
    assert decoded_ber > raw_ber, (
        f"Documented Section 16 miscorrection requires decoded BER ({decoded_ber:.4f}) > raw BER ({raw_ber:.4f})"
    )
    assert meta["corrected_errors"] > 0


def test_hamming_soft_ml_decoding():
    """Verify soft ML decoder outperforming hard decoding under Gaussian noise."""
    np.random.seed(42)
    n_blocks = 100
    msg = np.random.randint(0, 2, n_blocks * 4, dtype=np.uint8)
    coded = encode_hamming_7_4(msg)

    # BPSK modulation + AWGN at moderate SNR (e.g. 5 dB)
    tx = np.where(coded == 1, 1.0, -1.0)
    snr_lin = 10.0 ** (5.0 / 10.0)
    noise_var = 1.0 / snr_lin
    rx = tx + np.sqrt(noise_var / 2.0) * np.random.randn(len(tx))
    llrs = 4.0 * rx / noise_var

    # Soft ML decoding
    dec_soft, _ = decode_hamming_soft_ml(llrs)
    ber_soft = np.mean(dec_soft != msg)

    # Hard decoding
    hard_bits = (llrs > 0).astype(np.uint8)
    dec_hard, _ = decode_hamming_hard(hard_bits)
    ber_hard = np.mean(dec_hard != msg)

    # Soft ML should achieve lower or equal BER compared to hard decoding
    assert ber_soft <= ber_hard, f"Soft ML BER ({ber_soft}) should be <= Hard BER ({ber_hard})"


# -----------------------------------------------------------------------------
# 2. BCH (15,7) Tests
# -----------------------------------------------------------------------------

def test_bch_codebook_properties():
    """Verify BCH (15,7) generator polynomial g(x) produces 128 codewords with min distance 5."""
    assert len(BCH_CODEWORDS) == 128
    assert BCH_CODEWORDS.shape == (128, 15)

    # Minimum weight of non-zero codewords must be 5 (t=2)
    weights = [np.sum(cw) for cw in BCH_CODEWORDS if np.sum(cw) > 0]
    assert min(weights) == 5, f"BCH (15,7) minimum distance must be 5, got {min(weights)}"


def test_bch_syndrome_valid_and_random():
    """Verify valid BCH codewords have zero polynomial remainder and random rate ≈ 1/256."""
    np.random.seed(42)
    msg = np.random.randint(0, 2, 50 * 7)
    coded = encode_bch_15_7(msg)

    _, zero_frac = compute_bch_syndromes(coded)
    assert zero_frac == 1.0, f"Valid BCH codewords must have zero_syndrome_fraction 1.0, got {zero_frac}"

    # Random blocks
    rand_bits = np.random.randint(0, 2, 5000 * 15)
    _, rand_frac = compute_bch_syndromes(rand_bits)
    assert np.isclose(rand_frac, 1.0 / 256.0, atol=0.005)


def test_bch_alignment_search():
    """Verify BCH alignment search recovers true cyclic shift."""
    np.random.seed(42)
    msg = np.random.randint(0, 2, 50 * 7)
    coded = encode_bch_15_7(msg)

    offset_true = 7
    shifted = np.concatenate([np.random.randint(0, 2, offset_true), coded])

    best_offset, best_frac, _ = find_bch_alignment(shifted)
    assert best_offset == offset_true, f"Expected BCH offset {offset_true}, got {best_offset}"
    assert best_frac > 0.95


def test_bch_hard_and_soft_decoding():
    """Verify BCH hard nearest-codeword decoding and soft ML decoding."""
    np.random.seed(42)
    msg = np.random.randint(0, 2, 40 * 7, dtype=np.uint8)
    coded = encode_bch_15_7(msg)

    # 1. Clean test
    dec_hard, _ = decode_bch_hard(coded)
    assert np.array_equal(dec_hard, msg)

    # 2. Noisy test with BPSK + AWGN at 5 dB
    tx = np.where(coded == 1, 1.0, -1.0)
    snr_lin = 10.0 ** (5.0 / 10.0)
    noise_var = 1.0 / snr_lin
    rx = tx + np.sqrt(noise_var / 2.0) * np.random.randn(len(tx))
    llrs = 4.0 * rx / noise_var

    dec_soft, _ = decode_bch_soft_ml(llrs)
    ber_soft = np.mean(dec_soft != msg)

    dec_hard_noisy, _ = decode_bch_hard((llrs > 0).astype(np.uint8))
    ber_hard_noisy = np.mean(dec_hard_noisy != msg)

    # Soft ML decoding should outperform hard decoding as documented in Section 18
    assert ber_soft <= ber_hard_noisy


# -----------------------------------------------------------------------------
# 3. Family Separation & Lift Ranking Tests (No Scalar Fusion)
# -----------------------------------------------------------------------------

def test_family_identification_bch_vs_hamming():
    """Verify BCH lift >> Hamming lift on BCH stream, and vice-versa on Hamming stream."""
    np.random.seed(42)
    # A. BCH Stream
    msg_bch = np.random.randint(0, 2, 60 * 7)
    coded_bch = encode_bch_15_7(msg_bch)
    ev_h_on_bch = compute_hamming_evidence(coded_bch)
    ev_b_on_bch = compute_bch_evidence(coded_bch)

    # On BCH stream, BCH lift must be substantially higher than Hamming lift
    assert ev_b_on_bch["normalized_lift"] > ev_h_on_bch["normalized_lift"] * 2.0

    # B. Hamming Stream
    msg_h = np.random.randint(0, 2, 100 * 4)
    coded_h = encode_hamming_7_4(msg_h)
    ev_h_on_h = compute_hamming_evidence(coded_h)
    ev_b_on_h = compute_bch_evidence(coded_h)

    # On Hamming stream, Hamming lift must be substantially higher than BCH lift
    assert ev_h_on_h["normalized_lift"] > ev_b_on_h["normalized_lift"] * 2.0


# -----------------------------------------------------------------------------
# 4. Convolutional (K=3, r=1/2) Tests
# -----------------------------------------------------------------------------

def test_convolutional_encoding_and_viterbi_hard():
    """Verify convolutional encoding and hard Viterbi decoding for K=3, r=1/2."""
    np.random.seed(42)
    msg = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
    coded = conv_encode_k3_r12(msg)
    assert len(coded) == 2 * len(msg)

    # Hard decode clean
    decoded = viterbi_decode_hard(coded)
    assert np.array_equal(decoded, msg)


def test_convolutional_soft_viterbi_decoding():
    """Verify soft Viterbi decoding with LLR inputs."""
    np.random.seed(42)
    msg = np.random.randint(0, 2, 200, dtype=np.uint8)
    coded = conv_encode_k3_r12(msg)

    # Modulate + AWGN
    tx = np.where(coded == 1, 1.0, -1.0)
    noise = 0.4 * np.random.randn(len(tx))
    rx = tx + noise
    llrs = 4.0 * rx / (0.4 ** 2)

    decoded = viterbi_decode_soft(llrs)
    ber = np.mean(decoded != msg)
    assert ber < 0.02


def test_convolutional_reconstruction_mismatch_evidence():
    """Verify true generator polynomials [111, 101] produce lower mismatch than false candidates."""
    np.random.seed(42)
    msg = np.random.randint(0, 2, 300, dtype=np.uint8)
    coded = conv_encode_k3_r12(msg)

    # Add 5% noise errors
    corrupted = np.copy(coded)
    err_idx = np.random.choice(len(coded), int(0.05 * len(coded)), replace=False)
    corrupted[err_idx] ^= 1

    ev = compute_convolutional_evidence(corrupted)
    assert ev["best_candidate"] == "CONV_K3_R12_7_5_TRUE"
    assert ev["reconstruction_mismatch"] < 0.10

    # True candidate mismatch must be lower than any false candidate
    scores = ev["candidate_scores"]
    true_score = scores["CONV_K3_R12_7_5_TRUE"]
    for label, score in scores.items():
        if label != "CONV_K3_R12_7_5_TRUE":
            assert true_score < score


# -----------------------------------------------------------------------------
# 5. Structured Row-Column Interleaver Tests
# -----------------------------------------------------------------------------

def test_interleaver_roundtrip():
    """Verify row-column interleave and deinterleave roundtrip identity."""
    data = np.arange(100, dtype=np.uint8)
    width = 5
    interleaved = interleave_row_column(data, width=width)
    deinterleaved = deinterleave_row_column(interleaved, width=width)
    assert np.array_equal(data, deinterleaved)


def test_interleaver_width_search():
    """Verify candidate width search identifies true row-column interleaver width."""
    np.random.seed(42)
    n_words = 60
    msg = np.random.randint(0, 2, (n_words, 4), dtype=np.uint8)
    coded = encode_hamming_7_4(msg.flatten())  # 420 bits

    width_true = 5
    interleaved = interleave_row_column(coded, width=width_true)

    res = search_interleaver_width(interleaved, fec_family="HAMMING", candidate_widths=list(range(2, 11)))
    assert res["is_interleaved"] is True
    assert res["interleaver_type"] == "ROW_COLUMN"
    assert res["estimated_width"] == width_true, f"Expected width {width_true}, got {res['estimated_width']}"
    assert "Arbitrary permutation recovery is REJECTED" in res["rejected_approaches"][0]


# -----------------------------------------------------------------------------
# 6. Service Orchestration & Boundary Tests
# -----------------------------------------------------------------------------

def test_service_end_to_end_hamming():
    """Verify full end-to-end FEC identification and soft ML decoding on Hamming stream."""
    np.random.seed(42)
    msg = np.random.randint(0, 2, 80 * 4, dtype=np.uint8)
    coded = encode_hamming_7_4(msg)

    # BPSK + AWGN
    tx = np.where(coded == 1, 1.0, -1.0)
    llrs = tx + 0.3 * np.random.randn(len(tx))

    service = FecRecoveryService()
    res = service.process_fec(
        soft_bits=llrs,
        metric_type="LLR",
        ground_truth_info_bits=msg
    )

    assert res["status"] == "SUCCESS"
    assert res["detected_fec_family"] == "HAMMING"
    assert res["decoder_applied"] == "HAMMING_SOFT_ML"
    assert res["code_parameters"]["n"] == 7
    assert res["code_parameters"]["k"] == 4
    assert res["ber_diagnostics"]["decoded_ber"] == 0.0
    assert res["boundary"] == "MODULE_8_CONCLUDED_AT_FEC_AND_RECOVERED_BITS"


def test_service_end_to_end_bch():
    """Verify full end-to-end FEC identification and soft ML decoding on BCH stream."""
    np.random.seed(42)
    msg = np.random.randint(0, 2, 40 * 7, dtype=np.uint8)
    coded = encode_bch_15_7(msg)

    tx = np.where(coded == 1, 1.0, -1.0)
    llrs = tx + 0.3 * np.random.randn(len(tx))

    service = FecRecoveryService()
    res = service.process_fec(
        soft_bits=llrs,
        metric_type="LLR",
        ground_truth_info_bits=msg
    )

    assert res["status"] == "SUCCESS"
    assert res["detected_fec_family"] == "BCH"
    assert res["decoder_applied"] == "BCH_SOFT_ML"
    assert res["code_parameters"]["n"] == 15
    assert res["code_parameters"]["k"] == 7
    assert res["ber_diagnostics"]["decoded_ber"] == 0.0


def test_service_end_to_end_convolutional():
    """Verify full end-to-end FEC identification and Viterbi decoding on Convolutional stream."""
    np.random.seed(42)
    msg = np.random.randint(0, 2, 200, dtype=np.uint8)
    coded = conv_encode_k3_r12(msg)

    tx = np.where(coded == 1, 1.0, -1.0)
    llrs = tx + 0.3 * np.random.randn(len(tx))

    service = FecRecoveryService()
    res = service.process_fec(
        soft_bits=llrs,
        metric_type="LLR",
        ground_truth_info_bits=msg
    )

    assert res["status"] == "SUCCESS"
    assert res["detected_fec_family"] == "CONVOLUTIONAL"
    assert res["decoder_applied"] == "VITERBI_SOFT"
    assert res["code_parameters"]["k_constraint"] == 3
    assert res["ber_diagnostics"]["decoded_ber"] == 0.0


def test_service_uncoded_handling():
    """Verify random uncoded stream returns NO_FEC_DETECTED / UNCODED."""
    np.random.seed(42)
    rand_bits = np.random.randint(0, 2, 1000, dtype=np.uint8)

    service = FecRecoveryService()
    res = service.process_fec(hard_bits=rand_bits)

    assert res["status"] == "NO_FEC_DETECTED"
    assert res["detected_fec_family"] == "UNCODED"
    assert res["decoder_applied"] == "NONE_PASSTHROUGH"
    assert np.array_equal(res["recovered_information_bits"], rand_bits)


def test_service_insufficient_input_error():
    """Verify service returns explicit ERROR on empty inputs."""
    service = FecRecoveryService()
    res = service.process_fec(soft_bits=[], hard_bits=[])

    assert res["status"] == "ERROR"
    assert "INSUFFICIENT_INPUT" in res["error"]


def test_unsupported_thresholds_not_treated_as_scientific_constants():
    """Verify that family selection is explicitly marked as an UNVALIDATED_ENGINEERING_HEURISTIC,
    evidence mechanisms are kept separate without scalar fusion, candidate evaluations are reported
    for each family, and ambiguous evidence is not forced into a false positive."""
    np.random.seed(42)
    msg = np.random.randint(0, 2, 80 * 4, dtype=np.uint8)
    coded = encode_hamming_7_4(msg)

    service = FecRecoveryService()
    res = service.process_fec(hard_bits=coded)

    # 1. Selection policy must be explicitly labeled as unvalidated engineering heuristic
    assert res["selection_policy"] == "UNVALIDATED_ENGINEERING_HEURISTIC"
    assert "selection_policy_note" in res
    assert any("UNVALIDATED_ENGINEERING_HEURISTIC" in w for w in res["warnings"])

    # 2. Evidence mechanisms must remain separate without a universal scalar score
    ev = res["evidence_scores"]
    assert "hamming" in ev
    assert "bch" in ev
    assert "convolutional" in ev
    assert "fec_score" not in ev
    assert "scalar_score" not in ev

    # 3. Candidate evaluations for each family are independently available
    cands = res["candidate_evaluations"]
    assert "HAMMING" in cands
    assert "BCH" in cands
    assert "CONVOLUTIONAL" in cands
    assert cands["HAMMING"]["parameters"]["n"] == 7
    assert cands["BCH"]["parameters"]["n"] == 15
    assert cands["CONVOLUTIONAL"]["parameters"]["k_constraint"] == 3
    assert "candidate_decoders" in cands["HAMMING"]
    assert "candidate_decoders" in cands["BCH"]
    assert "candidate_decoders" in cands["CONVOLUTIONAL"]

