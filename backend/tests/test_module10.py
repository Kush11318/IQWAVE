"""Comprehensive Tests for Module 10: FEC, CRC & Interleaver Analysis.

Directly reproduces and verifies experimental results documented in:
# Module 10 — FEC, CRC & Interleave(2).txt

Test Categories:
A. CRC-16-CCITT & Candidate Polynomials (10A)
B. CRC Boundary Search & Soft Evidence (10A)
C. Hamming(7,4) Structure & Decoding (10B)
D. BCH(15,7) Codebook, Lift & Decoding (10B)
E. Reed-Solomon GF(16) RS(15,11) Decoding & Parameter Ranking (10B)
F. Controlled LDPC (6,12) Syndrome & Candidate Ranking (10C)
G. Convolutional (R=1/2, K=3) Encoder & Viterbi Decoder (10C)
H. Structured Row-Column Interleaver & Width Search (10D)
I. Joint FEC + Interleaver Hypothesis Engine (10D)
J. Decoder Cross-Validation & Miscorrection Safeguards
K. Excluded/Rejected Methods Safeguards
L. API Integration
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.modules.module10.crc import (
    compute_crc,
    verify_crc16_ccitt,
    append_crc16_ccitt,
    evaluate_crc_acceptance_rate,
    search_crc_boundaries,
    identify_crc_family,
    compute_soft_crc_evidence
)
from backend.modules.module10.hamming import (
    HAMMING_7_4_H,
    HAMMING_7_4_G,
    encode_hamming_7_4,
    decode_hamming_7_4,
    evaluate_hamming_evidence
)
from backend.modules.module10.bch import (
    encode_bch_15_7,
    decode_bch_15_7,
    evaluate_bch_evidence,
    BCH_CODEWORDS_128
)
from backend.modules.module10.reed_solomon import (
    encode_rs_15_11,
    decode_rs_15_11,
    evaluate_rs_parameter_candidates
)
from backend.modules.module10.ldpc import (
    LDPC_CONTROLLED_H,
    LDPC_VALID_CODEWORDS_64,
    evaluate_ldpc_syndrome,
    rank_ldpc_candidates
)
from backend.modules.module10.convolutional import (
    encode_convolutional_k3,
    viterbi_decode_hard_k3,
    identify_convolutional_generator
)
from backend.modules.module10.interleaver import (
    interleave_row_column,
    deinterleave_row_column,
    identify_interleaver_width
)
from backend.modules.module10.joint_engine import evaluate_joint_hypotheses
from backend.modules.module10.service import (
    default_fec_crc_service,
    process_fec_crc_analysis,
    MODULE10_REJECTED_METHODS
)

client = TestClient(app)


# ==============================================================================
# Category A: CRC-16-CCITT & Candidate Polynomials (10A)
# ==============================================================================

def test_crc16_clean_and_single_bit_flip():
    """Verify that CRC-16 passes on clean frame and fails on a single bit modification (Section 4.1)."""
    # 176 bits data (16 preamble + 32 header + 128 payload)
    rng = np.random.RandomState(42)
    data = rng.randint(0, 2, 176, dtype=np.uint8)
    frame = append_crc16_ccitt(data)

    assert len(frame) == 192
    assert verify_crc16_ccitt(frame, payload_end_idx=176, crc_width=16) is True

    # Single bit modification in data
    frame_corrupted = frame.copy()
    frame_corrupted[10] ^= 1
    assert verify_crc16_ccitt(frame_corrupted, payload_end_idx=176, crc_width=16) is False

    # Single bit modification in CRC itself
    frame_crc_corrupted = frame.copy()
    frame_crc_corrupted[180] ^= 1
    assert verify_crc16_ccitt(frame_crc_corrupted, payload_end_idx=176, crc_width=16) is False


def test_crc_acceptance_rate_under_ber():
    """Verify CRC acceptance rate collapses as BER increases (Section 4.2)."""
    rng = np.random.RandomState(99)
    M = 50
    frames_clean = np.zeros((M, 192), dtype=np.uint8)
    for m in range(M):
        data = rng.randint(0, 2, 176, dtype=np.uint8)
        frames_clean[m] = append_crc16_ccitt(data)

    # 0% BER -> 100% acceptance
    res_0 = evaluate_crc_acceptance_rate(frames_clean, payload_end_idx=176)
    assert res_0["acceptance_rate"] == 1.0

    # 10% BER -> 0% acceptance
    noise = rng.rand(M, 192) < 0.10
    frames_noisy = frames_clean ^ noise.astype(np.uint8)
    res_10 = evaluate_crc_acceptance_rate(frames_noisy, payload_end_idx=176)
    assert res_10["acceptance_rate"] == 0.0


# ==============================================================================
# Category B: CRC Boundary Search & Soft Evidence (10A)
# ==============================================================================

def test_crc_boundary_search():
    """Verify that candidate boundary search identifies correct payload/CRC boundary (Section 4.3)."""
    rng = np.random.RandomState(123)
    M = 20
    frames = np.zeros((M, 192), dtype=np.uint8)
    for m in range(M):
        data = rng.randint(0, 2, 176, dtype=np.uint8)
        frames[m] = append_crc16_ccitt(data)

    res = search_crc_boundaries(frames, candidate_boundaries=[160, 168, 176, 184])
    assert res["status"] == "SUCCESS"
    assert res["best_boundary"] == 176
    assert res["best_acceptance_rate"] == 1.0


def test_crc_family_identification():
    """Verify candidate CRC polynomial ranking (Section 4.4)."""
    rng = np.random.RandomState(55)
    M = 10
    frames = np.zeros((M, 192), dtype=np.uint8)
    for m in range(M):
        data = rng.randint(0, 2, 176, dtype=np.uint8)
        frames[m] = append_crc16_ccitt(data)

    res = identify_crc_family(frames, payload_end_idx=176)
    assert res["best_candidate"] == "CRC-16-CCITT"
    assert res["best_acceptance_rate"] == 1.0


def test_soft_crc_evidence():
    """Verify soft CRC parity evidence accumulation and semantic gating (Section 4.5)."""
    rng = np.random.RandomState(77)
    data = rng.randint(0, 2, 176, dtype=np.uint8)
    frame = append_crc16_ccitt(data)

    # Strong BPSK soft bits: +6.0 for bit 0, -6.0 for bit 1
    soft_bits = np.where(frame == 0, 6.0, -6.0)

    # 1. BPSK + calibrated LLR -> SUCCESS
    res_calibrated = compute_soft_crc_evidence(
        soft_bits, payload_end_idx=176, metric_type="LLR", modulation="BPSK"
    )
    assert res_calibrated["status"] == "SUCCESS"
    assert res_calibrated["hard_valid"] is True
    assert res_calibrated["soft_parity_score"] > 5.0

    # 2. Unknown metric type -> GATED_UNKNOWN_METRIC
    res_unknown = compute_soft_crc_evidence(soft_bits, payload_end_idx=176)
    assert res_unknown["status"] == "GATED_UNKNOWN_METRIC"
    assert res_unknown["soft_parity_score"] == 0.0

    # 3. SOFT_METRIC -> GATED_UNCALIBRATED_METRIC
    res_soft = compute_soft_crc_evidence(
        soft_bits, payload_end_idx=176, metric_type="SOFT_METRIC"
    )
    assert res_soft["status"] == "GATED_UNCALIBRATED_METRIC"
    assert res_soft["soft_parity_score"] == 0.0

    # 4. CPFSK/GFSK uncalibrated soft metric -> GATED_UNCALIBRATED_METRIC
    res_cpfsk = compute_soft_crc_evidence(
        soft_bits, payload_end_idx=176, metric_type="LLR", modulation="CPFSK"
    )
    assert res_cpfsk["status"] == "GATED_UNCALIBRATED_METRIC"
    assert res_cpfsk["soft_parity_score"] == 0.0



# ==============================================================================
# Category C: Hamming(7,4) Code (10B)
# ==============================================================================

def test_hamming_matrix_orthogonality_and_encoding():
    """Verify H * G^T = 0 mod 2 and encoding/syndrome properties (Section 5.1)."""
    prod = np.dot(HAMMING_7_4_H, HAMMING_7_4_G.T) % 2
    np.testing.assert_array_equal(prod, np.zeros((3, 4), dtype=np.uint8))

    info = [1, 0, 1, 1]
    coded = encode_hamming_7_4(info)
    assert len(coded) == 7

    ev = evaluate_hamming_evidence(coded)
    assert ev["zero_syndrome_fraction"] == 1.0


def test_hamming_bounded_distance_decoding():
    """Verify Hamming(7,4) single-error correction capability (Section 5.1)."""
    msg = [1, 1, 0, 1, 0, 1, 0, 0]  # 2 blocks of 4 bits
    coded = encode_hamming_7_4(msg)

    # Inject 1 bit flip in first block and 1 bit flip in second block
    corrupted = coded.copy()
    corrupted[2] ^= 1
    corrupted[9] ^= 1

    decoded, corrections = decode_hamming_7_4(corrupted)
    assert corrections == 2
    np.testing.assert_array_equal(decoded, msg)


# ==============================================================================
# Category D: BCH(15,7) Code (10B)
# ==============================================================================

def test_bch_codebook_and_decoding():
    """Verify BCH(15,7) codebook size (128) and decoding up to 2 bit errors (Section 6)."""
    assert BCH_CODEWORDS_128.shape == (128, 15)

    msg = [1, 0, 1, 1, 0, 0, 1]
    coded = encode_bch_15_7(msg)
    assert len(coded) == 15

    # 2 bit errors within t=2 capability
    corrupted = coded.copy()
    corrupted[3] ^= 1
    corrupted[8] ^= 1

    decoded, corrections = decode_bch_15_7(corrupted)
    assert corrections == 2
    np.testing.assert_array_equal(decoded, msg)


# ==============================================================================
# Category E: Reed-Solomon GF(16) Code (10B)
# ==============================================================================

def test_reed_solomon_15_11_decoding():
    """Verify RS(15,11) corrects <= 2 symbol errors and fails on >= 3 errors (Section 7.1)."""
    msg_syms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    codeword = encode_rs_15_11(msg_syms)
    assert len(codeword) == 15

    # 1. 2 symbol errors -> Correct recovery (100%)
    c_noisy = list(codeword)
    c_noisy[3] ^= 4
    c_noisy[9] ^= 7
    dec_syms, success, n_err = decode_rs_15_11(c_noisy)
    assert success is True
    assert n_err == 2
    assert dec_syms == msg_syms

    # 2. 3 symbol errors -> Exceeds t=2 capability
    c_noisy3 = list(codeword)
    c_noisy3[1] ^= 2
    c_noisy3[5] ^= 3
    c_noisy3[12] ^= 1
    dec_syms3, success3, _ = decode_rs_15_11(c_noisy3)
    assert success3 is False


def test_rs_parameter_candidate_evaluation():
    """Verify RS parameter identification across candidate pool (Section 7.2)."""
    msg_syms = [2, 4, 6, 8, 1, 3, 5, 7, 9, 11, 13]
    codeword = encode_rs_15_11(msg_syms)
    stream = codeword * 5  # 5 blocks of RS(15,11)

    res = evaluate_rs_parameter_candidates(stream)
    assert res["status"] == "SUCCESS"
    assert res["best_candidate"] == "RS(15,11)"


# ==============================================================================
# Category F: Controlled LDPC (6,12) (10C)
# ==============================================================================

def test_ldpc_syndrome_and_candidate_ranking():
    """Verify LDPC 6x12 matrix properties and candidate ranking (Section 8)."""
    assert LDPC_CONTROLLED_H.shape == (6, 12)
    assert LDPC_VALID_CODEWORDS_64.shape == (64, 12)

    # Clean stream of valid LDPC codewords
    clean_stream = LDPC_VALID_CODEWORDS_64[:10].flatten()
    ev = evaluate_ldpc_syndrome(clean_stream)
    assert ev["zero_syndrome_fraction"] == 1.0
    assert ev["mean_syndrome_weight"] == 0.0

    rank_res = rank_ldpc_candidates(clean_stream)
    assert rank_res["best_candidate"] == "LDPC_CONTROLLED_6x12"


# ==============================================================================
# Category G: Convolutional (R=1/2, K=3) (10C)
# ==============================================================================

def test_convolutional_deterministic_test_vector():
    """Verify deterministic encoder test vector from source Section 9.1:
    Input: [1, 0, 1, 1, 0] -> Encoded: [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0].
    """
    u = [1, 0, 1, 1, 0]
    encoded = encode_convolutional_k3(u, flush_tail=True)
    expected = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0]
    np.testing.assert_array_equal(encoded, expected)


def test_convolutional_viterbi_decoding_and_identification():
    """Verify Viterbi decoding and generator identification via reconstruction mismatch (Section 9.2, 9.3)."""
    rng = np.random.RandomState(42)
    info = rng.randint(0, 2, 50, dtype=np.uint8)
    encoded = encode_convolutional_k3(info, flush_tail=True)

    # Decode clean
    decoded, metric = viterbi_decode_hard_k3(encoded, flush_tail=True)
    np.testing.assert_array_equal(decoded, info)

    # Blind identification over candidate pool
    res = identify_convolutional_generator(encoded)
    assert res["status"] == "SUCCESS"
    assert res["best_candidate"] == "CONV_K3_G7_G5"
    assert res["best_mismatch"] == 0.0


# ==============================================================================
# Category H: Structured Row-Column Interleaver (10D)
# ==============================================================================

def test_row_column_interleaver_roundtrip():
    """Verify exact round-trip inversion of row-column interleaver (Section 10D.1)."""
    rng = np.random.RandomState(88)
    original = rng.randint(0, 2, 100, dtype=np.uint8)

    interleaved = interleave_row_column(original, width=20)
    recovered = deinterleave_row_column(interleaved, width=20)

    assert len(interleaved) == len(original)
    np.testing.assert_array_equal(recovered, original)


def test_interleaver_width_identification():
    """Verify width identification ranks true width 20 as Rank 1 and excludes width 1 (Section 11)."""
    rng = np.random.RandomState(101)
    # Generate Hamming blocks and interleave with width 20
    info = rng.randint(0, 2, 80, dtype=np.uint8)  # 20 blocks of 4
    coded = encode_hamming_7_4(info)  # 140 bits
    interleaved = interleave_row_column(coded, width=20)

    # 1. Default candidates: strictly {5, 10, 20, 25, 50}
    res_def = identify_interleaver_width(interleaved)
    assert res_def["status"] == "SUCCESS"
    assert res_def["best_width"] == 20
    assert res_def["best_score"] == 0.0  # Clean zero syndrome
    candidate_widths_def = [c["width"] for c in res_def["candidates"]]
    assert candidate_widths_def == [20, 5, 10, 25, 50] or set(candidate_widths_def) == {5, 10, 20, 25, 50}
    assert 1 not in candidate_widths_def

    # 2. If caller passes candidate_widths containing 1, width 1 must NOT appear in candidates
    res_with_1 = identify_interleaver_width(interleaved, candidate_widths=[1, 5, 10, 20, 25, 50])
    candidate_widths_with_1 = [c["width"] for c in res_with_1["candidates"]]
    assert 1 not in candidate_widths_with_1
    assert res_with_1["best_width"] == 20
    assert "identity_baseline" in res_with_1
    assert res_with_1["identity_baseline"]["width"] == 1
    assert res_with_1["identity_baseline"]["description"] == "IDENTITY / NO-INTERLEAVING BASELINE"
    assert res_with_1["identity_baseline"]["is_scientific_candidate"] is False


# ==============================================================================
# Category I: Joint FEC + Interleaver Hypothesis Engine (Section 12)
# ==============================================================================

def test_joint_fec_interleaver_ranking():
    """Verify joint Hamming(7,4) + width 20 hypothesis ranking and candidate set enforcement (Section 12)."""
    rng = np.random.RandomState(202)
    info = rng.randint(0, 2, 80, dtype=np.uint8)
    coded = encode_hamming_7_4(info)
    interleaved = interleave_row_column(coded, width=20)

    # 1. Default candidate set strictly {5, 10, 20, 25, 50}
    res = evaluate_joint_hypotheses(interleaved, fec_family="HAMMING")
    assert res["status"] == "SUCCESS"
    assert res["best_hypothesis"]["interleaver_width"] == 20
    assert res["best_hypothesis"]["rank"] == 1
    assert res["margin"] > 0.0
    hyp_widths = [h["interleaver_width"] for h in res["hypotheses"]]
    assert set(hyp_widths) == {5, 10, 20, 25, 50}
    assert 1 not in hyp_widths

    # 2. Exclude width 1 even if explicitly provided
    res_with_1 = evaluate_joint_hypotheses(interleaved, candidate_widths=[1, 5, 10, 20, 25, 50], fec_family="HAMMING")
    hyp_widths_with_1 = [h["interleaver_width"] for h in res_with_1["hypotheses"]]
    assert 1 not in hyp_widths_with_1
    assert res_with_1["best_hypothesis"]["interleaver_width"] == 20
    assert "identity_baseline" in res_with_1
    assert res_with_1["identity_baseline"]["width"] == 1
    assert res_with_1["identity_baseline"]["description"] == "IDENTITY / NO-INTERLEAVING BASELINE"
    assert res_with_1["identity_baseline"]["is_scientific_candidate"] is False



# ==============================================================================
# Category J: Decoder Cross-Validation & Miscorrection Safeguards
# ==============================================================================

def test_decoder_cross_validation_with_crc():
    """Verify decoded output is marked CRC_CONFIRMED when matching CRC, or UNVERIFIED otherwise."""
    rng = np.random.RandomState(303)
    # 176 bits data protected by CRC-16 -> 192 bits
    data = rng.randint(0, 2, 176, dtype=np.uint8)
    frame_with_crc = append_crc16_ccitt(data)

    # Encode with Hamming(7,4): 192 / 4 = 48 blocks -> 336 coded bits
    coded_bits = encode_hamming_7_4(frame_with_crc)

    res = process_fec_crc_analysis(bits=coded_bits)
    assert res["status"] == "SUCCESS"
    assert res["decoded_payload"]["crc_confirmed"] is True
    assert "CRC_CONFIRMED" in res["decoded_payload"]["validation_note"]


# ==============================================================================
# Category K: Excluded/Rejected Methods Safeguards
# ==============================================================================

def test_rejected_methods_safeguards():
    """Verify that rejected approaches are explicitly recorded and not implemented as universal solvers."""
    for rejected_key in [
        "arbitrary_permutation_interleaver_recovery",
        "universal_blind_crc_identification",
        "universal_ldpc_identification",
        "universal_fec_identification",
        "blind_acceptance_of_decoder_output",
        "fabricated_numerical_confidence"
    ]:
        assert rejected_key in MODULE10_REJECTED_METHODS


# ==============================================================================
# Category L: API Integration
# ==============================================================================

def test_module10_api_status():
    """Verify GET /api/module10/status."""
    resp = client.get("/api/module10/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["module"] == "MODULE_10_FEC_CRC_AND_INTERLEAVER"
    assert "10A_crc16_ccitt" in data["components"]
    assert "10B_hamming_7_4" in data["components"]
    assert "10B_bch_15_7" in data["components"]
    assert "10B_reed_solomon" in data["components"]
    assert "10C_ldpc" in data["components"]
    assert "10C_convolutional" in data["components"]
    assert "10D_interleaver" in data["components"]
    assert "rejected_methods" in data


def test_module10_api_analyze():
    """Verify POST /api/module10/analyze with bitstream."""
    rng = np.random.RandomState(404)
    bits = rng.randint(0, 2, 280, dtype=np.uint8).tolist()

    resp = client.post("/api/module10/analyze", json={"bits": bits})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert "fec" in data
    assert "interleaver" in data
    assert "joint_hypothesis" in data
    assert "decoded_payload" in data
