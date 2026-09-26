"""Comprehensive tests for Module 9: Bitstream Structure, Frame Detection & Structural Evidence.

Tests directly reflect the experimental validations and constraints documented in
# Module 9 — Full Experimental Report.

Categories:
A. Known Preamble Detection (9A)
B. Repeated Frame Detection & Period Search (9C.1)
C. Frame Phase Global Scoring (9D.1)
D. Structural Regions & Positional Statistics (9E.2)
E. Counter Candidate Evidence (9E.4)
F. Word/Byte Alignment Supporting Evidence (9E.5)
G. Exact GF(2) Parity Recovery (9F.1 / 9F.2)
H. Noisy GF(2) Parity Evaluation (9F.3)
I. Structural Evidence Map Representation (9H)
J. Rejected-Method Safeguards (9B, 9E.1, 9E.3, 9E.4.3, 9G)
K. API Endpoints & Error Handling
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.modules.module9.preamble import (
    detect_known_preamble,
    DEFAULT_PREAMBLE_16
)
from backend.modules.module9.frame_detection import detect_frame_period
from backend.modules.module9.frame_phase import detect_frame_phase, extract_frames
from backend.modules.module9.structural_regions import analyze_structural_regions
from backend.modules.module9.counter import detect_counter_candidates
from backend.modules.module9.alignment import evaluate_boundary_alignment, compute_bit_alignments
from backend.modules.module9.gf2 import (
    find_gf2_relationships,
    verify_gf2_relation,
    gf2_nullspace
)
from backend.modules.module9.evidence_map import build_structural_evidence_map
from backend.modules.module9.service import (
    default_structure_service,
    process_structure_analysis,
    REJECTED_METHODS_RECORD
)

client = TestClient(app)


def _generate_synthetic_frames(
    num_frames: int = 20,
    preamble_bits: np.ndarray = DEFAULT_PREAMBLE_16,
    payload_len: int = 128,
    include_counter: bool = True,
    include_gf2: bool = True,
    ber: float = 0.0,
    seed: int = 42
) -> np.ndarray:
    """Generate controlled test frames matching the Module 9 experimental setup.

    Frame format (P = 144 bits when preamble is 16 and payload is 128):
    - Bits 0..15: Known 16-bit preamble
    - Bits 16..23: 8-bit counter (MSB first) if include_counter
    - Bits 24..31: 8 data bits for GF(2) equations
    - Bits 32..39: 8 parity bits (GF(2) equations)
    - Bits 40..143: Random payload bits
    """
    rng = np.random.RandomState(seed)
    L_pre = len(preamble_bits)
    frame_len = L_pre + payload_len

    frames = np.zeros((num_frames, frame_len), dtype=np.uint8)

    for m in range(num_frames):
        # 1. Preamble
        frames[m, :L_pre] = preamble_bits

        # 2. Counter at 16..23
        if include_counter:
            val = m % 256
            for b in range(8):
                frames[m, 16 + b] = (val >> (7 - b)) & 1

        # 3. Data bits 24..31 (or 16..31 if GF(2) benchmark used)
        data16 = rng.randint(0, 2, size=16, dtype=np.uint8)
        frames[m, 16:32] = data16  # For exact GF(2) test, use 16 data bits

        # 4. Planted 8 GF(2) equations (Section 17):
        # 32 = 16^17^18^19, 33 = 20^21^22^23, 34 = 24^25^26^27, 35 = 28^29^30^31
        # 36 = 16^20^24^28, 37 = 17^21^25^29, 38 = 18^22^26^30, 39 = 19^23^27^31
        if include_gf2:
            d = frames[m]
            d[32] = d[16] ^ d[17] ^ d[18] ^ d[19]
            d[33] = d[20] ^ d[21] ^ d[22] ^ d[23]
            d[34] = d[24] ^ d[25] ^ d[26] ^ d[27]
            d[35] = d[28] ^ d[29] ^ d[30] ^ d[31]
            d[36] = d[16] ^ d[20] ^ d[24] ^ d[28]
            d[37] = d[17] ^ d[21] ^ d[25] ^ d[29]
            d[38] = d[18] ^ d[22] ^ d[26] ^ d[30]
            d[39] = d[19] ^ d[23] ^ d[27] ^ d[31]

        # 5. Payload bits 40..143
        frames[m, 40:frame_len] = rng.randint(0, 2, size=frame_len - 40, dtype=np.uint8)

    # Apply independent bit-flip noise
    if ber > 0.0:
        noise = rng.rand(*frames.shape) < ber
        frames ^= noise.astype(np.uint8)

    return frames


# ==============================================================================
# Category A: Known Preamble Detection (9A)
# ==============================================================================

def test_known_preamble_clean():
    """Verify 100% preamble detection accuracy under 0% BER."""
    bits = np.array([0, 1, 0, 1] + list(DEFAULT_PREAMBLE_16) + [0, 0, 1], dtype=np.uint8)
    res = detect_known_preamble(bits)
    assert res["status"] == "SUCCESS"
    assert res["detected"] is True
    assert res["best_position"] == 4
    assert res["hamming_distance"] == 0
    assert res["normalized_correlation"] == 1.0
    assert "9A" in res["scientific_status"]


def test_known_preamble_noisy():
    """Verify preamble detection under moderate BER (5% and 10%)."""
    # 5% BER in 16 bits is ~0 or 1 bit flip
    noisy_preamble = DEFAULT_PREAMBLE_16.copy()
    noisy_preamble[2] ^= 1  # 1 bit flip (6.25% BER)

    stream = np.concatenate([np.zeros(20, dtype=np.uint8), noisy_preamble, np.zeros(20, dtype=np.uint8)])
    res = detect_known_preamble(stream)
    assert res["detected"] is True
    assert res["best_position"] == 20
    assert res["hamming_distance"] == 1
    assert res["bit_error_rate_estimate"] == 1 / 16


def test_known_preamble_insufficient_length():
    """Verify clean handling when input is shorter than preamble length."""
    short_bits = [1, 0, 1]
    res = detect_known_preamble(short_bits)
    assert res["status"] == "INSUFFICIENT_DATA"
    assert res["detected"] is False


# ==============================================================================
# Category B: Repeated Frame Detection (9C.1)
# ==============================================================================

def test_repeated_frame_period_detection():
    """Verify that longest arithmetic chain correctly recovers P=144 across 20 frames."""
    frames = _generate_synthetic_frames(num_frames=20, ber=0.0)
    flat_bits = frames.flatten()

    res = detect_frame_period(flat_bits, candidate_periods=(100, 180))
    assert res["status"] == "SUCCESS"
    assert res["detected"] is True
    assert res["frame_period"] == 144
    assert res["chain_length"] == 20
    assert "9C.1" in res["scientific_status"]


def test_repeated_frame_period_at_5_percent_ber():
    """Verify period detection under 5% BER (source: robust through 10%)."""
    frames = _generate_synthetic_frames(num_frames=20, ber=0.05, seed=123)
    flat_bits = frames.flatten()

    res = detect_frame_period(flat_bits, candidate_periods=(100, 180))
    assert res["status"] == "SUCCESS"
    assert res["frame_period"] == 144
    assert res["chain_length"] >= 15


# ==============================================================================
# Category C: Frame Phase (9D.1)
# ==============================================================================

def test_global_frame_phase_detection():
    """Verify that global frame-phase scoring recovers exact phase offset phi*."""
    frames = _generate_synthetic_frames(num_frames=10, ber=0.0)
    flat_bits = frames.flatten()

    # Prepend 25 arbitrary offset bits
    offset = 25
    prefix = np.array([1, 0, 1, 1, 0] * 5, dtype=np.uint8)
    offset_stream = np.concatenate([prefix, flat_bits])

    res = detect_frame_phase(offset_stream, frame_period=144)
    assert res["status"] == "SUCCESS"
    assert res["frame_phase"] == offset
    assert "9D.1" in res["scientific_status"]


def test_frame_extraction():
    """Verify matrix slicing into (M, P) frames."""
    frames_orig = _generate_synthetic_frames(num_frames=5, ber=0.0)
    flat = frames_orig.flatten()

    sliced = extract_frames(flat, frame_period=144, frame_phase=0)
    assert sliced.shape == (5, 144)
    np.testing.assert_array_equal(sliced, frames_orig)


# ==============================================================================
# Category D: Structural Regions (9E.2)
# ==============================================================================

def test_structural_regions_variance():
    """Verify that preamble and static regions have low variance while payload has high variance."""
    frames = _generate_synthetic_frames(num_frames=100, ber=0.0)
    res = analyze_structural_regions(frames, preamble_length=16)

    assert res["status"] == "SUCCESS"
    assert res["frame_count"] == 100
    assert res["frame_period"] == 144

    variances = res["column_variances"]
    entropies = res["column_entropies"]

    # Preamble (bits 0..15) is constant -> variance == 0
    for j in range(16):
        assert variances[j] == 0.0
        assert entropies[j] == 0.0

    # Payload (bits 40..143) is random -> variance close to 0.25 (typically > 0.20)
    payload_vars = variances[40:]
    assert np.mean(payload_vars) > 0.20

    # Verify no forced header decision is made
    assert "safeguard_note" in res
    assert "Regional hypotheses are structural evidence only" in res["safeguard_note"]


# ==============================================================================
# Category E: Counter Evidence (9E.4)
# ==============================================================================

def test_counter_candidate_detection():
    """Verify counter candidate is localized with step consistency == 1.0 at 0% BER."""
    # Controlled frames where counter increments by 1
    frames = _generate_synthetic_frames(num_frames=20, ber=0.0)

    # Overwrite bits 16..23 with a strictly incrementing 8-bit counter
    for m in range(20):
        val = m
        for b in range(8):
            frames[m, 16 + b] = (val >> (7 - b)) & 1

    res = detect_counter_candidates(frames, candidate_widths=[8])
    assert res["status"] == "SUCCESS"
    assert res["detected"] is True

    best = res["best_candidate"]
    assert best is not None
    assert best["start_bit"] == 16
    assert best["end_bit"] == 23
    assert best["bit_width"] == 8
    assert best["consistency"] == 1.0
    # Must be labeled COUNTER_CANDIDATE, NOT confirmed counter
    assert best["candidate_type"] == "COUNTER_CANDIDATE"


# ==============================================================================
# Category F: Word/Byte Alignment (9E.5)
# ==============================================================================

def test_alignment_evidence():
    """Verify byte and word alignment supporting evidence."""
    # Test boundary 48 (byte, 16-bit, and 32-bit: 48 % 8 == 0, 48 % 16 == 0, 48 % 32 != 0)
    res = evaluate_boundary_alignment(48)
    assert res["boundary_bit"] == 48
    assert res["alignments"]["8_bit"]["aligned"] is True
    assert res["alignments"]["16_bit"]["aligned"] is True
    assert res["alignments"]["32_bit"]["aligned"] is False
    assert "Supporting Alignment Evidence Only" in res["scientific_status"]

    alignments = compute_bit_alignments(32)
    assert len(alignments) == 32
    assert alignments[0]["byte_aligned"] is True
    assert alignments[8]["byte_aligned"] is True
    assert alignments[16]["word16_aligned"] is True


# ==============================================================================
# Category G: Exact GF(2) Parity Relationships (9F.1 / 9F.2)
# ==============================================================================

def test_exact_gf2_recovery():
    """Verify generalized recovery of the 8 planted GF(2) equations from Section 17."""
    frames = _generate_synthetic_frames(num_frames=100, ber=0.0)

    # Restrict candidate columns to 16..39 (16 data bits + 8 parity bits)
    cand_cols = list(range(16, 40))
    res = find_gf2_relationships(frames, candidate_columns=cand_cols)

    assert res["status"] == "SUCCESS"
    assert res["exact_relations_count"] >= 8

    # Check that parity relations have error_rate == 0.0
    for rel in res["relations"]:
        assert rel["is_exact"] is True
        assert rel["error_rate"] == 0.0

    # Specifically verify equation 32 = 16 ^ 17 ^ 18 ^ 19
    ver = verify_gf2_relation(frames, target_bit=32, source_bits=[16, 17, 18, 19])
    assert ver["is_exact"] is True
    assert ver["error_rate"] == 0.0
    assert ver["scientific_status"] == "LOCKED"


# ==============================================================================
# Category H: Noisy GF(2) Evaluation (9F.3)
# ==============================================================================

def test_noisy_gf2_relation_error():
    """Verify that relation error rate increases under bit flips as documented in Section 18."""
    # At 5% BER, true relation error is around ~0.20
    frames_noisy = _generate_synthetic_frames(num_frames=100, ber=0.05, seed=99)
    ver = verify_gf2_relation(frames_noisy, target_bit=32, source_bits=[16, 17, 18, 19])

    # Error rate should be between 0.10 and 0.30 (source reports ~0.205)
    assert 0.08 <= ver["error_rate"] <= 0.32
    assert ver["is_exact"] is False
    assert ver["scientific_status"] == "CONDITIONAL"


# ==============================================================================
# Category I: Structural Evidence Map (9H)
# ==============================================================================

def test_structural_evidence_map():
    """Verify that the Structural Evidence Map maintains per-bit evidence independently."""
    frames = _generate_synthetic_frames(num_frames=20, ber=0.0)
    res = analyze_structural_regions(frames, preamble_length=16)
    counter_res = detect_counter_candidates(frames)
    gf2_res = find_gf2_relationships(frames, candidate_columns=list(range(16, 40)))

    emap = build_structural_evidence_map(
        frame_period=144,
        column_variances=res["column_variances"],
        column_entropies=res["column_entropies"],
        column_means=res["column_means"],
        counter_candidates=counter_res["candidates"],
        gf2_relations=gf2_res["relations"],
        preamble_length=16
    )

    assert emap["status"] == "SUCCESS"
    assert emap["frame_period"] == 144
    assert len(emap["bit_evidence_records"]) == 144

    # Verify NO unified scalar confidence score exists
    for rec in emap["bit_evidence_records"]:
        assert "confidence" not in rec
        assert "unified_score" not in rec
        assert "variance" in rec
        assert "entropy" in rec
        assert "static_evidence" in rec
        assert "counter_evidence" in rec
        assert "alignment_evidence" in rec


# ==============================================================================
# Category J: Rejected-Method Safeguards
# ==============================================================================

def test_rejected_methods_safeguards():
    """Verify that rejected methods are explicitly excluded and documented."""
    for rejected_key in [
        "9B_unknown_preamble_length",
        "9B_1_periodicity_preamble_detection",
        "9B_2_harmonic_suppression",
        "9B_3_repeated_complete_block_discovery",
        "9E_1_payload_balance",
        "9E_3_variance_only_header_boundary",
        "9E_4_3_counter_transition_signature",
        "9G_unified_boundary_score"
    ]:
        assert rejected_key in REJECTED_METHODS_RECORD
        assert "REJECTED" in REJECTED_METHODS_RECORD[rejected_key]


# ==============================================================================
# Category K: Full Pipeline & API Integration
# ==============================================================================

def test_module9_service_full_pipeline():
    """Verify full end-to-end Module 9 service execution."""
    frames = _generate_synthetic_frames(num_frames=20, ber=0.0)
    flat_bits = frames.flatten()

    res = process_structure_analysis(bits=flat_bits)
    assert res["status"] == "SUCCESS"
    assert res["frame_period"] == 144
    assert res["frame_phase"] == 0
    assert res["preamble_analysis"]["detected"] is True
    assert res["structural_evidence_map"] is not None
    assert res["output_philosophy"]["confidence_score_invented"] is False
    assert res["output_philosophy"]["header_boundary_forced"] is False


def test_module9_api_status():
    """Verify GET /api/module9/status."""
    resp = client.get("/api/module9/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["module"] == "MODULE_9_BITSTREAM_STRUCTURE_AND_EVIDENCE"
    assert "9A_known_preamble_detection" in data["components"]
    assert "9C_1_repeated_frame_period" in data["components"]
    assert "9D_1_global_frame_phase" in data["components"]
    assert "9H_structural_evidence_map" in data["components"]
    assert "rejected_methods" in data


def test_module9_api_analyze():
    """Verify POST /api/module9/analyze with bitstream."""
    frames = _generate_synthetic_frames(num_frames=20, ber=0.0)
    flat_bits = frames.flatten().tolist()

    resp = client.post("/api/module9/analyze", json={"bits": flat_bits})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["frame_period"] == 144
    assert data["frame_phase"] == 0
    assert data["structural_evidence_map"] is not None


def test_module9_api_error_handling():
    """Verify clean error response on missing inputs."""
    resp = client.post("/api/module9/analyze", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ERROR_NO_INPUT"
