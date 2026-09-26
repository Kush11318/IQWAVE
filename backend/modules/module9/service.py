"""Module 9 Service: Bitstream Structure, Frame Detection & Structural Evidence Orchestrator.

Scientific Status: IMPLEMENTED (Review Required before Lock).
Source: # Module 9 — Full Experimental Report.

The service executes the approved Module 9 pipeline:
1. Module 8 recovered information bits input
2. Known preamble detection (9A - LOCKED)
3. Repeated frame detection & arithmetic chain period estimation (9C.1 - LOCKED through 10% BER)
4. Frame phase global scoring & boundary extraction (9D.1 - LOCKED under tested conditions)
5. Positional structural statistics & regional hypotheses (9E.2 - CONDITIONAL)
6. Counter candidate localization (9E.4 - CONDITIONAL, reported as candidate)
7. Word / byte alignment evidence (9E.5 - CONDITIONAL supporting evidence)
8. Exact and noisy GF(2) parity relationship discovery (9F - LOCKED exact / CONDITIONAL noisy)
9. Structural Evidence Map assembly (9H - LOCKED evidence representation)

CRITICAL SCIENTIFIC RULES:
- NO unified confidence score is computed.
- Heterogeneous evidence is maintained independently per bit position.
- Header boundaries are NOT forced solely from variance thresholds.
- Protocol identity is NOT claimed.
- Rejected methods (9B, 9E.1, 9E.3, 9E.4.3, 9G) are strictly excluded from primary logic.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .preamble import detect_known_preamble, DEFAULT_PREAMBLE_16
from .frame_detection import detect_frame_period
from .frame_phase import detect_frame_phase, extract_frames
from .structural_regions import analyze_structural_regions
from .counter import detect_counter_candidates
from .alignment import evaluate_boundary_alignment, compute_bit_alignments
from .gf2 import find_gf2_relationships, verify_gf2_relation
from .evidence_map import build_structural_evidence_map


# Explicit record of experimentally rejected methods
REJECTED_METHODS_RECORD = {
    "9B_unknown_preamble_length": "REJECTED (harmonic ambiguity created severe length estimation errors)",
    "9B_1_periodicity_preamble_detection": "REJECTED (low accuracy 33-63%, harmonic candidates)",
    "9B_2_harmonic_suppression": "REJECTED (consistently favored false short periods e.g. 4 bits instead of 16)",
    "9B_3_repeated_complete_block_discovery": "REJECTED (unstable under noise, false blocks selected)",
    "9E_1_payload_balance": "REJECTED (statistically indistinguishable from random bits)",
    "9E_3_variance_only_header_boundary": "REJECTED (only 42.86% exact accuracy, heterogeneous fields resemble payload)",
    "9E_4_3_counter_transition_signature": "REJECTED (accuracy 42.67% at 0% BER, 5.33% at 20% BER)",
    "9G_unified_boundary_score": "REJECTED (systematic positional bias, top-1 accuracy only 14-28%)"
}


class BitstreamStructureService:
    """Service orchestrating blind bitstream structural analysis and evidence map generation."""

    def analyze_structure(
        self,
        module8_result: Optional[Dict[str, Any]] = None,
        bits: Optional[Union[np.ndarray, List[int]]] = None,
        preamble: Optional[Union[np.ndarray, List[int]]] = None,
        candidate_periods: Optional[Tuple[int, int]] = (100, 180),
        max_gf2_error: float = 0.0
    ) -> Dict[str, Any]:
        """Execute full Module 9 structural evidence pipeline.

        Args:
            module8_result: Output dictionary from Module 8 FEC recovery.
            bits: 1D array or list of binary bits (if direct bitstream input).
            preamble: Known preamble sequence. Defaults to standard 16-bit sequence.
            candidate_periods: Period search bounds (min, max). Defaults to (100, 180).
            max_gf2_error: Parity error threshold for GF(2) analysis (0.0 for exact only).

        Returns:
            Structured dictionary adhering strictly to the Module 9 contract.
        """
        # 1. Resolve input bits from Module 8 or direct argument
        rx_bits = bits
        if rx_bits is None and module8_result is not None:
            rx_bits = module8_result.get("recovered_information_bits")
            if rx_bits is None:
                rx_bits = module8_result.get("hard_bits")

        if rx_bits is None:
            return {
                "status": "ERROR_NO_INPUT_BITS",
                "message": "No input bitstream provided directly or via Module 8 output.",
                "scientific_status": "NOT_ANALYZED"
            }

        rx = np.asarray(rx_bits, dtype=np.uint8).flatten()
        p = np.asarray(preamble if preamble is not None else DEFAULT_PREAMBLE_16, dtype=np.uint8).flatten()

        N = len(rx)
        if N < 16:
            return {
                "status": "INSUFFICIENT_DATA",
                "stream_length": N,
                "message": "Bitstream too short for structural analysis (requires at least 16 bits).",
                "scientific_status": "INSUFFICIENT_DATA"
            }

        warnings: List[str] = []

        # 2. Known Preamble Detection (9A - LOCKED)
        preamble_res = detect_known_preamble(rx, preamble=p)

        # 3. Repeated Frame Detection & Period Search (9C.1 - LOCKED through 10% BER)
        frame_det_res = detect_frame_period(
            rx,
            preamble=p,
            candidate_periods=candidate_periods
        )
        detected_period = frame_det_res.get("frame_period")

        # Fallback period if no periodic chain found: cannot proceed with multi-frame statistics
        if detected_period is None or not frame_det_res.get("detected", False):
            warnings.append("No repeated frame period identified within search bounds.")
            return {
                "status": "PARTIAL_ANALYSIS_NO_PERIODICITY",
                "stream_length": N,
                "preamble_analysis": preamble_res,
                "frame_period_analysis": frame_det_res,
                "frame_phase_analysis": None,
                "structural_regions": None,
                "counter_analysis": None,
                "alignment_evidence": None,
                "gf2_analysis": None,
                "structural_evidence_map": None,
                "rejected_methods_safeguard": REJECTED_METHODS_RECORD,
                "warnings": warnings,
                "scientific_status": "CONDITIONAL (Periodicity Not Confirmed)"
            }

        # 4. Frame Phase Global Scoring (9D.1 - LOCKED under tested conditions)
        frame_phase_res = detect_frame_phase(
            rx,
            frame_period=detected_period,
            preamble=p
        )
        detected_phase = frame_phase_res.get("frame_phase", 0)

        # 5. Extract aligned frame matrix
        frames = extract_frames(rx, frame_period=detected_period, frame_phase=detected_phase)
        M, P = frames.shape

        if M < 2:
            warnings.append("Fewer than 2 complete frames extracted after phase alignment.")
            return {
                "status": "PARTIAL_ANALYSIS_INSUFFICIENT_FRAMES",
                "stream_length": N,
                "frame_period": detected_period,
                "frame_phase": detected_phase,
                "frames_count": int(M),
                "preamble_analysis": preamble_res,
                "frame_period_analysis": frame_det_res,
                "frame_phase_analysis": frame_phase_res,
                "structural_regions": None,
                "counter_analysis": None,
                "alignment_evidence": None,
                "gf2_analysis": None,
                "structural_evidence_map": None,
                "rejected_methods_safeguard": REJECTED_METHODS_RECORD,
                "warnings": warnings,
                "scientific_status": "CONDITIONAL (Insufficient Frames for Variance)"
            }

        # 6. Structural Region Evidence (9E.2 - CONDITIONAL)
        regions_res = analyze_structural_regions(frames, preamble_length=len(p))

        # 7. Counter Candidate Localization (9E.4 - CONDITIONAL)
        counter_res = detect_counter_candidates(frames)

        # 8. Word/Byte Alignment Supporting Evidence (9E.5 - CONDITIONAL)
        bit_alignments = compute_bit_alignments(detected_period)

        # 9. GF(2) Structural Relationship Discovery (9F.1 - LOCKED exact / 9F.3 - CONDITIONAL noisy)
        gf2_res = find_gf2_relationships(
            frames,
            max_error_threshold=max_gf2_error
        )

        # 10. Structural Evidence Map (9H - LOCKED as evidence representation)
        evidence_map_res = build_structural_evidence_map(
            frame_period=detected_period,
            column_variances=regions_res["column_variances"],
            column_entropies=regions_res["column_entropies"],
            column_means=regions_res["column_means"],
            counter_candidates=counter_res.get("candidates", []),
            gf2_relations=gf2_res.get("relations", []),
            preamble_length=len(p)
        )

        return {
            "status": "SUCCESS",
            "stream_length": N,
            "frame_period": int(detected_period),
            "frame_phase": int(detected_phase),
            "frames_count": int(M),
            "preamble_analysis": preamble_res,
            "frame_period_analysis": frame_det_res,
            "frame_phase_analysis": frame_phase_res,
            "structural_regions": regions_res,
            "counter_analysis": counter_res,
            "word_alignments": bit_alignments,
            "gf2_analysis": gf2_res,
            "structural_evidence_map": evidence_map_res,
            "rejected_methods_safeguard": REJECTED_METHODS_RECORD,
            "warnings": warnings,
            "output_philosophy": {
                "header_boundary_forced": False,
                "counter_confirmed": False,
                "protocol_identity_claimed": False,
                "confidence_score_invented": False,
                "note": "Module 9 outputs independent structural hypotheses and evidence for Module 10."
            },
            "scientific_status": "IMPLEMENTED (Review Required before Lock)"
        }


default_structure_service = BitstreamStructureService()


def process_structure_analysis(
    module8_result: Optional[Dict[str, Any]] = None,
    bits: Optional[Union[np.ndarray, List[int]]] = None,
    preamble: Optional[Union[np.ndarray, List[int]]] = None,
    candidate_periods: Optional[Tuple[int, int]] = (100, 180),
    max_gf2_error: float = 0.0
) -> Dict[str, Any]:
    """Convenience entry point for Module 9 bitstream structural analysis."""
    return default_structure_service.analyze_structure(
        module8_result=module8_result,
        bits=bits,
        preamble=preamble,
        candidate_periods=candidate_periods,
        max_gf2_error=max_gf2_error
    )
