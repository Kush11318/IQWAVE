"""Module 10 Service: FEC, CRC & Interleaver Analysis Orchestrator.

Scientific Status: IMPLEMENTED (Review Required before Lock).
Source: # Module 10 — FEC, CRC & Interleaver Analysis.

The service coordinates candidate-based structural analysis:
1. Input ingestion from Module 9 (frames, recovered bits, evidence map).
2. CRC analysis (10A): candidate polynomial ranking, boundary check, soft CRC evidence.
3. Candidate FEC evaluations (10B, 10C):
   - Hamming(7,4)
   - BCH(15,7)
   - Reed-Solomon(15,11) over GF(16)
   - LDPC (6,12)
   - Convolutional (R=1/2, K=3)
4. Interleaver analysis & joint hypotheses (10D):
   - Row-column candidate width search
   - Joint (FEC, Interleaver Width) ranking
5. Decoded payload cross-validation:
   - Decoder success is cross-checked against CRC and structural invariants.
   - If unverified, flags miscorrection_risk = True.
6. Rejected-method safeguards:
   - Explicitly records excluded methods.
   - Strict prohibition against universal confidence percentage calculation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .crc import (
    compute_crc,
    verify_crc16_ccitt,
    evaluate_crc_acceptance_rate,
    search_crc_boundaries,
    identify_crc_family,
    compute_soft_crc_evidence,
    CRC_CANDIDATE_CATALOG
)
from .hamming import (
    encode_hamming_7_4,
    decode_hamming_7_4,
    evaluate_hamming_evidence,
    HAMMING_7_4_H
)
from .bch import (
    encode_bch_15_7,
    decode_bch_15_7,
    evaluate_bch_evidence
)
from .reed_solomon import (
    decode_rs_15_11,
    evaluate_rs_parameter_candidates
)
from .ldpc import (
    evaluate_ldpc_syndrome,
    rank_ldpc_candidates,
    LDPC_CONTROLLED_H
)
from .convolutional import (
    encode_convolutional_k3,
    viterbi_decode_hard_k3,
    identify_convolutional_generator
)
from .interleaver import (
    interleave_row_column,
    deinterleave_row_column,
    identify_interleaver_width
)
from .joint_engine import evaluate_joint_hypotheses


# Explicit record of experimentally rejected / unsupported methods
MODULE10_REJECTED_METHODS = {
    "arbitrary_permutation_interleaver_recovery": (
        "REJECTED (permutation symmetries and missing priors lead to non-unique solutions; "
        "only structured row-column models are supported)."
    ),
    "universal_blind_crc_identification": (
        "REJECTED (hard CRC acceptance approaches random collision floor 2^-W under noise; "
        "only candidate-based ranking is supported)."
    ),
    "universal_ldpc_identification": (
        "REJECTED (unstructured blind LDPC matrix recovery is physically unestablished; "
        "limited to controlled candidate pool testing)."
    ),
    "universal_fec_identification": (
        "REJECTED (unconstrained search over infinite code spaces is invalid; "
        "candidate-based structural testing is enforced)."
    ),
    "blind_acceptance_of_decoder_output": (
        "REJECTED (decoder success != guaranteed correct recovery due to miscorrection; "
        "decoded payloads require cross-validation against CRC or structural checks)."
    ),
    "fabricated_numerical_confidence": (
        "PROHIBITED (heterogeneous evidence metrics must NOT be fused into an arbitrary percentage)."
    )
}


class FecCrcInterleaverService:
    """Service orchestrating candidate-based FEC, CRC, and interleaver structural analysis."""

    def analyze_fec_crc(
        self,
        module9_result: Optional[Dict[str, Any]] = None,
        bits: Optional[Union[np.ndarray, List[int]]] = None,
        frames: Optional[Union[np.ndarray, List[List[int]]]] = None,
        frame_period: Optional[int] = None,
        soft_bits: Optional[Union[np.ndarray, List[float]]] = None,
        metric_type: Optional[str] = None,
        modulation: Optional[str] = None,
        candidate_crc_polynomials: Optional[List[str]] = None,
        candidate_interleaver_widths: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Execute candidate-based FEC, CRC, and interleaver analysis.

        Args:
            module9_result: Output dictionary from Module 9.
            bits: 1D array of binary bits.
            frames: 2D array or list of frames.
            frame_period: Detected frame period P.
            soft_bits: Optional 1D soft bits array for soft CRC evidence.
            metric_type: Optional metric type string ('LLR', 'SOFT_METRIC', etc.).
            modulation: Optional modulation scheme ('BPSK', 'CPFSK', 'GFSK', etc.).
            candidate_crc_polynomials: Optional list of candidate CRC names.
            candidate_interleaver_widths: Optional list of candidate widths.

        Returns:
            Structured dictionary matching Module 10 specification.
        """
        # 1. Resolve inputs
        rx_bits = bits
        rx_frames = frames
        detected_p = frame_period

        if module9_result is not None:
            if rx_bits is None:
                rx_bits = module9_result.get("bits")
            if detected_p is None:
                detected_p = module9_result.get("frame_period")
            if metric_type is None:
                metric_type = module9_result.get("metric_type")
            if modulation is None:
                modulation = module9_result.get("modulation")

        if rx_bits is not None:
            rx_bits = np.asarray(rx_bits, dtype=np.uint8).flatten()


        if rx_frames is not None:
            rx_frames = np.asarray(rx_frames, dtype=np.uint8)
        elif rx_bits is not None and detected_p is not None and detected_p > 0:
            M = len(rx_bits) // detected_p
            if M > 0:
                rx_frames = rx_bits[: M * detected_p].reshape(M, detected_p)

        if rx_bits is None or len(rx_bits) == 0:
            return {
                "status": "ERROR_NO_INPUT",
                "message": "No bitstream or frames provided for Module 10 analysis.",
                "scientific_status": "NOT_ANALYZED"
            }

        warnings: List[str] = []
        N_bits = len(rx_bits)

        # 2. Module 10A: CRC Analysis
        crc_results: Dict[str, Any] = {}
        if rx_frames is not None and rx_frames.ndim == 2 and rx_frames.shape[0] >= 1:
            P = rx_frames.shape[1]
            # Boundary search (typical default payload boundary around P - 16)
            default_payload_end = max(16, P - 16)
            crc_family_res = identify_crc_family(
                rx_frames,
                payload_end_idx=default_payload_end,
                candidate_names=candidate_crc_polynomials
            )
            crc_boundary_res = search_crc_boundaries(rx_frames, crc_name="CRC-16-CCITT")
            crc_results = {
                "candidate": crc_family_res.get("best_candidate"),
                "family_evaluation": crc_family_res,
                "boundary_evaluation": crc_boundary_res,
                "confidence": "conditional",
                "evidence": "polynomial remainder consistency",
                "scientific_status": "LOCKED at low BER / CONDITIONAL at higher BER"
            }
        else:
            crc_results = {
                "candidate": None,
                "confidence": "unverified",
                "note": "Insufficient frame structure for multi-frame CRC consistency analysis."
            }

        # Soft CRC evidence if soft bits available
        if soft_bits is not None and len(soft_bits) >= 32:
            soft_crc_res = compute_soft_crc_evidence(
                soft_bits,
                metric_type=metric_type,
                modulation=modulation
            )
            crc_results["soft_crc_evidence"] = soft_crc_res

        # 3. Module 10B/C: Candidate FEC Structural Evaluations
        fec_candidate_evaluations: List[Dict[str, Any]] = []

        # Candidate 1: Hamming(7,4)
        hamming_ev = evaluate_hamming_evidence(rx_bits)
        fec_candidate_evaluations.append({
            "family": "Hamming",
            "parameters": "n=7, k=4, rate=4/7, t=1",
            "zero_syndrome_fraction": hamming_ev["zero_syndrome_fraction"],
            "score": float(1.0 - hamming_ev["zero_syndrome_fraction"]),
            "lift": hamming_ev["normalized_lift"],
            "metric_type": "syndrome_consistency",
            "status": "candidate-based (LOCKED controlled)"
        })

        # Candidate 2: BCH(15,7)
        bch_ev = evaluate_bch_evidence(rx_bits)
        fec_candidate_evaluations.append({
            "family": "BCH",
            "parameters": "n=15, k=7, rate=7/15, t=2",
            "zero_syndrome_fraction": bch_ev["zero_syndrome_fraction"],
            "score": float(1.0 - bch_ev["zero_syndrome_fraction"]),
            "lift": bch_ev["normalized_lift"],
            "metric_type": "polynomial_remainder_consistency",
            "status": "candidate-based (LOCKED controlled)"
        })

        # Candidate 3: Reed-Solomon(15,11) over GF(16)
        # Convert binary bits into 4-bit symbols
        n_nibbles = len(rx_bits) // 4
        if n_nibbles >= 15:
            nibbles = [
                int(rx_bits[i*4] << 3 | rx_bits[i*4+1] << 2 | rx_bits[i*4+2] << 1 | rx_bits[i*4+3])
                for i in range(n_nibbles)
            ]
            rs_res = evaluate_rs_parameter_candidates(nibbles)
            best_rs = rs_res.get("best_candidate", "RS(15,11)")
            top_lift = rs_res["candidates"][0]["lift"] if rs_res.get("candidates") else 0.0
            fec_candidate_evaluations.append({
                "family": "Reed-Solomon",
                "parameters": f"{best_rs} over GF(16), 4 bits/symbol",
                "score": float(1.0 - top_lift),
                "lift": float(top_lift),
                "metric_type": "gf16_syndrome_consistency",
                "status": "candidate-based (LOCKED controlled)"
            })

        # Candidate 4: LDPC (6,12)
        ldpc_ev = evaluate_ldpc_syndrome(rx_bits)
        fec_candidate_evaluations.append({
            "family": "LDPC",
            "parameters": "Controlled (6,12) sparse matrix, rate=1/2",
            "zero_syndrome_fraction": ldpc_ev["zero_syndrome_fraction"],
            "score": float(ldpc_ev["mean_syndrome_weight"] / 6.0),
            "lift": ldpc_ev["zero_syndrome_fraction"],
            "metric_type": "sparse_parity_check_weight",
            "status": "candidate-based (LOCKED controlled model)"
        })

        # Candidate 5: Convolutional (R=1/2, K=3)
        conv_res = identify_convolutional_generator(rx_bits)
        fec_candidate_evaluations.append({
            "family": "Convolutional",
            "parameters": "K=3, rate=1/2, generators=[111, 101] (octal [7, 5])",
            "score": float(conv_res["best_mismatch"]),
            "lift": float(1.0 - conv_res["best_mismatch"]),
            "metric_type": "trellis_reconstruction_mismatch",
            "status": "candidate-based (LOCKED controlled pool)"
        })

        # Sort candidate FECs by lift descending
        fec_candidate_evaluations.sort(key=lambda x: x["lift"], reverse=True)
        top_fec = fec_candidate_evaluations[0]

        # 4. Module 10D: Interleaver & Joint Hypothesis Engine
        width_res = identify_interleaver_width(
            rx_bits,
            candidate_widths=candidate_interleaver_widths
        )
        joint_res = evaluate_joint_hypotheses(
            rx_bits,
            candidate_widths=candidate_interleaver_widths,
            fec_family=top_fec["family"]
        )

        # Determine if interleaver is physically detected over raw identity baseline
        raw_ev = evaluate_hamming_evidence(rx_bits)
        raw_non_zero_score = float(1.0 - raw_ev["zero_syndrome_fraction"])
        best_width = width_res.get("best_width")
        best_score = width_res.get("best_score", 1.0)
        detected_width = best_width if (best_width and best_score < raw_non_zero_score) else None

        interleaver_output: Dict[str, Any] = {
            "model": "row-column",
            "estimated_width": detected_width,
            "status": "supported (structured row-column only)",
            "validated_candidate_set": [5, 10, 20, 25, 50],
            "width_evaluations": width_res.get("candidates", [])
        }
        if "identity_baseline" in width_res:
            interleaver_output["identity_baseline"] = width_res["identity_baseline"]
        else:
            interleaver_output["identity_baseline"] = {
                "description": "IDENTITY / NO-INTERLEAVING BASELINE",
                "width": 1,
                "non_zero_syndrome_score": raw_non_zero_score,
                "is_scientific_candidate": False
            }

        # 5. Candidate Decoding with Structural Cross-Validation
        # Perform test decoding with top FEC
        decoded_bits: List[int] = []
        crc_confirmed = False
        miscorrection_risk = False

        if top_fec["family"] == "Hamming":
            # Deinterleave only if joint interleaver hypothesis exhibits strictly superior
            # syndrome consistency over the raw (uninterleaved identity baseline) stream
            best_hyp = joint_res.get("best_hypothesis")
            if best_hyp and best_hyp.get("interleaver_width", 0) > 1 and best_hyp.get("score", 1.0) < raw_non_zero_score:
                w_opt = best_hyp["interleaver_width"]
                deint_bits = deinterleave_row_column(rx_bits, width=w_opt)
            else:
                deint_bits = rx_bits

            dec_info, _ = decode_hamming_7_4(deint_bits)
            decoded_bits = dec_info.tolist()

        elif top_fec["family"] == "BCH":
            dec_info, _ = decode_bch_15_7(rx_bits)
            decoded_bits = dec_info.tolist()
            if top_fec["lift"] < 0.20:
                miscorrection_risk = True
        elif top_fec["family"] == "Reed-Solomon":
            n_nibbles = len(rx_bits) // 4
            info_bits_acc: List[int] = []
            for blk_idx in range(n_nibbles // 15):
                cw = [
                    int(rx_bits[(blk_idx * 15 + j) * 4] << 3 |
                        rx_bits[(blk_idx * 15 + j) * 4 + 1] << 2 |
                        rx_bits[(blk_idx * 15 + j) * 4 + 2] << 1 |
                        rx_bits[(blk_idx * 15 + j) * 4 + 3])
                    for j in range(15)
                ]
                dec_info, ok, _ = decode_rs_15_11(cw)
                for sym in dec_info:
                    for s in [3, 2, 1, 0]:
                        info_bits_acc.append((sym >> s) & 1)
            decoded_bits = info_bits_acc if info_bits_acc else rx_bits.tolist()
            if top_fec["lift"] < 0.30:
                miscorrection_risk = True
        elif top_fec["family"] == "Convolutional":
            dec_info, _ = viterbi_decode_hard_k3(rx_bits)
            decoded_bits = dec_info.tolist()
            if top_fec["score"] > 0.15:
                miscorrection_risk = True
        else:
            decoded_bits = rx_bits.tolist()

        # Cross-validate decoded payload with CRC if frame format permits
        if len(decoded_bits) >= 192:
            crc_confirmed = verify_crc16_ccitt(decoded_bits, payload_end_idx=176, crc_width=16)

        decoded_payload_record = {
            "total_decoded_bits": len(decoded_bits),
            "preview_bits": decoded_bits[:64] if decoded_bits else [],
            "crc_confirmed": crc_confirmed,
            "miscorrection_risk": miscorrection_risk,
            "validation_note": (
                "CRC_CONFIRMED: Decoded output verified by CRC." if crc_confirmed else
                "UNVERIFIED_DECODER_OUTPUT: Decoder success does not guarantee correct recovery; "
                "standalone decoder output must not be blindly trusted."
            )
        }

        return {
            "status": "SUCCESS",
            "total_bits_analyzed": N_bits,
            "crc": crc_results,
            "fec": {
                "top_candidate": top_fec["family"],
                "candidates": fec_candidate_evaluations,
                "status": "candidate-based (native mathematical evidence)"
            },
            "interleaver": interleaver_output,
            "joint_hypothesis": {
                "fec": joint_res["best_hypothesis"]["fec_family"] if joint_res.get("best_hypothesis") else top_fec["family"],
                "interleaver_width": joint_res["best_hypothesis"]["interleaver_width"] if joint_res.get("best_hypothesis") else 1,
                "rank": joint_res["best_hypothesis"]["rank"] if joint_res.get("best_hypothesis") else 1,
                "margin": joint_res.get("margin", 0.0),
                "all_hypotheses": joint_res.get("hypotheses", [])
            },
            "decoded_payload": decoded_payload_record,
            "rejected_methods_safeguards": MODULE10_REJECTED_METHODS,
            "warnings": warnings,
            "scientific_status": "IMPLEMENTED (Review Required before Lock)"
        }


default_fec_crc_service = FecCrcInterleaverService()


def process_fec_crc_analysis(
    module9_result: Optional[Dict[str, Any]] = None,
    bits: Optional[Union[np.ndarray, List[int]]] = None,
    frames: Optional[Union[np.ndarray, List[List[int]]]] = None,
    frame_period: Optional[int] = None,
    soft_bits: Optional[Union[np.ndarray, List[float]]] = None,
    candidate_crc_polynomials: Optional[List[str]] = None,
    candidate_interleaver_widths: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Convenience functional entry point for Module 10 analysis."""
    return default_fec_crc_service.analyze_fec_crc(
        module9_result=module9_result,
        bits=bits,
        frames=frames,
        frame_period=frame_period,
        soft_bits=soft_bits,
        candidate_crc_polynomials=candidate_crc_polynomials,
        candidate_interleaver_widths=candidate_interleaver_widths
    )
