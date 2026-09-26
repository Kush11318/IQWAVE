"""Module 8 Service: Blind FEC & Interleaver Identification and Decoding Orchestrator.

Orchestrates the experimentally validated Module 8 pipeline:
1. FEC Candidate Evidence Generation:
   - Hamming (7,4): Parity-check syndrome consistency and normalized lift.
   - BCH (15,7): Polynomial remainder syndrome consistency and normalized lift.
   - Convolutional (K=3, r=1/2): Viterbi re-encoding reconstruction mismatch.
   CRITICAL CONSTRAINT: Evidence metrics remain mathematically separate.
   NO universal/generic scalar FEC score is created or invented.

2. Candidate Family Ranking & Selection:
   - Evaluates block code lifts (Hamming vs. BCH).
   - Evaluates convolutional re-encoding mismatch independently.
   - Selects validated family or flags NO_FEC_DETECTED / AMBIGUOUS_FEC.

3. Codeword Alignment Search:
   - Cyclic shift search over candidate offsets (0..6 for Hamming, 0..14 for BCH).

4. Interleaver Check:
   - Structured row-column interleaver width estimation.
   - Deinterleaves soft and hard bitstreams if row-column interleaving is identified.

5. Soft / Hard Decoding:
   - Soft ML decoding over codebooks when soft values / LLRs are available.
   - Hard decoding fallback when only hard bits are available.
   - Preserves documented Hamming low-SNR hard-decoder miscorrection limitation.

6. Output Contract & Boundary:
   - Concludes at recovered information bits and code parameters.
   - Module 9 bitstream structure / framing is strictly out of scope.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .hamming import (
    compute_hamming_evidence,
    decode_hamming_hard,
    decode_hamming_soft_ml,
    find_hamming_alignment
)
from .bch import (
    compute_bch_evidence,
    decode_bch_hard,
    decode_bch_soft_ml,
    find_bch_alignment
)
from .convolutional import (
    compute_convolutional_evidence,
    viterbi_decode_hard,
    viterbi_decode_soft
)
from .interleaver import (
    search_interleaver_width,
    deinterleave_row_column
)


class FecRecoveryService:
    """Service orchestrating blind FEC identification, interleaver detection, and decoding."""

    SUPPORTED_FAMILIES = ["HAMMING", "BCH", "CONVOLUTIONAL", "UNCODED"]

    def process_fec(
        self,
        module7_result: Optional[Dict[str, Any]] = None,
        soft_bits: Optional[Union[np.ndarray, List[float]]] = None,
        hard_bits: Optional[Union[np.ndarray, List[int]]] = None,
        metric_type: Optional[str] = None,
        mathematical_status: Optional[str] = None,
        ground_truth_info_bits: Optional[Union[np.ndarray, List[int]]] = None,
        candidate_interleaver_widths: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Execute Module 8 blind FEC identification, alignment, deinterleaving, and decoding.

        Args:
            module7_result: Output dictionary from Module 7.
            soft_bits: 1D array of signed soft values from Module 7.
            hard_bits: 1D array of hard decision bits from Module 7.
            metric_type: 'LLR' or 'SOFT_METRIC'.
            mathematical_status: Status string from Module 7.
            ground_truth_info_bits: Optional transmitted information bits for diagnostic BER.
            candidate_interleaver_widths: Optional widths to search.

        Returns:
            Structured dictionary matching Module 8 output contract.
        """
        warnings: List[str] = []
        m7 = module7_result or {}

        # 1. Resolve inputs
        rx_hard = hard_bits
        if rx_hard is None:
            rx_hard = m7.get("hard_bits")
        if rx_hard is not None:
            rx_hard = np.asarray(rx_hard, dtype=np.uint8).flatten()

        rx_soft = soft_bits
        if rx_soft is None:
            rx_soft = m7.get("soft_bits")
        if rx_soft is not None:
            rx_soft = np.asarray(rx_soft, dtype=np.float64).flatten()

        # If hard bits absent but soft bits present, derive hard decisions: soft > 0 -> 1
        if rx_hard is None and rx_soft is not None and len(rx_soft) > 0:
            rx_hard = (rx_soft > 0.0).astype(np.uint8)

        if rx_hard is None or len(rx_hard) == 0:
            return {
                "status": "ERROR",
                "error": "INSUFFICIENT_INPUT: Hard bits or soft bits must be provided.",
                "detected_fec_family": "UNKNOWN",
                "code_parameters": None,
                "recovered_information_bits": None,
                "warnings": ["NO_BITS_PROVIDED"]
            }

        resolved_metric_type = metric_type or m7.get("metric_type", "UNKNOWN")
        resolved_math_status = mathematical_status or m7.get("mathematical_status", "UNKNOWN")

        if resolved_metric_type == "SOFT_METRIC":
            warnings.append(
                "UNSCALED_SOFT_METRIC_INPUT: Received FSK soft metrics from Module 7. "
                "Soft ML decoding uses raw correlation differences without probabilistic calibration."
            )

        # 2. Evaluate Candidate Hypotheses Independently
        # A. Hamming (7,4)
        hamming_ev = compute_hamming_evidence(rx_hard)

        # B. BCH (15,7)
        bch_ev = compute_bch_evidence(rx_hard)

        # C. Convolutional (K=3, r=1/2)
        conv_ev = compute_convolutional_evidence(rx_hard)

        # Separate Evidence Scores (Strictly NOT combined into a scalar score)
        evidence_scores = {
            "hamming": {
                "zero_syndrome_fraction": hamming_ev["zero_syndrome_fraction"],
                "normalized_lift": hamming_ev["normalized_lift"],
                "random_baseline": hamming_ev["random_baseline"],
                "best_alignment_offset": hamming_ev["alignment_offset"]
            },
            "bch": {
                "zero_syndrome_fraction": bch_ev["zero_syndrome_fraction"],
                "normalized_lift": bch_ev["normalized_lift"],
                "random_baseline": bch_ev["random_baseline"],
                "best_alignment_offset": bch_ev["alignment_offset"]
            },
            "convolutional": {
                "reconstruction_mismatch": conv_ev["reconstruction_mismatch"],
                "best_candidate": conv_ev["best_candidate"],
                "candidate_scores": conv_ev["candidate_scores"]
            }
        }

        # Candidate Evaluations: Each candidate family is evaluated independently according
        # to its source-defined evidence metric (Sections 2, 9, 11).
        candidate_evaluations = {
            "HAMMING": {
                "family": "HAMMING",
                "code": "HAMMING_7_4",
                "parameters": {"family": "HAMMING", "code": "HAMMING_7_4", "n": 7, "k": 4, "t": 1, "rate": float(4 / 7), "parity_matrix": "H_3x7_standard"},
                "alignment_offset": int(hamming_ev["alignment_offset"]),
                "evidence_type": "NORMALIZED_SYNDROME_LIFT",
                "evidence_value": float(hamming_ev["normalized_lift"]),
                "zero_syndrome_fraction": float(hamming_ev["zero_syndrome_fraction"]),
                "raw_lift": float(hamming_ev["raw_lift"]),
                "random_baseline": float(hamming_ev["random_baseline"]),
                "candidate_decoders": ["HAMMING_HARD", "HAMMING_SOFT_ML"]
            },
            "BCH": {
                "family": "BCH",
                "code": "BCH_15_7",
                "parameters": {"family": "BCH", "code": "BCH_15_7", "n": 15, "k": 7, "t": 2, "rate": float(7 / 15), "generator_polynomial": "x^8+x^7+x^6+x^4+1"},
                "alignment_offset": int(bch_ev["alignment_offset"]),
                "evidence_type": "NORMALIZED_SYNDROME_LIFT",
                "evidence_value": float(bch_ev["normalized_lift"]),
                "zero_syndrome_fraction": float(bch_ev["zero_syndrome_fraction"]),
                "raw_lift": float(bch_ev["raw_lift"]),
                "random_baseline": float(bch_ev["random_baseline"]),
                "candidate_decoders": ["BCH_HARD", "BCH_SOFT_ML"]
            },
            "CONVOLUTIONAL": {
                "family": "CONVOLUTIONAL",
                "code": "CONV_K3_R12",
                "parameters": {"family": "CONVOLUTIONAL", "code": "CONV_K3_R12", "k_constraint": 3, "rate": 0.5, "generators_octal": [7, 5]},
                "alignment_offset": 0,
                "evidence_type": "VITERBI_RECONSTRUCTION_MISMATCH",
                "evidence_value": float(conv_ev["reconstruction_mismatch"]),
                "best_candidate_generator": conv_ev["best_candidate"],
                "candidate_scores": conv_ev["candidate_scores"],
                "candidate_decoders": ["VITERBI_HARD", "VITERBI_SOFT"]
            }
        }

        # 3. Family Selection & Decoder Routing Policy
        # SCIENTIFIC STATUS: UNVALIDATED_ENGINEERING_HEURISTIC
        # The source establishes relative candidate ranking within tested models:
        #   - Hamming vs BCH: correct family produces substantially larger normalized lift than incorrect family (Section 10).
        #   - Convolutional: true [7,5] candidate produces substantially lower Viterbi reconstruction mismatch
        #     than tested false candidates through 20% BER (Section 11).
        #   - The source explicitly REJECTS a universal scalar FEC score and does NOT establish universal
        #     scalar decision thresholds. Any deterministic selection is an engineering routing policy.
        warnings.append(
            "UNVALIDATED_ENGINEERING_HEURISTIC: Family selection was made via an implementation routing heuristic. "
            "Evidence scores (Hamming lift, BCH lift, Convolutional mismatch) are mathematically separate and "
            "must not be interpreted as a unified or thresholded scientific classification score."
        )

        detected_family = "UNCODED"
        status_val = "SUCCESS"
        code_params: Optional[Dict[str, Any]] = None
        alignment_offset = 0

        h_lift = hamming_ev["normalized_lift"]
        b_lift = bch_ev["normalized_lift"]
        c_mismatch = conv_ev["reconstruction_mismatch"]

        # Check candidate convolutional margin over false generators
        conv_margin = 0.0
        if conv_ev.get("candidate_scores"):
            false_mismatches = [
                v for k, v in conv_ev["candidate_scores"].items() if k != "CONV_K3_R12_7_5_TRUE"
            ]
            if false_mismatches:
                conv_margin = min(false_mismatches) - c_mismatch

        # Implementation routing heuristic based on relative candidate ranking
        has_hamming_evidence = (h_lift > 0.15 and h_lift > b_lift)
        has_bch_evidence = (b_lift > 0.15 and b_lift > h_lift)
        has_conv_evidence = (
            conv_ev["best_candidate"] == "CONV_K3_R12_7_5_TRUE"
            and (c_mismatch < 0.10 or (conv_margin >= 0.015 and c_mismatch <= 0.135))
            and h_lift <= 0.15
            and b_lift <= 0.15
        )

        if has_bch_evidence:
            detected_family = "BCH"
            alignment_offset = bch_ev["alignment_offset"]
            code_params = candidate_evaluations["BCH"]["parameters"]
        elif has_hamming_evidence:
            detected_family = "HAMMING"
            alignment_offset = hamming_ev["alignment_offset"]
            code_params = candidate_evaluations["HAMMING"]["parameters"]
        elif has_conv_evidence:
            detected_family = "CONVOLUTIONAL"
            alignment_offset = 0
            code_params = candidate_evaluations["CONVOLUTIONAL"]["parameters"]
        else:
            # Check for ambiguity vs uncoded
            if h_lift > 0.15 and b_lift > 0.15:
                status_val = "AMBIGUOUS_FEC"
                detected_family = "AMBIGUOUS"
                warnings.append("AMBIGUOUS_FEC: Hamming and BCH candidate lifts are comparable without a clear dominator.")
            else:
                status_val = "NO_FEC_DETECTED"
                detected_family = "UNCODED"
                code_params = {
                    "family": "UNCODED",
                    "n": 1,
                    "k": 1,
                    "rate": 1.0
                }

        # 4. Interleaver Check (Evaluated only if block FEC is detected)
        interleaver_info: Dict[str, Any] = {
            "is_interleaved": False,
            "interleaver_type": "NONE",
            "estimated_width": None
        }

        aligned_hard = rx_hard
        aligned_soft = rx_soft

        if detected_family in ["HAMMING", "BCH"]:
            int_res = search_interleaver_width(
                bits=rx_hard[alignment_offset:],
                fec_family=detected_family,
                candidate_widths=candidate_interleaver_widths
            )
            if int_res["is_interleaved"]:
                interleaver_info = {
                    "is_interleaved": True,
                    "interleaver_type": "ROW_COLUMN",
                    "estimated_width": int_res["estimated_width"],
                    "syndrome_improvement": float(int_res["best_syndrome_score"] - int_res["baseline_uninterleaved_score"])
                }
                # Deinterleave hard bits and soft bits
                w_est = int(int_res["estimated_width"])
                aligned_hard = deinterleave_row_column(rx_hard[alignment_offset:], width=w_est)
                if rx_soft is not None:
                    aligned_soft = deinterleave_row_column(rx_soft[alignment_offset:], width=w_est)
            else:
                aligned_hard = rx_hard[alignment_offset:]
                if rx_soft is not None:
                    aligned_soft = rx_soft[alignment_offset:]

        # 5. Decoder Selection & Execution
        recovered_info_bits: np.ndarray = np.array([], dtype=np.uint8)
        decoder_applied = "NONE"

        has_soft = aligned_soft is not None and len(aligned_soft) > 0

        if detected_family == "HAMMING":
            if has_soft:
                recovered_info_bits, dec_meta = decode_hamming_soft_ml(aligned_soft)
                decoder_applied = "HAMMING_SOFT_ML"
            else:
                recovered_info_bits, dec_meta = decode_hamming_hard(aligned_hard)
                decoder_applied = "HAMMING_HARD"
                warnings.append(
                    "HAMMING_HARD_DECODING: Single-bit correction applied. "
                    "Note that multi-bit errors at low SNR can cause miscorrection."
                )

        elif detected_family == "BCH":
            if has_soft:
                recovered_info_bits, dec_meta = decode_bch_soft_ml(aligned_soft)
                decoder_applied = "BCH_SOFT_ML"
            else:
                recovered_info_bits, dec_meta = decode_bch_hard(aligned_hard)
                decoder_applied = "BCH_HARD"

        elif detected_family == "CONVOLUTIONAL":
            if has_soft:
                recovered_info_bits = viterbi_decode_soft(aligned_soft)
                decoder_applied = "VITERBI_SOFT"
            else:
                recovered_info_bits = viterbi_decode_hard(aligned_hard)
                decoder_applied = "VITERBI_HARD"

        else:
            # UNCODED / Passthrough
            recovered_info_bits = aligned_hard
            decoder_applied = "NONE_PASSTHROUGH"

        # 6. Optional BER Diagnostics (Measured against reference information bits)
        ber_diag: Optional[Dict[str, Any]] = None
        if ground_truth_info_bits is not None and len(ground_truth_info_bits) > 0:
            gt = np.asarray(ground_truth_info_bits, dtype=np.uint8).flatten()
            n_comp = min(len(recovered_info_bits), len(gt))
            if n_comp > 0:
                dec_errors = int(np.sum(recovered_info_bits[:n_comp] != gt[:n_comp]))
                dec_ber = float(dec_errors / n_comp)

                # Raw bit comparison if applicable
                n_raw_comp = min(len(rx_hard), len(gt))
                raw_errors = int(np.sum(rx_hard[:n_raw_comp] != gt[:n_raw_comp]))
                raw_ber = float(raw_errors / n_raw_comp) if n_raw_comp > 0 else 0.0

                improvement = float(raw_ber / dec_ber) if dec_ber > 0 else (100.0 if raw_ber > 0 else 1.0)
                ber_diag = {
                    "compared_bits": n_comp,
                    "decoded_ber": dec_ber,
                    "raw_ber": raw_ber,
                    "improvement_ratio": improvement
                }

        return {
            "status": status_val,
            "detected_fec_family": detected_family,
            "selection_policy": "UNVALIDATED_ENGINEERING_HEURISTIC",
            "selection_policy_note": (
                "Family selection is an unvalidated engineering heuristic for downstream decoder routing. "
                "The source validates candidate evidence ranking within tested models, not universal decision thresholds."
            ),
            "code_parameters": code_params,
            "code_alignment": int(alignment_offset),
            "interleaver": interleaver_info,
            "decoder_applied": decoder_applied,
            "candidate_evaluations": candidate_evaluations,
            "evidence_scores": evidence_scores,
            "total_recovered_bits": int(len(recovered_info_bits)),
            "recovered_information_bits": recovered_info_bits.tolist(),
            "ber_diagnostics": ber_diag,
            "warnings": warnings,
            "boundary": "MODULE_8_CONCLUDED_AT_FEC_AND_RECOVERED_BITS"
        }


default_fec_service = FecRecoveryService()


def process_fec_recovery(
    module7_result: Optional[Dict[str, Any]] = None,
    soft_bits: Optional[Union[np.ndarray, List[float]]] = None,
    hard_bits: Optional[Union[np.ndarray, List[int]]] = None,
    metric_type: Optional[str] = None,
    mathematical_status: Optional[str] = None,
    ground_truth_info_bits: Optional[Union[np.ndarray, List[int]]] = None,
    candidate_interleaver_widths: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Functional convenience entry point for Module 8."""
    return default_fec_service.process_fec(
        module7_result=module7_result,
        soft_bits=soft_bits,
        hard_bits=hard_bits,
        metric_type=metric_type,
        mathematical_status=mathematical_status,
        ground_truth_info_bits=ground_truth_info_bits,
        candidate_interleaver_widths=candidate_interleaver_widths
    )
