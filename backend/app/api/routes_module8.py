"""API Endpoints for Module 8 (Blind FEC & Interleaver Identification and Decoding)."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np

from backend.modules.module1.canonical_iq import to_canonical_iq
from backend.modules.module2.service import observe_from_module1
from backend.modules.module3.service import classify_amc
from backend.modules.module4.service import default_parameter_service
from backend.modules.module5.service import default_recovery_service
from backend.modules.module6.service import default_snr_service
from backend.modules.module7.service import default_soft_bits_service
from backend.modules.module8.service import default_fec_service

router = APIRouter(prefix="/api/module8", tags=["Module 8"])


class FecDecodeRequest(BaseModel):
    # Direct bitstream inputs
    soft_bits: Optional[List[float]] = None
    hard_bits: Optional[List[int]] = None
    metric_type: Optional[str] = None
    mathematical_status: Optional[str] = None
    ground_truth_info_bits: Optional[List[int]] = None

    # Optional Raw IQ pipeline inputs
    i_channel: Optional[List[float]] = None
    q_channel: Optional[List[float]] = None
    modulation: Optional[str] = None
    sample_rate: Optional[float] = None
    samples_per_symbol: Optional[float] = None
    snr_db: Optional[float] = None


@router.get("/status")
def get_module8_status() -> Dict[str, Any]:
    """Report Module 8 blind FEC identification, interleaver detection, and decoder status."""
    return {
        "module": "MODULE_8_BLIND_FEC_AND_INTERLEAVER",
        "status": "LOCKED",
        "validated_families": {
            "HAMMING": {
                "parameters": "n=7, k=4, rate=4/7, t=1",
                "identification": "Parity-check syndrome consistency and normalized lift",
                "decoders": ["HAMMING_HARD (syndrome lookup)", "HAMMING_SOFT_ML (16 codewords)"],
                "status": "LOCKED"
            },
            "BCH": {
                "parameters": "n=15, k=7, rate=7/15, t=2, g(x)=x^8+x^7+x^6+x^4+1",
                "identification": "Polynomial remainder syndrome consistency and normalized lift",
                "decoders": ["BCH_HARD (min Hamming distance)", "BCH_SOFT_ML (128 codewords)"],
                "status": "LOCKED"
            },
            "CONVOLUTIONAL": {
                "parameters": "K=3, rate=1/2, generators=[111, 101] (octal [7, 5])",
                "identification": "Viterbi re-encoding reconstruction mismatch",
                "decoders": ["VITERBI_HARD", "VITERBI_SOFT"],
                "status": "LOCKED"
            },
            "INTERLEAVER": {
                "parameters": "Structured row-column interleaver",
                "identification": "Candidate width search W in {2..16} maximizing syndrome consistency",
                "status": "LOCKED"
            }
        },
        "rejected_approaches": [
            "Universal generic scalar FEC score: REJECTED (evidence metrics must remain mathematically separate).",
            "Universal decision thresholds: REJECTED (not established by experimental source; labeled as unvalidated engineering heuristics).",
            "Arbitrary permutation interleaver recognition: REJECTED (underdetermined due to code symmetries).",
            "General unknown interleaver phase recovery: REJECTED (parameterization ambiguity).",
            "Exact noisy null-space recovery for general linear codes: REJECTED as general noisy solver."
        ],
        "selection_policy": "UNVALIDATED_ENGINEERING_HEURISTIC",
        "scientific_constraints": [
            "Parity lifts (Hamming, BCH) and Convolutional reconstruction mismatch remain mathematically separate.",
            "The source validates relative candidate ranking within tested models, NOT universal decision thresholds.",
            "Family selection is an unvalidated engineering policy for decoder routing, not a scientifically proven decision boundary."
        ],
        "out_of_scope": [
            "LDPC and Reed-Solomon codes: Deferred to Module 10.",
            "Turbo codes: Not validated in Module 8.",
            "Frame period, preambles, and packet fields: Belong to Module 9.",
            "CRC verification: Belongs to Module 10."
        ],
        "boundary": "MODULE_8_CONCLUDED_AT_FEC_AND_RECOVERED_BITS"
    }


@router.post("/decode")
def post_decode_fec(req: FecDecodeRequest) -> Dict[str, Any]:
    """Execute Module 8 FEC identification and decoding pipeline."""
    # Direct bitstream path
    if req.soft_bits is not None or req.hard_bits is not None:
        return default_fec_service.process_fec(
            soft_bits=np.array(req.soft_bits, dtype=np.float64) if req.soft_bits is not None else None,
            hard_bits=np.array(req.hard_bits, dtype=np.uint8) if req.hard_bits is not None else None,
            metric_type=req.metric_type,
            mathematical_status=req.mathematical_status,
            ground_truth_info_bits=np.array(req.ground_truth_info_bits, dtype=np.uint8) if req.ground_truth_info_bits is not None else None
        )

    # Raw IQ pipeline path
    if req.i_channel is not None and req.q_channel is not None:
        canonical_iq, val_res = to_canonical_iq(req.i_channel, req.q_channel)
        if val_res["status"] == "INVALID":
            return {"status": "INVALID_INPUT", "validation": val_res}

        # M2
        mod2_res = observe_from_module1(canonical_iq, val_res)

        # M3 AMC
        active_mod = req.modulation
        if not active_mod:
            amc_res = classify_amc(canonical_iq, module2_observation=mod2_res.observation)
            rf_pred = amc_res.get("engines", {}).get("engine_a_rf", {}).get("predicted_class")
            cnn_pred = amc_res.get("engines", {}).get("engine_b_cnn", {}).get("predicted_class")
            active_mod = rf_pred or cnn_pred

        # M4 Parameters
        m4_res = default_parameter_service.estimate_parameters(
            canonical_iq=canonical_iq,
            modulation=active_mod,
            sample_rate=req.sample_rate
        )
        active_sps = req.samples_per_symbol or m4_res.get("samples_per_symbol")
        active_cfo = m4_res.get("cfo")

        # M5 Recovery
        m5_res = default_recovery_service.recover_signal(
            canonical_iq=canonical_iq,
            modulation=active_mod,
            module4_parameters=m4_res,
            sample_rate=req.sample_rate,
            samples_per_symbol=active_sps,
            cfo=active_cfo
        )

        # M6 SNR
        m6_res = default_snr_service.estimate_snr(
            canonical_iq=canonical_iq,
            modulation=active_mod,
            module4_parameters=m4_res,
            module5_recovery=m5_res
        )

        # M7 Soft Bits
        syms = np.array(m5_res.get("symbols", []), dtype=np.complex128) if m5_res.get("symbols") else None
        hard_b = np.array(m5_res.get("bits", []), dtype=np.uint8) if m5_res.get("bits") else None
        m7_res = default_soft_bits_service.generate_soft_bits(
            module5_recovery=m5_res,
            module6_snr=m6_res,
            symbol_samples=syms,
            reference_hard_bits=hard_b,
            modulation=active_mod
        )

        # M8 FEC & Decoding
        return default_fec_service.process_fec(
            module7_result=m7_res,
            hard_bits=hard_b,
            ground_truth_info_bits=np.array(req.ground_truth_info_bits, dtype=np.uint8) if req.ground_truth_info_bits is not None else None
        )


    return {
        "status": "ERROR",
        "error": "INSUFFICIENT_INPUT: Either (soft_bits/hard_bits) or (i_channel, q_channel) must be provided.",
        "warnings": ["NO_INPUT_PROVIDED"]
    }
