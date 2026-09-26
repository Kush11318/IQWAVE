"""API Endpoints for Module 9 (Bitstream Structure, Frame Detection & Structural Evidence)."""

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
from backend.modules.module9.service import (
    default_structure_service,
    REJECTED_METHODS_RECORD
)

router = APIRouter(prefix="/api/module9", tags=["Module 9"])


class StructureAnalyzeRequest(BaseModel):
    # Direct bitstream inputs
    bits: Optional[List[int]] = None
    preamble: Optional[List[int]] = None
    candidate_period_min: Optional[int] = 100
    candidate_period_max: Optional[int] = 180
    max_gf2_error: Optional[float] = 0.0

    # Optional upstream Module 8 dictionary
    module8_result: Optional[Dict[str, Any]] = None

    # Optional Raw IQ end-to-end pipeline inputs
    i_channel: Optional[List[float]] = None
    q_channel: Optional[List[float]] = None
    modulation: Optional[str] = None
    sample_rate: Optional[float] = None
    samples_per_symbol: Optional[float] = None
    snr_db: Optional[float] = None


@router.get("/status")
def get_module9_status() -> Dict[str, Any]:
    """Report Module 9 scientific status, validated components, conditional evidence, and rejected methods."""
    return {
        "module": "MODULE_9_BITSTREAM_STRUCTURE_AND_EVIDENCE",
        "scientific_status": "IMPLEMENTED (Review Required before Lock)",
        "components": {
            "9A_known_preamble_detection": {
                "method": "Sliding Hamming distance comparison",
                "status": "LOCKED",
                "validated_conditions": "BER 0%, 1%, 5%, 10% (robust); degrades at 20% BER"
            },
            "9C_1_repeated_frame_period": {
                "method": "Longest consistent arithmetic chain of candidate positions",
                "status": "LOCKED through 10% BER",
                "search_bound_note": "100-180 bit range is an implementation search bound derived from experiment",
                "limitation": "Degrades at 20% BER (accuracy ~43.67%)"
            },
            "9D_1_global_frame_phase": {
                "method": "Global frame-phase scoring over candidate offsets",
                "status": "LOCKED under tested conditions",
                "validated_conditions": "100% boundary accuracy across 0-20% BER in controlled setup"
            },
            "9E_2_structural_regions": {
                "method": "Frame-to-frame positional variance and entropy",
                "status": "CONDITIONAL",
                "constraint": "Output as regional hypotheses; header boundaries NOT forced"
            },
            "9E_4_counter_localization": {
                "method": "Frame-to-frame integer transition step consistency",
                "status": "CONDITIONAL",
                "constraint": "Reported strictly as COUNTER_CANDIDATE, not confirmed counter"
            },
            "9E_5_word_byte_alignment": {
                "method": "Alignment evaluation against 8, 16, 32-bit boundaries",
                "status": "CONDITIONAL",
                "constraint": "Supporting evidence only; does not independently decide boundaries"
            },
            "9F_1_exact_gf2_relationships": {
                "method": "Generalized GF(2) nullspace and linear relationship discovery",
                "status": "LOCKED (exact parity error = 0.0)"
            },
            "9F_3_noisy_gf2_relationships": {
                "method": "Parity check evaluation under bit errors",
                "status": "CONDITIONAL",
                "limitation": "Strong at 0-1%, conditional at 5%, insufficient at 10-20% BER"
            },
            "9H_structural_evidence_map": {
                "method": "Independent per-bit structural record without unified score",
                "status": "LOCKED as evidence representation"
            }
        },
        "rejected_methods": REJECTED_METHODS_RECORD,
        "output_philosophy": {
            "unified_confidence_score": "PROHIBITED (heterogeneous evidence kept independent)",
            "forced_header_boundaries": "PROHIBITED",
            "protocol_identity": "NOT_CLAIMED",
            "contract": "Produces structural hypotheses and evidence for Module 10"
        }
    }


@router.post("/analyze")
def post_analyze_structure(req: StructureAnalyzeRequest) -> Dict[str, Any]:
    """Execute Module 9 structural analysis and evidence map generation."""
    p_bounds = (req.candidate_period_min or 100, req.candidate_period_max or 180)

    # 1. Direct bits or Module 8 result provided
    if req.bits is not None or req.module8_result is not None:
        return default_structure_service.analyze_structure(
            module8_result=req.module8_result,
            bits=req.bits,
            preamble=req.preamble,
            candidate_periods=p_bounds,
            max_gf2_error=req.max_gf2_error or 0.0
        )

    # 2. Raw IQ pipeline execution through Modules 1 -> 8 -> 9
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
        m8_res = default_fec_service.process_fec(
            module7_result=m7_res,
            hard_bits=hard_b
        )

        # Fallback bits if M8 recovered 0 bits
        rec_bits = m8_res.get("recovered_information_bits")
        if not rec_bits and hard_b is not None and len(hard_b) > 0:
            rec_bits = hard_b.tolist()

        return default_structure_service.analyze_structure(
            module8_result=m8_res,
            bits=rec_bits,
            preamble=req.preamble,
            candidate_periods=p_bounds,
            max_gf2_error=req.max_gf2_error or 0.0
        )


    return {
        "status": "ERROR_NO_INPUT",
        "message": "Either 'bits', 'module8_result', or ('i_channel', 'q_channel') must be provided."
    }
