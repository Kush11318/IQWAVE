"""API Endpoints for Module 10 (FEC, CRC & Interleaver Analysis)."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np

from backend.modules.module10.service import (
    default_fec_crc_service,
    MODULE10_REJECTED_METHODS
)

router = APIRouter(prefix="/api/module10", tags=["Module 10"])


class FecCrcAnalyzeRequest(BaseModel):
    # Direct bitstream inputs
    bits: Optional[List[int]] = None
    frames: Optional[List[List[int]]] = None
    frame_period: Optional[int] = None
    soft_bits: Optional[List[float]] = None
    metric_type: Optional[str] = None
    modulation: Optional[str] = None

    # Upstream Module 9 dictionary
    module9_result: Optional[Dict[str, Any]] = None

    # Candidate parameters
    candidate_crc_polynomials: Optional[List[str]] = None
    candidate_interleaver_widths: Optional[List[int]] = None



@router.get("/status")
def get_module10_status() -> Dict[str, Any]:
    """Report Module 10 scientific status, validated components, conditional evidence, and rejected methods."""
    return {
        "module": "MODULE_10_FEC_CRC_AND_INTERLEAVER",
        "scientific_status": "IMPLEMENTED (Review Required before Lock)",
        "components": {
            "10A_crc16_ccitt": {
                "parameters": "Poly 0x1021, Init 0xFFFF",
                "status": "LOCKED (Controlled Validation)",
                "boundary_status": "LOCKED at low BER / CONDITIONAL at higher BER"
            },
            "10A_blind_crc_family": {
                "candidate_pool": ["CRC-8", "CRC-16-CCITT", "CRC-16-IBM", "CRC-32"],
                "status": "CONDITIONAL (Random collision floor under noise)"
            },
            "10A_soft_crc": {
                "method": "BPSK analytical LLR + box-plus parity accumulation",
                "status": "CONDITIONAL (Increases with SNR)"
            },
            "10B_hamming_7_4": {
                "parameters": "n=7, k=4, rate=4/7, t=1",
                "status": "LOCKED (Controlled Validation)",
                "equivalence_note": "Identifies equivalence class, not unique matrix representation"
            },
            "10B_bch_15_7": {
                "parameters": "n=15, k=7, rate=7/15, t=2, g(x)=x^8+x^7+x^6+x^4+1",
                "status": "LOCKED (Controlled Characterization)",
                "miscorrection_note": "Decoded BER exceeds raw BER at SNR <= 0 dB"
            },
            "10B_reed_solomon": {
                "parameters": "RS(15,11), t=2 over GF(16), primitive poly x^4+x+1",
                "status": "LOCKED (Controlled Validation)",
                "decoder_safeguard": "Decoder success != guaranteed correct recovery; requires cross-validation"
            },
            "10C_ldpc": {
                "parameters": "Controlled 6x12 sparse matrix, rate=1/2",
                "status": "LOCKED (Controlled Validation Model)",
                "limitation": "Restricted to candidate pool ranking with observation aggregation"
            },
            "10C_convolutional": {
                "parameters": "Rate 1/2, K=3, generators [111, 101] (octal [7, 5])",
                "status": "LOCKED (Controlled Candidate Pool)",
                "breakdown_note": "Viterbi decoding degrades below raw BER at BER >= 20%"
            },
            "10D_interleaver": {
                "model": "Structured row-column (transpose mapping)",
                "status": "LOCKED (Candidate widths {5, 10, 20, 25, 50})",
                "width_identification": "Downstream Hamming syndrome consistency"
            },
            "10_joint_hypotheses": {
                "method": "Joint (FEC, Interleaver Width) ranking",
                "status": "LOCKED (Controlled Validation)"
            }
        },
        "rejected_methods": MODULE10_REJECTED_METHODS,
        "output_philosophy": {
            "universal_confidence_score": "PROHIBITED (Heterogeneous evidence kept separate)",
            "blind_decoder_trust": "PROHIBITED (Cross-validation against structural checks required)",
            "unconstrained_blind_solvers": "REJECTED (Candidate-based structural testing enforced)"
        }
    }


@router.post("/analyze")
def post_analyze_fec_crc(req: FecCrcAnalyzeRequest) -> Dict[str, Any]:
    """Execute Module 10 candidate-based FEC, CRC, and interleaver analysis."""
    return default_fec_crc_service.analyze_fec_crc(
        module9_result=req.module9_result,
        bits=req.bits,
        frames=req.frames,
        frame_period=req.frame_period,
        soft_bits=req.soft_bits,
        metric_type=req.metric_type,
        modulation=req.modulation,
        candidate_crc_polynomials=req.candidate_crc_polynomials,
        candidate_interleaver_widths=req.candidate_interleaver_widths
    )

