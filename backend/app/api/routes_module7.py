"""API Endpoints for Module 7 (Soft Bits / LLR Generation)."""

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

router = APIRouter(prefix="/api/module7", tags=["Module 7"])


class SoftBitsRequest(BaseModel):
    i_channel: Optional[List[float]] = None
    q_channel: Optional[List[float]] = None
    symbol_i: Optional[List[float]] = None
    symbol_q: Optional[List[float]] = None
    modulation: Optional[str] = None
    n0: Optional[float] = None
    snr_db: Optional[float] = None
    sample_rate: Optional[float] = None
    samples_per_symbol: Optional[float] = None
    ground_truth_bits: Optional[List[int]] = None


@router.get("/status")
def get_module7_status() -> Dict[str, Any]:
    """Report Module 7 soft bit generation status and algorithms."""
    return {
        "module": "MODULE_7_SOFT_BITS_AND_LLR_GENERATION",
        "status": "LOCKED",
        "supported_classes": ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "CPFSK", "GFSK"],
        "algorithms": {
            "BPSK": {
                "method": "Analytical AWGN LLR (numerically equivalent to exact likelihood)",
                "metric_type": "LLR",
                "mathematical_status": "EXACT_UNDER_AWGN_MODEL",
                "calibrated_llr": False,
                "status": "LOCKED"
            },
            "QPSK": {
                "method": "Exact constellation likelihood LLR (analytical quadrant equivalent)",
                "metric_type": "LLR",
                "mathematical_status": "EXACT_UNDER_AWGN_MODEL",
                "calibrated_llr": False,
                "status": "LOCKED"
            },
            "8PSK": {
                "method": "Exact constellation likelihood LLR (natural binary mapping)",
                "metric_type": "LLR",
                "mathematical_status": "EXACT_UNDER_AWGN_MODEL",
                "calibrated_llr": False,
                "status": "LOCKED"
            },
            "QAM16": {
                "method": "Exact constellation likelihood LLR (Gray mapping)",
                "metric_type": "LLR",
                "mathematical_status": "EXACT_UNDER_AWGN_MODEL",
                "calibrated_llr": False,
                "status": "LOCKED"
            },
            "QAM64": {
                "method": "Exact constellation likelihood LLR (Gray mapping)",
                "metric_type": "LLR",
                "mathematical_status": "EXACT_UNDER_AWGN_MODEL",
                "calibrated_llr": False,
                "status": "LOCKED"
            },
            "CPFSK": {
                "method": "Real waveform correlation difference Re{<r,s1>} - Re{<r,s0>} (no 2/N0 scaling)",
                "metric_type": "SOFT_METRIC",
                "mathematical_status": "NOT_CALIBRATED_LLR",
                "calibrated_llr": False,
                "status": "LOCKED"
            },
            "GFSK": {
                "method": "Real waveform correlation difference Re{<r,s1>} - Re{<r,s0>} against BT=0.5 Gaussian templates (no 2/N0 scaling)",
                "metric_type": "SOFT_METRIC",
                "mathematical_status": "NOT_CALIBRATED_LLR",
                "calibrated_llr": False,
                "status": "LOCKED"
            }
        },
        "rejected_baselines": {
            "cpfsk_instantaneous_freq": "REJECTED_AS_PRIMARY (degrades at low SNR)",
            "cpfsk_symbol_center_freq": "REJECTED_AS_PRIMARY (degrades at low SNR)",
            "gfsk_frequency_state": "REJECTED_AS_PRIMARY (waveform correlation performed better)",
            "maxlog_qam": "CONDITIONAL_OPTIMIZATION_ONLY (hard decisions diverge below 20 dB)"
        },
        "scientific_rules": [
            "Strictly NO N0=1 fallback when noise parameter is unavailable for PSK/QAM; returns NOISE_PARAMETER_UNAVAILABLE.",
            "PSK/QAM outputs are exact likelihood under the AWGN model, but NOT claimed as empirical calibrated probabilities.",
            "CPFSK/GFSK outputs are raw waveform-correlation soft metrics with NO 2/N0 scaling.",
            "Sign convention: positive (>0) indicates bit 1, negative (<0) indicates bit 0.",
            "Reliability bins are strictly diagnostic verification tools, NOT confidence thresholds.",
            "Module 7 concludes strictly at soft bits and reliability summaries; Module 8 FEC/decoding is out of scope."
        ]
    }


@router.post("/soft_bits")
def post_generate_soft_bits(req: SoftBitsRequest) -> Dict[str, Any]:
    """Execute Module 7 soft bit generation given either raw IQ or explicit symbols/parameters."""
    # Direct symbol path
    if req.symbol_i is not None and req.symbol_q is not None:
        syms = np.array(req.symbol_i, dtype=np.float64) + 1j * np.array(req.symbol_q, dtype=np.float64)
        m6_dict = {"snr_db": req.snr_db} if req.snr_db is not None else {}
        return default_soft_bits_service.generate_soft_bits(
            symbol_samples=syms,
            modulation=req.modulation,
            n0=req.n0,
            module6_snr=m6_dict,
            ground_truth_bits=np.array(req.ground_truth_bits, dtype=np.uint8) if req.ground_truth_bits else None
        )

    # Raw IQ pipeline path
    if req.i_channel is not None and req.q_channel is not None:
        canonical_iq, val_res = to_canonical_iq(req.i_channel, req.q_channel)
        if val_res["status"] == "INVALID":
            return {"status": "INVALID_INPUT", "validation": val_res}

        # M2 Observation
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
        return default_soft_bits_service.generate_soft_bits(
            module5_recovery=m5_res,
            module6_snr=m6_res,
            symbol_samples=syms,
            reference_hard_bits=hard_b,
            modulation=active_mod,
            n0=req.n0,
            ground_truth_bits=np.array(req.ground_truth_bits, dtype=np.uint8) if req.ground_truth_bits else None
        )


    return {
        "status": "ERROR",
        "error": "MISSING_INPUT: Either (i_channel, q_channel) or (symbol_i, symbol_q) must be provided.",
        "warnings": ["NO_INPUT_PROVIDED"]
    }
