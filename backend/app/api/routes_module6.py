"""API Endpoints for Module 6 (SNR Estimation & Quality Assessment)."""

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

router = APIRouter(prefix="/api/module6", tags=["Module 6"])


class SnrEstimationRequest(BaseModel):
    i_channel: List[float]
    q_channel: List[float]
    modulation: Optional[str] = None
    sample_rate: Optional[float] = None
    samples_per_symbol: Optional[float] = None
    cfo: Optional[float] = None
    reference_clean_i: Optional[List[float]] = None
    reference_clean_q: Optional[List[float]] = None
    reference_noise_i: Optional[List[float]] = None
    reference_noise_q: Optional[List[float]] = None
    reference_bits: Optional[List[int]] = None


@router.get("/status")
def get_module6_status() -> Dict[str, Any]:
    """Report Module 6 SNR estimation and quality evaluation status."""
    return {
        "module": "MODULE_6_SNR_ESTIMATION_AND_CONFIDENCE",
        "status": "LOCKED",
        "supported_classes": ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "GFSK", "CPFSK"],
        "estimator_mapping": {
            "BPSK": {"estimator": "NDA-ML (Sun, Gong & Lu 2020)", "status": "CONDITIONAL"},
            "QPSK": {"estimator": "Residual ensemble", "status": "IMPLEMENTED_COMPONENTS_FUSION_RULE_UNSPECIFIED"},
            "8PSK": {"estimator": "Residual ensemble", "status": "IMPLEMENTED_COMPONENTS_FUSION_RULE_UNSPECIFIED"},
            "QAM16": {"estimator": "Robust median residual", "status": "CONDITIONAL"},
            "QAM64": {"estimator": "Robust median residual", "status": "CONDITIONAL"},
            "CPFSK": {"estimator": "Fourth-moment NDA", "status": "LOCKED_FOR_TESTED_AWGN_FSK"},
            "GFSK": {"estimator": "Fourth-moment NDA", "status": "LOCKED_FOR_TESTED_AWGN_FSK"}
        },
        "rejected_estimators": {
            "total_power": "REJECTED (MAE ≈ 17.0038 dB, measures total variance not true SNR)",
            "fsk_frequency_residual": "REJECTED_AS_PRIMARY (MAE ≈ 3.107 dB CPFSK, 6.379 dB GFSK)"
        },
        "scientific_boundaries": [
            "Oracle SNR is strictly for offline/synthetic validation testing; strictly prohibited in production.",
            "Total-power SNR is explicitly REJECTED.",
            "FSK frequency residual is REJECTED as a primary estimator.",
            "Residual ensemble fusion rule is NOT specified by the source; both Ordinary and Robust Residual are preserved independently with snr_db=null and no invented averaging.",
            "Residual SNR represents effective SNR / SINR under current processing assumptions, reflecting uncorrected CFO, timing jitter, and multipath.",
            "Quality rating (HIGH, MEDIUM, LOW, CONDITIONAL, UNAVAILABLE) is an engineering quality assessment, NOT a calibrated statistical probability.",
            "Synchronization quality is preserved strictly as qualitative status (HIGH / MEDIUM / LOW), never a fabricated probability score.",
            "Disagreement boundaries (2 dB, 4 dB), EVM threshold (0.35), and observation length thresholds are UNVALIDATED_ENGINEERING_HEURISTIC implementations."
        ]
    }


@router.post("/estimate")
def estimate_snr_endpoint(req: SnrEstimationRequest) -> Dict[str, Any]:
    """Execute end-to-end signal SNR estimation and confidence assessment."""
    canonical_iq, val_res = to_canonical_iq(req.i_channel, req.q_channel)

    if val_res["status"] == "INVALID":
        return {
            "status": "INVALID_INPUT",
            "validation": val_res
        }

    # Step A: Determine modulation if not supplied
    active_mod = req.modulation
    if not active_mod:
        mod2_res = observe_from_module1(canonical_iq, val_res)
        amc_res = classify_amc(canonical_iq, module2_observation=mod2_res.observation)
        rf_pred = amc_res.get("engines", {}).get("engine_a_rf", {}).get("predicted_class")
        cnn_pred = amc_res.get("engines", {}).get("engine_b_cnn", {}).get("predicted_class")
        if rf_pred:
            active_mod = rf_pred
        elif cnn_pred:
            active_mod = cnn_pred
        else:
            active_mod = None

    # Step B: Determine Module 4 parameters if needed
    active_sps = req.samples_per_symbol
    active_cfo = req.cfo
    m4_params = None

    if active_sps is None or active_cfo is None:
        m4_res = default_parameter_service.estimate_parameters(
            canonical_iq=canonical_iq,
            modulation=active_mod,
            sample_rate=req.sample_rate
        )
        m4_params = m4_res
        if active_sps is None:
            active_sps = m4_res.get("samples_per_symbol")
        if active_cfo is None:
            active_cfo = m4_res.get("cfo")

    # Step C: Execute Module 5 digital recovery for symbol/decision outputs
    m5_recovery = default_recovery_service.recover_signal(
        canonical_iq=canonical_iq,
        modulation=active_mod,
        module4_parameters=m4_params,
        sample_rate=req.sample_rate,
        samples_per_symbol=active_sps,
        cfo=active_cfo
    )

    # Step D: Construct reference clean and noise signals if provided
    ref_clean = None
    if req.reference_clean_i is not None and req.reference_clean_q is not None:
        if len(req.reference_clean_i) == len(req.reference_clean_q):
            ref_clean = np.array(req.reference_clean_i, dtype=np.float32) + 1j * np.array(req.reference_clean_q, dtype=np.float32)

    ref_noise = None
    if req.reference_noise_i is not None and req.reference_noise_q is not None:
        if len(req.reference_noise_i) == len(req.reference_noise_q):
            ref_noise = np.array(req.reference_noise_i, dtype=np.float32) + 1j * np.array(req.reference_noise_q, dtype=np.float32)

    # Step E: Execute Module 6 SNR estimation
    snr_result = default_snr_service.estimate_snr(
        canonical_iq=canonical_iq,
        modulation=active_mod,
        module4_parameters=m4_params,
        module5_recovery=m5_recovery,
        reference_clean_signal=ref_clean,
        reference_noise_signal=ref_noise,
        reference_bits=req.reference_bits
    )

    return snr_result
