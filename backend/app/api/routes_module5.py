"""API Endpoints for Module 5 (Synchronization & Digital Recovery)."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from backend.modules.module1.canonical_iq import to_canonical_iq
from backend.modules.module2.service import observe_from_module1
from backend.modules.module3.service import classify_amc
from backend.modules.module4.service import default_parameter_service
from backend.modules.module5.service import default_recovery_service

router = APIRouter(prefix="/api/module5", tags=["Module 5"])


class RecoveryRequest(BaseModel):
    i_channel: List[float]
    q_channel: List[float]
    modulation: Optional[str] = None
    sample_rate: Optional[float] = None
    samples_per_symbol: Optional[float] = None
    cfo: Optional[float] = None
    rrc_beta: Optional[float] = None


@router.get("/status")
def get_module5_status() -> Dict[str, Any]:
    """Report Module 5 synchronization and digital recovery statuses."""
    return {
        "module": "MODULE_5_SYNCHRONIZATION_AND_DIGITAL_RECOVERY",
        "status": "LOCKED",
        "supported_classes": ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "GFSK", "CPFSK"],
        "pipeline_stages": {
            "carrier_sync": "LOCKED (M-th-power FFT CFO estimation)",
            "phase_sync": "LOCKED (M-th-power phase estimation)",
            "timing_sync": "LOCKED (Gardner Timing Error Detector)",
            "matched_filter": "LOCKED (RRC matched filtering)",
            "resampling": "LOCKED (Rational polyphase resampling)",
            "demodulation": "LOCKED (Modulation-specific hard decisions)"
        },
        "scientific_boundaries": [
            "Demodulator output: recovered bits are raw demodulator output, NOT assumed to be original information bits.",
            "FEC, interleaving, and frame synchronization are outside Module 5.",
            "QAM requires blind RMS amplitude normalization for constellation slicing.",
            "FSK uses instantaneous-frequency decision, bypassing PSK constellation slicing."
        ]
    }


@router.post("/recover")
def recover_signal(req: RecoveryRequest) -> Dict[str, Any]:
    """Execute end-to-end signal recovery pipeline."""
    canonical_iq, val_res = to_canonical_iq(req.i_channel, req.q_channel)

    if val_res["status"] == "INVALID":
        return {
            "status": "INVALID_INPUT",
            "validation": val_res
        }

    # Step A: Ingest or determine modulation from Module 3
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

    # Step B: Ingest or determine parameters from Module 4
    m4_params = None
    active_sps = req.samples_per_symbol
    active_cfo = req.cfo

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

    # Step C: Execute Module 5 Recovery
    recovery_res = default_recovery_service.recover_signal(
        canonical_iq=canonical_iq,
        modulation=active_mod,
        module4_parameters=m4_params,
        sample_rate=req.sample_rate,
        samples_per_symbol=active_sps,
        cfo=active_cfo,
        rrc_beta=req.rrc_beta
    )

    recovery_res["validation"] = val_res
    return recovery_res
