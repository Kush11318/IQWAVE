"""API Endpoints for Module 4 (Signal Parameter Estimation)."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from backend.modules.module1.canonical_iq import to_canonical_iq
from backend.modules.module2.service import observe_from_module1
from backend.modules.module3.service import classify_amc
from backend.modules.module4.service import default_parameter_service

router = APIRouter(prefix="/api/module4", tags=["Module 4"])


class ParameterEstimateRequest(BaseModel):
    i_channel: List[float]
    q_channel: List[float]
    modulation: Optional[str] = None
    sample_rate: Optional[float] = None
    center_frequency: Optional[float] = None
    estimated_snr: Optional[float] = None


@router.get("/status")
def get_module4_status() -> Dict[str, Any]:
    """Report Module 4 estimator statuses and physical limitations."""
    return {
        "module": "MODULE_4_SIGNAL_PARAMETER_ESTIMATION",
        "status": "LOCKED",
        "estimators": {
            "symbol_rate": "LOCKED (Cyclic-correlation Ciblat)",
            "samples_per_symbol": "LOCKED (Fs / Rs when Fs available)",
            "psk_cfo": "LOCKED (M-th-power spectral estimator)",
            "qam16_cfo": "CONDITIONAL (Weighted full-rate 4th-power)",
            "qam64_cfo": "NOT_VALIDATED",
            "relative_center_frequency": "LOCKED (Adaptive PSD region)",
            "occupied_bandwidth": "LOCKED (Adaptive PSD edge detector)",
            "fsk_frequency_states": "LOCKED (Instantaneous-frequency clustering)",
            "fsk_frequency_separation": "LOCKED (Cluster separation Delta-f)",
            "absolute_rf_frequency": "METADATA_REQUIRED",
            "absolute_sampling_rate": "METADATA_REQUIRED"
        },
        "limitations": [
            "Absolute RF center frequency cannot be recovered from raw IQ samples alone without metadata/reference.",
            "Absolute physical sampling rate cannot be uniquely determined from raw IQ samples without metadata.",
            "QAM CFO is conditional and not universally valid across all QAM orders."
        ]
    }


@router.post("/estimate")
def estimate_parameters(req: ParameterEstimateRequest) -> Dict[str, Any]:
    """Execute end-to-end signal parameter estimation pipeline."""
    canonical_iq, val_res = to_canonical_iq(req.i_channel, req.q_channel)

    if val_res["status"] == "INVALID":
        return {
            "status": "INVALID_INPUT",
            "validation": val_res
        }

    # Modulation provided explicitly in request (Module 4 does not perform AMC)
    active_mod = req.modulation

    meta = {
        "sample_rate": req.sample_rate,
        "center_frequency": req.center_frequency
    }

    result = default_parameter_service.estimate_parameters(
        canonical_iq=canonical_iq,
        modulation=active_mod,
        metadata=meta,
        estimated_snr=req.estimated_snr
    )
    result["validation"] = val_res
    return result
