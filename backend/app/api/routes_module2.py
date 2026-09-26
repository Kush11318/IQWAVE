"""API Endpoints for Module 2 (Non-Destructive Observation)."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.modules.module1.canonical_iq import to_canonical_iq
from backend.modules.module2.service import (
    observe_signal,
    observe_from_module1,
    SignalRepresentationType
)

router = APIRouter(prefix="/api/module2", tags=["Module 2"])


class ObserveArraysRequest(BaseModel):
    i_channel: List[float]
    q_channel: List[float]
    metadata: Optional[Dict[str, Any]] = None


@router.get("/status")
def get_module2_status() -> Dict[str, Any]:
    """Report Module 2 architectural status and feature classifications."""
    return {
        "status": "LOCKED",
        "components": {
            "input_validation": "LOCKED",
            "non_destructive_observation": "LOCKED",
            "amplitude_features": "KEEP",
            "differential_phase_features": "KEEP",
            "differential_circular_variance": "CORE",
            "m20_m21": "POST-SYNC / supporting evidence",
            "zero_bin_fraction": "spectral evidence",
            "representation_gate": "LOCKED",
            "raw_signal_preservation": "LOCKED",
            "universal_dc_removal": "REJECTED",
            "universal_thresholds": "REJECTED",
            "blind_wav_hilbert": "REJECTED",
            "more_feature_invention": "STOPPED"
        }
    }


@router.post("/observe")
def observe_iq_arrays(req: ObserveArraysRequest) -> Dict[str, Any]:
    """Execute Module 2 non-destructive observation via canonical Module 1 conversion."""
    # Convert through Module 1A locked canonical representation
    canonical_iq, val_res = to_canonical_iq(req.i_channel, req.q_channel)

    if val_res["status"] == "INVALID":
        mod2_res = observe_from_module1(canonical_iq, val_res)
        return mod2_res.to_dict()

    mod2_res = observe_from_module1(canonical_iq, val_res)
    return mod2_res.to_dict()
