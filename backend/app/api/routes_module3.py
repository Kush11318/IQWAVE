"""API Endpoints for Module 3 (Automatic Modulation Classification)."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from backend.modules.module1.canonical_iq import to_canonical_iq
from backend.modules.module2.service import observe_from_module1
from backend.modules.module3.feature_extractor import FEATURE_NAMES_24, MODULATION_CLASSES
from backend.modules.module3.service import default_amc_service

router = APIRouter(prefix="/api/module3", tags=["Module 3"])


class AMCClassifyRequest(BaseModel):
    i_channel: List[float]
    q_channel: List[float]
    estimated_snr: Optional[float] = None


@router.get("/status")
def get_module3_status() -> Dict[str, Any]:
    """Report Module 3 engine status, artifact availability, and benchmark metrics."""
    return {
        "module": "MODULE_3_AUTOMATIC_MODULATION_CLASSIFICATION",
        "status": "LOCKED",
        "supported_classes": MODULATION_CLASSES,
        "feature_count": len(FEATURE_NAMES_24),
        "engine_a_rf": {
            "name": "24-Feature Random Forest",
            "status": "AVAILABLE" if default_amc_service.rf_engine.is_available else "MODEL_WEIGHTS_UNAVAILABLE",
            "artifact_path": default_amc_service.rf_engine.model_path,
            "configuration": default_amc_service.rf_engine.config,
            "validated_benchmark": {
                "accuracy": "69.00%",
                "macro_f1": "69.06%"
            }
        },
        "engine_b_cnn": {
            "name": "Raw-IQ 1D CNN",
            "status": "AVAILABLE" if default_amc_service.cnn_engine.is_available else "MODEL_WEIGHTS_UNAVAILABLE",
            "artifact_path": default_amc_service.cnn_engine.model_path,
            "parameter_count": 101319,
            "validated_benchmark": {
                "accuracy": "68.97%",
                "macro_f1": "68.63%"
            }
        },
        "engine_c_ensemble": {
            "name": "Deep Ensemble (3x Multi-Scale Dilated CNN)",
            "status": "AVAILABLE" if default_amc_service.ensemble_engine.is_available else "MODEL_WEIGHTS_UNAVAILABLE",
            "n_models_loaded": default_amc_service.ensemble_engine.n_models_loaded
        },
        "fusion_engine": {
            "name": "Dual-Branch Late Fusion Head",
            "status": "AVAILABLE" if default_amc_service.fusion_engine.is_available else "MODEL_WEIGHTS_UNAVAILABLE",
            "artifact_path": default_amc_service.fusion_engine.model_path
        },
        "fusion_status": "TRAINED_VALIDATED" if default_amc_service.fusion_engine.is_available else "NOT_YET_VALIDATED",
        "confidence_status": "CALIBRATED" if default_amc_service.fusion_engine.is_available else "NOT_YET_DEFINED"
    }


@router.post("/classify")
def classify_signal(req: AMCClassifyRequest) -> Dict[str, Any]:
    """End-to-end classification pipeline: Module 1 -> Module 2 -> Module 3."""
    canonical_iq, val_res = to_canonical_iq(req.i_channel, req.q_channel)

    if val_res["status"] == "INVALID":
        return {
            "status": "INVALID_INPUT",
            "validation": val_res,
            "fusion_status": "NOT_YET_VALIDATED"
        }

    # Module 2 Observation
    mod2_res = observe_from_module1(canonical_iq, val_res)

    # Module 3 AMC Execution
    amc_res = default_amc_service.classify_signal(
        canonical_iq=canonical_iq,
        module2_observation=mod2_res.observation,
        estimated_snr=req.estimated_snr
    )
    amc_res["validation"] = val_res
    return amc_res
