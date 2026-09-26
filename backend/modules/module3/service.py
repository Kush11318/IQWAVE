"""Module 3 Service: Unified AMC Orchestrator.

Integrates:
- Module 1: Canonical IQ ingestion
- Module 2: Non-destructive observation vector
- Module 3 Engine A: 24-feature Random Forest
- Module 3 Engine B: Raw-IQ 1D CNN with RMS normalization
- Module 3 Evidence Layer: Evidence aggregation with unvalidated fusion boundary.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from backend.modules.module1.canonical_iq import to_canonical_iq
from backend.modules.module2.observation import compute_observation_vector
from .feature_extractor import (
    FEATURE_NAMES_24,
    MODULATION_CLASSES,
    extract_24_features
)
from .preprocessing import preprocess_for_cnn
from .rf_engine import RFEngine
from .cnn_engine import CNNEngine
from .evidence_layer import EvidenceLayerResult


class AMCService:
    def __init__(
        self,
        rf_engine: Optional[RFEngine] = None,
        cnn_engine: Optional[CNNEngine] = None
    ):
        self.rf_engine = rf_engine or RFEngine()
        self.cnn_engine = cnn_engine or CNNEngine()

    def classify_signal(
        self,
        canonical_iq: np.ndarray,
        module2_observation: Optional[Dict[str, Any]] = None,
        estimated_snr: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute the complete Module 3 AMC classification pipeline.

        Non-destructive: original canonical_iq is never modified.
        """
        if canonical_iq is None or len(canonical_iq) == 0:
            return {
                "status": "INVALID_INPUT",
                "error": "EMPTY_OR_UNAVAILABLE_SIGNAL",
                "fusion_status": "NOT_YET_VALIDATED"
            }

        # 1. Extract 24 engineered features (Engine A input)
        features_24 = extract_24_features(
            iq_signal=canonical_iq,
            module2_observation=module2_observation
        )

        # 2. Run Engine A (Random Forest)
        rf_result = self.rf_engine.predict(features_24)

        # 3. Preprocess for Engine B (RMS normalization on copy)
        cnn_input, measured_power = preprocess_for_cnn(canonical_iq)

        # 4. Run Engine B (Raw-IQ CNN)
        cnn_result = self.cnn_engine.predict(cnn_input)

        # 5. Assemble Evidence Layer
        evidence_result = EvidenceLayerResult(
            rf_result=rf_result,
            cnn_result=cnn_result,
            structural_evidence=features_24,
            estimated_snr=estimated_snr
        )

        out = evidence_result.to_dict()
        out["status"] = "SUCCESS"
        out["features_24"] = features_24
        return out


# Global singleton service
default_amc_service = AMCService()


def classify_amc(
    canonical_iq: np.ndarray,
    module2_observation: Optional[Dict[str, Any]] = None,
    estimated_snr: Optional[float] = None
) -> Dict[str, Any]:
    """Convenience functional interface for AMC classification."""
    return default_amc_service.classify_signal(
        canonical_iq=canonical_iq,
        module2_observation=module2_observation,
        estimated_snr=estimated_snr
    )
