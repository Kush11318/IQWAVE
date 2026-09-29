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
    FEATURE_NAMES_32,
    MODULATION_CLASSES,
    extract_24_features,
    extract_features
)
from .preprocessing import preprocess_for_cnn
from .rf_engine import RFEngine
from .cnn_engine import CNNEngine
from .multiscale_cnn_engine import CNNEnsembleEngine, MultiScaleCNNEngine
from .fusion_engine import FusionEngine
from .evidence_layer import EvidenceLayerResult


class AMCService:
    def __init__(
        self,
        rf_engine: Optional[RFEngine] = None,
        cnn_engine: Optional[CNNEngine] = None,
        ensemble_engine: Optional[CNNEnsembleEngine] = None,
        fusion_engine: Optional[FusionEngine] = None
    ):
        self.rf_engine = rf_engine or RFEngine()
        self.cnn_engine = cnn_engine or CNNEngine()
        self.ensemble_engine = ensemble_engine or CNNEnsembleEngine()
        self.fusion_engine = fusion_engine or FusionEngine()

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

        # 1. Extract 32 engineered features and 24-feature backward-compatible slice
        features_32 = extract_features(
            iq_signal=canonical_iq,
            module2_observation=module2_observation
        )
        features_24 = {k: features_32.get(k) for k in FEATURE_NAMES_24}

        # 2. Run Engine A (Random Forest with 32 or 24 features)
        rf_result = self.rf_engine.predict(features_32)

        # 3. Preprocess for CNN (RMS normalization on copy)
        cnn_input, measured_power = preprocess_for_cnn(canonical_iq)

        # 4. Run Engine B (Raw-IQ CNN)
        cnn_result = self.cnn_engine.predict(cnn_input)

        # 5. Run Engine C (Ensemble CNN) & Extract sequence GAP features
        ensemble_result = None
        gap_features = None
        if self.ensemble_engine.is_available:
            ensemble_result = self.ensemble_engine.predict(cnn_input)
            gap_features = self.ensemble_engine.extract_gap_features(cnn_input)

        # 6. Run Trained Dual-Branch Fusion Head (Phase 5)
        fusion_result = None
        if self.fusion_engine.is_available and gap_features is not None:
            fusion_result = self.fusion_engine.predict(
                rf_features=features_32,
                cnn_gap_vector=gap_features
            )

        # 7. Assemble Evidence Layer
        evidence_result = EvidenceLayerResult(
            rf_result=rf_result,
            cnn_result=cnn_result,
            structural_evidence=features_32,
            estimated_snr=estimated_snr,
            fusion_result=fusion_result,
            ensemble_cnn_result=ensemble_result
        )

        out = evidence_result.to_dict()
        out["status"] = "SUCCESS"
        out["features_24"] = features_24
        out["features_32"] = features_32
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

