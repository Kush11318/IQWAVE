"""Module 3 Package: Automatic Modulation Classification (AMC)."""

from .feature_extractor import (
    FEATURE_NAMES_24,
    MODULATION_CLASSES,
    extract_24_features,
    feature_dict_to_vector
)
from .preprocessing import preprocess_for_cnn
from .rf_engine import RFEngine
from .cnn_engine import RawIQCNN, CNNEngine
from .evidence_layer import EvidenceLayerResult
from .service import AMCService, classify_amc, default_amc_service

__all__ = [
    "FEATURE_NAMES_24",
    "MODULATION_CLASSES",
    "extract_24_features",
    "feature_dict_to_vector",
    "preprocess_for_cnn",
    "RFEngine",
    "RawIQCNN",
    "CNNEngine",
    "EvidenceLayerResult",
    "AMCService",
    "classify_amc",
    "default_amc_service"
]
