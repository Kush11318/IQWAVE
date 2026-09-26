"""Module 2 Package: Non-Destructive Observation Layer."""

from .representation_gate import (
    SignalRepresentationType,
    RepresentationGateResult,
    process_representation_gate
)
from .amplitude_features import extract_amplitude_features
from .phase_features import extract_phase_features
from .circular_features import extract_circular_features
from .spectral_features import extract_spectral_features
from .observation import compute_observation_vector
from .service import Module2Result, observe_signal, observe_from_module1

__all__ = [
    "SignalRepresentationType",
    "RepresentationGateResult",
    "process_representation_gate",
    "extract_amplitude_features",
    "extract_phase_features",
    "extract_circular_features",
    "extract_spectral_features",
    "compute_observation_vector",
    "Module2Result",
    "observe_signal",
    "observe_from_module1"
]
