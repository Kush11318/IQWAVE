"""Module 2 Service: Non-Destructive Observation Pipeline.

Orchestrates:
1. Representation gating (checks representation, rejects blind WAV -> Hilbert conversion).
2. Extraction of non-destructive observation vector.
3. Retention of both raw IQ and observation vector.
4. Seamless consumption of canonical IQ from Module 1.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from .representation_gate import (
    SignalRepresentationType,
    RepresentationGateResult,
    process_representation_gate
)
from .observation import compute_observation_vector


class Module2Result:
    def __init__(
        self,
        status: str,
        observation: Dict[str, Any],
        raw_signal: Any,
        analysis_signal: Optional[np.ndarray] = None,
        rejection_reason: Optional[str] = None
    ):
        self.status = status
        self.observation = observation
        self.raw_signal = raw_signal
        self.analysis_signal = analysis_signal
        self.rejection_reason = rejection_reason

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response (raw_signal array excluded or summarized)."""
        return {
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "observation": self.observation
        }


def observe_signal(
    signal: Any,
    representation_type: Optional[SignalRepresentationType] = None,
    allow_explicit_analytic_conversion: bool = False
) -> Module2Result:
    """Execute Module 2 non-destructive observation on an input signal.

    Preserves raw_signal unconditionally.
    """
    gate_result: RepresentationGateResult = process_representation_gate(
        signal=signal,
        representation_type=representation_type,
        allow_explicit_analytic_conversion=allow_explicit_analytic_conversion
    )

    if not gate_result.is_admissible:
        obs = compute_observation_vector(None, gate_result.representation_info)
        obs["quality"]["rejection_reason"] = gate_result.rejection_reason
        return Module2Result(
            status="REJECTED",
            observation=obs,
            raw_signal=gate_result.raw_signal,
            analysis_signal=None,
            rejection_reason=gate_result.rejection_reason
        )

    # Compute observations on non-destructively guarded analysis copy
    obs = compute_observation_vector(
        analysis_signal=gate_result.analysis_signal,
        representation_info=gate_result.representation_info
    )

    return Module2Result(
        status="VALID",
        observation=obs,
        raw_signal=gate_result.raw_signal,
        analysis_signal=gate_result.analysis_signal
    )


def observe_from_module1(
    module1_canonical_iq: np.ndarray,
    module1_validation_result: Optional[Dict[str, Any]] = None
) -> Module2Result:
    """Consume canonical complex64 IQ directly from Module 1."""
    if module1_validation_result and module1_validation_result.get("status") == "INVALID":
        return Module2Result(
            status="REJECTED",
            observation=compute_observation_vector(
                None,
                {"input_representation": "INVALID_MODULE1_INPUT"}
            ),
            raw_signal=module1_canonical_iq,
            analysis_signal=None,
            rejection_reason="REJECTED_BY_MODULE1_VALIDATION"
        )

    return observe_signal(
        signal=module1_canonical_iq,
        representation_type=SignalRepresentationType.COMPLEX_IQ
    )
