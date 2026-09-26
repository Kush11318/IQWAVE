"""Module 2: Representation Gate.

Enforces representation checking and rejects blind WAV -> Hilbert conversion.
The experiments in Module 2 explicitly demonstrated that arbitrary
real waveform -> Hilbert -> analytic IQ substantially alters phase statistics
and M20/M21. Therefore:
- The representation must be explicitly identified.
- Blind conversion must NOT silently occur.
- The raw signal is preserved without permanent modification.
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.signal import hilbert


class SignalRepresentationType(str, Enum):
    COMPLEX_IQ = "COMPLEX_IQ"
    REAL_IF_PASSBAND = "REAL_IF_PASSBAND"
    UNKNOWN_REAL_WAV = "UNKNOWN_REAL_WAV"
    UNSUPPORTED = "UNSUPPORTED"


class RepresentationGateResult:
    def __init__(
        self,
        analysis_signal: Optional[np.ndarray],
        raw_signal: Any,
        representation_info: Dict[str, Any],
        is_admissible: bool,
        rejection_reason: Optional[str] = None
    ):
        self.analysis_signal = analysis_signal
        self.raw_signal = raw_signal
        self.representation_info = representation_info
        self.is_admissible = is_admissible
        self.rejection_reason = rejection_reason


def process_representation_gate(
    signal: Any,
    representation_type: Optional[SignalRepresentationType] = None,
    allow_explicit_analytic_conversion: bool = False
) -> RepresentationGateResult:
    """Inspect and route the input signal through the representation gate.

    Rules:
    1. If signal is complex IQ (np.iscomplexobj or dtype complex), it is passed directly
       as the common analysis signal.
    2. If signal is real and explicitly designated as REAL_IF_PASSBAND with
       allow_explicit_analytic_conversion=True, controlled Hilbert analytic conversion
       is performed for the analysis copy, while preserving the raw real signal.
    3. If signal is real/WAV without explicit verification or configuration, blind
       Hilbert conversion is REJECTED.
    """
    raw_signal = signal
    sig_arr = np.asarray(signal)

    if sig_arr.size == 0:
        return RepresentationGateResult(
            analysis_signal=None,
            raw_signal=raw_signal,
            representation_info={
                "input_representation": "EMPTY",
                "common_signal_type": "NONE",
                "analytic_conversion_applied": False
            },
            is_admissible=False,
            rejection_reason="EMPTY_SIGNAL"
        )

    # 1. Complex IQ input
    if np.iscomplexobj(sig_arr) or representation_type == SignalRepresentationType.COMPLEX_IQ:
        return RepresentationGateResult(
            analysis_signal=sig_arr.astype(np.complex64),
            raw_signal=raw_signal,
            representation_info={
                "input_representation": SignalRepresentationType.COMPLEX_IQ.value,
                "common_signal_type": "COMPLEX_IQ",
                "analytic_conversion_applied": False
            },
            is_admissible=True
        )

    # 2. Real signal input
    if not np.iscomplexobj(sig_arr):
        if (
            representation_type == SignalRepresentationType.REAL_IF_PASSBAND
            and allow_explicit_analytic_conversion
        ):
            # Explicit, deliberate analytic conversion
            analytic_copy = hilbert(sig_arr.astype(np.float32)).astype(np.complex64)
            return RepresentationGateResult(
                analysis_signal=analytic_copy,
                raw_signal=raw_signal,
                representation_info={
                    "input_representation": SignalRepresentationType.REAL_IF_PASSBAND.value,
                    "common_signal_type": "ANALYTIC_IQ",
                    "analytic_conversion_applied": True,
                    "warning": "Analytic conversion alters phase statistics and M20/M21 relative to true baseband IQ."
                },
                is_admissible=True
            )
        else:
            # Blind WAV / Real -> Hilbert conversion REJECTED
            return RepresentationGateResult(
                analysis_signal=None,
                raw_signal=raw_signal,
                representation_info={
                    "input_representation": SignalRepresentationType.UNKNOWN_REAL_WAV.value,
                    "common_signal_type": "NONE",
                    "analytic_conversion_applied": False
                },
                is_admissible=False,
                rejection_reason="BLIND_HILBERT_CONVERSION_REJECTED"
            )

    return RepresentationGateResult(
        analysis_signal=None,
        raw_signal=raw_signal,
        representation_info={
            "input_representation": SignalRepresentationType.UNSUPPORTED.value,
            "common_signal_type": "NONE",
            "analytic_conversion_applied": False
        },
        is_admissible=False,
        rejection_reason="UNSUPPORTED_SIGNAL_REPRESENTATION"
    )
