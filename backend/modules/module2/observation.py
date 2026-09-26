"""Module 2: Unified Observation Vector Extractor.

Assembles the full non-destructive observation vector:
- representation
- amplitude (mean, std, cv, skewness, kurtosis)
- phase (diff_std, gated_diff_std)
- circular (variance, m20_m21)
- spectral (zero_bin_fraction)
- quality (finite, power, observation_length)

Guarantees non-destructive processing: preserves original samples.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from .amplitude_features import extract_amplitude_features
from .phase_features import extract_phase_features
from .circular_features import extract_circular_features
from .spectral_features import extract_spectral_features


def compute_observation_vector(
    analysis_signal: Optional[np.ndarray],
    representation_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract the complete Module 2 observation vector from the analysis signal.

    Strictly non-destructive: does not modify analysis_signal.
    """
    if analysis_signal is None or len(analysis_signal) == 0:
        return {
            "representation": representation_info,
            "amplitude": {
                "mean": None,
                "std": None,
                "cv": None,
                "skewness": None,
                "kurtosis": None
            },
            "phase": {
                "diff_std": None,
                "gated_diff_std": None
            },
            "circular": {
                "variance": None,
                "m20_m21": None
            },
            "spectral": {
                "zero_bin_fraction": None
            },
            "quality": {
                "finite": False,
                "power": None,
                "observation_length": 0,
                "error": "EMPTY_OR_UNAVAILABLE_SIGNAL"
            }
        }

    n_samples = len(analysis_signal)
    is_finite = bool(np.all(np.isfinite(analysis_signal)))
    power = float(np.mean(np.abs(analysis_signal) ** 2)) if is_finite and n_samples > 0 else None

    quality = {
        "finite": is_finite,
        "power": power,
        "observation_length": n_samples
    }

    if not is_finite:
        quality["warning"] = "NON_FINITE_SAMPLES_DETECTED"

    if power == 0.0:
        quality["warning"] = "ZERO_POWER_SIGNAL"

    # Extract component features
    amplitude_dict = extract_amplitude_features(analysis_signal)
    phase_dict = extract_phase_features(analysis_signal)
    circular_dict = extract_circular_features(analysis_signal)
    spectral_dict = extract_spectral_features(analysis_signal)

    return {
        "representation": representation_info,
        "amplitude": amplitude_dict,
        "phase": phase_dict,
        "circular": circular_dict,
        "spectral": spectral_dict,
        "quality": quality
    }
