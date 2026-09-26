"""Module 2: Differential Phase Feature Extraction.

Extracts differential phase statistics:
- phase_diff_std: Standard deviation of consecutive phase differences.
- gated_phase_diff_std: Amplitude-gated phase difference standard deviation
  (filters out noise-dominated transitions near constellation zero-crossings).

Dropped from primary features per Module 2 specification:
- if_mean_norm (dropped)
- if_std_norm (dropped)
- phase_diff_mean (dropped as primary invariant due to CFO sensitivity)
"""

from typing import Dict, Optional
import numpy as np


def extract_phase_features(
    iq_signal: np.ndarray,
    gate_factor: float = 0.5
) -> Dict[str, Optional[float]]:
    """Compute differential phase features from the complex signal.

    Consecutive phase difference:
        Delta_phi[n] = angle(x[n] * conj(x[n-1])) in [-pi, pi]

    Gated differential phase:
        Evaluated on sample pairs where both |x[n]| and |x[n-1]| >= gate_factor * mean(|x|).
    """
    res: Dict[str, Optional[float]] = {
        "diff_std": None,
        "gated_diff_std": None
    }

    if iq_signal is None or len(iq_signal) < 2:
        return res

    if not np.all(np.isfinite(iq_signal)):
        return res

    # Consecutive complex conjugate product gives wrapped angular difference directly
    prod = iq_signal[1:] * np.conj(iq_signal[:-1])
    diff_phase = np.angle(prod)

    if len(diff_phase) > 0:
        res["diff_std"] = float(np.std(diff_phase))

    # Amplitude gating
    amp = np.abs(iq_signal)
    mean_amp = float(np.mean(amp))

    if mean_amp > 1e-12:
        threshold = gate_factor * mean_amp
        gate_mask = (amp[1:] >= threshold) & (amp[:-1] >= threshold)
        gated_diff_phase = diff_phase[gate_mask]

        if len(gated_diff_phase) >= 2:
            res["gated_diff_std"] = float(np.std(gated_diff_phase))
        else:
            res["gated_diff_std"] = None
    else:
        res["gated_diff_std"] = None

    return res
