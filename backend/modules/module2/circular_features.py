"""Module 2: Circular & Second-Order Feature Extraction.

Features:
- circular_variance: Core structural feature from differential circular statistics.
  Evaluated on unit differential phasors:
      V = 1 - |(1/M) * sum(exp(j * Delta_phi[n]))|
  CFO-robust and comparatively stable across RRC and multipath experiments.

- circularity_ratio (M20/M21): Supporting / post-sync evidence.
      M20 = E[x[n]^2]
      M21 = E[|x[n]|^2]
      Ratio = |M20| / M21
  Sensitive to CFO and impairments; retained strictly as supporting evidence.
"""

from typing import Dict, Optional
import numpy as np


def extract_circular_features(iq_signal: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute differential circular variance and M20/M21 circularity ratio.

    Does not modify the signal.
    """
    res: Dict[str, Optional[float]] = {
        "variance": None,
        "m20_m21": None
    }

    if iq_signal is None or len(iq_signal) == 0:
        return res

    if not np.all(np.isfinite(iq_signal)):
        return res

    n = len(iq_signal)

    # 1. Differential Circular Variance (CORE)
    if n >= 2:
        prod = iq_signal[1:] * np.conj(iq_signal[:-1])
        mag = np.abs(prod)
        valid_mask = mag > 1e-12

        if np.any(valid_mask):
            unit_phasors = prod[valid_mask] / mag[valid_mask]
            mean_resultant_vector = np.mean(unit_phasors)
            r_bar = float(np.abs(mean_resultant_vector))
            # Circular variance V = 1 - R_bar, bounded strictly in [0, 1]
            circ_var = float(np.clip(1.0 - r_bar, 0.0, 1.0))
            res["variance"] = circ_var
        else:
            res["variance"] = None
    else:
        res["variance"] = None

    # 2. Second-Order Circularity Ratio (Supporting / Post-Sync Evidence)
    m21 = float(np.mean(np.abs(iq_signal) ** 2))
    if m21 > 1e-12:
        m20 = complex(np.mean(iq_signal.astype(np.complex128) ** 2))
        res["m20_m21"] = float(np.abs(m20) / m21)
    else:
        res["m20_m21"] = None

    return res
