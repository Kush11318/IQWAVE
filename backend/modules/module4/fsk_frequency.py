"""Module 4B: FSK Instantaneous Frequency & State Clustering.

Estimates sample-by-sample instantaneous frequency:
    f[n] = (Fs / (2 * pi)) * angle(x[n] * conj(x[n-1]))

Performs 2-state frequency clustering to extract:
    frequency_state_0 (f0)
    frequency_state_1 (f1)
    frequency_separation (Delta f = |f1 - f0|)

Observation Length & Noise Invariant:
- At low SNR, exact recovery requires sufficient observation length (N >= 40,000 samples).
- If observation is short or clustering separation is low, flags are raised and
  confidence is adjusted accordingly.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans


def estimate_fsk_parameters(
    iq_signal: np.ndarray,
    sample_rate: Optional[float] = None
) -> Dict[str, Any]:
    """Estimate FSK frequency states f0, f1, and separation Delta f.

    If sample_rate is None, outputs are in normalized frequency units (cycles/sample).
    """
    res: Dict[str, Any] = {
        "frequency_state_0": None,
        "frequency_state_1": None,
        "frequency_separation": None,
        "confidence": "UNKNOWN",
        "method": "INSTANTANEOUS_FREQUENCY_CLUSTERING",
        "flags": []
    }

    if iq_signal is None or len(iq_signal) < 64:
        res["flags"].append("INSUFFICIENT_SAMPLES")
        res["confidence"] = "LOW"
        return res

    if not np.all(np.isfinite(iq_signal)):
        res["flags"].append("NON_FINITE_SAMPLES")
        res["confidence"] = "LOW"
        return res

    n = len(iq_signal)
    if n < 40000:
        res["flags"].append("SHORT_OBSERVATION_FOR_LOW_SNR_FSK")

    # 1. Sample-by-sample instantaneous frequency
    # Phase difference between adjacent samples in [-pi, pi]
    prod = iq_signal[1:] * np.conj(iq_signal[:-1])
    diff_phase = np.angle(prod)

    # Scale to Hz if Fs is known, else normalized cycles/sample in [-0.5, 0.5]
    fs_val = sample_rate if (sample_rate is not None and sample_rate > 0) else 1.0
    inst_freq = (fs_val / (2.0 * np.pi)) * diff_phase

    # Filter out outlier spikes caused by zero-crossings
    q25, q75 = np.percentile(inst_freq, [25, 75])
    iqr = q75 - q25
    valid_mask = (inst_freq >= (q25 - 3 * iqr)) & (inst_freq <= (q75 + 3 * iqr))
    clean_freq = inst_freq[valid_mask]

    if len(clean_freq) < 32:
        res["flags"].append("FAILED_OUTLIER_FILTERING")
        res["confidence"] = "LOW"
        return res

    # 2. Two-state clustering
    try:
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km.fit(clean_freq.reshape(-1, 1))
        centers = np.sort(km.cluster_centers_.flatten())

        f0 = float(centers[0])
        f1 = float(centers[1])
        delta_f = float(np.abs(f1 - f0))

        # Check cluster validity: separation must exceed intra-cluster variance
        c0_std = float(np.std(clean_freq[km.labels_ == 0]))
        c1_std = float(np.std(clean_freq[km.labels_ == 1]))
        spread = max(c0_std, c1_std)

        if delta_f > 2.0 * spread:
            res["confidence"] = "HIGH" if n >= 20000 else "MEDIUM"
        elif delta_f > spread:
            res["confidence"] = "MEDIUM"
        else:
            res["confidence"] = "LOW"
            res["flags"].append("POOR_CLUSTER_SEPARATION")

        if sample_rate is not None and sample_rate > 0:
            res["frequency_state_0"] = f0
            res["frequency_state_1"] = f1
            res["frequency_separation"] = delta_f
        else:
            # Normalized units
            res["frequency_state_0_normalized"] = f0
            res["frequency_state_1_normalized"] = f1
            res["frequency_separation_normalized"] = delta_f
            res["flags"].append("METADATA_REQUIRED_FOR_ABSOLUTE_FREQUENCIES")

    except Exception as e:
        res["flags"].append(f"CLUSTERING_FAILED: {str(e)}")
        res["confidence"] = "LOW"

    return res
