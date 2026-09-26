"""Module 4D: Relative Center Frequency & Occupied Bandwidth Estimator.

Implements the validated adaptive PSD edge-detection pipeline:
    IQ
     ↓
    Welch PSD
     ↓
    Gaussian smoothing (smoothing_sigma = 5)
     ↓
    Noise-floor estimation (lower_percentile = 40)
     ↓
    Adaptive threshold (threshold = noise_floor + k_sigma * noise_std, k_sigma = 1)
     ↓
    Strongest connected spectral region
     ↓
    Edges: f_left, f_right
     ↓
    Occupied bandwidth = f_right - f_left
    Relative center frequency = (f_left + f_right) / 2

CRITICAL PHYSICAL LIMITATION:
- Absolute RF frequency CANNOT be deduced from raw baseband IQ alone.
- If metadata provides an RF center frequency (e.g., 433 MHz), then:
    absolute_rf_frequency = metadata_center + relative_center_frequency
- Otherwise:
    absolute_rf_frequency = None, with flag "METADATA_REQUIRED".
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage


def estimate_relative_center_and_bandwidth(
    iq_signal: np.ndarray,
    sample_rate: Optional[float] = None,
    k_sigma: float = 1.0,
    lower_percentile: float = 40.0,
    smoothing_sigma: float = 5.0
) -> Dict[str, Any]:
    """Estimate relative spectral center and occupied bandwidth using the validated PSD parameters."""
    res: Dict[str, Any] = {
        "relative_center_frequency": None,
        "relative_center_normalized": None,
        "occupied_bandwidth": None,
        "occupied_bandwidth_normalized": None,
        "confidence": "UNKNOWN",
        "method": "ADAPTIVE_PSD_CONNECTED_REGION",
        "parameters": {
            "k_sigma": k_sigma,
            "lower_percentile": lower_percentile,
            "smoothing_sigma": smoothing_sigma
        },
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

    fs_val = sample_rate if (sample_rate is not None and sample_rate > 0) else 1.0

    # 1. Welch PSD estimation
    nperseg = min(1024, len(iq_signal))
    freqs, psd = signal.welch(
        iq_signal,
        fs=fs_val,
        nperseg=nperseg,
        return_onesided=False
    )
    # Sort frequencies monotonically from -Fs/2 to +Fs/2
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    psd = psd[sort_idx]

    # 2. Gaussian-style spectral smoothing
    smoothed_psd = ndimage.gaussian_filter1d(psd, sigma=smoothing_sigma)

    # 3. Noise-floor estimation (validated at 40th percentile)
    noise_floor = float(np.percentile(smoothed_psd, lower_percentile))
    lower_noise_samples = smoothed_psd[smoothed_psd <= noise_floor]
    noise_std = float(np.std(lower_noise_samples)) if len(lower_noise_samples) > 1 else 0.0

    # 4. Adaptive threshold
    threshold = noise_floor + (k_sigma * noise_std)

    # 5. Connected region identification
    mask = smoothed_psd > threshold
    labels, num_features = ndimage.label(mask)

    if num_features == 0:
        res["flags"].append("NO_SPECTRAL_REGION_ABOVE_THRESHOLD")
        res["confidence"] = "LOW"
        return res

    # Strongest connected region by integrated power
    region_powers = [np.sum(psd[labels == i]) for i in range(1, num_features + 1)]
    best_label = int(np.argmax(region_powers) + 1)
    region_freqs = freqs[labels == best_label]

    f_left = float(region_freqs[0])
    f_right = float(region_freqs[-1])
    est_bw = float(f_right - f_left)

    # Power-weighted spectral center (spectral centroid of the connected region)
    region_psd = psd[labels == best_label]
    region_total_p = np.sum(region_psd)
    if region_total_p > 1e-12:
        est_center = float(np.sum(region_freqs * region_psd) / region_total_p)
    else:
        est_center = float((f_left + f_right) / 2.0)

    # Quality check
    peak_val = np.max(smoothed_psd)
    snr_proxy = (peak_val - noise_floor) / (noise_std + 1e-12)
    if snr_proxy > 10.0:
        res["confidence"] = "HIGH"
    elif snr_proxy > 3.0:
        res["confidence"] = "MEDIUM"
    else:
        res["confidence"] = "LOW"
        res["flags"].append("LOW_SPECTRAL_SNR")

    if sample_rate is not None and sample_rate > 0:
        res["relative_center_frequency"] = est_center
        res["occupied_bandwidth"] = est_bw
    else:
        res["relative_center_normalized"] = est_center
        res["occupied_bandwidth_normalized"] = est_bw
        res["flags"].append("METADATA_REQUIRED_FOR_ABSOLUTE_HERTZ")

    return res
