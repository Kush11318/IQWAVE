"""Module 3: 24-Feature AMC Engineered Feature Extractor.

Extracts the exact 24 validated engineered features for Engine A (Random Forest).
Reuses the 10 non-destructive features from Module 2 and adds the 14 structural
features identified in the Module 3 experimental expansion:

Original 10 (from Module 2):
1. amp_mean
2. amp_std
3. amp_cv
4. amp_skewness
5. amp_kurtosis
6. phase_diff_std
7. gated_phase_diff_std
8. circular_variance
9. circularity_ratio
10. zero_bin_fraction

Additional 14:
11. phase_m2_concentration
12. phase_m4_concentration
13. phase_m8_concentration
14. phase_entropy
15. amp_p10
16. amp_p25
17. amp_p50
18. amp_p75
19. amp_p90
20. radial_iqr
21. radial_entropy
22. iq_eigen_ratio
23. c40_norm
24. c42_norm

Strictly non-destructive: operates on the input signal without permanent alteration.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from backend.modules.module2.observation import compute_observation_vector

FEATURE_NAMES_24: List[str] = [
    "amp_mean",
    "amp_std",
    "amp_cv",
    "amp_skewness",
    "amp_kurtosis",
    "phase_diff_std",
    "gated_phase_diff_std",
    "circular_variance",
    "circularity_ratio",
    "zero_bin_fraction",
    "phase_m2_concentration",
    "phase_m4_concentration",
    "phase_m8_concentration",
    "phase_entropy",
    "amp_p10",
    "amp_p25",
    "amp_p50",
    "amp_p75",
    "amp_p90",
    "radial_iqr",
    "radial_entropy",
    "iq_eigen_ratio",
    "c40_norm",
    "c42_norm"
]

MODULATION_CLASSES: List[str] = [
    "BPSK",
    "QPSK",
    "8PSK",
    "QAM16",
    "QAM64",
    "GFSK",
    "CPFSK"
]


def extract_24_features(
    iq_signal: np.ndarray,
    module2_observation: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[float]]:
    """Extract the complete 24-dimensional feature vector.

    Reuses Module 2 observations if already computed to avoid redundant DSP.
    """
    res: Dict[str, Optional[float]] = {name: None for name in FEATURE_NAMES_24}

    if iq_signal is None or len(iq_signal) == 0:
        return res

    if not np.all(np.isfinite(iq_signal)):
        return res

    n = len(iq_signal)
    if n < 4:
        return res

    # 1. Reuse or compute Module 2 retained features (1 to 10)
    if module2_observation is None or "amplitude" not in module2_observation:
        obs = compute_observation_vector(iq_signal, {"input_representation": "COMPLEX_IQ"})
    else:
        obs = module2_observation

    amp_obs = obs.get("amplitude", {})
    phase_obs = obs.get("phase", {})
    circ_obs = obs.get("circular", {})
    spec_obs = obs.get("spectral", {})

    res["amp_mean"] = amp_obs.get("mean")
    res["amp_std"] = amp_obs.get("std")
    res["amp_cv"] = amp_obs.get("cv")
    res["amp_skewness"] = amp_obs.get("skewness")
    res["amp_kurtosis"] = amp_obs.get("kurtosis")
    res["phase_diff_std"] = phase_obs.get("diff_std")
    res["gated_phase_diff_std"] = phase_obs.get("gated_diff_std")
    res["circular_variance"] = circ_obs.get("variance")
    res["circularity_ratio"] = circ_obs.get("m20_m21")
    res["zero_bin_fraction"] = spec_obs.get("zero_bin_fraction")

    # 2. Phase concentrations (11 to 13)
    # Evaluated via M-th power unit phasors: |E[exp(j * M * phi)]|
    phases = np.angle(iq_signal)
    for m in (2, 4, 8):
        m_phasor = np.mean(np.exp(1j * m * phases))
        res[f"phase_m{m}_concentration"] = float(np.abs(m_phasor))

    # 3. Phase entropy (14)
    # Normalized Shannon entropy over 32 bins across [-pi, pi)
    counts, _ = np.histogram(phases, bins=32, range=(-np.pi, np.pi))
    probs = counts / np.sum(counts)
    nonzero_probs = probs[probs > 0]
    raw_entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
    max_entropy = np.log2(32)  # 5.0
    res["phase_entropy"] = float(np.clip(raw_entropy / max_entropy, 0.0, 1.0))

    # 4. Amplitude percentiles (15 to 19)
    amp = np.abs(iq_signal).astype(np.float64)
    p10, p25, p50, p75, p90 = np.percentile(amp, [10, 25, 50, 75, 90])
    res["amp_p10"] = float(p10)
    res["amp_p25"] = float(p25)
    res["amp_p50"] = float(p50)
    res["amp_p75"] = float(p75)
    res["amp_p90"] = float(p90)

    # 5. Radial IQR and radial entropy (20 to 21)
    res["radial_iqr"] = float(p75 - p25)

    amp_min, amp_max = np.min(amp), np.max(amp)
    if amp_max - amp_min > 1e-12:
        r_counts, _ = np.histogram(amp, bins=32, range=(amp_min, amp_max))
        r_probs = r_counts / np.sum(r_counts)
        r_nonzero = r_probs[r_probs > 0]
        r_entropy = -np.sum(r_nonzero * np.log2(r_nonzero))
        res["radial_entropy"] = float(np.clip(r_entropy / max_entropy, 0.0, 1.0))
    else:
        res["radial_entropy"] = 0.0

    # 6. IQ Eigenvalue ratio (22)
    # Ratio of smallest to largest eigenvalue of the I/Q covariance matrix
    i_samples = iq_signal.real.astype(np.float64)
    q_samples = iq_signal.imag.astype(np.float64)
    cov_matrix = np.cov(i_samples, q_samples)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))
    if eigenvalues[1] > 1e-12:
        res["iq_eigen_ratio"] = float(np.clip(eigenvalues[0] / eigenvalues[1], 0.0, 1.0))
    else:
        res["iq_eigen_ratio"] = 0.0

    # 7. Fourth-order normalized cumulants c40_norm and c42_norm (23 to 24)
    p_signal = float(np.mean(amp ** 2))
    if p_signal > 1e-12:
        x_d = iq_signal.astype(np.complex128)
        m20 = np.mean(x_d ** 2)
        m40 = np.mean(x_d ** 4)
        m42 = np.mean((np.abs(x_d) ** 2) * (x_d ** 2))

        # C40 = Cum(x, x, x, x) = M40 - 3*(M20^2)
        c40 = m40 - 3.0 * (m20 ** 2)
        res["c40_norm"] = float(np.abs(c40) / (p_signal ** 2))

        # C42 = Cum(x, x, x*, x*) = E[|x|^4] - |M20|^2 - 2*(M21^2)
        # Note: E[|x|^4] = mean(amp^4)
        e_amp4 = np.mean(amp ** 4)
        c42 = e_amp4 - (np.abs(m20) ** 2) - 2.0 * (p_signal ** 2)
        res["c42_norm"] = float(np.abs(c42) / (p_signal ** 2))
    else:
        res["c40_norm"] = None
        res["c42_norm"] = None

    return res


def feature_dict_to_vector(feat_dict: Dict[str, Optional[float]]) -> np.ndarray:
    """Convert feature dictionary to ordered numpy 1D vector (length 24).

    Replaces None with 0.0 for model input formatting, while preserving
    the raw dict with None for statistical inspection.
    """
    vec = np.zeros(len(FEATURE_NAMES_24), dtype=np.float32)
    for idx, name in enumerate(FEATURE_NAMES_24):
        val = feat_dict.get(name)
        vec[idx] = float(val) if val is not None else 0.0
    return vec
