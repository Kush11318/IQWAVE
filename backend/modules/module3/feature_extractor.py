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

NEW_FEATURE_NAMES_8: List[str] = [
    "c60_norm",
    "c63_norm",
    "circular_skewness",
    "circular_kurtosis",
    "peak_to_avg_power_ratio",
    "spectral_flatness",
    "cyclic_freq_peak_ratio",
    "norm_phase_variance"
]

FEATURE_NAMES_32: List[str] = FEATURE_NAMES_24 + NEW_FEATURE_NAMES_8
FEATURE_NAMES: List[str] = FEATURE_NAMES_32

MODULATION_CLASSES: List[str] = [
    "BPSK",
    "QPSK",
    "8PSK",
    "QAM16",
    "QAM64",
    "GFSK",
    "CPFSK"
]


def extract_features(
    iq_signal: np.ndarray,
    module2_observation: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[float]]:
    """Extract the complete 32-dimensional research-backed feature vector.

    Reuses Module 2 observations if already computed to avoid redundant DSP.
    Includes the 24 base features plus 8 discriminators:
    - c60_norm, c63_norm (6th-order cumulants for QAM64 vs QAM16 / PSK vs QAM)
    - circular_skewness, circular_kurtosis (phase distribution for FSK / PSK)
    - peak_to_avg_power_ratio (PAPR)
    - spectral_flatness (Wiener entropy)
    - cyclic_freq_peak_ratio (cyclostationary symbol rate peak strength)
    - norm_phase_variance (CFO-robust phase spread)
    """
    res: Dict[str, Optional[float]] = {name: None for name in FEATURE_NAMES_32}

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
    phases = np.angle(iq_signal)
    for m in (2, 4, 8):
        m_phasor_raw = float(np.abs(np.mean(np.exp(1j * m * phases))))
        best_conc = m_phasor_raw
        if n >= 32:
            try:
                ym = (iq_signal.astype(np.complex64)) ** m
                n_fft = 2 ** int(np.ceil(np.log2(min(n, 1024))))
                fft_ym = np.abs(np.fft.fft(ym, n=n_fft))
                k = int(np.argmax(fft_ym[1:n_fft // 2])) + 1
                f_cfo = float(k / (n_fft * m))
                t = np.arange(n)
                derot_phases = phases - 2.0 * np.pi * f_cfo * t
                derot_conc = float(np.abs(np.mean(np.exp(1j * m * derot_phases))))
                if derot_conc > best_conc:
                    best_conc = derot_conc
            except Exception:
                pass
        res[f"phase_m{m}_concentration"] = best_conc

    # 3. Phase entropy (14)
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
    i_samples = iq_signal.real.astype(np.float64)
    q_samples = iq_signal.imag.astype(np.float64)
    cov_matrix = np.cov(i_samples, q_samples)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))
    if eigenvalues[1] > 1e-12:
        res["iq_eigen_ratio"] = float(np.clip(eigenvalues[0] / eigenvalues[1], 0.0, 1.0))
    else:
        res["iq_eigen_ratio"] = 0.0

    # 7. Fourth-order & Sixth-order normalized cumulants (23 to 26)
    p_signal = float(np.mean(amp ** 2))
    if p_signal > 1e-12:
        x_d = iq_signal.astype(np.complex128)
        m20 = np.mean(x_d ** 2)
        m40 = np.mean(x_d ** 4)
        m42 = np.mean((np.abs(x_d) ** 2) * (x_d ** 2))
        m60 = np.mean(x_d ** 6)
        m63 = np.mean(amp ** 6)

        # C40 = Cum(x, x, x, x) = M40 - 3*(M20^2)
        c40 = m40 - 3.0 * (m20 ** 2)
        c40_val = float(np.abs(c40) / (p_signal ** 2))

        # C60 = Cum(x, x, x, x, x, x) = M60 - 15*M40*M20 + 30*(M20^3)
        c60 = m60 - 15.0 * m40 * m20 + 30.0 * (m20 ** 3)
        c60_val = float(np.abs(c60) / (p_signal ** 3))

        if n >= 32:
            try:
                # CFO-compensated estimation
                y4 = x_d ** 4
                n_fft = 2 ** int(np.ceil(np.log2(min(n, 1024))))
                fft_y4 = np.abs(np.fft.fft(y4, n=n_fft))
                k4 = int(np.argmax(fft_y4[1:n_fft // 2])) + 1
                f_cfo4 = float(k4 / (n_fft * 4.0))
                t = np.arange(n)
                x_derot = x_d * np.exp(-1j * 2.0 * np.pi * f_cfo4 * t)
                m20_d = np.mean(x_derot ** 2)
                m40_d = np.mean(x_derot ** 4)
                m60_d = np.mean(x_derot ** 6)

                c40_d = float(np.abs(m40_d - 3.0 * (m20_d ** 2)) / (p_signal ** 2))
                if c40_d > c40_val:
                    c40_val = c40_d

                c60_d = float(np.abs(m60_d - 15.0 * m40_d * m20_d + 30.0 * (m20_d ** 3)) / (p_signal ** 3))
                if c60_d > c60_val:
                    c60_val = c60_d
            except Exception:
                pass

        res["c40_norm"] = c40_val
        res["c60_norm"] = c60_val

        # C42 = Cum(x, x, x*, x*) = E[|x|^4] - |M20|^2 - 2*(M21^2)
        e_amp4 = np.mean(amp ** 4)
        c42 = e_amp4 - (np.abs(m20) ** 2) - 2.0 * (p_signal ** 2)
        res["c42_norm"] = float(np.abs(c42) / (p_signal ** 2))

        # C63 = Cum(x, x, x, x*, x*, x*) = M63 - 9*M42*p + 12*p^3 - 3*|M20|^2*p
        c63 = m63 - 9.0 * e_amp4 * p_signal + 12.0 * (p_signal ** 3) - 3.0 * (np.abs(m20) ** 2) * p_signal
        res["c63_norm"] = float(np.abs(c63) / (p_signal ** 3))

        # Peak-to-Average Power Ratio (PAPR)
        res["peak_to_avg_power_ratio"] = float(np.max(amp ** 2) / p_signal)
    else:
        res["c40_norm"] = None
        res["c42_norm"] = None
        res["c60_norm"] = None
        res["c63_norm"] = None
        res["peak_to_avg_power_ratio"] = None

    # 8. Circular skewness and kurtosis (27 to 28)
    mean_sin = np.mean(np.sin(phases))
    mean_cos = np.mean(np.cos(phases))
    circ_mean = np.arctan2(mean_sin, mean_cos)
    centered_phases = phases - circ_mean
    res["circular_skewness"] = float(np.mean(np.sin(2.0 * centered_phases)))
    res["circular_kurtosis"] = float(np.mean(np.cos(2.0 * centered_phases)))

    # 9. Spectral flatness (Wiener entropy) (29)
    try:
        fft_mag2 = np.abs(np.fft.fft(iq_signal)) ** 2
        psd_norm = fft_mag2 / (np.mean(fft_mag2) + 1e-12)
        geom_mean = np.exp(np.mean(np.log(psd_norm + 1e-12)))
        arith_mean = np.mean(psd_norm) + 1e-12
        res["spectral_flatness"] = float(np.clip(geom_mean / arith_mean, 0.0, 1.0))
    except Exception:
        res["spectral_flatness"] = 0.5

    # 10. Cyclic frequency peak ratio (30)
    try:
        y_env = (amp ** 2) - p_signal
        n_fft = 2 ** int(np.ceil(np.log2(min(n, 1024))))
        fft_env = np.abs(np.fft.fft(y_env, n=n_fft))
        half_spec = fft_env[1:n_fft // 2]
        mean_spec = np.mean(half_spec)
        if mean_spec > 1e-12:
            res["cyclic_freq_peak_ratio"] = float(np.max(half_spec) / mean_spec)
        else:
            res["cyclic_freq_peak_ratio"] = 1.0
    except Exception:
        res["cyclic_freq_peak_ratio"] = 1.0

    # 11. Normalized phase variance (31)
    try:
        delta_phase = np.angle(iq_signal[1:] * np.conj(iq_signal[:-1]))
        var_delta = np.var(delta_phase)
        uniform_var = (np.pi ** 2) / 3.0  # ~3.289868
        res["norm_phase_variance"] = float(np.clip(var_delta / uniform_var, 0.0, 1.0))
    except Exception:
        res["norm_phase_variance"] = 0.5

    return res


def extract_24_features(
    iq_signal: np.ndarray,
    module2_observation: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[float]]:
    """Extract the 24 base engineered features (backwards-compatible)."""
    full_dict = extract_features(iq_signal, module2_observation)
    return {name: full_dict.get(name) for name in FEATURE_NAMES_24}


def feature_dict_to_vector(
    feat_dict: Dict[str, Optional[float]],
    feature_names: Optional[List[str]] = None
) -> np.ndarray:
    """Convert feature dictionary to ordered numpy 1D vector.

    If feature_names is not specified:
    - If all 32 features exist in feat_dict, returns vector of length 32.
    - Otherwise, returns vector of length 24 (backwards-compatible).
    """
    if feature_names is None:
        if all(k in feat_dict for k in FEATURE_NAMES_32):
            feature_names = FEATURE_NAMES_32
        else:
            feature_names = FEATURE_NAMES_24

    vec = np.zeros(len(feature_names), dtype=np.float32)
    for idx, name in enumerate(feature_names):
        val = feat_dict.get(name)
        vec[idx] = float(val) if val is not None else 0.0
    return vec
