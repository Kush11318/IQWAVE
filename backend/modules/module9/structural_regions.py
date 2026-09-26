"""Module 9E.2: Structural Region Evidence & Positional Statistics.

Scientific Status: CONDITIONAL (9E.2).
Source: # Module 9 — Full Experimental Report, Section 11 (9E.2), Section 12 (9E.3), Section 10 (9E.1).

Method:
Frame-to-Frame Positional Behavior Analysis.
Across M aligned frames of length P:
For each column position j in [0, P-1]:
    Mean bit value: mu_j = (1/M) * sum_{m=0}^{M-1} x_{m, j}
    Positional variance: sigma_j^2 = mu_j * (1 - mu_j)
    Binary entropy: H_j = -mu_j * log2(mu_j) - (1 - mu_j) * log2(1 - mu_j)

Experimental Regional Behaviors (from source):
    - Preamble: variance ~ 0 (0.0477 at 5% BER), entropy ~ 0.19.
    - Static header fields: variance ~ 0, entropy ~ 0.
    - Changing header fields (e.g., counters, flags): variance ~ 0.07–0.23, entropy ~ 0.76.
    - Payload: variance ~ 0.238–0.247, entropy ~ 0.98.

CRITICAL SCIENTIFIC CONSTRAINTS:
1. DO NOT output forced structural conclusions such as "HEADER = bits X-Y".
   The output must be structural hypotheses and positional statistics only.
2. DO NOT use variance-only thresholding to decide exact header boundaries.
   Section 12 (9E.3) experimentally REJECTED variance-only header boundary detection
   (exact accuracy 42.86%, MAE 3.29 bits).
3. Payload-balance detection (9E.1) was experimentally REJECTED (statistically indistinguishable).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def analyze_structural_regions(
    frames: np.ndarray,
    preamble_length: int = 16
) -> Dict[str, Any]:
    """Compute frame-to-frame positional statistics and regional hypotheses (9E.2).

    Args:
        frames: 2D binary numpy array of shape (M, P).
        preamble_length: Nominal preamble length (default 16 bits).

    Returns:
        Structured dictionary containing per-bit variance, entropy, means,
        and conservative regional hypotheses.
    """
    if not isinstance(frames, np.ndarray) or frames.ndim != 2:
        return {
            "status": "INVALID_INPUT",
            "frame_count": 0,
            "frame_period": 0,
            "column_means": [],
            "column_variances": [],
            "column_entropies": [],
            "regional_hypotheses": [],
            "scientific_status": "CONDITIONAL (9E.2: Positional Behavior)",
            "limitations": "Requires 2D matrix of shape (M, P)."
        }

    M, P = frames.shape
    if M < 2 or P < 1:
        return {
            "status": "INSUFFICIENT_FRAMES",
            "frame_count": int(M),
            "frame_period": int(P),
            "column_means": [],
            "column_variances": [],
            "column_entropies": [],
            "regional_hypotheses": [],
            "scientific_status": "CONDITIONAL (9E.2: Positional Behavior)",
            "limitations": "Need at least 2 frames to compute frame-to-frame variance."
        }

    # Positional statistics
    means = np.mean(frames, axis=0).astype(float)
    variances = (means * (1.0 - means)).astype(float)

    # Positional binary entropy in bits (with 0*log2(0) = 0 convention)
    entropies = np.zeros(P, dtype=float)
    eps = 1e-12
    p0 = np.clip(1.0 - means, eps, 1.0 - eps)
    p1 = np.clip(means, eps, 1.0 - eps)
    # Only compute entropy where neither p0 nor p1 is effectively 0 or 1
    mask = (means > 1e-6) & (means < 1.0 - 1e-6)
    entropies[mask] = -(means[mask] * np.log2(p1[mask]) + (1.0 - means[mask]) * np.log2(p0[mask]))

    # Identify regional candidate segments without forcing rigid header/payload decisions
    # We classify positional behavior into 3 observable categories:
    # 1. STATIC_CANDIDATE: variance < 0.06 (nearly invariant across frames, e.g. preamble/static fields)
    # 2. STRUCTURED_DYNAMIC_CANDIDATE: 0.06 <= variance < 0.22 (changing headers, counters, systematic bits)
    # 3. HIGH_ENTROPY_CANDIDATE: variance >= 0.22 (consistent with uncompressed/scrambled payload)
    position_classes = []
    for j in range(P):
        v = variances[j]
        if v < 0.06:
            pclass = "STATIC_CANDIDATE"
        elif v < 0.22:
            pclass = "STRUCTURED_DYNAMIC_CANDIDATE"
        else:
            pclass = "HIGH_ENTROPY_CANDIDATE"
        position_classes.append(pclass)

    # Segment contiguous spans for regional hypothesis reporting
    hypotheses = []
    if P > 0:
        curr_class = position_classes[0]
        start_idx = 0
        for j in range(1, P):
            if position_classes[j] != curr_class:
                hypotheses.append({
                    "start_bit": start_idx,
                    "end_bit": j - 1,
                    "length": j - start_idx,
                    "behavior_class": curr_class,
                    "mean_variance": float(np.mean(variances[start_idx:j])),
                    "mean_entropy": float(np.mean(entropies[start_idx:j])),
                    "interpretation": (
                        "Pre-payload static structure" if curr_class == "STATIC_CANDIDATE" and start_idx == 0
                        else "Dynamic header / structured candidate" if curr_class == "STRUCTURED_DYNAMIC_CANDIDATE"
                        else "High entropy / payload-like region"
                    )
                })
                curr_class = position_classes[j]
                start_idx = j
        hypotheses.append({
            "start_bit": start_idx,
            "end_bit": P - 1,
            "length": P - start_idx,
            "behavior_class": curr_class,
            "mean_variance": float(np.mean(variances[start_idx:P])),
            "mean_entropy": float(np.mean(entropies[start_idx:P])),
            "interpretation": (
                "Pre-payload static structure" if curr_class == "STATIC_CANDIDATE" and start_idx == 0
                else "Dynamic header / structured candidate" if curr_class == "STRUCTURED_DYNAMIC_CANDIDATE"
                else "High entropy / payload-like region"
            )
        })

    # Summary regional statistics (compatible with Section 21 report tables)
    L_pre = min(preamble_length, P)
    preamble_stats = {
        "region": "Preamble_Candidate",
        "span": [0, L_pre - 1],
        "mean_variance": float(np.mean(variances[:L_pre])) if L_pre > 0 else 0.0,
        "mean_entropy": float(np.mean(entropies[:L_pre])) if L_pre > 0 else 0.0
    }

    return {
        "status": "SUCCESS",
        "frame_count": int(M),
        "frame_period": int(P),
        "column_means": [float(m) for m in means],
        "column_variances": [float(v) for v in variances],
        "column_entropies": [float(h) for h in entropies],
        "position_classifications": position_classes,
        "regional_hypotheses": hypotheses,
        "preamble_region_stats": preamble_stats,
        "scientific_status": "CONDITIONAL (9E.2: Positional Behavior)",
        "safeguard_note": (
            "Regional hypotheses are structural evidence only. Exact header boundary is NOT forced. "
            "Variance-only blind header boundary detection (9E.3) was experimentally rejected."
        )
    }
