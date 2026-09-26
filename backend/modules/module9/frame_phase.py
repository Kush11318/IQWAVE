"""Module 9D: Frame Phase & Frame Boundary Determination.

Scientific Status: LOCKED under the tested conditions (9D.1).
Source: # Module 9 — Full Experimental Report, Section 7 (9D) and Section 8 (9D.1).

Method:
Global Frame-Phase Scoring.
Evaluates candidate phase offsets phi in {0, 1, ..., P-1} against detected frame periodicity P.
For each candidate offset phi:
    Partitions the bitstream into M = floor((N - phi) / P) frames.
    Evaluates global consistency score S(phi) combining:
    1. Known preamble alignment across all M frames (if preamble provided).
    2. Column agreement / polar consistency across all M frames:
       mean_j = (1/M) * sum_m x_{phi + m*P + j}
       column_polar_agreement = (1/P) * sum_j |2 * mean_j - 1|

The offset phi* maximizing the global score is selected as the frame phase.

Validated Conditions:
    BER: 0%, 1%, 5%, 10%, 20%.
    Frame-boundary accuracy: 100% across all tested conditions in the controlled experiment.

Important Limitation:
    Do not generalize the 100% controlled result into a universal guarantee across all
    channels, frame formats, or variable payloads.

Explicitly Rejected Methods (Section 7):
    - Initial residue-based phase estimation: REJECTED / FAILED under noise.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .preamble import DEFAULT_PREAMBLE_16


def detect_frame_phase(
    bits: Union[np.ndarray, List[int]],
    frame_period: int,
    preamble: Optional[Union[np.ndarray, List[int]]] = None
) -> Dict[str, Any]:
    """Determine frame start phase phi* via global frame-phase scoring (9D.1).

    Args:
        bits: 1D array of binary bits.
        frame_period: Detected frame period P (in bits).
        preamble: Optional known preamble bit pattern. Defaults to standard 16-bit sequence.

    Returns:
        Structured dictionary containing best phase offset phi*, phase scores,
        number of aligned frames M, and scientific status.
    """
    rx = np.asarray(bits, dtype=np.uint8).flatten()
    p = np.asarray(preamble if preamble is not None else DEFAULT_PREAMBLE_16, dtype=np.uint8).flatten()

    P = int(frame_period)
    N = len(rx)
    L = len(p)

    if P <= 0 or N < P:
        return {
            "status": "INSUFFICIENT_DATA",
            "frame_period": P,
            "frame_phase": None,
            "best_score": None,
            "aligned_frames_count": 0,
            "phase_scores": [],
            "scientific_status": "LOCKED under tested conditions (9D.1: Global Phase Scoring)",
            "limitations": "Bitstream length is smaller than one frame period."
        }

    # Evaluate all candidate offsets phi in [0, P - 1]
    candidate_scores = []
    best_phase = 0
    best_score = -1.0
    best_m_count = 0

    for phi in range(P):
        M = (N - phi) // P
        if M < 1:
            continue

        # Extract M frames of length P
        frames = rx[phi : phi + M * P].reshape(M, P)

        # 1. Preamble alignment score across all frames
        if L > 0 and L <= P:
            preamble_corr_per_frame = []
            for m in range(M):
                # Distance at position 0 within the frame
                dist = int(np.sum(p ^ frames[m, :L]))
                corr = 1.0 - (dist / L)
                preamble_corr_per_frame.append(corr)
            s_preamble = float(np.mean(preamble_corr_per_frame))
        else:
            s_preamble = 0.0

        # 2. Column polar agreement across all frames
        # |2 * p_j - 1|: 1.0 for constant bit 0 or 1, 0.0 for unbiased random bit
        col_means = np.mean(frames, axis=0)
        col_polar = np.abs(2.0 * col_means - 1.0)
        s_column = float(np.mean(col_polar))

        # Composite score: if preamble is available, weight preamble alignment heavily
        if L > 0 and L <= P:
            score = 0.7 * s_preamble + 0.3 * s_column
        else:
            score = s_column

        candidate_scores.append({
            "phase": phi,
            "global_score": float(score),
            "preamble_alignment": float(s_preamble),
            "column_agreement": float(s_column),
            "frame_count": int(M)
        })

        if score > best_score:
            best_score = score
            best_phase = phi
            best_m_count = M

    return {
        "status": "SUCCESS",
        "frame_period": P,
        "frame_phase": best_phase,
        "best_score": float(best_score),
        "aligned_frames_count": best_m_count,
        "phase_scores": candidate_scores,
        "scientific_status": "LOCKED under tested conditions (9D.1: Global Phase Scoring)",
        "limitations": "Tested on controlled repeated frames; 100% accuracy in controlled setup is not a universal guarantee."
    }


def extract_frames(
    bits: Union[np.ndarray, List[int]],
    frame_period: int,
    frame_phase: int
) -> np.ndarray:
    """Slice bitstream into a 2D matrix of shape (M, P) using detected phase and period.

    Args:
        bits: 1D array of binary bits.
        frame_period: Period P.
        frame_phase: Phase offset phi*.

    Returns:
        2D numpy array of shape (M, P) with binary values {0, 1}.
    """
    rx = np.asarray(bits, dtype=np.uint8).flatten()
    P = int(frame_period)
    phi = int(frame_phase)
    N = len(rx)

    if P <= 0 or phi < 0 or phi >= N:
        return np.empty((0, 0), dtype=np.uint8)

    M = (N - phi) // P
    if M < 1:
        return np.empty((0, 0), dtype=np.uint8)

    return rx[phi : phi + M * P].reshape(M, P)
