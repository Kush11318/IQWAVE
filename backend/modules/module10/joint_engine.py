"""Module 10 Joint FEC + Interleaver Hypothesis Engine.

Scientific Status: 🔒 LOCKED (Controlled Validation, Section 12).
Source: # Module 10 — FEC, CRC & Interleaver Analysis, Section 12.

Method:
Evaluates combinations of candidate FEC families and candidate interleaver widths.
For candidate (FEC, Width):
    1. Deinterleave using candidate width W.
    2. Evaluate candidate FEC structural consistency.
    3. Rank joint hypotheses by native FEC structural score.

Monte Carlo Benchmark (Section 12):
    Hamming(7,4) + width 20 vs. alternative widths:
    - 0% BER: Top-1 = 1.000, Mean Rank = 1.00, Margin = 0.8079
    - 1% BER: Top-1 = 1.000, Mean Rank = 1.00, Margin = 0.7455
    - 5% BER: Top-1 = 1.000, Mean Rank = 1.00, Margin = 0.5337
    - 10% BER: Top-1 = 1.000, Mean Rank = 1.00, Margin = 0.3244

CRITICAL SCIENTIFIC RULES:
1. Different code families use their NATIVE mathematical evidence (no invented universal scalar score).
2. Scores and margins are reported natively; NO arbitrary confidence percentage is fabricated.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .interleaver import deinterleave_row_column
from .hamming import evaluate_hamming_evidence
from .bch import evaluate_bch_evidence
from .ldpc import evaluate_ldpc_syndrome


VALIDATED_INTERLEAVER_CANDIDATE_WIDTHS: List[int] = [5, 10, 20, 25, 50]


def evaluate_joint_hypotheses(
    rx_bits: Union[np.ndarray, List[int]],
    candidate_widths: Optional[List[int]] = None,
    fec_family: str = "HAMMING"
) -> Dict[str, Any]:
    """Jointly evaluate candidate interleaver widths under a given candidate FEC family.

    The scientifically validated candidate set is strictly {5, 10, 20, 25, 50}.
    Width 1 represents an identity (no-interleaving) baseline and is never ranked as an interleaver candidate.

    Args:
        rx_bits: 1D array of binary bits.
        candidate_widths: List of candidate widths to test (default: [5, 10, 20, 25, 50]).
        fec_family: FEC candidate family ("HAMMING", "BCH", "LDPC").

    Returns:
        Structured dictionary containing ranked joint hypotheses, margins, and status.
    """
    rx = np.asarray(rx_bits, dtype=np.uint8).flatten()
    raw_cands = candidate_widths if candidate_widths is not None else VALIDATED_INTERLEAVER_CANDIDATE_WIDTHS
    # Strictly filter candidate widths > 1 (width=1 cannot be a scientific candidate)
    cands = [int(w) for w in raw_cands if int(w) > 1]

    identity_baseline: Optional[Dict[str, Any]] = None
    if candidate_widths is not None and 1 in [int(w) for w in candidate_widths]:
        if fec_family.upper() == "HAMMING":
            ev1 = evaluate_hamming_evidence(rx)
            score1 = float(1.0 - ev1["zero_syndrome_fraction"])
            lift1 = float(ev1["normalized_lift"])
            metric_name1 = "non_zero_syndrome_rate"
        elif fec_family.upper() == "BCH":
            ev1 = evaluate_bch_evidence(rx)
            score1 = float(1.0 - ev1["zero_syndrome_fraction"])
            lift1 = float(ev1["normalized_lift"])
            metric_name1 = "non_zero_syndrome_rate"
        elif fec_family.upper() == "LDPC":
            ev1 = evaluate_ldpc_syndrome(rx)
            score1 = float(ev1["mean_syndrome_weight"])
            lift1 = float(ev1["zero_syndrome_fraction"])
            metric_name1 = "mean_syndrome_weight"
        else:
            ev1 = evaluate_hamming_evidence(rx)
            score1 = float(1.0 - ev1["zero_syndrome_fraction"])
            lift1 = float(ev1["normalized_lift"])
            metric_name1 = "non_zero_syndrome_rate"

        identity_baseline = {
            "description": "IDENTITY / NO-INTERLEAVING BASELINE",
            "fec_family": fec_family.upper(),
            "width": 1,
            "score": score1,
            "metric_name": metric_name1,
            "lift": lift1,
            "is_scientific_candidate": False
        }

    ranked_hypotheses: List[Dict[str, Any]] = []

    for w in cands:
        deint = deinterleave_row_column(rx, width=w)

        if fec_family.upper() == "HAMMING":
            ev = evaluate_hamming_evidence(deint)
            # Primary score: non-zero syndrome fraction (lower is better consistency)
            score = float(1.0 - ev["zero_syndrome_fraction"])
            lift = float(ev["normalized_lift"])
            metric_name = "non_zero_syndrome_rate"
        elif fec_family.upper() == "BCH":
            ev = evaluate_bch_evidence(deint)
            score = float(1.0 - ev["zero_syndrome_fraction"])
            lift = float(ev["normalized_lift"])
            metric_name = "non_zero_syndrome_rate"
        elif fec_family.upper() == "LDPC":
            ev = evaluate_ldpc_syndrome(deint)
            score = float(ev["mean_syndrome_weight"])
            lift = float(ev["zero_syndrome_fraction"])
            metric_name = "mean_syndrome_weight"
        else:
            ev = evaluate_hamming_evidence(deint)
            score = float(1.0 - ev["zero_syndrome_fraction"])
            lift = float(ev["normalized_lift"])
            metric_name = "non_zero_syndrome_rate"

        ranked_hypotheses.append({
            "fec_family": fec_family.upper(),
            "interleaver_width": int(w),
            "score": score,
            "metric_name": metric_name,
            "lift": lift
        })

    # Sort ascending by score (lower error/syndrome is better)
    ranked_hypotheses.sort(key=lambda x: x["score"])

    # Assign ranks
    for rank_idx, hyp in enumerate(ranked_hypotheses):
        hyp["rank"] = rank_idx + 1

    best = ranked_hypotheses[0] if ranked_hypotheses else None
    second = ranked_hypotheses[1] if len(ranked_hypotheses) > 1 else None
    margin = float(second["score"] - best["score"]) if (best and second) else 0.0

    ret: Dict[str, Any] = {
        "status": "SUCCESS",
        "best_hypothesis": best,
        "margin": margin,
        "is_rank_1": bool(best and best.get("rank") == 1),
        "hypotheses": ranked_hypotheses,
        "scientific_status": "LOCKED for controlled Hamming+width-20 setup (Section 12)",
        "safeguards": (
            "Native separate evidence metrics; no universal scalar score is computed. "
            "Scientifically validated candidate set is strictly {5, 10, 20, 25, 50}. "
            "Width 1 is an identity / no-interleaving baseline and does not participate in candidate ranking."
        )
    }

    if identity_baseline is not None:
        ret["identity_baseline"] = identity_baseline

    return ret

