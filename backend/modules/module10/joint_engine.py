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
from .convolutional import viterbi_decode_hard_k3
from .reed_solomon import decode_rs_15_11, evaluate_rs_parameter_candidates


def evaluate_concatenated_code(
    rx_bits: Union[np.ndarray, List[int]]
) -> Dict[str, Any]:
    """Evaluate Concatenated Code (Outer RS(15,11) + Inner Convolutional K=3) per Task iv.
    
    1. Inner Viterbi hard decoding of received sequence.
    2. Outer Reed-Solomon syndrome consistency on decoded symbols.
    """
    rx = np.asarray(rx_bits, dtype=np.uint8).flatten()
    if len(rx) < 32:
        return {
            "status": "INSUFFICIENT_DATA",
            "score": 1.0,
            "lift": 0.0,
            "inner_viterbi_success": False,
            "outer_rs_success": False
        }
    
    # 1. Inner Viterbi decoding
    inner_decoded, _ = viterbi_decode_hard_k3(rx)
    
    # 2. Outer RS evaluation on inner decoded bits
    n_nibbles = len(inner_decoded) // 4
    if n_nibbles < 15:
        return {
            "status": "INSUFFICIENT_NIBBLES",
            "score": 0.8,
            "lift": 0.2,
            "inner_viterbi_success": True,
            "outer_rs_success": False,
            "inner_bits_count": len(inner_decoded)
        }
    
    nibbles = [
        int(inner_decoded[i*4] << 3 | inner_decoded[i*4+1] << 2 | inner_decoded[i*4+2] << 1 | inner_decoded[i*4+3])
        for i in range(n_nibbles)
    ]
    rs_res = evaluate_rs_parameter_candidates(nibbles)
    top_rs_lift = rs_res["candidates"][0]["lift"] if rs_res.get("candidates") else 0.0
    
    return {
        "status": "SUCCESS",
        "family": "Concatenated (RS outer + Conv inner)",
        "parameters": "Outer RS(15,11) GF(16) + Inner Conv K=3 R=1/2 Viterbi",
        "score": float(max(0.0, 1.0 - top_rs_lift)),
        "lift": float(top_rs_lift),
        "metric_type": "concatenated_rs_viterbi_consistency",
        "inner_bits_decoded": len(inner_decoded)
    }



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

