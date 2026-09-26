"""Module 9H: Structural Evidence Map Representation.

Scientific Status: LOCKED as evidence representation (9H).
Source: # Module 9 — Full Experimental Report, Section 20 (9H) and Section 21.

Architecture:
Maintains an independent record of structural evidence for EVERY bit position
in the detected frame period [0, P-1]:
    Bit position:
        ├── variance
        ├── static evidence
        ├── counter evidence (CONDITIONAL candidate only)
        ├── entropy evidence
        ├── alignment evidence
        ├── GF(2) evidence
        └── structural hypothesis

CRITICAL SCIENTIFIC RULES:
1. Do NOT invent a unified Module 9 confidence score.
2. Heterogeneous features MUST NOT be collapsed into one arbitrary scalar.
3. The unified score S(k) = gamma(pe)[0.25A + 0.35F + 0.25E + 0.15G] was REJECTED (Section 19).
4. Do NOT output forced protocol interpretations or forced header boundaries (e.g. "HEADER = 16–47").
5. Counter evidence is strictly a COUNTER_CANDIDATE.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def build_structural_evidence_map(
    frame_period: int,
    column_variances: List[float],
    column_entropies: List[float],
    column_means: List[float],
    counter_candidates: List[Dict[str, Any]],
    gf2_relations: List[Dict[str, Any]],
    preamble_length: int = 16
) -> Dict[str, Any]:
    """Construct the independent Structural Evidence Map across all bit positions in a frame.

    Args:
        frame_period: Frame period P.
        column_variances: List of per-column variances across frames.
        column_entropies: List of per-column binary entropies.
        column_means: List of per-column mean bit values.
        counter_candidates: List of detected candidate counter regions.
        gf2_relations: Discovered GF(2) parity equations.
        preamble_length: Length of nominal preamble region (default 16).

    Returns:
        Structured dictionary containing per-bit evidence records, summary statistics,
        and scientific diagnostics.
    """
    P = int(frame_period)
    records: List[Dict[str, Any]] = []

    # Map counter candidate presence per bit
    counter_map: Dict[int, List[Dict[str, Any]]] = {j: [] for j in range(P)}
    for cand in counter_candidates:
        s = cand.get("start_bit", 0)
        e = cand.get("end_bit", 0)
        for j in range(max(0, s), min(P, e + 1)):
            counter_map[j].append({
                "candidate_span": [s, e],
                "consistency": cand.get("consistency", 0.0),
                "bit_width": cand.get("bit_width", 0),
                "orientation": cand.get("orientation", "MSB_FIRST")
            })

    # Map GF(2) participation per bit (as target or source)
    gf2_map: Dict[int, List[Dict[str, Any]]] = {j: [] for j in range(P)}
    for rel in gf2_relations:
        t = rel.get("target_bit")
        srcs = rel.get("source_bits", [])
        if t is not None and 0 <= t < P:
            gf2_map[t].append({
                "role": "TARGET",
                "equation": rel.get("equation_str", ""),
                "is_exact": rel.get("is_exact", True),
                "error_rate": rel.get("error_rate", 0.0)
            })
        for s in srcs:
            if 0 <= s < P:
                gf2_map[s].append({
                    "role": "SOURCE",
                    "equation": rel.get("equation_str", ""),
                    "target_bit": t,
                    "is_exact": rel.get("is_exact", True)
                })

    for j in range(P):
        var_j = float(column_variances[j]) if j < len(column_variances) else 0.25
        ent_j = float(column_entropies[j]) if j < len(column_entropies) else 1.0
        mean_j = float(column_means[j]) if j < len(column_means) else 0.5

        # Static evidence: degree to which bit is constant across frames
        # In ideal constant bit, |2*mean - 1| == 1.0
        static_evidence = float(abs(2.0 * mean_j - 1.0))

        # Counter evidence: highest consistency among candidate counter windows spanning bit j
        cands_at_j = counter_map[j]
        counter_ev = float(max([c["consistency"] for c in cands_at_j])) if cands_at_j else 0.0

        # Word alignment properties
        alignment_ev = {
            "byte_aligned": (j % 8 == 0),
            "word16_aligned": (j % 16 == 0),
            "word32_aligned": (j % 32 == 0)
        }

        # GF(2) relations spanning this bit
        gf2_at_j = gf2_map[j]

        # Structural hypothesis (never forced protocol interpretation)
        if j < preamble_length and var_j < 0.08:
            hyp = "PREAMBLE_CANDIDATE"
        elif var_j < 0.06:
            hyp = "STATIC_FIELD_CANDIDATE"
        elif len(cands_at_j) > 0 and var_j < 0.22:
            hyp = "COUNTER_CANDIDATE_REGION"
        elif len(gf2_at_j) > 0 and var_j < 0.24:
            hyp = "PARITY_PROTECTED_CANDIDATE"
        elif var_j >= 0.22 and ent_j >= 0.90:
            hyp = "HIGH_ENTROPY_PAYLOAD_LIKE"
        else:
            hyp = "STRUCTURED_DYNAMIC_CANDIDATE"

        records.append({
            "bit_position": j,
            "mean": mean_j,
            "variance": var_j,
            "entropy": ent_j,
            "static_evidence": static_evidence,
            "counter_evidence": counter_ev,
            "alignment_evidence": alignment_ev,
            "gf2_evidence_count": len(gf2_at_j),
            "gf2_details": gf2_at_j,
            "structural_hypothesis": hyp
        })

    # Regional aggregate summaries (Section 21 table format)
    L_pre = min(preamble_length, P)
    preamble_recs = records[:L_pre]
    # Header candidate: e.g. bits after preamble up to 48 (or where payload starts)
    header_end = min(48, P)
    header_recs = records[L_pre:header_end]
    payload_recs = records[header_end:]

    def mean_of(field: str, recs: List[Dict[str, Any]]) -> float:
        return float(np.mean([r[field] for r in recs])) if recs else 0.0

    regional_summary = {
        "preamble_region": {
            "span": [0, L_pre - 1] if L_pre > 0 else [],
            "mean_variance": mean_of("variance", preamble_recs),
            "mean_entropy": mean_of("entropy", preamble_recs),
            "mean_static_evidence": mean_of("static_evidence", preamble_recs),
            "mean_counter_evidence": mean_of("counter_evidence", preamble_recs)
        },
        "header_candidate_region": {
            "span": [L_pre, header_end - 1] if header_end > L_pre else [],
            "mean_variance": mean_of("variance", header_recs),
            "mean_entropy": mean_of("entropy", header_recs),
            "mean_static_evidence": mean_of("static_evidence", header_recs),
            "mean_counter_evidence": mean_of("counter_evidence", header_recs)
        },
        "payload_candidate_region": {
            "span": [header_end, P - 1] if P > header_end else [],
            "mean_variance": mean_of("variance", payload_recs),
            "mean_entropy": mean_of("entropy", payload_recs),
            "mean_static_evidence": mean_of("static_evidence", payload_recs),
            "mean_counter_evidence": mean_of("counter_evidence", payload_recs)
        }
    }

    return {
        "status": "SUCCESS",
        "frame_period": P,
        "total_positions_mapped": len(records),
        "bit_evidence_records": records,
        "regional_summary": regional_summary,
        "scientific_status": "LOCKED as evidence representation (9H)",
        "safeguards": (
            "Evidence map preserves individual metrics independently. No unified confidence "
            "score is computed. Protocol identity is NOT claimed."
        )
    }
