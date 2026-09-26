"""Module 9E.5: Word and Byte Alignment Supporting Evidence.

Scientific Status: CONDITIONAL / SUPPORTING EVIDENCE ONLY (9E.5).
Source: # Module 9 — Full Experimental Report, Section 15 (9E.5).

Method:
Alignment Evaluation for Candidate Structural Boundaries.
Evaluates candidate boundary offsets k against common architectural word boundaries:
    - 8-bit (byte alignment): k mod 8 == 0
    - 16-bit (half-word alignment): k mod 16 == 0
    - 32-bit (word alignment): k mod 32 == 0

Controlled Experiment Result (Section 15):
    True boundary at bit 48 (16 preamble + 32 header).
    - 8-bit alignment best boundary: 40
    - 16-bit alignment best boundary: 48
    - 32-bit alignment best boundary: 48

CRITICAL SCIENTIFIC CONSTRAINT:
    Alignment is SUPPORTING EVIDENCE ONLY.
    It MUST NOT independently determine the frame structure, header length, or boundary.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


SUPPORTED_ALIGNMENTS = [8, 16, 32]


def evaluate_boundary_alignment(
    boundary_bit: int,
    word_sizes: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Evaluate whether a candidate structural bit boundary satisfies word/byte alignment.

    Args:
        boundary_bit: Bit index of candidate boundary.
        word_sizes: List of word sizes to evaluate (default [8, 16, 32]).

    Returns:
        Structured dictionary indicating alignment status for each word size.
    """
    k = int(boundary_bit)
    sizes = word_sizes if word_sizes is not None else SUPPORTED_ALIGNMENTS

    alignment_results = {}
    aligned_any = False
    for sz in sizes:
        is_aligned = (k % sz == 0)
        remainder = k % sz
        alignment_results[f"{sz}_bit"] = {
            "aligned": is_aligned,
            "word_size": sz,
            "remainder": remainder,
            "words_count": k // sz if is_aligned else None
        }
        if is_aligned:
            aligned_any = True

    return {
        "boundary_bit": k,
        "is_aligned_any": aligned_any,
        "alignments": alignment_results,
        "scientific_status": "CONDITIONAL (9E.5: Supporting Alignment Evidence Only)",
        "safeguard_note": (
            "Alignment is supporting evidence only. It must NOT independently force "
            "or decide frame structure or header boundaries."
        )
    }


def compute_bit_alignments(frame_period: int) -> List[Dict[str, Any]]:
    """Compute word/byte alignment properties for all bit positions in a frame period.

    Args:
        frame_period: Frame period P.

    Returns:
        List of dictionaries with alignment flags for each bit position j in [0, P-1].
    """
    P = int(frame_period)
    records = []
    for j in range(P):
        records.append({
            "bit_position": j,
            "byte_aligned": (j % 8 == 0),
            "word16_aligned": (j % 16 == 0),
            "word32_aligned": (j % 32 == 0)
        })
    return records
