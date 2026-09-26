"""Module 9: Bitstream Structure, Frame Detection & Structural Evidence.

Scientific Status: IMPLEMENTED (Review Required before Lock).
Source: # Module 9 — Full Experimental Report.

Exports the core functions and services for:
- 9A Known Preamble Detection
- 9C/9C.1 Repeated Frame Detection & Period Search
- 9D/9D.1 Global Frame Phase Scoring & Frame Extraction
- 9E.2 Positional Variance & Regional Hypotheses
- 9E.4 Counter Candidate Localization
- 9E.5 Word / Byte Alignment Supporting Evidence
- 9F Exact & Noisy GF(2) Parity Relationship Discovery
- 9H Structural Evidence Map Assembly
- Module 9 Orchestration Service
"""

from .preamble import detect_known_preamble, DEFAULT_PREAMBLE_16
from .frame_detection import detect_frame_period
from .frame_phase import detect_frame_phase, extract_frames
from .structural_regions import analyze_structural_regions
from .counter import detect_counter_candidates
from .alignment import evaluate_boundary_alignment, compute_bit_alignments
from .gf2 import find_gf2_relationships, verify_gf2_relation, gf2_nullspace
from .evidence_map import build_structural_evidence_map
from .service import (
    BitstreamStructureService,
    default_structure_service,
    process_structure_analysis,
    REJECTED_METHODS_RECORD
)

__all__ = [
    "detect_known_preamble",
    "DEFAULT_PREAMBLE_16",
    "detect_frame_period",
    "detect_frame_phase",
    "extract_frames",
    "analyze_structural_regions",
    "detect_counter_candidates",
    "evaluate_boundary_alignment",
    "compute_bit_alignments",
    "find_gf2_relationships",
    "verify_gf2_relation",
    "gf2_nullspace",
    "build_structural_evidence_map",
    "BitstreamStructureService",
    "default_structure_service",
    "process_structure_analysis",
    "REJECTED_METHODS_RECORD"
]
