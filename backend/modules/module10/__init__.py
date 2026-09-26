"""Module 10: FEC, CRC & Interleaver Analysis.

Scientific Status: IMPLEMENTED (Review Required before Lock).
Source: # Module 10 — FEC, CRC & Interleaver Analysis.

Exports:
- 10A: CRC Analysis (compute_crc, verify_crc16_ccitt, search_crc_boundaries, identify_crc_family, compute_soft_crc_evidence)
- 10B: Hamming(7,4) (encode_hamming_7_4, decode_hamming_7_4, evaluate_hamming_evidence)
- 10B: BCH(15,7) (encode_bch_15_7, decode_bch_15_7, evaluate_bch_evidence)
- 10B: Reed-Solomon over GF(16) (encode_rs_15_11, decode_rs_15_11, evaluate_rs_parameter_candidates)
- 10C: LDPC (6,12) (evaluate_ldpc_syndrome, rank_ldpc_candidates)
- 10C: Convolutional (encode_convolutional_k3, viterbi_decode_hard_k3, identify_convolutional_generator)
- 10D: Interleaver (interleave_row_column, deinterleave_row_column, identify_interleaver_width)
- Joint Hypothesis Engine (evaluate_joint_hypotheses)
- Orchestration Service (FecCrcInterleaverService, default_fec_crc_service, process_fec_crc_analysis)
- Rejection Safeguards (MODULE10_REJECTED_METHODS)
"""

from .crc import (
    compute_crc,
    verify_crc16_ccitt,
    append_crc16_ccitt,
    evaluate_crc_acceptance_rate,
    search_crc_boundaries,
    identify_crc_family,
    compute_soft_crc_evidence,
    CRC_CANDIDATE_CATALOG
)
from .hamming import (
    HAMMING_7_4_H,
    HAMMING_7_4_G,
    encode_hamming_7_4,
    decode_hamming_7_4,
    evaluate_hamming_evidence,
    compute_hamming_syndrome
)
from .bch import (
    BCH_15_7_G,
    encode_bch_15_7,
    decode_bch_15_7,
    evaluate_bch_evidence
)
from .reed_solomon import (
    encode_rs_15_11,
    decode_rs_15_11,
    evaluate_rs_parameter_candidates
)
from .ldpc import (
    LDPC_CONTROLLED_H,
    evaluate_ldpc_syndrome,
    rank_ldpc_candidates
)
from .convolutional import (
    CONV_CANDIDATE_POOL_K3,
    encode_convolutional_k3,
    viterbi_decode_hard_k3,
    identify_convolutional_generator
)
from .interleaver import (
    interleave_row_column,
    deinterleave_row_column,
    identify_interleaver_width
)
from .joint_engine import evaluate_joint_hypotheses
from .service import (
    FecCrcInterleaverService,
    default_fec_crc_service,
    process_fec_crc_analysis,
    MODULE10_REJECTED_METHODS
)

__all__ = [
    "compute_crc",
    "verify_crc16_ccitt",
    "append_crc16_ccitt",
    "evaluate_crc_acceptance_rate",
    "search_crc_boundaries",
    "identify_crc_family",
    "compute_soft_crc_evidence",
    "CRC_CANDIDATE_CATALOG",
    "HAMMING_7_4_H",
    "HAMMING_7_4_G",
    "encode_hamming_7_4",
    "decode_hamming_7_4",
    "evaluate_hamming_evidence",
    "compute_hamming_syndrome",
    "BCH_15_7_G",
    "encode_bch_15_7",
    "decode_bch_15_7",
    "evaluate_bch_evidence",
    "encode_rs_15_11",
    "decode_rs_15_11",
    "evaluate_rs_parameter_candidates",
    "LDPC_CONTROLLED_H",
    "evaluate_ldpc_syndrome",
    "rank_ldpc_candidates",
    "CONV_CANDIDATE_POOL_K3",
    "encode_convolutional_k3",
    "viterbi_decode_hard_k3",
    "identify_convolutional_generator",
    "interleave_row_column",
    "deinterleave_row_column",
    "identify_interleaver_width",
    "evaluate_joint_hypotheses",
    "FecCrcInterleaverService",
    "default_fec_crc_service",
    "process_fec_crc_analysis",
    "MODULE10_REJECTED_METHODS"
]
