"""Module 8: Blind FEC & Interleaver Identification, Recovery, and Decoding.

Determines whether a recovered bitstream contains an error-correcting code (FEC),
identifies candidate code families (Hamming, BCH, Convolutional), estimates
interleaver width (structured row-column), and performs hard or soft ML decoding.
"""

from .service import FecRecoveryService, default_fec_service, process_fec_recovery
from .hamming import (
    HAMMING_H,
    HAMMING_G,
    encode_hamming_7_4,
    compute_hamming_syndromes,
    find_hamming_alignment,
    compute_hamming_evidence,
    decode_hamming_hard,
    decode_hamming_soft_ml,
)
from .bch import (
    BCH_G_POLY,
    encode_bch_15_7,
    compute_bch_syndromes,
    find_bch_alignment,
    compute_bch_evidence,
    decode_bch_hard,
    decode_bch_soft_ml,
)
from .convolutional import (
    conv_encode_k3_r12,
    viterbi_decode_hard,
    viterbi_decode_soft,
    compute_convolutional_evidence,
)
from .interleaver import (
    interleave_row_column,
    deinterleave_row_column,
    search_interleaver_width,
)

__all__ = [
    "FecRecoveryService",
    "default_fec_service",
    "process_fec_recovery",
    "HAMMING_H",
    "HAMMING_G",
    "encode_hamming_7_4",
    "compute_hamming_syndromes",
    "find_hamming_alignment",
    "compute_hamming_evidence",
    "decode_hamming_hard",
    "decode_hamming_soft_ml",
    "BCH_G_POLY",
    "encode_bch_15_7",
    "compute_bch_syndromes",
    "find_bch_alignment",
    "compute_bch_evidence",
    "decode_bch_hard",
    "decode_bch_soft_ml",
    "conv_encode_k3_r12",
    "viterbi_decode_hard",
    "viterbi_decode_soft",
    "compute_convolutional_evidence",
    "interleave_row_column",
    "deinterleave_row_column",
    "search_interleaver_width",
]
