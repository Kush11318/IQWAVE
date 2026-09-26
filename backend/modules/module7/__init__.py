"""Module 7: Soft Bits / LLR Generation.

Converts recovered digital symbols and waveforms into signed soft bit reliability metrics
for downstream error-correcting codes (Module 8).
"""

from .service import SoftBitsService, default_soft_bits_service, generate_soft_bits
from .psk_llr import (
    compute_bpsk_analytical_llr,
    compute_bpsk_exact_llr,
    compute_qpsk_analytical_llr,
    compute_qpsk_exact_llr,
    compute_8psk_exact_llr,
    compute_8psk_maxlog_llr,
)
from .qam_llr import (
    get_qam16_constellation,
    compute_qam16_exact_llr,
    compute_qam16_maxlog_llr,
    get_qam64_constellation,
    compute_qam64_exact_llr,
    compute_qam64_maxlog_llr,
)
from .fsk_soft import (
    generate_cpfsk_templates,
    generate_gfsk_templates,
    compute_fsk_waveform_correlation_soft,
    compute_rejected_instantaneous_freq_metric,
    compute_rejected_symbol_center_freq_metric,
)
from .reliability import compute_reliability_summary

__all__ = [
    "SoftBitsService",
    "default_soft_bits_service",
    "generate_soft_bits",
    "compute_bpsk_analytical_llr",
    "compute_bpsk_exact_llr",
    "compute_qpsk_analytical_llr",
    "compute_qpsk_exact_llr",
    "compute_8psk_exact_llr",
    "compute_8psk_maxlog_llr",
    "get_qam16_constellation",
    "compute_qam16_exact_llr",
    "compute_qam16_maxlog_llr",
    "get_qam64_constellation",
    "compute_qam64_exact_llr",
    "compute_qam64_maxlog_llr",
    "generate_cpfsk_templates",
    "generate_gfsk_templates",
    "compute_fsk_waveform_correlation_soft",
    "compute_rejected_instantaneous_freq_metric",
    "compute_rejected_symbol_center_freq_metric",
    "compute_reliability_summary",
]
