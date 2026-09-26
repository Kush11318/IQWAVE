"""Module 6: SNR Estimation & Quality Assessment.

Implements blind SNR estimators:
- Oracle SNR (Validation only)
- Total Power (REJECTED)
- NDA-ML for BPSK (CONDITIONAL)
- Ordinary & Robust Residual for QPSK, 8PSK, QAM16, QAM64
- Fourth-Moment NDA for CPFSK and GFSK (LOCKED for tested AWGN)
- Disagreement, EVM, and BER engineering quality metrics
"""

from .service import SnrEstimationService, default_snr_service, estimate_signal_snr
from .oracle_snr import compute_oracle_snr
from .total_power import compute_total_power_snr
from .nda_ml import estimate_bpsk_nda_ml
from .residual_snr import estimate_residual_snr, compute_evm
from .fourth_moment import estimate_fourth_moment_snr
from .fsk_snr import estimate_fsk_snr
from .quality_metrics import compute_estimator_disagreement, assess_snr_quality

__all__ = [
    "SnrEstimationService",
    "default_snr_service",
    "estimate_signal_snr",
    "compute_oracle_snr",
    "compute_total_power_snr",
    "estimate_bpsk_nda_ml",
    "estimate_residual_snr",
    "compute_evm",
    "estimate_fourth_moment_snr",
    "estimate_fsk_snr",
    "compute_estimator_disagreement",
    "assess_snr_quality",
]
