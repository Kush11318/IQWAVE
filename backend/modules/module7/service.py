"""Module 7 Service: Soft Bits / LLR Generation Orchestrator.

Processes synchronized symbol/waveform outputs from Module 5 and noise parameters from Module 6
to produce signed soft bits and diagnostic reliability summaries for all seven modulation classes:
- BPSK (Analytical AWGN LLR, exact equivalent)
- QPSK (Exact constellation likelihood LLR)
- 8PSK (Exact constellation likelihood LLR, natural binary mapping)
- QAM16 (Exact constellation likelihood LLR)
- QAM64 (Exact constellation likelihood LLR)
- CPFSK (Waveform-correlation soft metric, NOT calibrated LLR)
- GFSK (Waveform-correlation soft metric, NOT calibrated LLR)

Enforces strict scientific constraints:
1. If required noise parameter N0 is unavailable for PSK/QAM:
   - Does NOT substitute N0=1.0.
   - Returns NOISE_PARAMETER_UNAVAILABLE.
   - Does NOT fabricate a scaled LLR.
2. PSK/QAM metric_type='LLR', mathematical_status='EXACT_UNDER_AWGN_MODEL', calibrated_llr=False.
3. CPFSK/GFSK metric_type='SOFT_METRIC', mathematical_status='NOT_CALIBRATED_LLR', calibrated_llr=False,
   with strictly NO 2/N0 scaling.
4. Sign convention: positive (>0) -> bit 1, negative (<0) -> bit 0.
5. Reliability bins are strictly diagnostic, not confidence thresholds.
6. Module 7 terminates at soft bits and reliability diagnostics (no Module 8 FEC/decoding).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .psk_llr import (
    compute_bpsk_analytical_llr,
    compute_bpsk_exact_llr,
    compute_qpsk_analytical_llr,
    compute_qpsk_exact_llr,
    compute_8psk_exact_llr,
    compute_8psk_maxlog_llr
)
from .qam_llr import (
    compute_qam16_exact_llr,
    compute_qam16_maxlog_llr,
    compute_qam64_exact_llr,
    compute_qam64_maxlog_llr
)
from .fsk_soft import (
    compute_fsk_waveform_correlation_soft,
    compute_rejected_instantaneous_freq_metric,
    compute_rejected_symbol_center_freq_metric
)
from .reliability import compute_reliability_summary


class SoftBitsService:
    """Service generating modulation-specific soft bit representations."""

    SUPPORTED_MODULATIONS = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "CPFSK", "GFSK"]

    def generate_soft_bits(
        self,
        module5_recovery: Optional[Dict[str, Any]] = None,
        module6_snr: Optional[Dict[str, Any]] = None,
        symbol_samples: Optional[np.ndarray] = None,
        signal_waveform: Optional[np.ndarray] = None,
        modulation: Optional[str] = None,
        sps: Optional[int] = None,
        sample_rate: Optional[float] = None,
        n0: Optional[float] = None,
        reference_hard_bits: Optional[np.ndarray] = None,
        ground_truth_bits: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Generate soft bits / LLRs from digital recovery and SNR parameters.

        Args:
            module5_recovery: Output dictionary from Module 5.
            module6_snr: Output dictionary from Module 6.
            symbol_samples: Explicit symbol samples r_k (for PSK/QAM).
            signal_waveform: Explicit synchronized waveform (for FSK).
            modulation: Active modulation string override.
            sps: Samples per symbol override.
            sample_rate: Sampling frequency override.
            n0: Direct noise spectral density / variance override.
            reference_hard_bits: Reference hard bits for agreement diagnostic.
            ground_truth_bits: Ground-truth bits for empirical bin BER.

        Returns:
            Structured dictionary matching Module 7 contract.
        """
        warnings: List[str] = []
        m5 = module5_recovery or {}
        m6 = module6_snr or {}

        # 1. Determine active modulation
        active_mod = modulation or m5.get("modulation") or m6.get("modulation")
        if not active_mod:
            return {
                "status": "ERROR",
                "modulation": None,
                "error": "MISSING_MODULATION: Modulation class must be specified.",
                "soft_bits": None,
                "hard_bits": None,
                "warnings": ["MODULATION_NOT_SPECIFIED"]
            }

        mod_upper = str(active_mod).upper()
        if mod_upper not in self.SUPPORTED_MODULATIONS:
            return {
                "status": "UNSUPPORTED_MODULATION",
                "modulation": active_mod,
                "error": f"Modulation '{active_mod}' is not one of the seven supported classes.",
                "soft_bits": None,
                "hard_bits": None,
                "warnings": [f"UNSUPPORTED_MODULATION_{mod_upper}"]
            }

        # 2. Extract symbol samples and waveform
        syms = symbol_samples
        if syms is None:
            syms = m5.get("symbol_samples")
            if syms is None and mod_upper not in ["CPFSK", "GFSK"]:
                # If neither provided, check decided_symbols or iq_signal
                syms = m5.get("decided_symbols")

        waveform = signal_waveform
        if waveform is None:
            waveform = m5.get("signal_waveform")
            if waveform is None:
                waveform = m5.get("canonical_iq")

        sps_val = sps or m5.get("sps") or 4
        fs_val = sample_rate or m5.get("sample_rate") or 1000000.0

        upstream_hard = reference_hard_bits
        if upstream_hard is None:
            raw_hard = m5.get("demodulated_bits")
            if raw_hard is not None:
                upstream_hard = np.asarray(raw_hard, dtype=np.uint8)

        # 3. Extract Noise Parameter N0 (Required for PSK / QAM)
        resolved_n0 = None
        if n0 is not None and float(n0) > 0.0 and not np.isnan(n0) and not np.isinf(n0):
            resolved_n0 = float(n0)
        elif m6:
            # Check explicit n0 or noise_var
            m6_n0 = m6.get("n0") or m6.get("noise_var")
            if m6_n0 is not None and float(m6_n0) > 0.0:
                resolved_n0 = float(m6_n0)
            elif m6.get("snr_db") is not None:
                snr_db_val = float(m6["snr_db"])
                if not np.isnan(snr_db_val) and not np.isinf(snr_db_val):
                    snr_lin = 10.0 ** (snr_db_val / 10.0)
                    if snr_lin > 1e-12:
                        # Compute symbol power from symbols if available, else nominal 1.0
                        p_s = 1.0
                        if syms is not None and len(syms) > 0:
                            p_s = float(np.mean(np.abs(syms) ** 2.0))
                            if p_s <= 0.0 or np.isnan(p_s):
                                p_s = 1.0
                        resolved_n0 = p_s / snr_lin

        # 4. Enforce Missing-Noise Failure Path for PSK / QAM
        is_fsk = mod_upper in ["CPFSK", "GFSK"]
        if not is_fsk and resolved_n0 is None:
            # Strictly NO N0=1 fallback
            msg = (
                "NOISE_PARAMETER_UNAVAILABLE: Exact likelihood scaling requires a valid upstream "
                "noise parameter (N0, noise_var, or SNR) from Module 6. No fallback value (e.g. N0=1) "
                "was substituted to avoid fabricating unscaled or falsely calibrated LLRs."
            )
            warnings.append(msg)
            return {
                "status": "NOISE_PARAMETER_UNAVAILABLE",
                "modulation": mod_upper,
                "soft_bits": None,
                "hard_bits": upstream_hard.tolist() if upstream_hard is not None else None,
                "metric_type": "LLR",
                "mathematical_status": "EXACT_UNDER_AWGN_MODEL",
                "calibrated_llr": False,
                "noise_parameter_used": None,
                "warnings": warnings,
                "reliability_summary": None,
                "boundary": "MODULE_7_CONCLUDED_AT_SOFT_BITS_AND_RELIABILITY_DIAGNOSTICS"
            }

        # 5. Modulation-Specific Execution
        soft_bits_arr: np.ndarray = np.array([], dtype=np.float64)
        per_bit_mat: Optional[np.ndarray] = None
        algo_used = ""
        metric_type = "LLR"
        math_status = "EXACT_UNDER_AWGN_MODEL"
        bits_per_symbol = 1

        if mod_upper == "BPSK":
            if syms is None or len(syms) == 0:
                return self._empty_error("NO_SYMBOLS_FOR_BPSK", mod_upper)
            soft_bits_arr = compute_bpsk_analytical_llr(syms, n0=resolved_n0)
            algo_used = "ANALYTICAL_AWGN_LLR"
            metric_type = "LLR"
            math_status = "EXACT_UNDER_AWGN_MODEL"
            bits_per_symbol = 1

        elif mod_upper == "QPSK":
            if syms is None or len(syms) == 0:
                return self._empty_error("NO_SYMBOLS_FOR_QPSK", mod_upper)
            soft_bits_arr = compute_qpsk_exact_llr(syms, n0=resolved_n0)
            algo_used = "EXACT_CONSTELLATION_LIKELIHOOD"
            metric_type = "LLR"
            math_status = "EXACT_UNDER_AWGN_MODEL"
            bits_per_symbol = 2
            per_bit_mat = soft_bits_arr.reshape(-1, 2)

        elif mod_upper == "8PSK":
            if syms is None or len(syms) == 0:
                return self._empty_error("NO_SYMBOLS_FOR_8PSK", mod_upper)
            soft_bits_arr, per_bit_mat = compute_8psk_exact_llr(syms, n0=resolved_n0)
            algo_used = "EXACT_CONSTELLATION_LIKELIHOOD"
            metric_type = "LLR"
            math_status = "EXACT_UNDER_AWGN_MODEL"
            bits_per_symbol = 3

        elif mod_upper == "QAM16":
            if syms is None or len(syms) == 0:
                return self._empty_error("NO_SYMBOLS_FOR_QAM16", mod_upper)
            soft_bits_arr, per_bit_mat = compute_qam16_exact_llr(syms, n0=resolved_n0)
            algo_used = "EXACT_CONSTELLATION_LIKELIHOOD"
            metric_type = "LLR"
            math_status = "EXACT_UNDER_AWGN_MODEL"
            bits_per_symbol = 4

        elif mod_upper == "QAM64":
            if syms is None or len(syms) == 0:
                return self._empty_error("NO_SYMBOLS_FOR_QAM64", mod_upper)
            soft_bits_arr, per_bit_mat = compute_qam64_exact_llr(syms, n0=resolved_n0)
            algo_used = "EXACT_CONSTELLATION_LIKELIHOOD"
            metric_type = "LLR"
            math_status = "EXACT_UNDER_AWGN_MODEL"
            bits_per_symbol = 6

        elif mod_upper in ["CPFSK", "GFSK"]:
            # Uses waveform correlation
            target_signal = waveform if waveform is not None and len(waveform) > 0 else syms
            if target_signal is None or len(target_signal) < sps_val:
                return self._empty_error(f"INSUFFICIENT_SAMPLES_FOR_{mod_upper}", mod_upper)
            soft_bits_arr = compute_fsk_waveform_correlation_soft(
                iq_signal=np.asarray(target_signal, dtype=np.complex128),
                sps=int(sps_val),
                sample_rate=float(fs_val),
                modulation=mod_upper
            )
            algo_used = "WAVEFORM_CORRELATION"
            metric_type = "SOFT_METRIC"
            math_status = "NOT_CALIBRATED_LLR"
            bits_per_symbol = 1
            if resolved_n0 is None:
                warnings.append("NOISE_PARAMETER_NOT_PROVIDED: FSK waveform correlation operates without N0 scaling.")

        # 6. Hard decisions from soft values: sign convention (positive -> 1, negative -> 0)
        derived_hard_bits = (soft_bits_arr > 0.0).astype(np.uint8)

        # 7. Diagnostic Reliability Summary
        reliability = compute_reliability_summary(
            soft_bits=soft_bits_arr,
            ground_truth_bits=ground_truth_bits,
            per_bit_matrix=per_bit_mat,
            reference_hard_bits=upstream_hard
        )

        return {
            "status": "SUCCESS",
            "modulation": mod_upper,
            "algorithm_used": algo_used,
            "metric_type": metric_type,
            "mathematical_status": math_status,
            "calibrated_llr": False,
            "noise_parameter_used": resolved_n0,
            "bits_per_symbol": bits_per_symbol,
            "total_soft_bits": int(len(soft_bits_arr)),
            "soft_bits": soft_bits_arr.tolist(),
            "hard_bits": derived_hard_bits.tolist(),
            "reliability_summary": reliability,
            "warnings": warnings,
            "boundary": "MODULE_7_CONCLUDED_AT_SOFT_BITS_AND_RELIABILITY_DIAGNOSTICS"
        }

    def _empty_error(self, code: str, modulation: str) -> Dict[str, Any]:
        return {
            "status": "ERROR",
            "modulation": modulation,
            "error": code,
            "soft_bits": None,
            "hard_bits": None,
            "warnings": [code]
        }


default_soft_bits_service = SoftBitsService()


def generate_soft_bits(
    module5_recovery: Optional[Dict[str, Any]] = None,
    module6_snr: Optional[Dict[str, Any]] = None,
    symbol_samples: Optional[np.ndarray] = None,
    signal_waveform: Optional[np.ndarray] = None,
    modulation: Optional[str] = None,
    sps: Optional[int] = None,
    sample_rate: Optional[float] = None,
    n0: Optional[float] = None,
    reference_hard_bits: Optional[np.ndarray] = None,
    ground_truth_bits: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Convenience functional interface for Module 7."""
    return default_soft_bits_service.generate_soft_bits(
        module5_recovery=module5_recovery,
        module6_snr=module6_snr,
        symbol_samples=symbol_samples,
        signal_waveform=signal_waveform,
        modulation=modulation,
        sps=sps,
        sample_rate=sample_rate,
        n0=n0,
        reference_hard_bits=reference_hard_bits,
        ground_truth_bits=ground_truth_bits
    )
