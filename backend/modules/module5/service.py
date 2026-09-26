"""Module 5 Service: Synchronization & Digital Symbol/Bit Recovery Orchestrator.

Consumes:
- Modulation from Module 3.
- Parameter estimates from Module 4 (symbol_rate, samples_per_symbol, cfo, sample_rate, etc.).

Executes the validated DSP receiver pipeline:
    1. 5A: Carrier-frequency synchronization (CFO correction).
    2. 5B: Carrier-phase synchronization (Phase offset correction).
    3. 5C: Symbol-timing recovery (Gardner TED).
    4. 5D: RRC matched filtering.
    5. 5E: Rational resampling (for non-integer SPS).
    6. Symbol sampling.
    7. 5F: Modulation-specific demodulation.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from .carrier_sync import synchronize_carrier
from .phase_sync import synchronize_phase
from .timing_sync import synchronize_timing_gardner
from .matched_filter import apply_matched_filter
from .resampler import rational_resample
from .demodulators import demodulate_signal


class DigitalRecoveryService:
    def recover_signal(
        self,
        canonical_iq: np.ndarray,
        modulation: Optional[str] = None,
        module4_parameters: Optional[Dict[str, Any]] = None,
        sample_rate: Optional[float] = None,
        samples_per_symbol: Optional[float] = None,
        cfo: Optional[float] = None,
        rrc_beta: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute end-to-end synchronization and digital recovery.

        Args:
            canonical_iq: Complex canonical IQ numpy array.
            modulation: Modulation class (e.g. 'BPSK', 'QPSK', 'CPFSK').
            module4_parameters: Output contract dictionary from Module 4.
            sample_rate: Physical sampling rate in Hz (if available).
            samples_per_symbol: SPS from Module 4 (if available).
            cfo: Estimated CFO from Module 4 (if available).
            rrc_beta: Roll-off factor beta.

        Returns:
            Structured dictionary matching Section 28 output contract.
        """
        warnings: List[str] = []
        m4 = module4_parameters or {}

        # 1. Numerical & Input Guards
        if canonical_iq is None or len(canonical_iq) == 0:
            return {
                "status": "INVALID_INPUT",
                "error": "EMPTY_OR_UNAVAILABLE_SIGNAL",
                "modulation": modulation,
                "symbols": [],
                "bits": [],
                "num_symbols": 0,
                "num_bits": 0,
                "warnings": ["EMPTY_SIGNAL"]
            }

        if not np.all(np.isfinite(canonical_iq)):
            return {
                "status": "INVALID_INPUT",
                "error": "NON_FINITE_SAMPLES",
                "modulation": modulation,
                "symbols": [],
                "bits": [],
                "num_symbols": 0,
                "num_bits": 0,
                "warnings": ["NON_FINITE_SAMPLES"]
            }

        power = float(np.mean(np.abs(canonical_iq) ** 2.0))
        if power < 1e-15:
            return {
                "status": "INVALID_INPUT",
                "error": "ZERO_POWER_SIGNAL",
                "modulation": modulation,
                "symbols": [],
                "bits": [],
                "num_symbols": 0,
                "num_bits": 0,
                "warnings": ["ZERO_POWER_SIGNAL"]
            }

        if len(canonical_iq) < 32:
            return {
                "status": "INSUFFICIENT_SAMPLES",
                "error": "INSUFFICIENT_OBSERVATION_LENGTH",
                "modulation": modulation,
                "symbols": [],
                "bits": [],
                "num_symbols": 0,
                "num_bits": 0,
                "warnings": ["INSUFFICIENT_SAMPLES"]
            }

        # 2. Extract and Validate Required Parameters
        active_mod = modulation or m4.get("modulation")
        if not active_mod or active_mod in ["UNKNOWN", "NONE", "MODEL_WEIGHTS_UNAVAILABLE"]:
            return {
                "status": "INSUFFICIENT_PARAMETERS",
                "error": "MISSING_OR_UNAVAILABLE_MODULATION_CLASS",
                "modulation": None,
                "symbols": [],
                "bits": [],
                "num_symbols": 0,
                "num_bits": 0,
                "warnings": ["MODULATION_CLASS_REQUIRED_FROM_MODULE3"]
            }

        active_mod_upper = active_mod.upper()
        supported = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "GFSK", "CPFSK"]
        if active_mod_upper not in supported:
            return {
                "status": "UNSUPPORTED_MODULATION",
                "error": f"MODULATION_{active_mod_upper}_NOT_SUPPORTED",
                "modulation": active_mod,
                "symbols": [],
                "bits": [],
                "num_symbols": 0,
                "num_bits": 0,
                "warnings": ["UNSUPPORTED_MODULATION_CLASS"]
            }

        fs = sample_rate if sample_rate is not None else m4.get("metadata", {}).get("sample_rate")
        sps = samples_per_symbol if samples_per_symbol is not None else m4.get("samples_per_symbol")
        cfo_val = cfo if cfo is not None else m4.get("cfo")
        is_fsk = active_mod_upper in ["GFSK", "CPFSK"]

        # Validate SPS
        if sps is None or sps <= 0:
            return {
                "status": "INSUFFICIENT_PARAMETERS",
                "error": "MISSING_OR_INVALID_SAMPLES_PER_SYMBOL",
                "modulation": active_mod,
                "symbols": [],
                "bits": [],
                "num_symbols": 0,
                "num_bits": 0,
                "warnings": ["SAMPLES_PER_SYMBOL_REQUIRED_FROM_MODULE4"]
            }

        curr_iq = canonical_iq.copy()

        # 3. FSK Dedicated Branch
        if is_fsk:
            # FSK uses instantaneous frequency without constellation slicing
            fs_fsk = fs if (fs is not None and fs > 0) else 1000000.0
            sps_int = max(2, int(round(sps)))
            try:
                dec_syms, bits = demodulate_signal(
                    signal_waveform=curr_iq,
                    modulation=active_mod_upper,
                    sps=sps_int,
                    sample_rate=fs_fsk
                )
            except Exception as e:
                return {
                    "status": "DEMODULATION_FAILURE",
                    "error": str(e),
                    "modulation": active_mod,
                    "symbols": [],
                    "bits": [],
                    "num_symbols": 0,
                    "num_bits": 0,
                    "warnings": ["FSK_FREQUENCY_EXTRACTION_FAILED"]
                }

            return {
                "status": "SUCCESS",
                "modulation": active_mod,
                "synchronization": {
                    "cfo_estimate": 0.0,
                    "cfo_applied": 0.0,
                    "phase_estimate": 0.0,
                    "phase_applied": 0.0,
                    "timing_offset": 0.0,
                    "timing_quality": "NOT_APPLICABLE_TO_FSK",
                    "matched_filter": "BYPASS_FSK",
                    "resampling": "BYPASS_FSK"
                },
                "symbols": [complex(s) for s in dec_syms.tolist()],
                "bits": [int(b) for b in bits.tolist()],
                "num_symbols": len(dec_syms),
                "num_bits": len(bits),
                "quality": {
                    "cfo": "NOT_APPLICABLE_TO_FSK",
                    "phase": "NOT_APPLICABLE_TO_FSK",
                    "timing": "NOT_APPLICABLE_TO_FSK",
                    "overall": "HIGH" if len(bits) > 0 else "LOW"
                },
                "warnings": warnings
            }

        # 4. PSK / QAM Branch

        # 4A. Carrier-Frequency Synchronization
        cfo_res = synchronize_carrier(
            iq_signal=curr_iq,
            modulation=active_mod_upper,
            sample_rate=fs,
            initial_cfo=cfo_val
        )
        curr_iq = cfo_res["corrected_signal"]
        warnings.extend(cfo_res.get("warnings", []))

        # 4B. Carrier-Phase Synchronization
        phase_res = synchronize_phase(
            iq_signal=curr_iq,
            modulation=active_mod_upper
        )
        curr_iq = phase_res["corrected_signal"]
        warnings.extend(phase_res.get("warnings", []))

        # 4C. Timing Recovery (Gardner TED)
        timing_res = synchronize_timing_gardner(
            iq_signal=curr_iq,
            samples_per_symbol=sps
        )
        curr_iq = timing_res["corrected_signal"]
        warnings.extend(timing_res.get("warnings", []))

        # 4D. RRC Matched Filtering
        mf_res = apply_matched_filter(
            iq_signal=curr_iq,
            modulation=active_mod_upper,
            samples_per_symbol=sps,
            beta=rrc_beta
        )
        curr_iq = mf_res["filtered_signal"]
        warnings.extend(mf_res.get("warnings", []))

        # 4E. Rational Resampling (if non-integer SPS)
        resamp_res = rational_resample(
            iq_signal=curr_iq,
            samples_per_symbol=sps
        )
        curr_iq = resamp_res["resampled_signal"]
        effective_sps = int(round(resamp_res["new_sps"]))
        warnings.extend(resamp_res.get("warnings", []))

        # 4F. Demodulation
        try:
            dec_syms, bits = demodulate_signal(
                signal_waveform=curr_iq,
                modulation=active_mod_upper,
                sps=effective_sps,
                sample_rate=fs
            )
        except Exception as e:
            return {
                "status": "DEMODULATION_FAILURE",
                "error": str(e),
                "modulation": active_mod,
                "symbols": [],
                "bits": [],
                "num_symbols": 0,
                "num_bits": 0,
                "warnings": warnings + ["CONSTELLATION_DECISION_FAILED"]
            }

        # Overall quality aggregation
        q_list = [cfo_res.get("quality"), phase_res.get("quality"), timing_res.get("quality")]
        if "LOW" in q_list:
            overall_q = "LOW"
        elif "MEDIUM" in q_list:
            overall_q = "MEDIUM"
        else:
            overall_q = "HIGH"

        out: Dict[str, Any] = {
            "status": "SUCCESS",
            "modulation": active_mod,
            "synchronization": {
                "cfo_estimate": cfo_res.get("cfo_estimate"),
                "cfo_applied": cfo_res.get("cfo_applied"),
                "phase_estimate": phase_res.get("phase_estimate_deg"),
                "phase_applied": phase_res.get("phase_applied_rad"),
                "timing_offset": timing_res.get("timing_offset_samples"),
                "timing_quality": timing_res.get("quality"),
                "matched_filter": mf_res.get("method"),
                "resampling": resamp_res.get("method")
            },
            "symbols": [complex(s) for s in dec_syms.tolist()],
            "bits": [int(b) for b in bits.tolist()],
            "num_symbols": len(dec_syms),
            "num_bits": len(bits),
            "quality": {
                "cfo": cfo_res.get("quality", "UNKNOWN"),
                "phase": phase_res.get("quality", "UNKNOWN"),
                "timing": timing_res.get("quality", "UNKNOWN"),
                "overall": overall_q
            },
            "warnings": list(set(warnings))
        }

        return out


default_recovery_service = DigitalRecoveryService()


def recover_digital_symbols_and_bits(
    canonical_iq: np.ndarray,
    modulation: Optional[str] = None,
    module4_parameters: Optional[Dict[str, Any]] = None,
    sample_rate: Optional[float] = None,
    samples_per_symbol: Optional[float] = None,
    cfo: Optional[float] = None,
    rrc_beta: Optional[float] = None
) -> Dict[str, Any]:
    """Convenience functional interface."""
    return default_recovery_service.recover_signal(
        canonical_iq=canonical_iq,
        modulation=modulation,
        module4_parameters=module4_parameters,
        sample_rate=sample_rate,
        samples_per_symbol=samples_per_symbol,
        cfo=cfo,
        rrc_beta=rrc_beta
    )
