"""Module 6 Service: SNR Estimation & Confidence Orchestrator.

Consumes:
- Module 3 AMC modulation.
- Module 4 parameters (Rs, SPS, Fs, etc.).
- Module 5 digital recovery outputs (received symbols r_k, detected symbols s_hat_k,
  demodulated bits, synchronization quality).

Applies modulation-specific estimator selection:
- BPSK: NDA-ML (CONDITIONAL)
- QPSK: Residual ensemble (LOCKED; both ordinary and robust preserved independently;
  fusion rule is explicitly reported as FUSION_RULE_NOT_SPECIFIED_BY_SOURCE rather
  than inventing an average).
- 8PSK: Residual ensemble (LOCKED; same independent preservation rule).
- QAM16: Robust residual (CONDITIONAL)
- QAM64: Robust residual (CONDITIONAL)
- CPFSK: Fourth-moment NDA (LOCKED for tested AWGN)
- GFSK: Fourth-moment NDA (LOCKED for tested AWGN)

Disagreement, EVM, and BER are used strictly as engineering quality features.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from .oracle_snr import compute_oracle_snr
from .total_power import compute_total_power_snr
from .nda_ml import estimate_bpsk_nda_ml
from .residual_snr import estimate_residual_snr, compute_evm
from .fourth_moment import estimate_fourth_moment_snr
from .fsk_snr import estimate_fsk_snr
from .quality_metrics import compute_estimator_disagreement, assess_snr_quality


class SnrEstimationService:
    def estimate_snr(
        self,
        canonical_iq: Optional[np.ndarray] = None,
        modulation: Optional[str] = None,
        module4_parameters: Optional[Dict[str, Any]] = None,
        module5_recovery: Optional[Dict[str, Any]] = None,
        reference_clean_signal: Optional[np.ndarray] = None,
        reference_noise_signal: Optional[np.ndarray] = None,
        reference_bits: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Execute Module 6 SNR estimation and quality evaluation.

        Args:
            canonical_iq: Raw canonical complex IQ array.
            modulation: Active modulation class.
            module4_parameters: Parameter output dictionary from Module 4.
            module5_recovery: Recovery output dictionary from Module 5.
            reference_clean_signal: Clean transmitted signal (validation test only).
            reference_noise_signal: Injected noise signal (validation test only).
            reference_bits: Known transmitted bits for BER calculation.

        Returns:
            Structured dictionary matching Module 6 output contract.
        """
        warnings: List[str] = []
        limitations: List[str] = [
            "Decision-directed residual SNR is subject to decision-error bias below 5 dB.",
            "Residual energy reflects effective SNR under channel assumptions (including multipath/ISI), not pure thermal noise alone.",
            "Fourth-moment FSK estimator is locked only for tested AWGN constant-envelope setups and not arbitrary multipath/fading channels.",
            "Disagreement boundaries (2 dB, 4 dB), EVM threshold (0.35), and observation length thresholds are UNVALIDATED_ENGINEERING_HEURISTIC thresholds.",
            "Oracle SNR is restricted strictly to offline test benchmarking and prohibited from production use."
        ]

        m4 = module4_parameters or {}
        m5 = module5_recovery or {}

        active_mod = modulation or m5.get("modulation") or m4.get("modulation")
        if not active_mod or active_mod in ["UNKNOWN", "NONE", "MODEL_WEIGHTS_UNAVAILABLE"]:
            return {
                "status": "INSUFFICIENT_PARAMETERS",
                "error": "MISSING_OR_UNAVAILABLE_MODULATION_CLASS",
                "modulation": None,
                "snr_db": None,
                "estimator_used": "NONE",
                "estimator_status": "UNAVAILABLE",
                "quality": "UNAVAILABLE",
                "estimator_outputs": {},
                "disagreement": {"range": None, "std": None, "mad": None},
                "evm": None,
                "ber": None,
                "warnings": ["MODULATION_CLASS_REQUIRED_FROM_UPSTREAM"],
                "limitations": limitations
            }

        mod_upper = active_mod.upper()
        supported = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "GFSK", "CPFSK"]
        if mod_upper not in supported:
            return {
                "status": "UNSUPPORTED_MODULATION",
                "error": f"MODULATION_{mod_upper}_NOT_SUPPORTED",
                "modulation": active_mod,
                "snr_db": None,
                "estimator_used": "NONE",
                "estimator_status": "UNSUPPORTED",
                "quality": "UNAVAILABLE",
                "estimator_outputs": {},
                "disagreement": {"range": None, "std": None, "mad": None},
                "evm": None,
                "ber": None,
                "warnings": ["UNSUPPORTED_MODULATION_CLASS"],
                "limitations": limitations
            }

        # Check IQ finiteness if present
        if canonical_iq is not None:
            if len(canonical_iq) == 0:
                warnings.append("EMPTY_CANONICAL_IQ")
            elif not np.all(np.isfinite(canonical_iq)):
                warnings.append("NON_FINITE_CANONICAL_IQ")

        # Extract symbols and sync info from Module 5
        detected_syms = m5.get("detected_symbols") if m5.get("detected_symbols") is not None else m5.get("symbols", [])
        received_syms = m5.get("received_symbols")
        recovered_bits = m5.get("recovered_bits") if m5.get("recovered_bits") is not None else m5.get("bits", [])
        sync_diag = m5.get("synchronization", {})
        raw_sq = m5.get("synchronization_quality") or m5.get("quality", {}).get("overall") or "UNKNOWN"
        if isinstance(raw_sq, dict):
            sync_quality = str(raw_sq.get("overall") or raw_sq.get("status") or "UNKNOWN").upper()
        elif isinstance(raw_sq, str):
            sync_quality = raw_sq.upper()
        else:
            sync_quality = "UNKNOWN"

        # For PSK/QAM residual calculation, symbol samples r_k can come from m5 or canonical_iq
        r_k = None
        s_hat = None
        if detected_syms is not None and len(detected_syms) > 0:
            s_hat = np.asarray(detected_syms, dtype=np.complex128)

        if received_syms is not None and len(received_syms) > 0:
            r_k = np.asarray(received_syms, dtype=np.complex128)
            if s_hat is not None and len(r_k) != len(s_hat):
                min_len = min(len(r_k), len(s_hat))
                r_k = r_k[:min_len]
                s_hat = s_hat[:min_len]
        elif s_hat is not None:
            # If canonical_iq is provided, extract symbol-spaced samples
            sps = m4.get("samples_per_symbol")
            if canonical_iq is not None and sps is not None and sps >= 1:
                sps_int = int(round(sps))
                r_k = canonical_iq[::sps_int][:len(s_hat)]
            else:
                r_k = s_hat

        # 1. Oracle ground truth calculation (validation only)
        oracle_out = None
        if reference_clean_signal is not None and reference_noise_signal is not None:
            oracle_out = compute_oracle_snr(reference_clean_signal, reference_noise_signal)
            if oracle_out.get("snr_db") is not None:
                warnings.append("ORACLE_SNR_COMPUTED_FOR_VALIDATION_ONLY")

        # 2. Fourth-Moment NDA Estimator
        fm_out = None
        if canonical_iq is not None and len(canonical_iq) > 0 and np.all(np.isfinite(canonical_iq)):
            fm_out = estimate_fourth_moment_snr(canonical_iq, modulation=mod_upper)

        # 3. Decision-Directed Residual Estimator (Ordinary & Robust) + EVM
        res_out = None
        evm_val = None
        if r_k is not None and s_hat is not None and len(r_k) > 0 and len(r_k) == len(s_hat):
            res_out = estimate_residual_snr(r_k, s_hat, modulation=mod_upper)
            evm_val = res_out.get("evm")
            warnings.extend(res_out.get("warnings", []))

        # 4. Measured BER (when reference bits are provided)
        ber_val = None
        if reference_bits is not None and len(reference_bits) > 0 and len(recovered_bits) > 0:
            cmp_len = min(len(reference_bits), len(recovered_bits))
            ref_arr = np.array(reference_bits[:cmp_len], dtype=np.uint8)
            rec_arr = np.array(recovered_bits[:cmp_len], dtype=np.uint8)
            ber_val = float(np.mean(ref_arr != rec_arr))
        else:
            warnings.append("BER_UNAVAILABLE_WITHOUT_REFERENCE_BITSTREAM")

        # 5. Modulation-Specific Estimator Selection
        final_snr_db: Optional[float] = None
        primary_estimator = "UNKNOWN"
        estimator_status = "UNKNOWN"
        ensemble_status: Optional[str] = None
        obs_len = len(canonical_iq) if canonical_iq is not None else len(detected_syms)

        estimator_outputs: Dict[str, Any] = {
            "ordinary_residual": {
                "snr_db": res_out.get("ordinary_snr_db") if res_out else None,
                "status": "SUPPORTING"
            },
            "robust_residual": {
                "snr_db": res_out.get("robust_snr_db") if res_out else None,
                "status": "CONDITIONAL" if mod_upper in ["QAM16", "QAM64"] else "SUPPORTING"
            },
            "fourth_moment": {
                "snr_db": fm_out.get("snr_db") if fm_out else None,
                "status": "LOCKED_FOR_TESTED_AWGN" if mod_upper in ["CPFSK", "GFSK"] else "SUPPORTING"
            },
            "nda_ml": {
                "snr_db": None,
                "status": "CONDITIONAL"
            },
            "oracle": {
                "snr_db": oracle_out.get("snr_db") if oracle_out else None,
                "status": "OFFLINE_VALIDATION_ONLY"
            }
        }

        # Branching logic
        if mod_upper == "BPSK":
            primary_estimator = "BPSK_NDA_ML"
            estimator_status = "CONDITIONAL"
            # Compute NDA-ML
            real_data = None
            if canonical_iq is not None and len(canonical_iq) > 0:
                real_data = np.real(canonical_iq)
            elif s_hat is not None and len(s_hat) > 0:
                real_data = np.real(s_hat)

            if real_data is not None:
                ml_out = estimate_bpsk_nda_ml(real_data)
                final_snr_db = ml_out.get("snr_db")
                estimator_outputs["nda_ml"]["snr_db"] = final_snr_db
                warnings.extend(ml_out.get("warnings", []))
            else:
                warnings.append("NO_SAMPLES_FOR_BPSK_NDA_ML")

        elif mod_upper in ["QPSK", "8PSK"]:
            # RESIDUAL ENSEMBLE:
            # Source specifies "Residual ensemble" but does NOT provide a validated mathematical
            # fusion rule. Both Ordinary Residual and Robust Residual are preserved independently.
            # No averaging, weights, or heuristic selector may be invented.
            primary_estimator = "RESIDUAL_ENSEMBLE"
            estimator_status = "IMPLEMENTED_COMPONENTS_FUSION_RULE_UNSPECIFIED"
            ensemble_status = "FUSION_RULE_NOT_SPECIFIED_BY_SOURCE"
            warnings.append("RESIDUAL_ENSEMBLE_FUSION_RULE_UNSPECIFIED_BY_SOURCE")
            final_snr_db = None  # Not fused; callers inspect ordinary_residual and robust_residual

        elif mod_upper in ["QAM16", "QAM64"]:
            primary_estimator = "ROBUST_RESIDUAL"
            estimator_status = "CONDITIONAL"
            if res_out:
                final_snr_db = res_out.get("robust_snr_db")
            if mod_upper == "QAM64":
                warnings.append("QAM64_HIGH_DECISION_ERROR_SUSCEPTIBILITY")

        elif mod_upper in ["CPFSK", "GFSK"]:
            primary_estimator = "FOURTH_MOMENT_NDA"
            estimator_status = "LOCKED_FOR_TESTED_AWGN_FSK"
            sps_val = int(round(m4.get("samples_per_symbol") or 4))
            fs_val = float(m4.get("metadata", {}).get("sample_rate") or 1000000.0)
            if canonical_iq is not None and len(canonical_iq) > 0:
                fsk_res = estimate_fsk_snr(canonical_iq, modulation=mod_upper, sps=sps_val, sample_rate=fs_val)
                final_snr_db = fsk_res.get("snr_db")
                warnings.extend(fsk_res.get("warnings", []))
            else:
                warnings.append("NO_IQ_SAMPLES_FOR_FSK_SNR")

        # Check upstream sync warnings
        if sync_diag.get("cfo_applied") is None or sync_quality == "LOW":
            warnings.append("UNCORRECTED_CFO_CONTAMINATES_RESIDUAL_SNR")
        if sync_diag.get("timing_offset") is None:
            warnings.append("TIMING_OFFSET_CONTAMINATES_RESIDUAL_SNR")

        # 6. Disagreement Metrics
        candidates = [
            v["snr_db"] for k, v in estimator_outputs.items()
            if k != "oracle" and isinstance(v, dict) and v.get("snr_db") is not None
        ]
        disagreement = compute_estimator_disagreement(candidates)

        # 7. Quality Assessment
        eval_snr = final_snr_db if final_snr_db is not None else estimator_outputs.get("robust_residual", {}).get("snr_db")
        quality_rating = assess_snr_quality(
            snr_db=eval_snr,
            disagreement=disagreement,
            evm=evm_val,
            ber=ber_val,
            synchronization_quality=sync_quality,
            observation_length=obs_len,
            estimator_status=estimator_status
        )

        out: Dict[str, Any] = {
            "status": "SUCCESS",
            "modulation": active_mod,
            "snr_db": final_snr_db,
            "estimator_used": primary_estimator,
            "estimator_status": estimator_status,
            "ensemble_status": ensemble_status,
            "quality": quality_rating,
            "estimator_outputs": estimator_outputs,
            "disagreement": disagreement,
            "evm": evm_val,
            "ber": ber_val,
            "synchronization_quality": sync_quality,
            "observation_length": obs_len,
            "warnings": list(set(warnings)),
            "limitations": limitations
        }

        return out


default_snr_service = SnrEstimationService()


def estimate_signal_snr(
    canonical_iq: Optional[np.ndarray] = None,
    modulation: Optional[str] = None,
    module4_parameters: Optional[Dict[str, Any]] = None,
    module5_recovery: Optional[Dict[str, Any]] = None,
    reference_clean_signal: Optional[np.ndarray] = None,
    reference_noise_signal: Optional[np.ndarray] = None,
    reference_bits: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Convenience functional interface."""
    return default_snr_service.estimate_snr(
        canonical_iq=canonical_iq,
        modulation=modulation,
        module4_parameters=module4_parameters,
        module5_recovery=module5_recovery,
        reference_clean_signal=reference_clean_signal,
        reference_noise_signal=reference_noise_signal,
        reference_bits=reference_bits
    )
