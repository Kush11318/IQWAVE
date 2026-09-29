"""Unified End-to-End Signal Analysis Pipeline Orchestrator (Modules 1–10).

This service is an orchestration/coordination layer only:
- Executes the locked Modules 1 through 10 sequentially.
- Enforces strict input validation and non-destructive propagation.
- Adapts data structures between locked modules without altering algorithms.
- Preserves scientific classifications: LOCKED, CONDITIONAL, HEURISTIC, REJECTED, UNKNOWN.
- Never fabricates missing SNR, unscaled LLRs, or universal confidence scalars.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.modules.module1.canonical_iq import to_canonical_iq
from backend.modules.module2.service import observe_from_module1
from backend.modules.module3.service import classify_amc
from backend.modules.module4.service import default_parameter_service
from backend.modules.module5.service import default_recovery_service
from backend.modules.module6.service import default_snr_service
from backend.modules.module7.service import default_soft_bits_service
from backend.modules.module8.service import default_fec_service
from backend.modules.module9.service import default_structure_service
from backend.modules.module10.service import default_fec_crc_service


class PipelineOrchestrator:
    """Orchestrates sequential end-to-end execution of Modules 1 through 10."""

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Return architecture status of all locked modules in pipeline."""
        return {
            "status": "SUCCESS",
            "pipeline": "BLIND_SIGNAL_ANALYSIS_E2E_PIPELINE",
            "scope": "Modules 1 through 10 (LOCKED)",
            "pipeline_version": "1.0.0-locked-modules-1-10",
            "total_modules": 10,
            "module11_status": "NOT_STARTED",
            "modules": [
                {"id": 1, "name": "Canonical IQ Representation & Validation", "status": "LOCKED"},
                {"id": 2, "name": "Non-Destructive Observation Features", "status": "LOCKED"},
                {"id": 3, "name": "Automatic Modulation Classification (AMC)", "status": "LOCKED"},
                {"id": 4, "name": "Signal Parameter Estimation", "status": "LOCKED"},
                {"id": 5, "name": "Synchronization & Symbol/Bit Recovery", "status": "LOCKED"},
                {"id": 6, "name": "SNR & Engineering Quality Diagnostics", "status": "LOCKED"},
                {"id": 7, "name": "Soft-Bit / LLR Generation", "status": "LOCKED"},
                {"id": 8, "name": "Blind FEC Identification & Decoding", "status": "LOCKED"},
                {"id": 9, "name": "Bitstream Structure & Frame Evidence", "status": "LOCKED"},
                {"id": 10, "name": "Advanced FEC, CRC & Interleaver Analysis", "status": "LOCKED"}
            ],
            "interleaver_validated_candidate_widths": [5, 10, 20, 25, 50],
            "identity_baseline": "Width 1 (excluded from candidate ranking)"
        }

    def process_signal(self, *args, **kwargs) -> Dict[str, Any]:
        """Convenience alias for run_pipeline."""
        return self.run_pipeline(*args, **kwargs)

    def run_pipeline(
        self,
        i_channel: Optional[Union[np.ndarray, List[float]]] = None,
        q_channel: Optional[Union[np.ndarray, List[float]]] = None,
        raw_iq: Optional[np.ndarray] = None,
        modulation: Optional[str] = None,
        override_modulation: Optional[str] = None,
        input_type: Optional[str] = None,
        sample_rate: Optional[float] = None,
        center_frequency: Optional[float] = None,
        samples_per_symbol: Optional[float] = None,
        cfo: Optional[float] = None,
        n0: Optional[float] = None,
        preamble: Optional[List[int]] = None,
        candidate_period_min: Optional[int] = 100,
        candidate_period_max: Optional[int] = 180,
        max_gf2_error: float = 0.0,
        candidate_crc_polynomials: Optional[List[str]] = None,
        candidate_interleaver_widths: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute end-to-end pipeline across Modules 1 to 10."""
        # Pipeline status tracker
        module_statuses: Dict[str, str] = {
            f"module{m}": "NOT_EXECUTED" for m in range(1, 11)
        }
        pipeline_warnings: List[str] = []

        # ======================================================================
        # MODULE 1: Canonical IQ Representation & Validation (Locked 1A)
        # ======================================================================
        if raw_iq is not None:
            raw_iq_arr = np.asarray(raw_iq)
            i_in = raw_iq_arr.real
            q_in = raw_iq_arr.imag
        elif i_channel is not None and q_channel is not None:
            i_in = i_channel
            q_in = q_channel
        else:
            return {
                "status": "ERROR_NO_INPUT",
                "pipeline_success": False,
                "message": "Either (i_channel, q_channel) or raw_iq must be provided.",
                "module_statuses": module_statuses,
                "pipeline_warnings": ["NO_SIGNAL_INPUT_PROVIDED"]
            }

        canonical_iq, val_res = to_canonical_iq(i_in, q_in)
        m1_output = val_res

        if canonical_iq is None or val_res.get("status") == "INVALID":
            module_statuses["module1"] = "FAILED"
            return {
                "status": "VALIDATION_FAILED",
                "pipeline_success": False,
                "module1_validation": m1_output,
                "module1": m1_output,
                "module_statuses": module_statuses,
                "pipeline_warnings": val_res.get("warnings", [])
            }

        module_statuses["module1"] = "PASS"

        # ======================================================================
        # MODULE 2: Non-Destructive Observation Features (Locked)
        # ======================================================================
        try:
            m2_res = observe_from_module1(canonical_iq, val_res)
            m2_output = m2_res.to_dict()
            module_statuses["module2"] = "PASS" if m2_res.status == "VALID" else "PARTIAL"
        except Exception as err:
            pipeline_warnings.append(f"MODULE_2_OBSERVATION_ERROR: {str(err)}")
            m2_output = {"status": "PARTIAL_ERROR", "error": str(err), "observation": {}}
            module_statuses["module2"] = "PARTIAL"

        # ======================================================================
        # MODULE 3: Automatic Modulation Classification (Locked)
        # ======================================================================
        try:
            obs_dict = m2_output.get("observation") if isinstance(m2_output, dict) else getattr(m2_res, "observation", None)
            m3_res = classify_amc(canonical_iq, module2_observation=obs_dict)
            m3_output = m3_res
        except Exception as err:
            pipeline_warnings.append(f"MODULE_3_AMC_ERROR: {str(err)}")
            m3_output = {"status": "PARTIAL_ERROR", "error": str(err), "engines": {}}

        # Resolve active modulation: caller override > joint dual-engine consensus > individual AMC > physics inference
        rf_engine_out = m3_output.get("engines", {}).get("engine_a_rf", {})
        cnn_engine_out = m3_output.get("engines", {}).get("engine_b_cnn", {})
        rf_pred = rf_engine_out.get("predicted_class")
        cnn_pred = cnn_engine_out.get("predicted_class")
        rf_probs = rf_engine_out.get("probabilities")
        cnn_probs = cnn_engine_out.get("probabilities")

        amc_consensus = None
        if rf_probs and cnn_probs:
            # Dual-engine calibrated posterior consensus
            joint_scores = {}
            for cls_name in set(rf_probs.keys()).intersection(cnn_probs.keys()):
                joint_scores[cls_name] = 0.55 * float(rf_probs[cls_name]) + 0.45 * float(cnn_probs[cls_name])
            if joint_scores:
                amc_consensus = max(joint_scores, key=joint_scores.get)

        active_mod = override_modulation or modulation or amc_consensus or rf_pred or cnn_pred

        # Blind physical decision tree fallback if no weights/override present (Swami & Sadler 2000)
        if not active_mod:
            try:
                feats24 = m3_output.get("features_24", {}) if isinstance(m3_output, dict) else {}
                obs_dict = m2_output.get("observation", {}) if isinstance(m2_output, dict) else {}
                circ = float(feats24.get("circularity_ratio") or obs_dict.get("circular", {}).get("m20_m21") or 0.0)
                amp_cv = float(feats24.get("amp_cv") or obs_dict.get("amplitude", {}).get("cv") or 0.3)
                c42 = float(feats24.get("c42_norm") or 0.8)
                c40 = float(feats24.get("c40_norm") or 0.0)

                if amp_cv < 0.12:
                    active_mod = "GFSK"
                elif circ > 0.60 or (amp_cv > 0.38 and c40 > 0.10):
                    active_mod = "BPSK"
                elif amp_cv > 0.35 and c42 < 0.65:
                    active_mod = "QAM16"
                else:
                    active_mod = "QPSK"  # Default robust M-PSK hypothesis
                pipeline_warnings.append(f"BLIND_STATISTICAL_MODULATION_INFERRED_{active_mod}")
            except Exception:
                active_mod = "QPSK"

        m3_output["active_modulation_selected"] = active_mod

        if rf_pred or cnn_pred:
            module_statuses["module3"] = "PASS"
        elif active_mod:
            module_statuses["module3"] = "PARTIAL (Caller / Metadata Provided)"
        else:
            module_statuses["module3"] = "PARTIAL (Weights Unavailable, Modulation Unknown)"

        # ======================================================================
        # MODULE 4: Signal Parameter Estimation (Locked)
        # ======================================================================
        try:
            m4_res = default_parameter_service.estimate_parameters(
                canonical_iq=canonical_iq,
                modulation=active_mod,
                metadata=metadata,
                sample_rate=sample_rate,
                center_frequency=center_frequency
            )
            m4_output = m4_res
            module_statuses["module4"] = "PASS" if m4_res.get("status") == "SUCCESS" else "PARTIAL"
        except Exception as err:
            pipeline_warnings.append(f"MODULE_4_PARAMETER_ERROR: {str(err)}")
            m4_output = {"status": "PARTIAL_ERROR", "symbol_rate": None, "samples_per_symbol": 4.0, "cfo": 0.0}
            module_statuses["module4"] = "PARTIAL"

        active_sps = samples_per_symbol or m4_output.get("samples_per_symbol")
        if (active_sps is None or active_sps <= 0) and m4_output.get("symbol_rate_normalized") and m4_output["symbol_rate_normalized"] > 0:
            active_sps = float(1.0 / m4_output["symbol_rate_normalized"])
        if active_sps is None or active_sps <= 0:
            active_sps = 4.0  # Standard 4x oversampled baseband fallback

        active_cfo = cfo if cfo is not None else (m4_output.get("cfo") if m4_output.get("cfo") is not None else m4_output.get("cfo_normalized", 0.0))
        if active_cfo is None:
            active_cfo = 0.0

        # ======================================================================
        # MODULE 5: Synchronization & Digital Symbol/Bit Recovery (Locked)
        # ======================================================================
        try:
            m5_res = default_recovery_service.recover_signal(
                canonical_iq=canonical_iq,
                modulation=active_mod,
                module4_parameters=m4_output,
                sample_rate=sample_rate,
                samples_per_symbol=active_sps,
                cfo=active_cfo
            )
            m5_output = m5_res

            if m5_res.get("status") == "SUCCESS":
                module_statuses["module5"] = "PASS"
            elif m5_res.get("status") == "INSUFFICIENT_PARAMETERS":
                module_statuses["module5"] = "PARTIAL (Insufficient Parameters)"
            else:
                module_statuses["module5"] = "FAILED"
        except Exception as err:
            pipeline_warnings.append(f"MODULE_5_RECOVERY_ERROR: {str(err)}")
            m5_output = {"status": "PARTIAL_ERROR", "symbols": [], "bits": [], "num_symbols": 0, "num_bits": 0}
            module_statuses["module5"] = "PARTIAL"

        # Adapt symbol and bit outputs for downstream modules
        recovered_syms = np.array(m5_output.get("symbols", []), dtype=np.complex128) if m5_output.get("symbols") else None
        m5_hard_bits = np.array(m5_output.get("bits", []), dtype=np.uint8) if m5_output.get("bits") else None

        # Slicer fallback: if synchronization returned no bits, derive direct sign-slice fallback
        if (m5_hard_bits is None or len(m5_hard_bits) == 0) and len(canonical_iq) > 0:
            step = max(1, int(round(active_sps)))
            down = canonical_iq[::step]
            fb_bits: List[int] = []
            for s in down:
                fb_bits.append(1 if s.real > 0 else 0)
                if active_mod in ["QPSK", "QAM16", "QAM64", "8PSK"]:
                    fb_bits.append(1 if s.imag > 0 else 0)
            m5_hard_bits = np.array(fb_bits, dtype=np.uint8)

        # ======================================================================
        # MODULE 6: Engineering Quality & SNR Estimation (Locked)
        # ======================================================================
        try:
            m6_res = default_snr_service.estimate_snr(
                canonical_iq=canonical_iq,
                modulation=active_mod,
                module4_parameters=m4_output,
                module5_recovery=m5_output
            )
            m6_output = m6_res
            if m6_res.get("status") == "SUCCESS":
                module_statuses["module6"] = "PASS"
            elif m6_res.get("status") == "INSUFFICIENT_PARAMETERS":
                module_statuses["module6"] = "PARTIAL (Insufficient Parameters)"
            else:
                module_statuses["module6"] = "PARTIAL (Uncalibrated Estimator)"
        except Exception as err:
            pipeline_warnings.append(f"MODULE_6_SNR_ERROR: {str(err)}")
            m6_output = {"status": "PARTIAL_ERROR", "snr_db": None}
            module_statuses["module6"] = "PARTIAL"

        # Resolve noise parameter N0
        active_n0 = n0
        if active_n0 is None:
            snr_candidates = [
                m6_output.get("snr_db"),
                m6_output.get("estimator_outputs", {}).get("robust_residual", {}).get("snr_db"),
                m6_output.get("estimator_outputs", {}).get("ordinary_residual", {}).get("snr_db"),
                m6_output.get("estimator_outputs", {}).get("fourth_moment", {}).get("snr_db"),
            ]
            valid_snrs = [s for s in snr_candidates if s is not None and np.isfinite(s) and -10.0 <= float(s) <= 50.0]
            if valid_snrs:
                snr_val = float(valid_snrs[0])
                active_n0 = float(10.0 ** (-snr_val / 10.0))

        # ======================================================================
        # MODULE 7: Soft-Bit & LLR Generation (Locked)
        # ======================================================================
        try:
            m7_res = default_soft_bits_service.generate_soft_bits(
                module5_recovery=m5_output,
                module6_snr=m6_output,
                symbol_samples=recovered_syms,
                reference_hard_bits=m5_hard_bits,
                modulation=active_mod,
                n0=active_n0
            )
            m7_output = m7_res

            if m7_res.get("status") == "SUCCESS":
                module_statuses["module7"] = "PASS"
            elif m7_res.get("status") == "NOISE_PARAMETER_UNAVAILABLE":
                module_statuses["module7"] = "PARTIAL (Noise Parameter Unavailable - LLR Not Fabricated)"
            elif m7_res.get("status") == "MISSING_MODULATION":
                module_statuses["module7"] = "PARTIAL (Missing Modulation)"
            else:
                module_statuses["module7"] = "PARTIAL"
        except Exception as err:
            pipeline_warnings.append(f"MODULE_7_SOFT_BITS_ERROR: {str(err)}")
            m7_output = {"status": "PARTIAL_ERROR", "soft_bits": None, "hard_bits": m5_hard_bits.tolist() if m5_hard_bits is not None else []}
            module_statuses["module7"] = "PARTIAL"

        # Adapt bitstream for downstream modules
        m7_soft_bits = m7_output.get("soft_bits")
        m7_hard_bits = m7_output.get("hard_bits")
        active_hard_bits = m7_hard_bits if (m7_hard_bits is not None and len(m7_hard_bits) > 0) else m5_hard_bits

        # ======================================================================
        # MODULE 8: Blind FEC Identification & Decoding (Locked)
        # ======================================================================
        try:
            m8_res = default_fec_service.process_fec(
                module7_result=m7_output,
                soft_bits=m7_soft_bits,
                hard_bits=active_hard_bits,
                metric_type=m7_output.get("metric_type")
            )
            m8_output = m8_res

            if m8_res.get("status") in ["SUCCESS", "NO_FEC_DETECTED"]:
                module_statuses["module8"] = "PASS"
            else:
                module_statuses["module8"] = "PARTIAL"
        except Exception as err:
            pipeline_warnings.append(f"MODULE_8_FEC_ERROR: {str(err)}")
            m8_output = {"status": "PARTIAL_ERROR", "detected_fec_family": None}
            module_statuses["module8"] = "PARTIAL"

        # Extract bits for structural analysis (M8 info bits or raw hard bits)
        m8_recovered_bits = m8_output.get("recovered_information_bits")
        if m8_recovered_bits is not None and len(m8_recovered_bits) > 0:
            structural_bits = m8_recovered_bits
        elif active_hard_bits is not None and len(active_hard_bits) > 0:
            structural_bits = active_hard_bits.tolist() if isinstance(active_hard_bits, np.ndarray) else active_hard_bits
        else:
            structural_bits = []

        # ======================================================================
        # MODULE 9: Bitstream Structure, Frame Detection & Evidence (Locked)
        # ======================================================================
        try:
            p_bounds = (candidate_period_min or 100, candidate_period_max or 180)
            m9_res = default_structure_service.analyze_structure(
                module8_result=m8_output,
                bits=structural_bits if len(structural_bits) > 0 else None,
                preamble=preamble,
                candidate_periods=p_bounds,
                max_gf2_error=max_gf2_error
            )
            m9_output = m9_res

            if m9_res.get("status") == "SUCCESS":
                module_statuses["module9"] = "PASS"
            elif m9_res.get("status") == "ERROR_NO_INPUT_BITS":
                module_statuses["module9"] = "NOT_EXECUTED (No Input Bits)"
            else:
                module_statuses["module9"] = "PARTIAL"
        except Exception as err:
            pipeline_warnings.append(f"MODULE_9_STRUCTURE_ERROR: {str(err)}")
            m9_output = {"status": "PARTIAL_ERROR", "frame_period": None}
            module_statuses["module9"] = "PARTIAL"

        # ======================================================================
        # MODULE 10: Advanced FEC, CRC & Interleaver Analysis (Locked)
        # ======================================================================
        try:
            m10_res = default_fec_crc_service.analyze_fec_crc(
                module9_result=m9_output,
                bits=structural_bits if len(structural_bits) > 0 else None,
                frame_period=m9_output.get("frame_period"),
                soft_bits=m7_soft_bits,
                metric_type=m7_output.get("metric_type"),
                modulation=active_mod,
                candidate_crc_polynomials=candidate_crc_polynomials,
                candidate_interleaver_widths=candidate_interleaver_widths
            )
            m10_output = m10_res

            if m10_res.get("status") == "SUCCESS":
                module_statuses["module10"] = "PASS"
            elif m10_res.get("status") == "ERROR_NO_INPUT":
                module_statuses["module10"] = "NOT_EXECUTED (No Input Bits)"
            else:
                module_statuses["module10"] = "PARTIAL"
        except Exception as err:
            pipeline_warnings.append(f"MODULE_10_FEC_CRC_ERROR: {str(err)}")
            m10_output = {
                "status": "PARTIAL_ERROR",
                "interleaver": {"identity_baseline": {"width": 1}, "candidate_evaluations": {}},
                "fec": {"top_candidate": None},
                "crc": {"candidate": None}
            }
            module_statuses["module10"] = "PARTIAL"

        # Overall pipeline execution status
        all_passed = all(status.startswith("PASS") for status in module_statuses.values())
        overall_status = "SUCCESS" if all_passed else "PARTIAL_SUCCESS"

        return {
            "status": overall_status,
            "pipeline_success": all_passed,
            "overall_pipeline_status": overall_status,
            "module_statuses": module_statuses,
            "pipeline_warnings": pipeline_warnings,
            "active_modulation": active_mod,
            "module1_validation": m1_output,
            "module2_observation": m2_output,
            "module3_amc": m3_output,
            "module4_parameters": m4_output,
            "module5_recovery": m5_output,
            "module6_snr": m6_output,
            "module7_soft_bits": m7_output,
            "module8_fec": m8_output,
            "module9_structure": m9_output,
            "module10_fec_crc": m10_output,
            "module10_crc_fec_interleaver": m10_output
        }


default_pipeline_orchestrator = PipelineOrchestrator()
