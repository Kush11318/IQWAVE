"""Module 3: AMC Evidence Layer & Fusion Interface.

Assembles independent evidence from both validated engines:
- Engine A: 24-Feature Random Forest probabilities
- Engine B: Raw-IQ 1D CNN probabilities
- Module 2: Structural and physical observations (phase concentrations, circular variance, envelope)
- Estimated SNR (if provided)

CRITICAL INVARIANTS:
1. Fusion Status: Strictly "NOT_YET_VALIDATED".
   Per the Module 3 specification, RF and CNN were benchmarked independently.
   No heuristic fusion formula (weighted average, thresholding, majority vote, SNR-switch)
   has been experimentally validated.
2. Confidence: Raw model probabilities are NOT calibrated confidence.
   Calibrated final confidence is marked "NOT_YET_DEFINED".
"""

from typing import Any, Dict, List, Optional
from .feature_extractor import MODULATION_CLASSES


class EvidenceLayerResult:
    def __init__(
        self,
        rf_result: Dict[str, Any],
        cnn_result: Dict[str, Any],
        structural_evidence: Dict[str, Any],
        estimated_snr: Optional[float] = None,
        fusion_result: Optional[Dict[str, Any]] = None,
        ensemble_cnn_result: Optional[Dict[str, Any]] = None
    ):
        self.rf_result = rf_result
        self.cnn_result = cnn_result
        self.structural_evidence = structural_evidence
        self.estimated_snr = estimated_snr
        self.fusion_result = fusion_result
        self.ensemble_cnn_result = ensemble_cnn_result

        if fusion_result and fusion_result.get("status") == "PREDICTION_SUCCESSFUL":
            self.fusion_status = "TRAINED_VALIDATED"
            self.confidence_status = "CALIBRATED"
            self.calibrated_confidence = fusion_result.get("confidence")
            self.confidence_level = fusion_result.get("confidence_level", "UNCERTAIN")
        else:
            self.fusion_status = "NOT_YET_VALIDATED"
            self.confidence_status = "NOT_YET_DEFINED"
            self.calibrated_confidence = None
            self.confidence_level = "UNCERTAIN"

    def to_dict(self) -> Dict[str, Any]:
        """Produce the standardized Module 3 output contract."""
        rf_probs = self.rf_result.get("probabilities")
        cnn_probs = self.cnn_result.get("probabilities")

        engines_dict: Dict[str, Any] = {
            "engine_a_rf": {
                "status": self.rf_result.get("status"),
                "predicted_class": self.rf_result.get("predicted_class"),
                "probabilities": rf_probs,
                "validation_benchmark": {
                    "accuracy": "69.00%",
                    "macro_f1": "69.06%"
                }
            },
            "engine_b_cnn": {
                "status": self.cnn_result.get("status"),
                "predicted_class": self.cnn_result.get("predicted_class"),
                "probabilities": cnn_probs,
                "validation_benchmark": {
                    "accuracy": "68.97%",
                    "macro_f1": "68.63%"
                }
            }
        }

        if self.ensemble_cnn_result:
            engines_dict["engine_c_cnn_ensemble"] = self.ensemble_cnn_result

        if self.fusion_result:
            engines_dict["dual_branch_fusion"] = self.fusion_result

        out: Dict[str, Any] = {
            "fusion_status": self.fusion_status,
            "calibrated_confidence": self.calibrated_confidence,
            "confidence_status": self.confidence_status,
            "confidence_level": self.confidence_level,
            "supported_classes": MODULATION_CLASSES,
            "snr": self.estimated_snr,
            "engines": engines_dict,
            "evidence": {
                "rf_prediction": self.rf_result.get("predicted_class"),
                "cnn_prediction": self.cnn_result.get("predicted_class"),
                "structural_phase": {
                    "circular_variance": self.structural_evidence.get("circular_variance"),
                    "circular_skewness": self.structural_evidence.get("circular_skewness"),
                    "circular_kurtosis": self.structural_evidence.get("circular_kurtosis"),
                    "norm_phase_variance": self.structural_evidence.get("norm_phase_variance"),
                    "phase_diff_std": self.structural_evidence.get("phase_diff_std"),
                    "phase_m2_concentration": self.structural_evidence.get("phase_m2_concentration"),
                    "phase_m4_concentration": self.structural_evidence.get("phase_m4_concentration"),
                    "phase_m8_concentration": self.structural_evidence.get("phase_m8_concentration"),
                    "phase_entropy": self.structural_evidence.get("phase_entropy")
                },
                "structural_amplitude": {
                    "amp_cv": self.structural_evidence.get("amp_cv"),
                    "radial_iqr": self.structural_evidence.get("radial_iqr"),
                    "radial_entropy": self.structural_evidence.get("radial_entropy"),
                    "iq_eigen_ratio": self.structural_evidence.get("iq_eigen_ratio"),
                    "peak_to_avg_power_ratio": self.structural_evidence.get("peak_to_avg_power_ratio"),
                    "spectral_flatness": self.structural_evidence.get("spectral_flatness"),
                    "cyclic_freq_peak_ratio": self.structural_evidence.get("cyclic_freq_peak_ratio")
                },
                "higher_order_statistics": {
                    "circularity_ratio_m20_m21": self.structural_evidence.get("circularity_ratio"),
                    "c40_norm": self.structural_evidence.get("c40_norm"),
                    "c42_norm": self.structural_evidence.get("c42_norm"),
                    "c60_norm": self.structural_evidence.get("c60_norm"),
                    "c63_norm": self.structural_evidence.get("c63_norm")
                }
            }
        }

        if self.fusion_result and self.fusion_result.get("status") == "PREDICTION_SUCCESSFUL":
            out["predicted_class"] = self.fusion_result.get("predicted_class")
            out["fused_probabilities"] = self.fusion_result.get("probabilities")
            out["uncertainty"] = self.fusion_result.get("uncertainty")
        elif self.rf_result.get("predicted_class"):
            out["predicted_class"] = self.rf_result.get("predicted_class")

        return out

