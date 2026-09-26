"""Module 4 Service: Signal Parameter Estimation Orchestrator.

Orchestrates:
1. Ingestion of canonical IQ from Module 1 (with metadata).
2. Branch selection driven strictly by Module 3 modulation classification.
3. Symbol rate estimation (Ciblat cyclic correlation).
4. SPS calculation (Fs / Rs when trustworthy Fs is known; otherwise UNKNOWN).
5. CFO estimation:
   - PSK: M-th power spectral
   - QAM16: Conditional weighted 4th power
   - QAM64: NOT_VALIDATED
   - FSK: Tracked via states
6. Relative center frequency and occupied bandwidth (adaptive PSD region).
7. Absolute RF frequency determination (computed if metadata is provided; otherwise UNKNOWN).
8. FSK-specific frequency state estimation (f0, f1, Delta f).
"""

from typing import Any, Dict, List, Optional
import numpy as np

from .symbol_rate import estimate_symbol_rate
from .fsk_frequency import estimate_fsk_parameters
from .cfo_estimator import estimate_cfo
from .spectral_estimator import estimate_relative_center_and_bandwidth


class ParameterEstimationService:
    def estimate_parameters(
        self,
        canonical_iq: np.ndarray,
        modulation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sample_rate: Optional[float] = None,
        center_frequency: Optional[float] = None,
        estimated_snr: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute Module 4 parameter estimation pipeline."""
        meta = metadata or {}
        fs = sample_rate if sample_rate is not None else meta.get("sample_rate")
        rf_center_meta = center_frequency if center_frequency is not None else meta.get("center_frequency")

        flags: List[str] = []

        if canonical_iq is None or len(canonical_iq) == 0:
            return {
                "status": "INVALID_INPUT",
                "error": "EMPTY_OR_UNAVAILABLE_SIGNAL",
                "flags": ["EMPTY_SIGNAL"]
            }

        if not np.all(np.isfinite(canonical_iq)):
            return {
                "status": "INVALID_INPUT",
                "error": "NON_FINITE_SAMPLES",
                "symbol_rate": None,
                "samples_per_symbol": None,
                "cfo": None,
                "relative_center_frequency": None,
                "absolute_rf_frequency": None,
                "occupied_bandwidth": None,
                "confidence": {
                    "symbol_rate": "LOW",
                    "cfo": "LOW",
                    "center_frequency": "LOW",
                    "bandwidth": "LOW",
                    "fsk_states": "NOT_APPLICABLE"
                },
                "flags": ["NON_FINITE_SAMPLES"],
                "metadata": {
                    "sample_rate": fs,
                    "center_frequency": rf_center_meta
                }
            }

        mod_upper = (modulation or "").upper()
        if not mod_upper or mod_upper in ["UNKNOWN", "NONE", "MODEL_WEIGHTS_UNAVAILABLE"]:
            flags.append("MODULATION_BRANCH_UNKNOWN_OR_UNAVAILABLE")
        is_fsk = mod_upper in ["GFSK", "CPFSK"]

        # 1. Symbol Rate & SPS (4A)
        sym_res = estimate_symbol_rate(canonical_iq, sample_rate=fs)
        flags.extend(sym_res.get("flags", []))

        # 2. Spectral Analysis: Relative Center & Bandwidth (4D)
        spec_res = estimate_relative_center_and_bandwidth(canonical_iq, sample_rate=fs)
        flags.extend(spec_res.get("flags", []))

        # Absolute RF Center Frequency calculation
        rel_center = spec_res.get("relative_center_frequency")
        if rf_center_meta is not None and rel_center is not None:
            abs_rf = float(rf_center_meta + rel_center)
        else:
            abs_rf = None
            flags.append("METADATA_REQUIRED_FOR_ABSOLUTE_RF_CENTER")

        # 3. CFO / Frequency State Analysis
        fsk_res = None
        cfo_res = None

        if is_fsk:
            # FSK branch (4B)
            fsk_res = estimate_fsk_parameters(canonical_iq, sample_rate=fs)
            flags.extend(fsk_res.get("flags", []))
            cfo_val = None
            cfo_conf = "NOT_APPLICABLE_TO_FSK"
        else:
            # PSK / QAM branch (4C)
            cfo_res = estimate_cfo(canonical_iq, modulation_type=mod_upper, sample_rate=fs)
            flags.extend(cfo_res.get("flags", []))
            cfo_val = cfo_res.get("cfo")
            cfo_conf = cfo_res.get("confidence", "UNKNOWN")

        # Assemble Output Contract
        out: Dict[str, Any] = {
            "status": "SUCCESS",
            "modulation": modulation,
            "symbol_rate": sym_res.get("symbol_rate"),
            "symbol_rate_normalized": sym_res.get("symbol_rate_normalized"),
            "samples_per_symbol": sym_res.get("samples_per_symbol"),

            "cfo": cfo_val,
            "cfo_normalized": cfo_res.get("cfo_normalized") if cfo_res else None,

            "relative_center_frequency": rel_center,
            "relative_center_normalized": spec_res.get("relative_center_normalized"),
            "absolute_rf_frequency": abs_rf,

            "occupied_bandwidth": spec_res.get("occupied_bandwidth"),
            "occupied_bandwidth_normalized": spec_res.get("occupied_bandwidth_normalized"),

            # FSK-only parameters
            "frequency_state_0": fsk_res.get("frequency_state_0") if fsk_res else None,
            "frequency_state_1": fsk_res.get("frequency_state_1") if fsk_res else None,
            "frequency_separation": fsk_res.get("frequency_separation") if fsk_res else None,

            "confidence": {
                "symbol_rate": sym_res.get("confidence", "UNKNOWN"),
                "cfo": cfo_conf,
                "center_frequency": spec_res.get("confidence", "UNKNOWN"),
                "bandwidth": spec_res.get("confidence", "UNKNOWN"),
                "fsk_states": fsk_res.get("confidence") if fsk_res else "NOT_APPLICABLE"
            },

            "estimator_methods": {
                "symbol_rate": sym_res.get("method"),
                "cfo": cfo_res.get("method") if cfo_res else "FSK_CARRIER_TRACKED_VIA_STATES",
                "center_and_bandwidth": spec_res.get("method"),
                "fsk_parameters": fsk_res.get("method") if fsk_res else "NOT_APPLICABLE"
            },

            "flags": list(set(flags)),

            "metadata": {
                "sample_rate": fs,
                "center_frequency": rf_center_meta
            }
        }
        return out


default_parameter_service = ParameterEstimationService()


def estimate_signal_parameters(
    canonical_iq: np.ndarray,
    modulation: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    sample_rate: Optional[float] = None,
    center_frequency: Optional[float] = None,
    estimated_snr: Optional[float] = None
) -> Dict[str, Any]:
    """Convenience functional interface."""
    return default_parameter_service.estimate_parameters(
        canonical_iq=canonical_iq,
        modulation=modulation,
        metadata=metadata,
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        estimated_snr=estimated_snr
    )
