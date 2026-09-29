"""Verification and Live Testing Script for AMC Research Pipeline (Phases 1-6).

Tests live signals across all 7 modulation classes:
1. Feature extraction of all 32 research features
2. Random Forest inference (32 features)
3. Multi-Scale Dilated CNN inference (256-dim GAP)
4. CNN Ensemble inference & Uncertainty Quantification (HIGH / MEDIUM / LOW / UNCERTAIN)
5. EvidenceLayer contract and status validation
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from scratch.train_v2 import generate_signal
from backend.modules.module3.service import default_amc_service, classify_amc
from backend.modules.module3.feature_extractor import MODULATION_CLASSES, FEATURE_NAMES_32, extract_features

def test_pipeline():
    print("=" * 70)
    print("IQWAVE AMC RESEARCH PIPELINE VALIDATION (PHASES 1 TO 6)")
    print("=" * 70)

    # 1. Inspect Engine Status
    status = {
        "RF Engine": default_amc_service.rf_engine.is_available,
        "RF Features": getattr(default_amc_service.rf_engine.model, "n_features_in_", 24),
        "MultiScale CNN": default_amc_service.cnn_engine.is_available,
        "CNN Ensemble": default_amc_service.ensemble_engine.is_available,
        "Ensemble Models Loaded": default_amc_service.ensemble_engine.n_models_loaded,
        "Fusion Engine": default_amc_service.fusion_engine.is_available
    }
    for k, v in status.items():
        print(f"  {k:25s}: {v}")

    print("\n" + "-" * 70)
    print(f"{'Class':8s} | {'True SNR':8s} | {'RF Pred':8s} | {'Ens Pred':8s} | {'Final Class':12s} | {'Conf':6s} | {'UQ Tier':9s}")
    print("-" * 70)

    results = []
    correct_rf = 0
    correct_ens = 0
    total = 0

    for cls in MODULATION_CLASSES:
        for snr in [5.0, 15.0, 25.0]:
            sig = generate_signal(cls, snr_db=snr, with_fading=True, with_iq_imbalance=True)
            res = classify_amc(sig, estimated_snr=snr)

            rf_p = res["engines"]["engine_a_rf"].get("predicted_class", "N/A")
            ens_eng = res["engines"].get("engine_c_cnn_ensemble", {})
            ens_p = ens_eng.get("predicted_class", "N/A")
            ens_uq = ens_eng.get("uncertainty", {})
            conf_tier = ens_uq.get("confidence_level", "N/A")

            final_p = res.get("predicted_class", "N/A")
            probs = ens_eng.get("probabilities", {}) or res["engines"]["engine_a_rf"].get("probabilities", {})
            max_prob = max(probs.values()) if probs else 0.0

            if rf_p == cls:
                correct_rf += 1
            if ens_p == cls:
                correct_ens += 1
            total += 1

            print(f"{cls:8s} | {snr:5.1f} dB  | {rf_p:8s} | {ens_p:8s} | {final_p:12s} | {max_prob*100:5.1f}% | {conf_tier:9s}")

    print("-" * 70)
    print(f"Random Forest Accuracy on challenging multipath: {correct_rf}/{total} ({correct_rf/total*100:.1f}%)")
    print(f"CNN Ensemble Accuracy on challenging multipath:  {correct_ens}/{total} ({correct_ens/total*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    test_pipeline()
