"""Test Suite for Module 3: Automatic Modulation Classification (AMC).

Verifies:
1. Feature extraction: exact 24 feature names, finite/safe calculations, reuse of Module 2.
2. RF engine: configuration (300 estimators, leaf 2, balanced, seed 42), 7 classes.
3. CNN preprocessing: RMS normalization, shape (128, 2), non-destructive preservation.
4. CNN architecture: 101,319 parameter count, forward pass shape (batch, 7).
5. Module integration: Module 1 -> Module 2 -> Module 3 end-to-end pipeline.
6. Missing model artifacts: explicitly returns MODEL_WEIGHTS_UNAVAILABLE.
7. Fusion interface: fusion_status is strictly NOT_YET_VALIDATED; calibrated confidence NOT_YET_DEFINED.
8. Non-destructive behavior: original IQ samples remain completely unchanged.
"""

import numpy as np
import pytest
import torch

from backend.modules.module1.canonical_iq import to_canonical_iq
from backend.modules.module2.observation import compute_observation_vector
from backend.modules.module3.feature_extractor import (
    FEATURE_NAMES_24,
    MODULATION_CLASSES,
    extract_24_features,
    feature_dict_to_vector
)
from backend.modules.module3.preprocessing import preprocess_for_cnn
from backend.modules.module3.rf_engine import RFEngine
from backend.modules.module3.cnn_engine import RawIQCNN, CNNEngine
from backend.modules.module3.evidence_layer import EvidenceLayerResult
from backend.modules.module3.service import AMCService, classify_amc


# -----------------------------------------------------------------------------
# 1. Feature Extraction (Exact 24 Features)
# -----------------------------------------------------------------------------

def test_feature_extraction_24_names_and_types():
    """Verify exact 24 feature names and clean extraction on a synthetic signal."""
    assert len(FEATURE_NAMES_24) == 24
    assert len(MODULATION_CLASSES) == 7

    rng = np.random.default_rng(42)
    # 256 samples of 16-QAM signal (multi-level envelope with variance > 0)
    qam16_symbols = (
        rng.choice([-3.0, -1.0, 1.0, 3.0], size=256) +
        1j * rng.choice([-3.0, -1.0, 1.0, 3.0], size=256)
    ).astype(np.complex64)

    feats = extract_24_features(qam16_symbols)

    for name in FEATURE_NAMES_24:
        assert name in feats, f"Missing feature: {name}"
        assert feats[name] is not None, f"Feature {name} is None on valid 16-QAM signal"

    # Verify vectorization preserves order and dimension
    vec = feature_dict_to_vector(feats)
    assert vec.shape == (24,)
    assert vec.dtype == np.float32


def test_feature_extraction_reuse_module2():
    """Verify that Module 3 feature extractor correctly reuses pre-computed Module 2 observations."""
    n = 128
    iq_signal = (np.ones(n, dtype=np.float32) + 1j * np.zeros(n, dtype=np.float32)).astype(np.complex64)
    mod2_obs = compute_observation_vector(iq_signal, {"input_representation": "COMPLEX_IQ"})

    feats = extract_24_features(iq_signal, module2_observation=mod2_obs)

    # amp_mean must match Module 2 directly
    assert feats["amp_mean"] == mod2_obs["amplitude"]["mean"]
    assert feats["circular_variance"] == mod2_obs["circular"]["variance"]
    assert feats["circularity_ratio"] == mod2_obs["circular"]["m20_m21"]


# -----------------------------------------------------------------------------
# 2. Random Forest Engine Configuration
# -----------------------------------------------------------------------------

def test_rf_engine_configuration():
    """Verify the exact validated Random Forest configuration."""
    rf = RFEngine(model_path="nonexistent_rf.joblib")
    assert rf.classes == MODULATION_CLASSES
    assert rf.config["n_estimators"] == 300
    assert rf.config["min_samples_leaf"] == 2
    assert rf.config["class_weight"] == "balanced"
    assert rf.config["random_state"] == 42

    unfitted = rf.build_unfitted_model()
    assert unfitted.n_estimators == 300
    assert unfitted.min_samples_leaf == 2
    assert unfitted.class_weight == "balanced"
    assert unfitted.random_state == 42


def test_rf_missing_weights_handling():
    """Verify that when serialized weights are absent, engine reports MODEL_WEIGHTS_UNAVAILABLE."""
    rf = RFEngine(model_path="nonexistent_rf.joblib")
    assert not rf.is_available

    dummy_feats = {name: 0.5 for name in FEATURE_NAMES_24}
    res = rf.predict(dummy_feats)
    assert res["status"] == "MODEL_WEIGHTS_UNAVAILABLE"
    assert res["predicted_class"] is None
    assert res["probabilities"] is None


# -----------------------------------------------------------------------------
# 3. CNN Preprocessing & RMS Normalization
# -----------------------------------------------------------------------------

def test_cnn_preprocessing_rms_normalization():
    """Verify CNN RMS normalization formula and non-destructive invariant."""
    n = 128
    # Constant amplitude signal with known power
    iq_signal = (2.0 * np.ones(n, dtype=np.float32) + 1j * 2.0 * np.ones(n, dtype=np.float32)).astype(np.complex64)
    copy_iq = np.copy(iq_signal)

    cnn_input, measured_power = preprocess_for_cnn(iq_signal, target_length=128)

    # 1. Output shape must be (128, 2)
    assert cnn_input.shape == (128, 2)
    assert cnn_input.dtype == np.float32

    # 2. Check channel 0 = I, channel 1 = Q
    assert np.allclose(cnn_input[:, 0], cnn_input[:, 1])

    # 3. Normalized frame power: (1 / 2N) * sum(I_norm^2 + Q_norm^2) should be approximately 1.0
    norm_power = np.mean(cnn_input[:, 0] ** 2 + cnn_input[:, 1] ** 2) / 2.0
    assert pytest.approx(norm_power, rel=1e-3) == 1.0

    # 4. Invariant: original signal array must be completely unaltered
    np.testing.assert_array_equal(iq_signal, copy_iq)


# -----------------------------------------------------------------------------
# 4. CNN Architecture & Parameter Count
# -----------------------------------------------------------------------------

def test_cnn_architecture_and_parameter_count():
    """Verify the exact 101,319 parameter count of the validated CNN architecture."""
    model = RawIQCNN(num_classes=7)
    param_summary = model.get_parameter_summary()

    # Exact research benchmark parameters:
    # 100,679 trainable weights + 640 BatchNorm moving statistics = 101,319 total parameters
    assert param_summary["trainable_parameters"] == 100679
    assert param_summary["bn_running_parameters"] == 640
    assert param_summary["total_parameters"] == 101319

    # Verify forward pass with shape (batch, 128, 2)
    batch_size = 4
    x_test = torch.randn(batch_size, 128, 2)
    model.eval()
    with torch.no_grad():
        out = model(x_test)

    assert out.shape == (batch_size, 7)
    # Softmax output probabilities must sum to 1.0 per sample
    row_sums = out.sum(dim=-1).numpy()
    np.testing.assert_allclose(row_sums, np.ones(batch_size), atol=1e-5)


def test_cnn_missing_weights_handling():
    """Verify that when CNN weights file is absent, engine reports MODEL_WEIGHTS_UNAVAILABLE."""
    cnn = CNNEngine(model_path="nonexistent_cnn.pt")
    assert not cnn.is_available

    dummy_input = np.zeros((128, 2), dtype=np.float32)
    res = cnn.predict(dummy_input)
    assert res["status"] == "MODEL_WEIGHTS_UNAVAILABLE"
    assert res["predicted_class"] is None
    assert res["probabilities"] is None
    assert res["parameter_count"] == 101319


# -----------------------------------------------------------------------------
# 5. Evidence Layer & Fusion Rules
# -----------------------------------------------------------------------------

def test_evidence_layer_unvalidated_fusion_boundary():
    """Verify that evidence layer preserves fusion_status='NOT_YET_VALIDATED' and no invented weights."""
    rf_mock = {
        "status": "PREDICTION_SUCCESSFUL",
        "predicted_class": "QPSK",
        "probabilities": {"QPSK": 0.85, "8PSK": 0.15}
    }
    cnn_mock = {
        "status": "PREDICTION_SUCCESSFUL",
        "predicted_class": "8PSK",
        "probabilities": {"8PSK": 0.70, "QPSK": 0.30}
    }
    struct_evidence = {"circular_variance": 0.12, "amp_cv": 0.05}

    layer = EvidenceLayerResult(
        rf_result=rf_mock,
        cnn_result=cnn_mock,
        structural_evidence=struct_evidence,
        estimated_snr=10.0
    )
    data = layer.to_dict()

    # Rule: Fusion must remain explicitly NOT_YET_VALIDATED
    assert data["fusion_status"] == "NOT_YET_VALIDATED"
    assert data["confidence_status"] == "NOT_YET_DEFINED"
    assert data["calibrated_confidence"] is None
    assert data["engines"]["engine_a_rf"]["predicted_class"] == "QPSK"
    assert data["engines"]["engine_b_cnn"]["predicted_class"] == "8PSK"


# -----------------------------------------------------------------------------
# 6. Pipeline Integration: Module 1 -> Module 2 -> Module 3
# -----------------------------------------------------------------------------

def test_end_to_end_module1_to_module3_pipeline():
    """Verify end-to-end integration:
    Module 1 canonical IQ -> Module 2 observation -> Module 3 AMC classification.
    """
    rng = np.random.default_rng(42)
    i_raw = [float(x) for x in rng.standard_normal(128)]
    q_raw = [float(x) for x in rng.standard_normal(128)]

    # Module 1
    canonical_iq, val_res = to_canonical_iq(i_raw, q_raw)
    assert val_res["status"] == "VALID"

    # Module 2
    mod2_obs = compute_observation_vector(canonical_iq, {"input_representation": "COMPLEX_IQ"})
    assert mod2_obs["quality"]["observation_length"] == 128

    # Module 3
    amc_result = classify_amc(
        canonical_iq=canonical_iq,
        module2_observation=mod2_obs,
        estimated_snr=18.0
    )

    assert amc_result["status"] == "SUCCESS"
    assert amc_result["fusion_status"] in ("NOT_YET_VALIDATED", "TRAINED_VALIDATED")
    assert "features_24" in amc_result
    assert len(amc_result["features_24"]) == 24
    assert "engines" in amc_result
    assert "engine_a_rf" in amc_result["engines"]
    assert "engine_b_cnn" in amc_result["engines"]
