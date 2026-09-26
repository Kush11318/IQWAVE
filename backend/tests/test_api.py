"""Integration Tests for Module 1 FastAPI Endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_api_status_endpoint():
    """Verify status reports 1A as LOCKED, and 1B/1C/1D as NOT_YET_VALIDATED."""
    response = client.get("/api/module1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["module_1a_status"] == "LOCKED"
    assert data["module_1b_status"] == "NOT_YET_VALIDATED"
    assert data["module_1c_status"] == "NOT_YET_VALIDATED"
    assert data["module_1d_status"] == "NOT_YET_VALIDATED"


def test_api_baseline_radioml():
    """Verify endpoint reproduces validated RadioML2016.10a QPSK @ 18 dB baseline."""
    response = client.get("/api/module1/baseline-radioml")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "VALID"
    assert data["num_samples"] == 128
    assert data["nan_count"] == 0
    assert data["inf_count"] == 0
    assert pytest.approx(data["power"], rel=1e-3) == 6.103946e-05
    assert pytest.approx(data["diagnostics"]["rms"], rel=1e-3) == 0.007812776


def test_api_validate_valid_payload():
    """Test validating valid numerical arrays through the API."""
    payload = {
        "i_channel": [0.1, -0.2, 0.3],
        "q_channel": [0.4, 0.5, -0.6],
        "metadata": {"sample_rate": 1000000.0}
    }
    response = client.post("/api/module1/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "VALID"
    assert data["num_samples"] == 3
    assert data["metadata"]["sample_rate"] == 1000000.0


def test_api_validate_failure_cases():
    """Test failure cases via the API."""
    # Zero power
    res = client.post("/api/module1/validate", json={
        "i_channel": [0.0, 0.0],
        "q_channel": [0.0, 0.0]
    })
    assert res.json()["status"] == "INVALID"
    assert "ZERO_POWER" in res.json()["warnings"]

    # Length mismatch
    res = client.post("/api/module1/validate", json={
        "i_channel": [0.1],
        "q_channel": [0.1, 0.2]
    })
    assert res.json()["status"] == "INVALID"
    assert "I_Q_LENGTH_MISMATCH" in res.json()["warnings"]


def test_api_file_upload_wav_handling():
    """Verify uploading a WAV file returns NOT_YET_VALIDATED explicitly."""
    files = {"file": ("test_capture.wav", b"RIFFfakeWAVEfmt ", "audio/wav")}
    response = client.post("/api/module1/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NOT_YET_VALIDATED"
    assert data["module"] == "1B"
    assert "MODULE_1B_NOT_YET_VALIDATED" in data["warnings"]


def test_api_file_upload_raw_iq_handling():
    """Verify uploading raw .iq binary returns NOT_YET_VALIDATED explicitly."""
    files = {"file": ("capture.iq", b"\x00\x01\x02\x03", "application/octet-stream")}
    response = client.post("/api/module1/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NOT_YET_VALIDATED"
    assert data["module"] == "1C"
    assert "MODULE_1C_NOT_YET_VALIDATED" in data["warnings"]


def test_api_file_upload_unsupported_handling():
    """Verify uploading unsupported file type returns UNSUPPORTED explicitly."""
    files = {"file": ("notes.txt", b"plain text", "text/plain")}
    response = client.post("/api/module1/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UNSUPPORTED"
    assert "UNSUPPORTED_FILE_FORMAT" in data["warnings"]


def test_api_module2_status():
    """Verify Module 2 status endpoint returns component states."""
    response = client.get("/api/module2/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "LOCKED"
    assert data["components"]["differential_circular_variance"] == "CORE"
    assert data["components"]["blind_wav_hilbert"] == "REJECTED"


def test_api_module2_observe_valid():
    """Verify Module 2 observe endpoint extracts full observation vector."""
    payload = {
        "i_channel": [0.5, -0.2, 0.8, -0.4, 0.3],
        "q_channel": [0.1, -0.4, 0.3, 0.5, -0.2]
    }
    response = client.post("/api/module2/observe", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "VALID"
    obs = data["observation"]
    assert "amplitude" in obs
    assert "phase" in obs
    assert "circular" in obs
    assert "spectral" in obs
    assert "quality" in obs
    assert obs["quality"]["observation_length"] == 5
    assert obs["circular"]["variance"] is not None


def test_api_module3_status():
    """Verify Module 3 status endpoint returns dual engine details and benchmarks."""
    response = client.get("/api/module3/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "LOCKED"
    assert data["feature_count"] == 24
    assert len(data["supported_classes"]) == 7
    assert data["engine_b_cnn"]["parameter_count"] == 101319
    assert data["fusion_status"] == "NOT_YET_VALIDATED"


def test_api_module3_classify_pipeline():
    """Verify Module 3 classify endpoint integrates Module 1, Module 2, and dual AMC engines."""
    payload = {
        "i_channel": [0.5, -0.2, 0.8, -0.4, 0.3, 0.2, -0.1, 0.4] * 16,
        "q_channel": [0.1, -0.4, 0.3, 0.5, -0.2, -0.3, 0.2, 0.1] * 16,
        "estimated_snr": 12.0
    }
    response = client.post("/api/module3/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["fusion_status"] == "NOT_YET_VALIDATED"
    assert "features_24" in data
    assert len(data["features_24"]) == 24
    assert "engines" in data
    assert "evidence" in data


def test_api_module4_status():
    """Verify Module 4 status endpoint returns estimator statuses and limitations."""
    response = client.get("/api/module4/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "LOCKED"
    assert "symbol_rate" in data["estimators"]
    assert "psk_cfo" in data["estimators"]
    assert len(data["limitations"]) > 0


def test_api_module4_estimate_endpoint():
    """Verify Module 4 estimate endpoint runs estimation pipeline."""
    payload = {
        "i_channel": [0.5, -0.2, 0.8, -0.4, 0.3, 0.2, -0.1, 0.4] * 16,
        "q_channel": [0.1, -0.4, 0.3, 0.5, -0.2, -0.3, 0.2, 0.1] * 16,
        "modulation": "QPSK",
        "sample_rate": 1000000.0,
        "center_frequency": 433000000.0
    }
    response = client.post("/api/module4/estimate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "symbol_rate" in data
    assert "occupied_bandwidth" in data
    assert "cfo" in data
    assert "confidence" in data
    assert data["metadata"]["sample_rate"] == 1000000.0


def test_api_module4_estimate_without_modulation():
    """Verify Module 4 handles missing modulation without inventing one."""
    payload = {
        "i_channel": [0.5, -0.2, 0.8, -0.4, 0.3, 0.2, -0.1, 0.4] * 16,
        "q_channel": [0.1, -0.4, 0.3, 0.5, -0.2, -0.3, 0.2, 0.1] * 16,
        "sample_rate": 1000000.0
    }
    response = client.post("/api/module4/estimate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["modulation"] is None
    assert "MODULATION_BRANCH_UNKNOWN_OR_UNAVAILABLE" in data["flags"]


def test_api_module5_status():
    """Verify Module 5 status endpoint returns pipeline stages and boundaries."""
    response = client.get("/api/module5/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "LOCKED"
    assert "carrier_sync" in data["pipeline_stages"]
    assert "timing_sync" in data["pipeline_stages"]
    assert len(data["scientific_boundaries"]) > 0


def test_api_module5_recover():
    """Verify Module 5 recovery endpoint returns symbols, bits, and synchronization diagnostics."""
    payload = {
        "i_channel": [0.5, -0.2, 0.8, -0.4, 0.3, 0.2, -0.1, 0.4] * 16,
        "q_channel": [0.1, -0.4, 0.3, 0.5, -0.2, -0.3, 0.2, 0.1] * 16,
        "modulation": "QPSK",
        "sample_rate": 1000000.0,
        "samples_per_symbol": 4.0
    }
    response = client.post("/api/module5/recover", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["num_symbols"] > 0
    assert data["num_bits"] > 0
    assert "synchronization" in data
    assert "quality" in data


def test_api_module6_status():
    """Verify Module 6 status endpoint returns estimator mapping and scientific boundaries."""
    response = client.get("/api/module6/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "LOCKED"
    assert "estimator_mapping" in data
    assert "rejected_estimators" in data
    assert "scientific_boundaries" in data
    assert "total_power" in data["rejected_estimators"]
    assert "fsk_frequency_residual" in data["rejected_estimators"]


def test_api_module6_estimate():
    """Verify Module 6 estimate endpoint executes pipeline and returns full contract."""
    payload = {
        "i_channel": [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0] * 32,
        "q_channel": [0.0] * 256,
        "modulation": "BPSK",
        "sample_rate": 1000000.0,
        "samples_per_symbol": 4.0
    }
    response = client.post("/api/module6/estimate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["modulation"] == "BPSK"
    assert data["estimator_used"] == "BPSK_NDA_ML"
    assert data["estimator_status"] == "CONDITIONAL"
    assert "estimator_outputs" in data
    assert "disagreement" in data
    assert "quality" in data
    assert "warnings" in data


def test_api_module7_status():
    """Verify Module 7 status endpoint returns algorithm mappings and scientific rules."""
    response = client.get("/api/module7/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "LOCKED"
    assert "algorithms" in data
    assert "rejected_baselines" in data
    assert "scientific_rules" in data
    assert len(data["supported_classes"]) == 7
    assert data["algorithms"]["BPSK"]["metric_type"] == "LLR"
    assert data["algorithms"]["CPFSK"]["metric_type"] == "SOFT_METRIC"


def test_api_module7_soft_bits_direct():
    """Verify Module 7 soft_bits endpoint given direct symbol samples."""
    payload = {
        "symbol_i": [1.0, -1.0, 1.0, -1.0],
        "symbol_q": [0.0, 0.0, 0.0, 0.0],
        "modulation": "BPSK",
        "n0": 0.1
    }
    response = client.post("/api/module7/soft_bits", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["metric_type"] == "LLR"
    assert data["mathematical_status"] == "EXACT_UNDER_AWGN_MODEL"
    assert data["calibrated_llr"] is False
    assert len(data["soft_bits"]) == 4
    assert data["hard_bits"] == [1, 0, 1, 0]
    assert "reliability_summary" in data


def test_api_module7_missing_noise():
    """Verify Module 7 soft_bits endpoint cleanly rejects missing noise for PSK."""
    payload = {
        "symbol_i": [1.0, -1.0],
        "symbol_q": [0.0, 0.0],
        "modulation": "BPSK"
    }
    response = client.post("/api/module7/soft_bits", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NOISE_PARAMETER_UNAVAILABLE"
    assert data["soft_bits"] is None


def test_api_module8_status():
    """Verify Module 8 status endpoint reports LOCKED and validated/rejected scope."""
    response = client.get("/api/module8/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "LOCKED"
    assert data["module"] == "MODULE_8_BLIND_FEC_AND_INTERLEAVER"
    assert "HAMMING" in data["validated_families"]
    assert "BCH" in data["validated_families"]
    assert "CONVOLUTIONAL" in data["validated_families"]
    assert "INTERLEAVER" in data["validated_families"]
    assert len(data["rejected_approaches"]) >= 4
    assert data["boundary"] == "MODULE_8_CONCLUDED_AT_FEC_AND_RECOVERED_BITS"


def test_api_module8_decode_direct_hamming():
    """Verify Module 8 decode endpoint processes direct Hamming bitstream."""
    from backend.modules.module8.hamming import encode_hamming_7_4
    import numpy as np

    np.random.seed(42)
    msg = [1, 0, 1, 1, 0, 1, 0, 0] * 10
    coded = encode_hamming_7_4(np.array(msg, dtype=np.uint8)).tolist()

    payload = {
        "hard_bits": coded,
        "ground_truth_info_bits": msg
    }
    response = client.post("/api/module8/decode", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["detected_fec_family"] == "HAMMING"
    assert data["code_parameters"]["code"] == "HAMMING_7_4"
    assert data["decoder_applied"] == "HAMMING_HARD"
    assert "evidence_scores" in data
    assert data["ber_diagnostics"]["decoded_ber"] == 0.0


def test_api_module8_decode_uncoded():
    """Verify Module 8 decode endpoint reports NO_FEC_DETECTED on random stream."""
    import numpy as np
    np.random.seed(42)
    rand_bits = np.random.randint(0, 2, 1000).tolist()

    payload = {
        "hard_bits": rand_bits
    }
    response = client.post("/api/module8/decode", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NO_FEC_DETECTED"
    assert data["detected_fec_family"] == "UNCODED"


def test_api_module8_decode_insufficient_input():
    """Verify Module 8 decode endpoint rejects empty payload."""
    response = client.post("/api/module8/decode", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ERROR"
    assert "INSUFFICIENT_INPUT" in data["error"]


def test_api_module9_status():
    """Verify Module 9 status endpoint."""
    response = client.get("/api/module9/status")
    assert response.status_code == 200
    data = response.json()
    assert data["module"] == "MODULE_9_BITSTREAM_STRUCTURE_AND_EVIDENCE"
    assert "9A_known_preamble_detection" in data["components"]
    assert "9C_1_repeated_frame_period" in data["components"]
    assert "9D_1_global_frame_phase" in data["components"]
    assert "9H_structural_evidence_map" in data["components"]


def test_api_module9_analyze_bitstream():
    """Verify Module 9 analyze endpoint with bitstream."""
    from backend.tests.test_module9 import _generate_synthetic_frames
    frames = _generate_synthetic_frames(num_frames=10, ber=0.0)
    flat_bits = frames.flatten().tolist()

    response = client.post("/api/module9/analyze", json={"bits": flat_bits})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["frame_period"] == 144
    assert data["frame_phase"] == 0
    assert data["structural_evidence_map"] is not None
    assert "regional_summary" in data["structural_evidence_map"]


def test_api_module10_status():
    """Verify Module 10 status endpoint."""
    response = client.get("/api/module10/status")
    assert response.status_code == 200
    data = response.json()
    assert data["module"] == "MODULE_10_FEC_CRC_AND_INTERLEAVER"
    assert "10A_crc16_ccitt" in data["components"]
    assert "10B_hamming_7_4" in data["components"]
    assert "10B_bch_15_7" in data["components"]
    assert "10B_reed_solomon" in data["components"]
    assert "10C_ldpc" in data["components"]
    assert "10C_convolutional" in data["components"]
    assert "10D_interleaver" in data["components"]
    assert "rejected_methods" in data


def test_api_module10_analyze_bitstream():
    """Verify Module 10 analyze endpoint with bitstream."""
    import numpy as np
    from backend.modules.module10.crc import append_crc16_ccitt
    from backend.modules.module10.hamming import encode_hamming_7_4

    rng = np.random.RandomState(42)
    data = rng.randint(0, 2, 176, dtype=np.uint8)
    frame = append_crc16_ccitt(data)
    coded = encode_hamming_7_4(frame).tolist()

    response = client.post("/api/module10/analyze", json={"bits": coded})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "fec" in data
    assert "interleaver" in data
    assert "joint_hypothesis" in data
    assert "decoded_payload" in data
