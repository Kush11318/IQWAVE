"""
Tests for End-to-End Modules 1–10 Pipeline Integration & Final Report Generator.

CRITICAL DISTINCTION:
- RadioML baseline is the locked research baseline from Module 1.
- Synthetic fixtures in this file are clearly labeled:
  INTEGRATION TEST FIXTURE - NOT SCIENTIFIC VALIDATION
  Used solely for verifying software plumbing, data handoffs, and report generation.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.pipeline.pipeline_service import default_pipeline_orchestrator, PipelineOrchestrator
from backend.app.reporting.report_generator import default_report_generator, EngineeringReportGenerator


@pytest.fixture
def client():
    return TestClient(app)


def create_integration_test_fixture(num_symbols=300, sps=4, snr_db=20.0):
    """
    INTEGRATION TEST FIXTURE - NOT SCIENTIFIC VALIDATION.
    Creates a basic QPSK pulse-shaped waveform with known bits for testing
    software plumbing between Modules 1 and 10.
    """
    rng = np.random.RandomState(42)
    bits = rng.randint(0, 2, size=num_symbols * 2).tolist()
    
    # QPSK Gray mapping
    constellation = {
        (0, 0): complex(1, 1) / np.sqrt(2),
        (0, 1): complex(-1, 1) / np.sqrt(2),
        (1, 1): complex(-1, -1) / np.sqrt(2),
        (1, 0): complex(1, -1) / np.sqrt(2)
    }
    symbols = []
    for k in range(0, len(bits), 2):
        b_pair = (bits[k], bits[k+1])
        symbols.append(constellation[b_pair])
    symbols = np.array(symbols, dtype=np.complex128)
    
    # Upsample by sps
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    
    # Square root raised cosine pulse (simple truncated filter for plumbing)
    t = np.arange(-8, 9)
    pulse = np.sinc(t / sps)
    pulse /= np.sqrt(np.sum(pulse**2))
    tx_signal = np.convolve(upsampled, pulse, mode='same')
    
    # Add AWGN
    sig_pwr = np.mean(np.abs(tx_signal)**2)
    noise_pwr = sig_pwr / (10**(snr_db / 10.0))
    noise = (rng.normal(0, np.sqrt(noise_pwr / 2), len(tx_signal)) +
             1j * rng.normal(0, np.sqrt(noise_pwr / 2), len(tx_signal)))
    rx_signal = tx_signal + noise
    
    return rx_signal.real.tolist(), rx_signal.imag.tolist(), bits


class TestPipelineOrchestrator:
    """Test suite for PipelineOrchestrator software plumbing."""

    def test_pipeline_status(self):
        orchestrator = PipelineOrchestrator()
        status = orchestrator.get_pipeline_status()
        assert status["total_modules"] == 10
        assert status["pipeline_version"] == "1.0.0-locked-modules-1-10"
        for m in status["modules"]:
            assert m["status"] in ("LOCKED", "VALIDATED", "EXPERIMENTAL_RECONCILED")

    def test_pipeline_zero_power_rejection(self):
        orchestrator = PipelineOrchestrator()
        result = orchestrator.process_signal(
            i_channel=[0.0] * 128,
            q_channel=[0.0] * 128,
            input_type="CUSTOM_IQ"
        )
        assert result["pipeline_success"] is False
        assert result["module_statuses"]["module1"] == "FAILED"
        assert result["module_statuses"]["module2"] == "NOT_EXECUTED"
        assert result["module1_validation"]["status"] == "INVALID"

    def test_pipeline_empty_rejection(self):
        orchestrator = PipelineOrchestrator()
        result = orchestrator.process_signal(
            i_channel=[],
            q_channel=[],
            input_type="CUSTOM_IQ"
        )
        assert result["pipeline_success"] is False
        assert result["module_statuses"]["module1"] == "FAILED"

    def test_pipeline_nan_rejection(self):
        orchestrator = PipelineOrchestrator()
        result = orchestrator.process_signal(
            i_channel=[1.0, float('nan'), 0.5],
            q_channel=[0.0, 0.0, 0.0],
            input_type="CUSTOM_IQ"
        )
        assert result["pipeline_success"] is False
        assert result["module_statuses"]["module1"] == "FAILED"

    def test_pipeline_radioml_baseline_execution(self, client):
        # Fetch actual RadioML baseline vector from Module 1
        base_resp = client.get("/api/module1/baseline-radioml")
        assert base_resp.status_code == 200
        base_data = base_resp.json()
        assert "full_samples" in base_data
        
        i_samples = base_data["full_samples"]["i"]
        q_samples = base_data["full_samples"]["q"]
        
        orchestrator = PipelineOrchestrator()
        result = orchestrator.process_signal(
            i_channel=i_samples,
            q_channel=q_samples,
            sample_rate=1000000.0,
            center_frequency=433000000.0,
            input_type="CANONICAL_IQ",
            override_modulation="QPSK",
            metadata={"source": "RadioML2016.10a_QPSK_18dB"}
        )
        
        # Modules 1–10 must all have recorded execution statuses
        statuses = result["module_statuses"]
        assert statuses["module1"] == "PASS"
        assert statuses["module2"] == "PASS"
        assert statuses["module3"].startswith("PASS") or statuses["module3"].startswith("PARTIAL")
        assert statuses["module4"] == "PASS"
        assert statuses["module5"] == "PASS"
        assert statuses["module6"] == "PASS"
        assert statuses["module7"].startswith("PASS") or statuses["module7"].startswith("PARTIAL")
        assert statuses["module8"].startswith("PASS") or statuses["module8"].startswith("PARTIAL")
        assert statuses["module9"].startswith("PASS") or statuses["module9"].startswith("PARTIAL")
        assert statuses["module10"].startswith("PASS") or statuses["module10"].startswith("PARTIAL")
        
        # Check scientific constraints on Module 10
        m10 = result["module10_crc_fec_interleaver"]
        assert m10["status"] == "SUCCESS"
        il = m10["interleaver"]
        candidates = il.get("candidate_evaluations", {})
        # Only validated candidates allowed in scientific ranking
        for cand_w in candidates.keys():
            assert cand_w in ("5", "10", "20", "25", "50", 5, 10, 20, 25, 50)
        assert il.get("identity_baseline") is not None

    def test_pipeline_integration_fixture_execution(self):
        """Execute pipeline on INTEGRATION TEST FIXTURE - NOT SCIENTIFIC VALIDATION."""
        i_samples, q_samples, _ = create_integration_test_fixture(num_symbols=400, sps=4, snr_db=22.0)
        orchestrator = PipelineOrchestrator()
        result = orchestrator.process_signal(
            i_channel=i_samples,
            q_channel=q_samples,
            sample_rate=1000000.0,
            input_type="INTEGRATION_TEST_FIXTURE_NOT_SCIENTIFIC_VALIDATION",
            override_modulation="QPSK"
        )
        assert result["status"] in ("SUCCESS", "PARTIAL_SUCCESS")
        assert result["module_statuses"]["module1"] == "PASS"
        assert result["module_statuses"]["module5"] == "PASS"
        assert result["module5_recovery"]["num_bits"] > 0
        assert result["module_statuses"]["module6"] == "PASS"
        assert result["module_statuses"]["module7"].startswith("PASS") or result["module_statuses"]["module7"].startswith("PARTIAL")


class TestReportGenerator:
    """Test suite for EngineeringReportGenerator aggregation layer."""

    def test_report_generation_from_pipeline_result(self, client):
        base_resp = client.get("/api/module1/baseline-radioml")
        base_data = base_resp.json()
        
        orchestrator = PipelineOrchestrator()
        pipeline_result = orchestrator.process_signal(
            i_channel=base_data["full_samples"]["i"],
            q_channel=base_data["full_samples"]["q"],
            sample_rate=1000000.0,
            override_modulation="QPSK",
            input_type="CANONICAL_IQ"
        )
        
        generator = EngineeringReportGenerator()
        
        # 1. Test Structured JSON Report
        json_report = generator.generate_structured_report(pipeline_result)
        assert "executive_summary" in json_report
        assert "input_signal_information" in json_report
        assert "module2_observation" in json_report
        assert "module3_amc" in json_report
        assert "module4_parameters" in json_report
        assert "module5_recovery" in json_report
        assert "module6_snr" in json_report
        assert "module7_soft_bits" in json_report
        assert "module8_fec" in json_report
        assert "module9_structure" in json_report
        assert "module10_crc_fec_interleaver" in json_report
        assert "scientific_status_and_limitations" in json_report
        assert "final_pipeline_status" in json_report
        assert "reproducibility_and_evidence" in json_report
        
        # 2. Test Markdown Report
        md_report = generator.generate_markdown_report(pipeline_result)
        assert "FINAL BLIND SIGNAL ANALYSIS ENGINEERING REPORT" in md_report
        assert "## 1. Executive Summary" in md_report
        assert "## 11. Advanced FEC, CRC & Interleaver (Module 10)" in md_report
        assert "## 12. Scientific Status & Limitations" in md_report
        assert "## 13. Pipeline Module Execution Status Table" in md_report
        
        # 3. Test HTML Report
        html_report = generator.generate_html_report(pipeline_result)
        assert "<!DOCTYPE html>" in html_report
        assert "Blind Signal Analysis System" in html_report
        assert "11. Advanced FEC, CRC & Interleaver (Module 10)" in html_report
        assert "12. Scientific Status & Limitations" in html_report

    def test_report_generation_defensive_on_empty(self):
        generator = EngineeringReportGenerator()
        empty_result = {
            "status": "FAILED",
            "pipeline_success": False,
            "module_statuses": {"module1": "FAILED"},
            "module1_validation": {"status": "INVALID", "warnings": ["ZERO_POWER_SIGNAL"]},
            "module2_observation": None,
            "module3_amc": None,
            "module4_parameters": None,
            "module5_recovery": None,
            "module6_snr": None,
            "module7_soft_bits": None,
            "module8_fec": None,
            "module9_structure": None,
            "module10_crc_fec_interleaver": None
        }
        json_report = generator.generate_structured_report(empty_result)
        assert json_report["executive_summary"]["pipeline_execution_status"] == "FAILED"
        assert json_report["final_pipeline_status"]["Module 1 (Ingestion & Validation)"] == "FAILED"
        
        md_report = generator.generate_markdown_report(empty_result)
        assert "FAILED" in md_report
        
        html_report = generator.generate_html_report(empty_result)
        assert "ZERO_POWER_SIGNAL" in html_report


class TestPipelineRoutes:
    """Test suite for FastAPI /api/pipeline/* endpoints."""

    def test_get_status_route(self, client):
        resp = client.get("/api/pipeline/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert data["pipeline_status"]["total_modules"] == 10

    def test_run_pipeline_route(self, client):
        base_resp = client.get("/api/module1/baseline-radioml")
        base_data = base_resp.json()
        
        resp = client.post("/api/pipeline/run", json={
            "i_channel": base_data["full_samples"]["i"],
            "q_channel": base_data["full_samples"]["q"],
            "sample_rate": 1000000.0,
            "override_modulation": "QPSK",
            "input_type": "CANONICAL_IQ"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["module_statuses"]["module1"] == "PASS"
        assert "module10_crc_fec_interleaver" in data

    def test_report_routes_all_formats(self, client):
        # 1. Run pipeline
        base_resp = client.get("/api/module1/baseline-radioml")
        base_data = base_resp.json()
        
        pipe_resp = client.post("/api/pipeline/run", json={
            "i_channel": base_data["full_samples"]["i"],
            "q_channel": base_data["full_samples"]["q"],
            "override_modulation": "QPSK"
        })
        pipeline_output = pipe_resp.json()
        
        # 2. JSON report
        r_json = client.post("/api/pipeline/report?format=json", json=pipeline_output)
        assert r_json.status_code == 200
        assert r_json.headers["content-type"] == "application/json"
        assert "executive_summary" in r_json.json()
        
        # 3. HTML report
        r_html = client.post("/api/pipeline/report?format=html", json=pipeline_output)
        assert r_html.status_code == 200
        assert "text/html" in r_html.headers["content-type"]
        assert "<!DOCTYPE html>" in r_html.text
        
        # 4. Markdown report
        r_md = client.post("/api/pipeline/report?format=markdown", json=pipeline_output)
        assert r_md.status_code == 200
        assert "text/markdown" in r_md.headers["content-type"]
        assert "FINAL BLIND SIGNAL ANALYSIS ENGINEERING REPORT" in r_md.text

    def test_serve_frontend_index_and_static(self, client):
        # 1. Index.html
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Blind Signal Analysis System" in resp.text
        assert "Modules 1–10" in resp.text

        # 2. Static CSS
        resp_css = client.get("/static/style.css")
        assert resp_css.status_code == 200
        assert "--accent-cyan" in resp_css.text

        # 3. Static JS
        resp_js = client.get("/static/app.js")
        assert resp_js.status_code == 200
        assert "btnRunAnalysis" in resp_js.text

    def test_pipeline_upload_json_file(self, client):
        import json as py_json
        import io
        json_content = py_json.dumps({
            "i": [0.5, -0.2, 0.8, -0.4, 0.1, 0.6, -0.7, 0.3],
            "q": [0.1, -0.4, 0.3, 0.5, -0.6, 0.2, -0.1, 0.4],
            "modulation": "QPSK"
        }).encode("utf-8")
        files = {"file": ("test_signal.json", io.BytesIO(json_content), "application/json")}
        resp = client.post("/api/pipeline/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["module_statuses"]["module1"] == "PASS"

    def test_pipeline_upload_wav_unvalidated(self, client):
        import io
        # Dummy wav bytes
        dummy_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00"
        files = {"file": ("recording.wav", io.BytesIO(dummy_wav), "audio/wav")}
        resp = client.post("/api/pipeline/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "NOT_YET_VALIDATED"
        assert data["module_statuses"]["module1"] == "NOT_YET_VALIDATED"
        assert data["module_statuses"]["module2"] == "NOT_EXECUTED"

