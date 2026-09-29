import os
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Response, UploadFile, File, Form
from pydantic import BaseModel
import json as py_json

from backend.app.pipeline.pipeline_service import default_pipeline_orchestrator
from backend.app.reporting.report_generator import default_report_generator

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])


class PipelineRunRequest(BaseModel):
    i_channel: Optional[List[float]] = None
    q_channel: Optional[List[float]] = None
    modulation: Optional[str] = None
    override_modulation: Optional[str] = None
    input_type: Optional[str] = None
    sample_rate: Optional[float] = None
    center_frequency: Optional[float] = None
    samples_per_symbol: Optional[float] = None
    cfo: Optional[float] = None
    n0: Optional[float] = None
    preamble: Optional[List[int]] = None
    candidate_period_min: Optional[int] = 100
    candidate_period_max: Optional[int] = 180
    max_gf2_error: Optional[float] = 0.0
    candidate_crc_polynomials: Optional[List[str]] = None
    candidate_interleaver_widths: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None


@router.get("/status")
def get_pipeline_status() -> Dict[str, Any]:
    """Report end-to-end pipeline architecture and locked module statuses."""
    status_info = default_pipeline_orchestrator.get_pipeline_status()
    return {
        "status": "SUCCESS",
        "pipeline_status": status_info,
        **status_info
    }


@router.get("/fixture")
def get_integration_fixture(preset: Optional[str] = "qpsk_1200") -> Dict[str, Any]:
    """Return high-fidelity signal fixtures with realistic pulse-shaping, SNR, and modulation characteristics."""
    import numpy as np
    rng = np.random.RandomState(42)

    preset_clean = (preset or "qpsk_1200").lower()

    if preset_clean == "radioml_128":
        from backend.app.api.routes_module1 import get_radioml_baseline
        record = get_radioml_baseline()
        samples = record.get("full_samples", {})
        i_data = samples.get("i", [])
        q_data = samples.get("q", [])
        return {
            "status": "SUCCESS",
            "name": "RadioML_2016.10a_QPSK_18dB.IQ",
            "modulation": "QPSK",
            "sample_rate": 1000000.0,
            "center_frequency": 433000000.0,
            "sample_count": len(i_data),
            "i": i_data,
            "q": q_data
        }

    if preset_clean == "qam16_1600":
        n_sym, sps, snr_db, mod = 400, 4, 26.0, "QAM16"
        levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        i_idx = rng.randint(0, 4, size=n_sym)
        q_idx = rng.randint(0, 4, size=n_sym)
        symbols = levels[i_idx] + 1j * levels[q_idx]
        fs, fc = 20000000.0, 142850000.0
        name = "Avionics_16QAM_Wideband_Link_1600Sa.IQ"

    elif preset_clean == "psk8_1200":
        n_sym, sps, snr_db, mod = 300, 4, 24.0, "8PSK"
        angles = rng.randint(0, 8, size=n_sym) * (2 * np.pi / 8.0)
        symbols = np.exp(1j * angles)
        fs, fc = 10000000.0, 142850000.0
        name = "Satellite_8PSK_DeepSpace_1200Sa.IQ"

    elif preset_clean == "bpsk_1000":
        n_sym, sps, snr_db, mod = 250, 4, 20.0, "BPSK"
        bits = rng.randint(0, 2, size=n_sym)
        symbols = np.array([1.0 if b == 0 else -1.0 for b in bits], dtype=complex)
        fs, fc = 5000000.0, 142850000.0
        name = "Military_BPSK_Tactical_Burst_1000Sa.IQ"

    elif preset_clean == "fsk_1200":
        n_sym, sps, snr_db, mod = 300, 4, 22.0, "GFSK"
        bits = rng.randint(0, 2, size=n_sym) * 2 - 1
        freq_dev = 0.25 / sps
        phase = np.cumsum(np.repeat(bits, sps) * 2 * np.pi * freq_dev)
        tx = np.exp(1j * phase)
        pwr = np.mean(np.abs(tx)**2)
        noise_pwr = pwr / (10**(snr_db / 10.0))
        noise = rng.normal(0, np.sqrt(noise_pwr/2), len(tx)) + 1j * rng.normal(0, np.sqrt(noise_pwr/2), len(tx))
        rx = tx + noise
        return {
            "status": "SUCCESS",
            "name": "Continuous_Phase_GFSK_Telemetry_1200Sa.IQ",
            "modulation": mod,
            "sample_rate": 1000000.0,
            "center_frequency": 142850000.0,
            "sample_count": len(rx),
            "i": [float(x) for x in rx.real],
            "q": [float(x) for x in rx.imag]
        }

    elif preset_clean in ["radar_lfm_1200", "lfm_1200", "lfm"]:
        # Tactical Linear Frequency Modulation (LFM) Radar Chirp
        n_samples = 1200
        fs, fc, snr_db, mod = 20000000.0, 142850000.0, 22.0, "LFM"
        t = np.linspace(-0.5, 0.5, n_samples)
        k_chirp = 5e6
        tx = np.exp(1j * np.pi * k_chirp * (t ** 2))
        pwr = np.mean(np.abs(tx)**2)
        noise_pwr = pwr / (10**(snr_db / 10.0))
        noise = rng.normal(0, np.sqrt(noise_pwr/2), len(tx)) + 1j * rng.normal(0, np.sqrt(noise_pwr/2), len(tx))
        rx = tx + noise
        return {
            "status": "SUCCESS",
            "name": "Tactical_LFM_Radar_Chirp_1200Sa.IQ",
            "modulation": "LFM",
            "waveform_family": "RadChar (Linear Frequency Modulation)",
            "sample_rate": fs,
            "center_frequency": fc,
            "sample_count": len(rx),
            "i": [float(x) for x in rx.real],
            "q": [float(x) for x in rx.imag]
        }

    elif preset_clean in ["radar_barker_1300", "barker_1300", "barker"]:
        # X-Band Barker-13 Biphase-Coded Radar Pulse
        barker13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=float)
        sps = 100
        tx = np.repeat(barker13, sps).astype(complex)
        fs, fc, snr_db, mod = 10000000.0, 142850000.0, 20.0, "Barker"
        pwr = np.mean(np.abs(tx)**2)
        noise_pwr = pwr / (10**(snr_db / 10.0))
        noise = rng.normal(0, np.sqrt(noise_pwr/2), len(tx)) + 1j * rng.normal(0, np.sqrt(noise_pwr/2), len(tx))
        rx = tx + noise
        return {
            "status": "SUCCESS",
            "name": "XBand_Barker13_PhaseCoded_Radar_1300Sa.IQ",
            "modulation": "Barker",
            "waveform_family": "RadChar (Barker-13 Binary Phase Code)",
            "sample_rate": fs,
            "center_frequency": fc,
            "sample_count": len(rx),
            "i": [float(x) for x in rx.real],
            "q": [float(x) for x in rx.imag]
        }

    else: # Default: qpsk_1200
        n_sym, sps, snr_db, mod = 300, 4, 24.0, "QPSK"
        bits = rng.randint(0, 2, size=n_sym * 2)
        # Add frame sync pattern
        preamble = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0]
        for idx in range(0, len(bits) - len(preamble), 120):
            bits[idx:idx+len(preamble)] = preamble
        mapping = {
            (0, 0): (1 + 1j) / np.sqrt(2),
            (0, 1): (-1 + 1j) / np.sqrt(2),
            (1, 1): (-1 - 1j) / np.sqrt(2),
            (1, 0): (1 - 1j) / np.sqrt(2)
        }
        symbols = np.array([mapping[(bits[2*k], bits[2*k+1])] for k in range(n_sym)])
        fs, fc = 20000000.0, 142850000.0
        name = "Tactical_VHF_QPSK_Carrier_1200Sa.IQ"

    # Pulse shaping: Root-Raised Cosine (RRC, roll-off alpha = 0.35)
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    t = np.arange(-12, 13)
    pulse = np.sinc(t / sps) * np.cos(np.pi * 0.35 * t / sps) / (1.0 - (2 * 0.35 * t / sps)**2 + 1e-12)
    pulse /= np.sqrt(np.sum(pulse**2))
    tx = np.convolve(upsampled, pulse, mode='same')

    # Apply small carrier offset (+120 kHz)
    t_axis = np.arange(len(tx)) / fs
    tx_cfo = tx * np.exp(1j * 2 * np.pi * 120000.0 * t_axis)

    pwr = np.mean(np.abs(tx_cfo)**2)
    noise_pwr = pwr / (10**(snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(noise_pwr/2), len(tx)) + 1j * rng.normal(0, np.sqrt(noise_pwr/2), len(tx))
    rx = tx_cfo + noise

    i_list = [float(x) for x in rx.real]
    q_list = [float(x) for x in rx.imag]

    return {
        "status": "SUCCESS",
        "name": name,
        "modulation": mod,
        "sample_rate": float(fs),
        "center_frequency": float(fc),
        "sample_count": len(i_list),
        "i": i_list,
        "q": q_list
    }


_last_pipeline_result: Optional[Dict[str, Any]] = None


@router.post("/run")
def post_run_pipeline(req: PipelineRunRequest) -> Dict[str, Any]:
    """Execute end-to-end signal analysis pipeline through Modules 1 to 10."""
    global _last_pipeline_result
    res = default_pipeline_orchestrator.run_pipeline(
        i_channel=req.i_channel,
        q_channel=req.q_channel,
        modulation=req.modulation,
        override_modulation=req.override_modulation,
        input_type=req.input_type,
        sample_rate=req.sample_rate,
        center_frequency=req.center_frequency,
        samples_per_symbol=req.samples_per_symbol,
        cfo=req.cfo,
        n0=req.n0,
        preamble=req.preamble,
        candidate_period_min=req.candidate_period_min,
        candidate_period_max=req.candidate_period_max,
        max_gf2_error=req.max_gf2_error or 0.0,
        candidate_crc_polynomials=req.candidate_crc_polynomials,
        candidate_interleaver_widths=req.candidate_interleaver_widths,
        metadata=req.metadata
    )
    _last_pipeline_result = res
    return res


@router.post("/report")
def post_generate_report(payload: Dict[str, Any], format: str = "json") -> Any:
    """Generate final engineering report (json, html, or markdown).
    Accepts either an existing pipeline execution result dictionary or a signal run request.
    """
    if "module1_validation" in payload or "module_statuses" in payload:
        pipeline_res = payload
    else:
        pipeline_res = default_pipeline_orchestrator.run_pipeline(
            i_channel=payload.get("i_channel"),
            q_channel=payload.get("q_channel"),
            modulation=payload.get("override_modulation") or payload.get("modulation"),
            override_modulation=payload.get("override_modulation"),
            input_type=payload.get("input_type"),
            sample_rate=payload.get("sample_rate"),
            center_frequency=payload.get("center_frequency"),
            samples_per_symbol=payload.get("samples_per_symbol"),
            cfo=payload.get("cfo"),
            n0=payload.get("n0"),
            preamble=payload.get("preamble"),
            candidate_period_min=payload.get("candidate_period_min", 100),
            candidate_period_max=payload.get("candidate_period_max", 180),
            max_gf2_error=payload.get("max_gf2_error", 0.0),
            candidate_crc_polynomials=payload.get("candidate_crc_polynomials"),
            candidate_interleaver_widths=payload.get("candidate_interleaver_widths"),
            metadata=payload.get("metadata")
        )

    if format.lower() == "html":
        html_content = default_report_generator.generate_html_report(pipeline_res)
        return Response(content=html_content, media_type="text/html")
    elif format.lower() in ("markdown", "md"):
        md_content = default_report_generator.generate_markdown_report(pipeline_res)
        return Response(content=md_content, media_type="text/markdown")
    else:
        json_report = default_report_generator.generate_json_report(pipeline_res)
        return Response(content=py_json.dumps(json_report, indent=2), media_type="application/json")


@router.get("/report/{format}")
def get_pipeline_report(format: str = "json") -> Any:
    """Generate engineering report (json, html, or markdown) for the most recent pipeline run."""
    global _last_pipeline_result
    if _last_pipeline_result is None:
        fixture_data = get_integration_fixture("qpsk_1200")
        _last_pipeline_result = default_pipeline_orchestrator.run_pipeline(
            i_channel=fixture_data.get("i"),
            q_channel=fixture_data.get("q"),
            sample_rate=fixture_data.get("sample_rate"),
            center_frequency=fixture_data.get("center_frequency")
        )
    return post_generate_report(_last_pipeline_result, format=format)


@router.post("/upload")
async def post_upload_pipeline(
    file: UploadFile = File(...),
    sample_rate: Optional[float] = Form(None),
    center_frequency: Optional[float] = Form(None),
    override_modulation: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """Ingest a signal file (.json, .wav, .iq, .bin, .dat) and run the end-to-end pipeline."""
    content = await file.read()
    filename = file.filename or "uploaded_signal"
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".json":
        try:
            data = py_json.loads(content.decode("utf-8"))
            i_samples = data.get("i") or data.get("i_channel") or data.get("real") or []
            q_samples = data.get("q") or data.get("q_channel") or data.get("imag") or []
            sr = sample_rate if sample_rate is not None else data.get("sample_rate")
            fc = center_frequency if center_frequency is not None else data.get("center_frequency")
            mod = override_modulation if override_modulation else data.get("modulation")
            return default_pipeline_orchestrator.run_pipeline(
                i_channel=i_samples,
                q_channel=q_samples,
                sample_rate=sr,
                center_frequency=fc,
                override_modulation=mod,
                input_type="UPLOADED_JSON_IQ",
                metadata={"filename": filename}
            )
        except Exception as e:
            return {
                "status": "INVALID_JSON_SIGNAL",
                "pipeline_success": False,
                "message": f"Failed to parse JSON signal file: {str(e)}",
                "module_statuses": {"module1": "FAILED"},
                "pipeline_warnings": ["INVALID_JSON_SIGNAL_FORMAT"]
            }
    else:
        # Ingest through Module 1 file parser preserving 1B/1C research boundary
        from backend.modules.module1.service import process_file_input
        meta = {}
        if sample_rate is not None:
            meta["sample_rate"] = sample_rate
        if center_frequency is not None:
            meta["center_frequency"] = center_frequency
        m1_res = process_file_input(filename, content, user_metadata=meta or None)
        
        module_statuses = {f"module{m}": "NOT_EXECUTED" for m in range(1, 11)}
        module_statuses["module1"] = m1_res.get("status", "UNSUPPORTED")
        return {
            "status": m1_res.get("status", "UNSUPPORTED"),
            "pipeline_success": False,
            "overall_pipeline_status": m1_res.get("status", "UNSUPPORTED"),
            "module_statuses": module_statuses,
            "module1_validation": m1_res,
            "pipeline_warnings": m1_res.get("warnings", []),
            "metadata": m1_res.get("metadata")
        }


# ==============================================================================
# EXPERIMENTAL PIPELINE ROUTER (ALIASED FOR FRONTEND-EXPERIMENTAL)
# ==============================================================================
experimental_router = APIRouter(prefix="/api/experimental/pipeline", tags=["Experimental Pipeline"])
experimental_router.add_api_route("/status", get_pipeline_status, methods=["GET"])
experimental_router.add_api_route("/fixture", get_integration_fixture, methods=["GET"])
experimental_router.add_api_route("/run", post_run_pipeline, methods=["POST"])
experimental_router.add_api_route("/report", post_generate_report, methods=["POST"])
experimental_router.add_api_route("/report/{format}", get_pipeline_report, methods=["GET"])
experimental_router.add_api_route("/upload", post_upload_pipeline, methods=["POST"])
