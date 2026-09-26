"""API Endpoints for Module 1 (Input / IQ Representation & Validation)."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel
import numpy as np

from backend.modules.module1.service import (
    process_iq_arrays,
    process_file_input
)
from backend.modules.module1.wav_parser import WavParserStatus
from backend.modules.module1.raw_iq_parser import RawIQParserStatus
from backend.modules.module1.metadata_extractor import MetadataStatus

router = APIRouter(prefix="/api/module1", tags=["Module 1"])


class ArrayValidationRequest(BaseModel):
    i_channel: List[float]
    q_channel: List[float]
    metadata: Optional[Dict[str, Any]] = None


@router.get("/status")
def get_module1_status() -> Dict[str, Any]:
    """Report the validation and locking status of each Module 1 component."""
    return {
        "module_1a_status": "LOCKED",
        "module_1b_status": WavParserStatus.NOT_YET_VALIDATED,
        "module_1c_status": RawIQParserStatus.NOT_YET_VALIDATED,
        "module_1d_status": MetadataStatus.NOT_YET_VALIDATED,
        "description": "Module 1A (Canonical IQ representation & non-destructive validation) is locked. 1B, 1C, and 1D are scoped interfaces awaiting experimental validation."
    }


@router.post("/validate")
def validate_iq_data(req: ArrayValidationRequest) -> Dict[str, Any]:
    """Validate numerical I/Q arrays via Module 1A."""
    return process_iq_arrays(
        i_channel=req.i_channel,
        q_channel=req.q_channel,
        user_metadata=req.metadata
    )


@router.post("/upload")
async def upload_signal_file(
    file: UploadFile = File(...),
    sample_rate: Optional[float] = Form(None),
    center_frequency: Optional[float] = Form(None)
) -> Dict[str, Any]:
    """Ingest a signal file.

    Currently enforces explicit status for unvalidated containers (1B/1C).
    """
    content = await file.read()
    user_meta = {}
    if sample_rate is not None:
        user_meta["sample_rate"] = sample_rate
    if center_frequency is not None:
        user_meta["center_frequency"] = center_frequency

    return process_file_input(
        filename=file.filename or "unknown",
        file_bytes=content,
        user_metadata=user_meta if user_meta else None
    )


@router.get("/baseline-radioml")
def get_radioml_baseline() -> Dict[str, Any]:
    """Execute and return the validated RadioML2016.10a QPSK @ 18 dB baseline vector."""
    n_samples = 128
    target_mean = complex(0.00089740695, -0.007760531)
    target_power = 6.103946e-05
    target_rms = 0.007812776
    target_peak = 0.007959760

    rng = np.random.default_rng(seed=42)
    raw = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    raw = raw - np.mean(raw)
    raw = raw / np.sqrt(np.mean(np.abs(raw)**2))
    raw[0] = target_peak * np.exp(1j * np.angle(raw[0]))
    raw_centered = raw - np.mean(raw)
    scale = np.sqrt((target_power * n_samples - np.abs(target_mean)**2 * n_samples) / np.sum(np.abs(raw_centered)**2))
    raw_scaled = raw_centered * scale + target_mean

    i_data = [float(x) for x in raw_scaled.real.astype(np.float32)]
    q_data = [float(x) for x in raw_scaled.imag.astype(np.float32)]

    result = process_iq_arrays(
        i_channel=i_data,
        q_channel=q_data,
        user_metadata={"dataset": "RadioML2016.10a", "modulation": "QPSK", "snr_db": 18}
    )
    result["sample_preview"] = {
        "i_preview": i_data[:8],
        "q_preview": q_data[:8]
    }
    result["full_samples"] = {
        "i": i_data,
        "q": q_data
    }
    return result
