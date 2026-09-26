"""Module 1 Service: Unified interface for signal ingestion and validation."""

from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np

from .canonical_iq import to_canonical_iq, validate_iq
from .wav_parser import parse_wav_file, WavParserStatus
from .raw_iq_parser import parse_raw_iq_bytes, RawIQParserStatus
from .metadata_extractor import extract_metadata, MetadataStatus


def process_iq_arrays(
    i_channel: Any,
    q_channel: Any,
    user_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Ingest and validate numerical I and Q arrays via Module 1A (LOCKED).

    Does NOT modify the signal in any way.
    Nonzero mean is recorded, never subtracted.
    """
    iq_canonical, val_result = to_canonical_iq(i_channel, q_channel)
    meta = extract_metadata(user_metadata=user_metadata)

    response = {
        "status": val_result["status"],
        "num_samples": val_result.get("num_samples", 0),
        "nan_count": val_result.get("nan_count", 0),
        "inf_count": val_result.get("inf_count", 0),
        "power": val_result.get("power"),
        "diagnostics": val_result.get("diagnostics"),
        "metadata": meta,
        "warnings": val_result.get("warnings", [])
    }
    return response


def process_file_input(
    filename: str,
    file_bytes: bytes,
    user_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Ingest file data.

    Preserves clear boundaries:
    - 1B (WAV) and 1C (Raw IQ) are NOT YET VALIDATED.
    - Unsupported formats are rejected explicitly.
    """
    ext = os.path.splitext(filename.lower())[1]

    if ext == ".wav":
        iq, parse_info = parse_wav_file(file_bytes)
        return {
            "status": parse_info["status"],
            "module": "1B",
            "filename": filename,
            "message": parse_info["message"],
            "warnings": parse_info["warnings"],
            "metadata": extract_metadata(user_metadata=user_metadata)
        }
    elif ext in [".iq", ".bin", ".dat"]:
        iq, parse_info = parse_raw_iq_bytes(file_bytes)
        return {
            "status": parse_info["status"],
            "module": "1C",
            "filename": filename,
            "message": parse_info["message"],
            "warnings": parse_info["warnings"],
            "metadata": extract_metadata(user_metadata=user_metadata)
        }
    else:
        return {
            "status": "UNSUPPORTED",
            "module": "1",
            "filename": filename,
            "message": f"Unsupported file extension '{ext}'. Only .wav, .iq, .bin, .dat are recognized ingestion formats.",
            "warnings": ["UNSUPPORTED_FILE_FORMAT"],
            "metadata": extract_metadata(user_metadata=user_metadata)
        }


def process_signal_input(
    i_channel: Optional[Any] = None,
    q_channel: Optional[Any] = None,
    filename: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    user_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Primary pipeline dispatcher for Module 1."""
    if i_channel is not None and q_channel is not None:
        return process_iq_arrays(i_channel, q_channel, user_metadata=user_metadata)
    elif filename is not None and file_bytes is not None:
        return process_file_input(filename, file_bytes, user_metadata=user_metadata)
    else:
        return {
            "status": "INVALID",
            "num_samples": 0,
            "nan_count": 0,
            "inf_count": 0,
            "power": None,
            "diagnostics": None,
            "metadata": extract_metadata(user_metadata=user_metadata),
            "warnings": ["NO_INPUT_PROVIDED"]
        }
