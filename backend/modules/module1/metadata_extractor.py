"""Module 1D: Metadata Extraction & Schema Interface (STATUS: NOT_YET_VALIDATED).

This module defines the metadata handling rules:
- Records sample rate, center frequency, data type, channels, sample count when explicitly available.
- If metadata is unavailable, it is strictly set to None / UNKNOWN.
- NEVER invent sample rate, absolute center frequency, or other physical parameters
  from raw IQ samples.
"""

from typing import Any, Dict, Optional


class MetadataStatus:
    NOT_YET_VALIDATED = "NOT_YET_VALIDATED"
    UNKNOWN = "UNKNOWN"


def extract_metadata(
    raw_header: Optional[Dict[str, Any]] = None,
    user_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Compile metadata from verified headers or explicit caller configuration.

    Guarantees:
    - Never invents sample_rate or center_frequency.
    - Fields not explicitly verified remain None / UNKNOWN.
    """
    header = raw_header or {}
    user = user_metadata or {}

    sample_rate = user.get("sample_rate", header.get("sample_rate", None))
    center_frequency = user.get("center_frequency", header.get("center_frequency", None))
    data_type = user.get("data_type", header.get("data_type", MetadataStatus.UNKNOWN))
    channels = user.get("channels", header.get("channels", None))

    return {
        "status": MetadataStatus.NOT_YET_VALIDATED,
        "sample_rate": sample_rate,
        "center_frequency": center_frequency,
        "data_type": data_type,
        "channels": channels
    }
