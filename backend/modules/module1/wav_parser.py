"""Module 1B: WAV Parsing Interface (STATUS: NOT_YET_VALIDATED).

This module defines the ingestion interface for WAV container files.
Per the project specification, WAV parsing requirements (channel mapping,
bit depth conversion, mono vs stereo semantics) have NOT yet been experimentally
validated or locked.

Any attempt to parse without an approved, validated specification is flagged
explicitly as NOT_YET_VALIDATED or UNSUPPORTED.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


class WavParserStatus:
    NOT_YET_VALIDATED = "NOT_YET_VALIDATED"
    UNSUPPORTED = "UNSUPPORTED"
    UNKNOWN = "UNKNOWN"
    SUPPORTED = "SUPPORTED"


def parse_wav_file(
    file_bytes: bytes,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Interface for WAV file parsing.

    Returns:
        (canonical_iq, result_dict)
    """
    # Explicit status: NOT_YET_VALIDATED
    return None, {
        "status": WavParserStatus.NOT_YET_VALIDATED,
        "module": "1B",
        "message": (
            "WAV parsing is NOT YET VALIDATED. "
            "Specific channel mapping, bit-depth normalization, and RF-WAV format "
            "rules require experimental validation before locking."
        ),
        "warnings": ["MODULE_1B_NOT_YET_VALIDATED"]
    }
