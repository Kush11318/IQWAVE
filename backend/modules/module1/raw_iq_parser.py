"""Module 1C: Raw IQ Binary Parsing Interface (STATUS: NOT_YET_VALIDATED).

This module defines the ingestion interface for raw binary IQ streams.
Per the project specification, raw IQ format conventions (interleaved vs planar,
sample datatypes, endianness, framing) have NOT yet been experimentally validated
or locked.

No automatic guessing or default assumptions are made.
Any attempt to parse without an approved, validated specification is flagged
explicitly as NOT_YET_VALIDATED or UNSUPPORTED.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


class RawIQParserStatus:
    NOT_YET_VALIDATED = "NOT_YET_VALIDATED"
    UNSUPPORTED = "UNSUPPORTED"
    UNKNOWN = "UNKNOWN"
    SUPPORTED = "SUPPORTED"


def parse_raw_iq_bytes(
    raw_bytes: bytes,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Interface for Raw IQ binary parsing.

    Returns:
        (canonical_iq, result_dict)
    """
    # Explicit status: NOT_YET_VALIDATED
    return None, {
        "status": RawIQParserStatus.NOT_YET_VALIDATED,
        "module": "1C",
        "message": (
            "Raw IQ binary parsing is NOT YET VALIDATED. "
            "Data types, interleaving scheme, and byte order must not be guessed "
            "without explicit experimental specification and validation."
        ),
        "warnings": ["MODULE_1C_NOT_YET_VALIDATED"]
    }
