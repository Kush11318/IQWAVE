"""Module 1 Package: Input / IQ Representation & Validation."""

from .canonical_iq import to_canonical_iq, validate_iq, compute_diagnostics
from .wav_parser import parse_wav_file, WavParserStatus
from .raw_iq_parser import parse_raw_iq_bytes, RawIQParserStatus
from .metadata_extractor import extract_metadata, MetadataStatus
from .service import process_signal_input

__all__ = [
    "to_canonical_iq",
    "validate_iq",
    "compute_diagnostics",
    "parse_wav_file",
    "WavParserStatus",
    "parse_raw_iq_bytes",
    "RawIQParserStatus",
    "extract_metadata",
    "MetadataStatus",
    "process_signal_input",
]
