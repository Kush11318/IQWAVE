"""Module 5: Synchronization & Digital Symbol/Bit Recovery.

Submodules:
- carrier_sync: 5A Carrier-frequency synchronization
- phase_sync: 5B Carrier-phase synchronization
- timing_sync: 5C Gardner symbol-timing recovery
- matched_filter: 5D Root-Raised-Cosine (RRC) matched filter
- resampler: 5E Rational polyphase resampling
- demodulators: 5F Modulation-specific hard-decision demodulation
- service: Orchestrator integrating Modules 1-4 into recovered symbols and bits
"""

from .service import DigitalRecoveryService, recover_digital_symbols_and_bits

__all__ = ["DigitalRecoveryService", "recover_digital_symbols_and_bits"]
