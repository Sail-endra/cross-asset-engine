"""Signal engine: momentum, carry, regime, and a shared registry.

Importing this package registers every signal in SIGNAL_REGISTRY, so the
backtester and briefing can iterate over signals generically.
"""

from .base import SIGNAL_REGISTRY, SignalResult, list_signals, register_signal
from . import momentum, carry, regime  # noqa: F401  (import triggers registration)

__all__ = ["SIGNAL_REGISTRY", "SignalResult", "list_signals", "register_signal", "momentum", "carry", "regime"]
