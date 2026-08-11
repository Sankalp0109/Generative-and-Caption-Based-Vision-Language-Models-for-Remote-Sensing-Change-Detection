"""Phase 1 baseline model components.

Keeping this file makes ``src.models.baseline`` an explicit Python package.
That prevents stale modules or bytecode named ``baseline`` from shadowing the
directory on cluster environments.
"""

from .baseline import RSICCformerBaseline
from .decoder import SimpleDecoder
from .encoder import SimpleEncoder
from .interface import ChangeCaptioningModel

__all__ = [
    "ChangeCaptioningModel",
    "SimpleEncoder",
    "SimpleDecoder",
    "RSICCformerBaseline",
]
