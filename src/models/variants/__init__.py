"""Custom model variants for individual researchers."""

from .remoteclip_difference import (
    RemoteCLIPDifferenceModel,
    RemoteCLIPEncoder,
    resolve_remoteclip_checkpoint,
)

__all__ = [
    "RemoteCLIPEncoder",
    "RemoteCLIPDifferenceModel",
    "resolve_remoteclip_checkpoint",
]
