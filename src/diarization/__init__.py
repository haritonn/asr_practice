"""Speaker diarization implementations and transcript reconciliation."""

from .base import BaseDiarizer
from .transcript import assign_speakers

__all__ = ["BaseDiarizer", "assign_speakers"]

try:
    from .pyannote import PyannoteCommunityDiarizer
except ModuleNotFoundError as error:
    if error.name != "pyannote.audio":
        raise
else:
    __all__.append("PyannoteCommunityDiarizer")
