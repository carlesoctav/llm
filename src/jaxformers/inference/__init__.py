"""Inference runtime package."""

from .llm_client import make, SameProcessTPUInferenceClient


__all__ = [
    "SameProcessTPUInferenceClient",
    "make",
]
