"""External services integration (Gemini API and similar).

This package isolates third-party API clients from the rest of
Lythéa. The cognition pipeline depends only on stable abstractions
exposed here, never on httpx, anthropic SDK, google-generativeai,
or other vendor specifics.

Current contents:

* :mod:`gemini_client` — Google Gemini API wrapper used by the
  cascade generator (V3.9).
"""

from rune.external.gemini_client import (
    GeminiClient,
    GeminiClientError,
    GeminiQuotaExceededError,
    GeminiUnauthorizedError,
    mask_api_key,
)

__all__ = [
    "GeminiClient",
    "GeminiClientError",
    "GeminiQuotaExceededError",
    "GeminiUnauthorizedError",
    "mask_api_key",
]
