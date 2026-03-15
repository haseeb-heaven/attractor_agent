"""Gemini provider profile (Section 3.6)."""

from __future__ import annotations

from typing import Any
from attractor.agent.profiles.base import BaseProfile


class GeminiProfile(BaseProfile):
    """Gemini-aligned profile (gemini-cli style)."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        super().__init__(model)
        self.id = "gemini"
        self._register_gemini_tools()

    def _register_gemini_tools(self) -> None:
        # Gemini often uses the same core tools as base, 
        # but could have unique ones like web_search or google_search rounding.
        pass

    def _get_base_instructions(self) -> str:
        return """You are a coding assistant using Google's Gemini models.
Follow the GEMINI.md conventions for project-specific guidance."""

    def provider_options(self) -> dict[str, Any] | None:
        return {
            "gemini": {
                "safety_settings": [],
                "grounding": True
            }
        }
