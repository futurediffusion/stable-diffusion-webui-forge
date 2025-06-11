from __future__ import annotations

"""Compatibility wrappers for prompt style functions.

This module exposes the API expected by older code that imports
``modules.prompt_style_system``. The actual implementations live in
``modules.styles`` and the shared :data:`modules.shared.prompt_styles`
instance.
"""

from modules.styles import (
    PromptStyle,
    StyleDatabase,
    apply_styles_to_prompt,
    extract_style_text_from_prompt,
    extract_original_prompts,
)
from modules import shared


def apply_negative_styles_to_prompt(prompt: str, styles: list[str]) -> str:
    """Apply negative style texts to ``prompt`` using the shared
    :class:`~modules.styles.StyleDatabase`.
    """
    if shared.prompt_styles is None:
        # Lazily create database if not initialized
        shared.prompt_styles = StyleDatabase(shared.styles_filename)
    return shared.prompt_styles.apply_negative_styles_to_prompt(prompt, styles)

