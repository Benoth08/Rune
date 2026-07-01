"""Vision Active — API publique (couche fine).

V5.7.1 — La détection est maintenant assurée par vision_semantic
(embeddings multilingues). Ce fichier reste comme façade pour la
compatibilité d'import (hippocampe.py importait depuis vision_active).

Toute la logique est dans vision_semantic.py.
"""
from __future__ import annotations

# Façade : ré-exporte les symboles publics depuis vision_semantic.
from rune.cognition.vision_semantic import (
    ZoomTrigger,
    SemanticVisionDetector,
    detect_zoom_trigger,
    detect_perceptual_uncertainty,
    build_zoom_prompt,
    format_zoom_block,
    build_visual_warning_block,
    looks_like_visual_question,
    get_detector,
)


# Pour rétro-compat avec V5.7.0 :
class ZoomResult:
    """Conservé pour rétro-compat — non utilisé directement."""
    def __init__(self, image_id: str = "", region_hint: str = "",
                 raw_vlm_output: str = "", prompt_block: str = ""):
        self.image_id = image_id
        self.region_hint = region_hint
        self.raw_vlm_output = raw_vlm_output
        self.prompt_block = prompt_block


__all__ = [
    "ZoomTrigger",
    "ZoomResult",
    "SemanticVisionDetector",
    "detect_zoom_trigger",
    "detect_perceptual_uncertainty",
    "build_zoom_prompt",
    "format_zoom_block",
    "build_visual_warning_block",
    "looks_like_visual_question",
    "get_detector",
]
