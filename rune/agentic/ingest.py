"""Ingestion de pièces jointes — mécanique commune au chat ET à l'agent.

Une seule logique décide quoi faire d'un fichier joint, selon la capacité du
modèle chargé (le principe « une brique, deux consommateurs ») :

- doc texte (pdf/docx/…) : le texte est déjà extrait en amont par le pipeline
  d'upload → on l'écrit tel quel dans la mission ;
- image + cerveau NON multimodal (Qwen Coder, Qwen3-thinking) : on passe par
  le captioner (Qwen2-VL/BLIP) → une description textuelle que l'agent lit ;
- image + cerveau multimodal natif (Gemma 4…) : PAS de captioner — les pixels
  iront directement au modèle (le captioner ferait doublon et appauvrirait).

Ce module ne fait pas de génération ; il classe et, pour les images en mode
captioner, délègue à un callable ``caption_fn`` fourni par l'appelant. Les
fonctions de classification sont pures → testables hors GPU.
"""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass, field

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
BINARY_DOC_EXT = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx",
                  ".ppt", ".odt", ".rtf"}


def classify_attachment(filename: str) -> str:
    """'image' | 'doc' | 'text' selon l'extension."""
    ext = os.path.splitext(str(filename or ""))[1].lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in BINARY_DOC_EXT:
        return "doc"
    return "text"


@dataclass
class IngestResult:
    """Résultat d'ingestion pour UNE pièce jointe."""
    filename: str                 # nom final (réécrit en .txt si doc/caption)
    text: str | None = None       # contenu texte à semer (doc ou caption)
    pil_image: object | None = None   # image à passer en natif (cerveau VLM)
    note: str = ""                # ligne d'info pour le contexte agent

    def as_dict(self) -> dict:
        return {"filename": self.filename, "has_text": self.text is not None,
                "native_image": self.pil_image is not None, "note": self.note}


def _decode_image(content) -> object | None:
    """Tente de décoder ``content`` (base64 ou bytes) en PIL.Image."""
    try:
        from PIL import Image
        import io
        if isinstance(content, str):
            raw = base64.b64decode(content, validate=False)
        elif isinstance(content, (bytes, bytearray)):
            raw = bytes(content)
        else:
            return None
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (binascii.Error, OSError, ValueError):
        return None
    except Exception:  # noqa: BLE001 — PIL absent / format exotique
        return None


def ingest_attachment(att: dict, *, native_multimodal: bool,
                      caption_fn=None) -> IngestResult | None:
    """Ingère une pièce jointe selon la capacité du modèle.

    ``native_multimodal`` : le cerveau voit-il les pixels ?
    ``caption_fn(pil_image) -> str`` : captioner fourni par l'appelant (utilisé
    seulement pour une image quand le cerveau n'est PAS multimodal).
    """
    fname = os.path.basename(str((att or {}).get("filename") or "").strip())
    content = (att or {}).get("content")
    if not fname or content is None:
        return None
    stem, ext = os.path.splitext(fname)
    kind = classify_attachment(fname)

    if kind == "doc":
        # texte déjà extrait en amont → semé en .txt
        return IngestResult(filename=f"{stem}.txt", text=str(content),
                            note=f"{fname} (document, texte extrait)")

    if kind == "image":
        if native_multimodal:
            img = _decode_image(content)
            if img is not None:
                return IngestResult(filename=fname, pil_image=img,
                                    note=f"{fname} (image, vision native)")
            # décodage raté → on retombe sur le captioner si dispo
        if caption_fn is not None:
            img = _decode_image(content)
            if img is not None:
                try:
                    desc = caption_fn(img) or ""
                except Exception:  # noqa: BLE001
                    desc = ""
                if desc:
                    return IngestResult(
                        filename=f"{stem}.txt",
                        text=f"[Description de l'image {fname}]\n{desc}",
                        note=f"{fname} (image décrite par le captioner)")
        # ni natif ni captioner exploitable
        return IngestResult(filename=f"{stem}.txt",
                            text=f"[Image {fname} non interprétable]",
                            note=f"{fname} (image non interprétée)")

    # texte brut
    return IngestResult(filename=fname, text=str(content),
                        note=f"{fname} (texte)")


@dataclass
class IngestBatch:
    seeded_text: list = field(default_factory=list)   # (filename, text)
    native_images: list = field(default_factory=list)  # pil images
    notes: list = field(default_factory=list)
