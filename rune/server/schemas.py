"""Pydantic v2 schemas for the Lythéa API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ImageInput(BaseModel):
    """Base64-encoded image attachment."""

    data: str = Field(..., description="base64-encoded image data")
    mime: Literal["image/png", "image/jpeg", "image/webp"] = "image/png"
    model_config = {"extra": "forbid"}


class DocumentAttachment(BaseModel):
    """Document attaché à un message (texte déjà extrait côté serveur).

    Produit par ``/api/upload/document`` en mode ``attach`` puis renvoyé
    dans le message suivant. Le texte est injecté en contexte ; le nom
    et la mention de l'ingestion (le cas échéant) sont passés à
    l'orchestrateur pour qu'il en tienne compte dans le prompt.
    """

    filename: str = Field(..., max_length=300)
    # mode = "attach" → ``text`` est injecté en contexte
    # mode = "ingest" → l'ingestion a déjà eu lieu côté serveur ;
    #   ``text`` peut être vide, on garde juste la trace nominale
    mode: Literal["attach", "ingest"] = "attach"
    text: str = Field(default="", max_length=200_000)
    model_config = {"extra": "forbid"}


class ChatRequest(BaseModel):
    """Chat message from the user.

    V6.0.0-rc : le message peut être vide si AU MOINS un document ou
    une image est joint. Avant, l'utilisateur ne pouvait pas juste
    "déposer un fichier sans rien dire" — il était obligé d'écrire au
    moins un caractère, ce qui forçait des messages bidon type "voilà"
    ou un espace. Maintenant, joindre = action implicite, Lythéa réagit
    en demandant quoi faire du fichier.
    """

    session_id: str
    message: str = Field(default="", max_length=8000)
    images: list[ImageInput] = Field(default_factory=list, max_length=4)
    documents: list[DocumentAttachment] = Field(default_factory=list, max_length=8)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _require_message_or_attachment(self):
        """Au moins un input non-vide : message texte OU image OU document."""
        if not self.message.strip() and not self.images and not self.documents:
            raise ValueError(
                "Au moins un message, une image ou un document doit être fourni."
            )
        return self


class ModelLoadRequest(BaseModel):
    """Request to load a model."""

    model_id: str
    model_config = {"extra": "forbid"}


class SessionPatch(BaseModel):
    """Partial update for a session."""

    title: str | None = None
    pinned: bool | None = None
    archived: bool | None = None
    model_config = {"extra": "forbid"}


class GitConfigRequest(BaseModel):
    """Git sync configuration."""

    token: str
    user: str
    repo: str
    model_config = {"extra": "forbid"}


class EntropyConfigRequest(BaseModel):
    """Entropy threshold update."""

    threshold: float = Field(..., ge=0.1, le=1.0)
    model_config = {"extra": "forbid"}


class WebModeRequest(BaseModel):
    """Web search mode update."""

    mode: Literal["off", "auto", "always"]
    model_config = {"extra": "forbid"}


class ReasoningConfigRequest(BaseModel):
    """Reasoning toggle — un seul flag pour activer le raisonnement."""

    enabled: bool
    model_config = {"extra": "forbid"}


class SamplingConfigRequest(BaseModel):
    """Sampling parameters update.

    All fields are optional — only the fields actually provided in the
    request body are updated, others keep their current value. This
    lets the UI send partial updates (e.g. moving just the temperature
    slider) without having to read and resend the whole profile.

    ``None`` for ``top_p``/``top_k``/``min_p`` means: disable that
    sampling step (the model uses pure temperature/repetition_penalty).
    """

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0, le=200)
    min_p: float | None = Field(default=None, ge=0.0, le=0.5)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=2.0)
    max_new_tokens: int | None = Field(default=None, ge=16, le=8192)
    model_config = {"extra": "forbid"}


class CodegenRequest(BaseModel):
    """Raw assistant Markdown to extract code files from (then zip)."""

    text: str = Field(..., max_length=2_000_000)
    model_config = {"extra": "forbid"}


class CodegenCommitRequest(BaseModel):
    """Materialise the code files of an answer into the workspace sandbox.

    ``subdir`` optionally namespaces the project under a folder (e.g.
    ``projets/mon-api``) so successive generations don't collide at root.
    """

    text: str = Field(..., max_length=2_000_000)
    subdir: str = Field(default="", max_length=200)
    model_config = {"extra": "forbid"}


class AgentAttachment(BaseModel):
    """A user-provided file seeded into the mission dir (text content)."""

    filename: str = Field(..., min_length=1, max_length=255)
    content: str = Field(default="", max_length=200_000)
    model_config = {"extra": "forbid"}


class AgentRunRequest(BaseModel):
    """Launch a bounded agentic run on a task (V6 Phase 1)."""

    task: str = Field(..., min_length=1, max_length=8000)
    subdir: str = Field(default="", max_length=200)
    session_id: str = Field(default="", max_length=64)
    react: bool | None = Field(default=None)   # UI toggle; None = server default
    attachments: list[AgentAttachment] = Field(default_factory=list, max_length=8)
    model_config = {"extra": "forbid"}


class AgentInterjectRequest(BaseModel):
    """Inject a new instruction into a running agent (absorbed between steps)."""

    run_id: str = Field(..., max_length=64)
    text: str = Field(..., min_length=1, max_length=8000)
    model_config = {"extra": "forbid"}


class AgentStopRequest(BaseModel):
    """Request a running agent to stop after the current step."""

    run_id: str = Field(..., max_length=64)
    model_config = {"extra": "forbid"}


class SteeringApplyRequest(BaseModel):
    """Attach/adjust/detach a steering axis (V6 beta).

    ``alpha == 0`` detaches (neutral voice). Sign selects the axis
    direction; magnitude is clamped server-side for coherence.
    """

    axis: str = Field(..., max_length=40)
    alpha: float = Field(default=0.0, ge=-8.0, le=8.0)
    model_config = {"extra": "forbid"}


class SteeringMixRequest(BaseModel):
    """Engage several steering axes at once (V6 beta).

    ``mix`` maps axis name → alpha (0 / absent = off). Per-layer nudges are
    summed and clamped to a global budget server-side for coherence.
    """

    mix: dict[str, float] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}


class SteeringCalibrateRequest(BaseModel):
    """Trigger (re)calibration of a steering axis on the loaded model."""

    axis: str = Field(..., max_length=40)
    model_config = {"extra": "forbid"}
