"""Tests for the FastAPI REST endpoints."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path

# We need to mock heavy dependencies before importing the app
import sys

# Minimal test that checks the API structure without loading actual models


def test_schemas_validation():
    """Pydantic schemas should validate correctly."""
    from rune.server.schemas import ChatRequest, ImageInput, SessionPatch

    req = ChatRequest(session_id="sess_123", message="Hello")
    assert req.session_id == "sess_123"
    assert req.images == []

    img = ImageInput(data="base64data", mime="image/png")
    assert img.mime == "image/png"

    patch = SessionPatch(title="New title", pinned=True)
    assert patch.title == "New title"
    assert patch.archived is None


def test_schemas_reject_extra():
    """Extra fields should be rejected."""
    from rune.server.schemas import ChatRequest
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ChatRequest(session_id="s", message="hi", extra_field="bad")


def test_schemas_message_length():
    """Message should respect length constraints."""
    from rune.server.schemas import ChatRequest
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ChatRequest(session_id="s", message="")

    # Max 8000 chars
    long_msg = "a" * 8001
    with pytest.raises(pydantic.ValidationError):
        ChatRequest(session_id="s", message=long_msg)


def test_schemas_max_images():
    """Maximum 4 images per request."""
    from rune.server.schemas import ChatRequest, ImageInput
    import pydantic

    images = [ImageInput(data=f"img{i}") for i in range(5)]
    with pytest.raises(pydantic.ValidationError):
        ChatRequest(session_id="s", message="hi", images=images)


def test_session_manager_crud():
    """Session CRUD operations should work."""
    from rune.sessions import SessionManager, Message

    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))

        # Create
        session = sm.create(title="Test Chat")
        assert session.session_id.startswith("sess_")
        assert session.title == "Test Chat"

        # List
        sessions = sm.list_sessions()
        assert len(sessions) == 1

        # Get
        loaded = sm.get(session.session_id)
        assert loaded is not None
        assert loaded.title == "Test Chat"

        # Add message
        msg = Message(role="user", content="Hello Rune")
        sm.add_message(session.session_id, msg)

        loaded2 = sm.get(session.session_id)
        assert len(loaded2.messages) == 1
        assert loaded2.messages[0].content == "Hello Rune"

        # Update
        sm.update(session.session_id, title="Renamed", pinned=True)
        loaded3 = sm.get(session.session_id)
        assert loaded3.title == "Renamed"
        assert loaded3.pinned is True

        # Delete
        sm.delete(session.session_id)
        assert sm.get(session.session_id) is None
        assert len(sm.list_sessions()) == 0


def test_session_pagination():
    """Session messages should support offset/limit."""
    from rune.sessions import SessionManager, Message

    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))
        session = sm.create()

        for i in range(20):
            sm.add_message(session.session_id, Message(role="user", content=f"msg_{i}"))

        # Load first 5
        s = sm.get(session.session_id, offset=0, limit=5)
        assert len(s.messages) == 5
        assert s.messages[0].content == "msg_0"

        # Load next 5
        s2 = sm.get(session.session_id, offset=5, limit=5)
        assert len(s2.messages) == 5
        assert s2.messages[0].content == "msg_5"


def test_session_export_markdown():
    """Export should produce valid markdown."""
    from rune.sessions import SessionManager, Message

    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(sessions_dir=Path(tmp))
        session = sm.create(title="Export Test")
        sm.add_message(session.session_id, Message(role="user", content="Question"))
        sm.add_message(session.session_id, Message(role="assistant", content="Réponse"))

        md = sm.export_markdown(session.session_id)
        assert "Export Test" in md
        assert "Question" in md
        assert "Réponse" in md
