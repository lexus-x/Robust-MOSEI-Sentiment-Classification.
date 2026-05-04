from __future__ import annotations

import pytest

from multimod.video_review import (
    DEFAULT_VIDEO_REVIEW_SCHEMA,
    build_video_review_prompt,
    wait_for_uploaded_video,
)


def test_build_video_review_prompt_includes_focus_and_checklist():
    prompt = build_video_review_prompt(
        focus="safety",
        checklist=["Worker wears PPE", "No blocked exit"],
    )

    assert "Primary focus: safety." in prompt
    assert "- Worker wears PPE" in prompt
    assert "- No blocked exit" in prompt
    assert "MM:SS" in prompt


def test_video_review_schema_has_required_top_level_fields():
    required = set(DEFAULT_VIDEO_REVIEW_SCHEMA["required"])

    assert {"focus", "summary", "needs_human_review", "events", "violations", "recommended_actions"} <= required


class _DummyState:
    def __init__(self, name: str):
        self.name = name


class _DummyError:
    def __init__(self, message: str):
        self.message = message


class _DummyUploadedFile:
    def __init__(self, name: str, state_name: str, error: _DummyError | None = None):
        self.name = name
        self.state = _DummyState(state_name)
        self.error = error


class _DummyClient:
    def __init__(self, states: list[_DummyUploadedFile]):
        self._states = iter(states)
        self.files = self

    def get(self, name: str) -> _DummyUploadedFile:
        return next(self._states)


def test_wait_for_uploaded_video_returns_when_active(monkeypatch):
    monkeypatch.setattr("multimod.video_review.time.sleep", lambda _: None)
    client = _DummyClient([_DummyUploadedFile("video-1", "ACTIVE")])
    uploaded = _DummyUploadedFile("video-1", "PROCESSING")

    result = wait_for_uploaded_video(client, uploaded, poll_seconds=0, timeout_seconds=1)

    assert result.state.name == "ACTIVE"


def test_wait_for_uploaded_video_raises_on_terminal_failure(monkeypatch):
    monkeypatch.setattr("multimod.video_review.time.sleep", lambda _: None)
    client = _DummyClient(
        [_DummyUploadedFile("video-1", "FAILED", error=_DummyError("unsupported codec"))]
    )
    uploaded = _DummyUploadedFile("video-1", "PROCESSING")

    with pytest.raises(RuntimeError, match="FAILED: unsupported codec"):
        wait_for_uploaded_video(client, uploaded, poll_seconds=0, timeout_seconds=1)
