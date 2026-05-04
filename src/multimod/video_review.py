"""Video-to-report helpers for a narrow, real-world review task."""

from __future__ import annotations

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_VIDEO_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "focus": {
            "type": "string",
            "enum": ["safety", "incident", "compliance", "operations"],
            "description": "Primary review lens used for this report.",
        },
        "summary": {
            "type": "string",
            "description": "Short factual summary of what happened in the video.",
        },
        "needs_human_review": {
            "type": "boolean",
            "description": "True when the video is ambiguous or the stakes are high enough for manual review.",
        },
        "events": {
            "type": "array",
            "description": "Chronological events with timestamps in MM:SS format.",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Event timestamp in MM:SS."},
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Operational severity of the event.",
                    },
                    "description": {
                        "type": "string",
                        "description": "What is visibly happening at this point in the video.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Visible cues that support the event description.",
                    },
                },
                "required": ["timestamp", "severity", "description", "evidence"],
                "additionalProperties": False,
            },
        },
        "violations": {
            "type": "array",
            "description": "Checklist failures, unsafe acts, or compliance problems visible in the video.",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Violation timestamp in MM:SS."},
                    "category": {
                        "type": "string",
                        "description": "Short category such as PPE, trip hazard, unsafe driving, or blocked exit.",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Risk severity if the violation is real.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Specific factual statement about the issue.",
                    },
                },
                "required": ["timestamp", "category", "severity", "description"],
                "additionalProperties": False,
            },
        },
        "recommended_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Immediate follow-up actions grounded in the visible evidence.",
        },
    },
    "required": [
        "focus",
        "summary",
        "needs_human_review",
        "events",
        "violations",
        "recommended_actions",
    ],
    "additionalProperties": False,
}


def build_video_review_prompt(focus: str, checklist: list[str] | None = None) -> str:
    checklist = checklist or []
    checklist_text = "\n".join(f"- {item}" for item in checklist) if checklist else "- No custom checklist provided."
    return (
        "Review this uploaded video and produce a strict JSON incident report.\n"
        f"Primary focus: {focus}.\n"
        "Rules:\n"
        "- Stay factual. Do not guess beyond what is visible or audible.\n"
        "- Use MM:SS timestamps.\n"
        "- If evidence is ambiguous, set needs_human_review to true.\n"
        "- Keep the summary concise and operational.\n"
        "- Recommended actions must be grounded in the observed events.\n"
        "Custom checklist:\n"
        f"{checklist_text}\n"
    )


def _guess_mime_type(video_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(video_path.name)
    return mime_type or "video/mp4"


def _require_genai() -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. Install with: pip install -e '.[video]'"
        ) from exc
    return genai


def _uploaded_file_state_name(uploaded_file: Any) -> str | None:
    state = getattr(uploaded_file, "state", None)
    return getattr(state, "name", None)


def _uploaded_file_error_message(uploaded_file: Any) -> str | None:
    error = getattr(uploaded_file, "error", None)
    if error is None:
        return None
    message = getattr(error, "message", None)
    if message:
        return str(message)
    return str(error)


def wait_for_uploaded_video(
    client: Any,
    uploaded_file: Any,
    poll_seconds: int = 5,
    timeout_seconds: int = 600,
) -> Any:
    started = time.time()
    current = uploaded_file
    terminal_failure_states = {"FAILED", "ERROR", "CANCELLED"}
    while _uploaded_file_state_name(current) != "ACTIVE":
        state_name = _uploaded_file_state_name(current)
        if state_name in terminal_failure_states:
            message = _uploaded_file_error_message(current)
            if message:
                raise RuntimeError(f"Video processing failed in state {state_name}: {message}")
            raise RuntimeError(f"Video processing failed in state {state_name}: {current.name}")
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for video processing: {current.name}")
        time.sleep(poll_seconds)
        current = client.files.get(name=current.name)
    return current


def review_video_with_gemini(
    video_path: str | Path,
    *,
    focus: str = "safety",
    checklist: list[str] | None = None,
    model: str = "gemini-2.5-flash",
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    genai = _require_genai()
    if "GEMINI_API_KEY" not in os.environ:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    client = genai.Client()
    uploaded = client.files.upload(
        file=str(video_path),
        config={"mime_type": _guess_mime_type(video_path)},
    )
    uploaded = wait_for_uploaded_video(
        client=client,
        uploaded_file=uploaded,
        timeout_seconds=timeout_seconds,
    )

    response = client.models.generate_content(
        model=model,
        contents=[uploaded, build_video_review_prompt(focus=focus, checklist=checklist)],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": DEFAULT_VIDEO_REVIEW_SCHEMA,
        },
    )
    return json.loads(response.text)
