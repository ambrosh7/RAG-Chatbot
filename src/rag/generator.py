"""Ask Groq for claims tied to chunk ids. The model cannot supply a citation.

Temperature is 0. The reply is JSON. Prose, a markdown fence, or trailing
commas are not claims: they become a generation failure. Publisher and URL
fields on the model object are ignored.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from src.constants import GROQ_DEFAULT_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE
from src.rag.models import DraftClaim, DraftDocument
from src.rag.retriever import DocumentHits

# The validator enforces the same three rules.
RULES: tuple[str, ...] = (
    "A claim's chunk_ids are chosen from the ids in that document's group.",
    "A claim states what that document's chunks say.",
    "An empty claims list means that document does not answer the question.",
)

SYSTEM_PROMPT = (
    "You write claims from the retrieved guidance passages for one question.\n"
    "Return only a JSON object of this shape and no other text:\n"
    '{"documents":[{"document_id":"...","claims":[{"text":"...","chunk_ids":["..."]}]}]}\n'
    "There is no field for publisher, year, or URL. Do not include them.\n"
    "Do not add a sentence about what the guidelines say.\n"
    "Rules:\n"
    + "\n".join(f"- {rule}" for rule in RULES)
)


class GenerationError(Exception):
    """The model did not return parseable claims. This is not an empty corpus."""


class ChatClient(Protocol):
    """One completion. Tests pass a fake. The Groq client is the default."""

    def complete(self, *, system: str, user: str) -> str: ...


def groq_model_name() -> str:
    """``GROQ_MODEL`` when it is set to a non-empty value, otherwise the default."""
    configured = os.environ.get("GROQ_MODEL", "").strip()
    return configured or GROQ_DEFAULT_MODEL


def _load_local_env() -> None:
    """Read `.env` for a local run. Variables already set in the process stay put."""
    from dotenv import load_dotenv

    from src.constants import PROJECT_ROOT

    load_dotenv(PROJECT_ROOT / ".env")


class GroqChatClient:
    """Chat completion at temperature 0. Quota and transport failures raise."""

    def __init__(self, create: Callable[..., Any] | None = None) -> None:
        self._create = create

    def complete(self, *, system: str, user: str) -> str:
        from groq import APIError, Groq

        _load_local_env()
        kwargs = {
            "model": groq_model_name(),
            "temperature": GROQ_TEMPERATURE,
            "max_tokens": GROQ_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        try:
            completion = (
                self._create(**kwargs)
                if self._create is not None
                else Groq().chat.completions.create(**kwargs)
            )
        except APIError as exc:
            raise GenerationError("groq request failed") from exc
        content = _completion_text(completion)
        if content is None:
            raise GenerationError("groq returned no completion")
        return content


def generate(
    message: str,
    groups: Sequence[DocumentHits],
    *,
    client: ChatClient | None = None,
) -> tuple[DraftDocument, ...]:
    """Ask for claims. ``groups`` is the only passage text the model sees.

    A malformed reply is asked once more. A transport or quota failure is not.
    """
    chat = client or GroqChatClient()
    user = build_user_prompt(message, groups)
    raw = chat.complete(system=SYSTEM_PROMPT, user=user)
    try:
        return parse_draft(raw)
    except GenerationError:
        raw = chat.complete(system=SYSTEM_PROMPT, user=user)
        return parse_draft(raw)


def build_user_prompt(message: str, groups: Sequence[DocumentHits]) -> str:
    """The question and the retrieved chunks. Nothing from outside ``groups``."""
    lines = [f"Question:\n{message.strip()}", ""]
    for group in groups:
        lines.append(group.document_id)
        for hit in group.hits:
            lines.append(f"[{hit.chunk_id}]")
            lines.append(hit.text)
            lines.append("")
    return "\n".join(lines).strip()


def parse_draft(raw: str) -> tuple[DraftDocument, ...]:
    """Parse the model object. Invalid JSON raises. Extra fields are ignored."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GenerationError("model output is not JSON") from exc
    if not isinstance(payload, dict):
        raise GenerationError("model output is not a JSON object")
    documents = payload.get("documents")
    if not isinstance(documents, list):
        raise GenerationError("model output has no documents array")
    parsed: list[DraftDocument] = []
    for entry in documents:
        if not isinstance(entry, dict):
            raise GenerationError("document entry is not an object")
        document_id = entry.get("document_id")
        claims = entry.get("claims")
        if not isinstance(document_id, str) or not isinstance(claims, list):
            raise GenerationError("document entry is missing document_id or claims")
        parsed.append(
            DraftDocument(document_id=document_id, claims=tuple(_parse_claim(claim) for claim in claims))
        )
    return tuple(parsed)


def _parse_claim(claim: object) -> DraftClaim:
    if not isinstance(claim, dict):
        raise GenerationError("claim is not an object")
    text = claim.get("text")
    chunk_ids = claim.get("chunk_ids")
    if not isinstance(text, str) or not isinstance(chunk_ids, list):
        raise GenerationError("claim is missing text or chunk_ids")
    if not all(isinstance(chunk_id, str) for chunk_id in chunk_ids):
        raise GenerationError("chunk_ids must be strings")
    return DraftClaim(text=text, chunk_ids=tuple(chunk_ids))


def _completion_text(completion: Any) -> str | None:
    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    if not isinstance(content, str) or not content.strip():
        return None
    return content
