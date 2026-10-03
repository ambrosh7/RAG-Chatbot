"""POST /chat client for the chat page."""

from __future__ import annotations

import httpx


class ChatApiError(Exception):
    """The page could not read a chat response. This is not a corpus refusal."""


def post_chat(
    base_url: str,
    message: str,
    document_id: str | None,
    *,
    timeout: float = 120.0,
) -> dict[str, object]:
    """Send one question. ``document_id`` None is JSON null, which searches all."""
    url = base_url.rstrip("/") + "/chat"
    try:
        response = httpx.post(
            url,
            json={"message": message, "document_id": document_id},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise ChatApiError("The chat API could not be reached.") from exc
    if response.status_code >= 400:
        raise ChatApiError(f"The chat API returned {response.status_code}.")
    try:
        body = response.json()
    except ValueError as exc:
        raise ChatApiError("The chat API returned an unexpected response.") from exc
    if not isinstance(body, dict):
        raise ChatApiError("The chat API returned an unexpected response.")
    return body
