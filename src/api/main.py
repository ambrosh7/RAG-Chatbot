"""HTTP entry for the guidance chat.

``POST /chat`` returns the pipeline JSON unchanged. An empty message or a
``document_id`` outside the registry is rejected before the pipeline runs.
``GET /health`` is ok when the Chroma directory and the index manifest exist.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from src.constants import DOCUMENT_IDS, INDEX_DIR, INDEX_MANIFEST_PATH
from src.rag.pipeline import answer

_PIPELINE_LOGGER = "src.rag.pipeline"


class ChatRequest(BaseModel):
    """One question, optionally limited to a registry document."""

    message: str
    document_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_must_have_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message is empty")
        return value

    @field_validator("document_id")
    @classmethod
    def document_must_be_registered(cls, value: str | None) -> str | None:
        if value is not None and value not in DOCUMENT_IDS:
            raise ValueError(f"document_id {value} is not in the registry")
        return value


def configure_logging() -> None:
    """Show pipeline scope, hit count, and refusal lines under uvicorn."""
    chat_log = logging.getLogger(_PIPELINE_LOGGER)
    chat_log.setLevel(logging.INFO)
    if any(getattr(handler, "_guidance_chat", False) for handler in chat_log.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    handler._guidance_chat = True  # type: ignore[attr-defined]
    chat_log.addHandler(handler)


def index_is_ready() -> bool:
    """The published Chroma directory and its manifest are both on disk."""
    return (
        INDEX_DIR.is_dir()
        and (INDEX_DIR / "chroma.sqlite3").is_file()
        and INDEX_MANIFEST_PATH.is_file()
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    yield


app = FastAPI(title="Dietary guidance", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if index_is_ready():
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "unavailable"}, status_code=503)


@app.post("/chat")
def chat(body: ChatRequest) -> JSONResponse:
    return JSONResponse(answer(body.message, body.document_id))
