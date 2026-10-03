"""Answer a question from the corpus, or refuse it.

Order is fixed. An empty message is rejected. The scope guard runs next,
and a match returns without retrieval. Otherwise the document scope is the
request id, or the single alias in the message. Retrieval with no surviving
hit returns not-in-corpus and does not call Groq. Surviving hits are sent
for claims, cited from chunk metadata, and checked. A Groq failure is
generation_unavailable, which is a different response from an empty corpus.
Each turn logs the document scope, the kept hit count, and any refusal
reason. Dropped claims log reason codes only, not the model JSON.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.rag.citations import render
from src.rag.classifier import EmptyMessageError, classify
from src.rag.generator import ChatClient, GenerationError, generate
from src.rag.models import ChatResponse, SearchedDocument
from src.rag.refusal import (
    NO_SURVIVING_CLAIMS,
    generation_unavailable,
    not_in_corpus,
    out_of_scope,
)
from src.rag.retriever import RetrievalResult, retrieve
from src.rag.validator import validate

logger = logging.getLogger(__name__)


def answer(
    message: str,
    document_id: str | None = None,
    *,
    retriever: Callable[..., RetrievalResult] | None = None,
    client: ChatClient | None = None,
) -> dict[str, object]:
    """Return the architecture JSON for one user message."""
    if not isinstance(message, str) or not message.strip():
        raise EmptyMessageError("message is empty")
    decision = classify(message, document_id)
    if decision.reason is not None:
        return _log_turn(out_of_scope(decision.reason), document_scope="none", hit_count=0)
    search = retriever or retrieve
    found = search(message, decision.document_id)
    scope_ids = [document.document_id for document in found.scope]
    document_scope = decision.document_id or "all"
    hit_count = sum(len(group.hits) for group in found.groups)
    if not found.groups:
        return _log_turn(
            not_in_corpus(scope_ids),
            document_scope=document_scope,
            hit_count=hit_count,
        )
    try:
        draft = generate(message, found.groups, client=client)
    except GenerationError as exc:
        cause = exc.__cause__
        logger.info(
            "generation_error=%s status=%s",
            exc,
            getattr(cause, "status_code", ""),
        )
        return _log_turn(
            generation_unavailable(),
            document_scope=document_scope,
            hit_count=hit_count,
        )
    checked = validate(render(draft, found.groups), found.groups)
    # Reason codes only. The model JSON and the claim text stay out of the log.
    logger.info(
        "dropped_claim_reasons=%s",
        ",".join(item.reason for item in checked.dropped),
    )
    if not checked.sections:
        return _log_turn(
            not_in_corpus(scope_ids, reason=NO_SURVIVING_CLAIMS),
            document_scope=document_scope,
            hit_count=hit_count,
        )
    return _log_turn(
        ChatResponse(
            type="answer",
            sections=checked.sections,
            searched=tuple(
                SearchedDocument(
                    document_id=document.document_id,
                    document_name=document.document_name,
                    publisher=document.publisher,
                    year=document.year,
                )
                for document in found.scope
            ),
        ).as_dict(),
        document_scope=document_scope,
        hit_count=hit_count,
    )


def _log_turn(
    body: dict[str, object],
    *,
    document_scope: str,
    hit_count: int,
) -> dict[str, object]:
    """Log scope, kept hits, and a refusal reason. The body is unchanged."""
    logger.info(
        "document_scope=%s hit_count=%s type=%s reason=%s",
        document_scope,
        hit_count,
        body.get("type", ""),
        body.get("reason") or "",
    )
    return body
