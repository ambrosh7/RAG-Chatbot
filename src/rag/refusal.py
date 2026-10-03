"""Fixed refusal text. These builders do not retrieve and do not call a model.

Out-of-scope replies have an empty ``searched`` list: the corpus was not
consulted. ``not_in_corpus`` names the documents that were searched.
``generation_unavailable`` says the answer could not be generated. It does
not say the guidance was searched and found empty.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.constants import CORPUS
from src.rag.classifier import NUTRIENT_LOOKUP, PERSONAL_REASONS, UnknownDocumentError

# Architecture §8.2 and §9. The apostrophe is U+2019, matching those strings.
PROFESSIONAL_REFERRAL = (
    "I can\u2019t help with medical advice or with personal calorie or weight targets. "
    "Please speak with a qualified health professional."
)
NUTRIENT_LOOKUP_MESSAGE = (
    "This assistant answers from dietary guidance documents and does not look up "
    "nutrient numbers for individual foods."
)
NOT_IN_CORPUS_MESSAGE = "The guidance I searched does not cover that."
GENERATION_UNAVAILABLE_MESSAGE = (
    "I couldn\u2019t generate an answer just now. Please try again."
)
NO_SUPPORTING_CHUNKS = "no_supporting_chunks"
NO_SURVIVING_CLAIMS = "no_surviving_claims"
MODEL_ERROR = "model_error"

_REGISTRY = {document.document_id: document for document in CORPUS}


def out_of_scope(reason: str) -> dict[str, object]:
    """Decline a question the guard refused. ``searched`` is empty."""
    if reason == NUTRIENT_LOOKUP:
        message = NUTRIENT_LOOKUP_MESSAGE
    elif reason in PERSONAL_REASONS:
        message = PROFESSIONAL_REFERRAL
    else:
        raise ValueError(f"unknown out-of-scope reason: {reason}")
    return _body("out_of_scope", reason, message, searched=[])


def not_in_corpus(
    document_ids: Sequence[str] | None = None,
    *,
    reason: str = NO_SUPPORTING_CHUNKS,
) -> dict[str, object]:
    """The question is in scope and the searched guidance does not cover it.

    ``document_ids`` None is the whole corpus, in registry order. A subset
    is also returned in registry order, so a filtered miss names only the
    document that was searched.
    """
    if not reason:
        raise ValueError("not_in_corpus needs a reason")
    return _body(
        "not_in_corpus",
        reason,
        NOT_IN_CORPUS_MESSAGE,
        searched=searched_entries(document_ids),
    )


def generation_unavailable(reason: str = MODEL_ERROR) -> dict[str, object]:
    """The model did not return claims. This is not a corpus miss."""
    if not reason:
        raise ValueError("generation_unavailable needs a reason")
    return _body("generation_unavailable", reason, GENERATION_UNAVAILABLE_MESSAGE, searched=[])


def searched_entries(document_ids: Sequence[str] | None = None) -> list[dict[str, object]]:
    """Name, publisher, and year for the searched scope, in registry order."""
    if document_ids is None:
        documents = CORPUS
    else:
        if len(document_ids) == 0:
            raise ValueError("not_in_corpus needs the searched scope")
        unknown = [document_id for document_id in document_ids if document_id not in _REGISTRY]
        if unknown:
            raise UnknownDocumentError(
                f"document_id {unknown[0]} is not in the registry"
            )
        wanted = set(document_ids)
        documents = tuple(document for document in CORPUS if document.document_id in wanted)
    return [
        {
            "document_id": document.document_id,
            "document_name": document.document_name,
            "publisher": document.publisher,
            "year": document.year,
        }
        for document in documents
    ]


def _body(
    type_: str,
    reason: str,
    message: str,
    *,
    searched: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "type": type_,
        "reason": reason,
        "message": message,
        "sections": [],
        "searched": searched,
    }
