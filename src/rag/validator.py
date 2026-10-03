"""Drop claims the retrieved chunks do not support.

A claim survives only when every cited chunk was retrieved for that same
document, the citation URL is the registry URL, every number in the claim
appears in those chunks, and at least half of its content words appear
there too. Personal medical, calorie, and weight wording is dropped even
when a chunk mentions the same words. A claim that says what the guidelines
say is dropped. The renderer does not add that sentence; this stops the
model from adding it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.constants import CORPUS, MIN_CLAIM_TERM_OVERLAP
from src.rag.classifier import PERSONAL_REASONS, scope_reason
from src.rag.models import Claim, Section
from src.rag.retriever import DocumentHits, Hit

EMPTY_TEXT = "empty_text"
MISSING_CHUNK = "missing_chunk"
FOREIGN_CHUNK = "foreign_chunk"
BAD_CITATION_URL = "bad_citation_url"
BLENDED_VOICE = "blended_voice"
UNSUPPORTED_NUMBER = "unsupported_number"
LOW_TERM_OVERLAP = "low_term_overlap"

_REGISTRY = {document.document_id: document for document in CORPUS}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z]+")
_BLENDED = re.compile(r"\bthe guidelines(?:\s+collectively)?\s+say\b", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "by",
        "that",
        "this",
        "it",
        "its",
        "as",
        "at",
        "from",
        "into",
        "than",
        "then",
        "so",
        "if",
        "but",
        "not",
        "no",
        "can",
        "should",
        "would",
        "could",
        "may",
        "might",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "their",
        "there",
        "these",
        "those",
        "such",
        "per",
        "about",
        "over",
        "under",
        "less",
        "more",
        "your",
        "you",
    }
)


@dataclass(frozen=True)
class DroppedClaim:
    """Why a claim was removed. Phase 7 logs the reason code."""

    reason: str
    document_id: str
    text: str


@dataclass(frozen=True)
class ValidationResult:
    """Surviving sections, in registry order, and the claims that were removed."""

    sections: tuple[Section, ...]
    dropped: tuple[DroppedClaim, ...]


def validate(sections: Sequence[Section], groups: Sequence[DocumentHits]) -> ValidationResult:
    """Keep claims the cited chunks support. Omit a document when none remain."""
    own_hits = {group.document_id: {hit.chunk_id: hit for hit in group.hits} for group in groups}
    all_hits = {hit.chunk_id: hit for group in groups for hit in group.hits}
    kept: list[Section] = []
    dropped: list[DroppedClaim] = []
    for section in sections:
        registry_url = _REGISTRY[section.document_id].source_url
        available = own_hits.get(section.document_id, {})
        surviving: list[Claim] = []
        for claim in section.claims:
            reason = _drop_reason(claim, section, available, all_hits, registry_url)
            if reason is not None:
                dropped.append(
                    DroppedClaim(reason=reason, document_id=section.document_id, text=claim.text)
                )
                continue
            surviving.append(claim)
        if surviving:
            kept.append(_section_from_hits(section, surviving, available))
    return ValidationResult(sections=tuple(kept), dropped=tuple(dropped))


def _drop_reason(
    claim: Claim,
    section: Section,
    available: dict[str, Hit],
    all_hits: dict[str, Hit],
    registry_url: str,
) -> str | None:
    if not claim.text.strip():
        return EMPTY_TEXT
    membership = _membership(claim.chunk_ids, available, all_hits)
    if membership is not None:
        return membership
    cited = [available[chunk_id] for chunk_id in claim.chunk_ids]
    if section.source_url != registry_url or any(hit.source_url != registry_url for hit in cited):
        return BAD_CITATION_URL
    advice = scope_reason(claim.text)
    if advice in PERSONAL_REASONS:
        return advice
    if _BLENDED.search(claim.text):
        return BLENDED_VOICE
    support = "\n".join(hit.text for hit in cited)
    if not _numbers_supported(claim.text, support):
        return UNSUPPORTED_NUMBER
    if not _overlap_supported(claim.text, support):
        return LOW_TERM_OVERLAP
    return None


def _membership(
    chunk_ids: tuple[str, ...],
    available: dict[str, Hit],
    all_hits: dict[str, Hit],
) -> str | None:
    if not chunk_ids:
        return MISSING_CHUNK
    missing = False
    for chunk_id in chunk_ids:
        if chunk_id in available:
            continue
        if chunk_id in all_hits:
            return FOREIGN_CHUNK
        missing = True
    if missing:
        return MISSING_CHUNK
    return None


def _section_from_hits(
    section: Section,
    claims: list[Claim],
    available: dict[str, Hit],
) -> Section:
    """Name, publisher, year, and URL come from the cited chunk, not the model."""
    anchor = available[claims[0].chunk_ids[0]]
    headed = tuple(
        Claim(
            text=claim.text,
            chunk_ids=claim.chunk_ids,
            section_heading=available[claim.chunk_ids[0]].section_heading,
        )
        for claim in claims
    )
    return Section(
        document_id=section.document_id,
        document_name=anchor.document_name,
        publisher=anchor.publisher,
        year=anchor.year,
        source_url=anchor.source_url,
        claims=headed,
    )


def _numbers_supported(claim: str, support: str) -> bool:
    for match in _NUMBER.finditer(claim):
        number = re.escape(match.group())
        if re.search(rf"(?<!\d){number}(?!\d)", support) is None:
            return False
    return True


def _overlap_supported(claim: str, support: str) -> bool:
    words = _content_words(claim)
    if not words:
        return True
    support_words = set(_content_words(support))
    matched = sum(1 for word in words if word in support_words)
    return matched / len(words) >= MIN_CLAIM_TERM_OVERLAP


def _content_words(text: str) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for word in _WORD.findall(text.casefold()):
        if word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        found.append(word)
    return tuple(found)
