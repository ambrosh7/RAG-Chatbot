"""Turn a ``POST /chat`` body into the blocks the page renders.

Answers stay one block per document, in registry order. A citation uses
that section's source URL. Out of scope never carries a searched list.
Not-in-corpus does. Generation failure is its own kind.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.constants import CORPUS

_ORDER = {document.document_id: index for index, document in enumerate(CORPUS)}


class ReplyShapeError(ValueError):
    """The chat response is missing a field the page renders."""


@dataclass(frozen=True)
class ClaimView:
    """One claim and the citation printed under it."""

    text: str
    section_heading: str
    document_name: str
    publisher: str
    year: int
    source_url: str


@dataclass(frozen=True)
class SectionView:
    """One document's claims. The link is that document's source URL."""

    document_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    claims: tuple[ClaimView, ...]


@dataclass(frozen=True)
class SearchedView:
    """A document named in a not-in-corpus reply."""

    document_name: str
    publisher: str
    year: int


@dataclass(frozen=True)
class ReplyView:
    """What the page draws for one assistant turn.

    ``kind`` is the response ``type``. ``searched`` is filled only for
    ``not_in_corpus``.
    """

    kind: str
    message: str
    sections: tuple[SectionView, ...]
    searched: tuple[SearchedView, ...]


def present(body: dict[str, object]) -> ReplyView:
    """Build a view from the pipeline JSON. Unknown shapes raise."""
    kind = body.get("type")
    if kind == "answer":
        return ReplyView(
            kind="answer",
            message="",
            sections=_sections(body.get("sections")),
            searched=(),
        )
    if kind == "not_in_corpus":
        return ReplyView(
            kind="not_in_corpus",
            message=_message(body),
            sections=(),
            searched=_searched(body.get("searched")),
        )
    if kind == "out_of_scope":
        return ReplyView(
            kind="out_of_scope",
            message=_message(body),
            sections=(),
            searched=(),
        )
    if kind == "generation_unavailable":
        return ReplyView(
            kind="generation_unavailable",
            message=_message(body),
            sections=(),
            searched=(),
        )
    raise ReplyShapeError("response type is missing")


def _sections(raw: object) -> tuple[SectionView, ...]:
    if not isinstance(raw, list):
        raise ReplyShapeError("answer is missing sections")
    sections: list[SectionView] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ReplyShapeError("section is not an object")
        document_id = _text(entry.get("document_id"), "section document_id")
        document_name = _text(entry.get("document_name"), "section document_name")
        publisher = _text(entry.get("publisher"), "section publisher")
        year = _year(entry.get("year"))
        source_url = _text(entry.get("source_url"), "section source_url")
        claims_raw = entry.get("claims")
        if not isinstance(claims_raw, list) or not claims_raw:
            raise ReplyShapeError("section is missing claims")
        claims: list[ClaimView] = []
        for claim in claims_raw:
            if not isinstance(claim, dict):
                raise ReplyShapeError("claim is not an object")
            claims.append(
                ClaimView(
                    text=_text(claim.get("text"), "claim text"),
                    section_heading=_text(claim.get("section_heading"), "claim section_heading"),
                    document_name=document_name,
                    publisher=publisher,
                    year=year,
                    source_url=source_url,
                )
            )
        sections.append(
            SectionView(
                document_id=document_id,
                document_name=document_name,
                publisher=publisher,
                year=year,
                source_url=source_url,
                claims=tuple(claims),
            )
        )
    sections.sort(key=lambda section: _ORDER.get(section.document_id, len(_ORDER)))
    return tuple(sections)


def _searched(raw: object) -> tuple[SearchedView, ...]:
    if not isinstance(raw, list):
        raise ReplyShapeError("not_in_corpus is missing searched")
    rows: list[SearchedView] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ReplyShapeError("searched entry is not an object")
        rows.append(
            SearchedView(
                document_name=_text(entry.get("document_name"), "searched document_name"),
                publisher=_text(entry.get("publisher"), "searched publisher"),
                year=_year(entry.get("year")),
            )
        )
    return tuple(rows)


def _message(body: dict[str, object]) -> str:
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ReplyShapeError("response is missing a message")
    return message


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplyShapeError(f"{label} is missing")
    return value


def _year(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplyShapeError("year is missing")
    return value
