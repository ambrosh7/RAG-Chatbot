"""Response shapes for POST /chat, and the claim draft the model is allowed to return.

The model draft has text and chunk ids only. Name, publisher, year, source
URL, and section heading are filled later from the chunk record.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftClaim:
    """One claim as the model returned it. Publisher and URL are not fields."""

    text: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class DraftDocument:
    """Claims the model attributed to one document."""

    document_id: str
    claims: tuple[DraftClaim, ...]


@dataclass(frozen=True)
class Claim:
    """A claim that survived validation, with the heading of its cited chunk."""

    text: str
    chunk_ids: tuple[str, ...]
    section_heading: str

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "chunk_ids": list(self.chunk_ids),
            "section_heading": self.section_heading,
        }


@dataclass(frozen=True)
class Section:
    """One document's surviving claims. The link is that document's source URL."""

    document_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    claims: tuple[Claim, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "publisher": self.publisher,
            "year": self.year,
            "source_url": self.source_url,
            "claims": [claim.as_dict() for claim in self.claims],
        }


@dataclass(frozen=True)
class SearchedDocument:
    """One document in the scope that was searched."""

    document_id: str
    document_name: str
    publisher: str
    year: int

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "publisher": self.publisher,
            "year": self.year,
        }


@dataclass(frozen=True)
class ChatResponse:
    """An answer, or one of the three refusals.

    ``answer`` has sections and searched, and no reason. A refusal has a
    reason and a fixed message. ``searched`` is empty when the corpus was
    not consulted.
    """

    type: str
    sections: tuple[Section, ...] = ()
    searched: tuple[SearchedDocument, ...] = ()
    reason: str | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, object]:
        sections = [section.as_dict() for section in self.sections]
        searched = [document.as_dict() for document in self.searched]
        if self.type == "answer":
            return {"type": self.type, "sections": sections, "searched": searched}
        return {
            "type": self.type,
            "reason": self.reason,
            "message": self.message,
            "sections": sections,
            "searched": searched,
        }
