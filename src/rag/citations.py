"""Attach citations from chunk metadata. The model text is not rewritten.

Sections follow the corpus registry. A document with no claims is omitted.
The claim text is copied as the model wrote it. This step does not add a
sentence about what the guidelines say, and it does not read a publisher or
URL off the model object.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.constants import CORPUS
from src.rag.models import Claim, DraftClaim, DraftDocument, Section
from src.rag.retriever import DocumentHits, Hit

_REGISTRY = {document.document_id: document for document in CORPUS}


def render(
    drafts: Sequence[DraftDocument],
    groups: Sequence[DocumentHits],
) -> tuple[Section, ...]:
    """One section per document that the model gave claims for, in registry order."""
    by_chunk = {hit.chunk_id: hit for group in groups for hit in group.hits}
    by_document: dict[str, list[DraftClaim]] = {}
    for draft in drafts:
        by_document.setdefault(draft.document_id, []).extend(draft.claims)
    group_hits = {group.document_id: group.hits for group in groups}
    sections: list[Section] = []
    for document in CORPUS:
        claims = by_document.get(document.document_id)
        if not claims:
            continue
        own_hits = group_hits.get(document.document_id, ())
        anchor = own_hits[0] if own_hits else None
        rendered = tuple(_claim(claim, by_chunk, document.document_id) for claim in claims)
        sections.append(
            Section(
                document_id=document.document_id,
                document_name=anchor.document_name if anchor else document.document_name,
                publisher=anchor.publisher if anchor else document.publisher,
                year=anchor.year if anchor else document.year,
                source_url=anchor.source_url if anchor else document.source_url,
                claims=rendered,
            )
        )
    return tuple(sections)


def _claim(claim: DraftClaim, by_chunk: dict[str, Hit], document_id: str) -> Claim:
    cited = [
        by_chunk[chunk_id]
        for chunk_id in claim.chunk_ids
        if chunk_id in by_chunk and by_chunk[chunk_id].document_id == document_id
    ]
    heading = cited[0].section_heading if cited else ""
    return Claim(text=claim.text, chunk_ids=claim.chunk_ids, section_heading=heading)
