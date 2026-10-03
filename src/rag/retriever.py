"""Vector search over the published guidance collection.

Unfiltered search reads one collection, keeps a candidate window, then
caps how many chunks one document can contribute. Filtered search stays
inside one registry document and does not apply that cap. Near-duplicate
rows are not collapsed: the chunk text is what distinguishes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import chromadb

from src.constants import (
    BGE_MAX_SEQUENCE_LENGTH,
    BGE_QUERY_PREFIX,
    CANDIDATE_K,
    CORPUS,
    MAX_CHUNKS_PER_DOCUMENT,
    MIN_SIMILARITY,
    TOP_K_ALL,
    TOP_K_FILTERED,
)
from src.ingestion.indexer import BgeEmbedder, open_collection

_REGISTRY = {document.document_id: document for document in CORPUS}
_REGISTRY_ORDER = {document.document_id: index for index, document in enumerate(CORPUS)}

_embedder: BgeEmbedder | None = None
_collection: chromadb.Collection | None = None


class RetrievalError(Exception):
    """The query cannot be searched. Chroma was not called."""


class Embedder(Protocol):
    """Token count and one embedding. Passages and queries both use ``embed``."""

    def count_tokens(self, text: str) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorCollection(Protocol):
    """The part of a Chroma collection this search calls."""

    def query(self, **kwargs: Any) -> chromadb.QueryResult: ...


@dataclass(frozen=True)
class ScopeDocument:
    """One document in the searched scope."""

    document_id: str
    document_name: str
    publisher: str
    year: int


@dataclass(frozen=True)
class Hit:
    """One neighbour. ``similarity`` is ``1 - distance``."""

    chunk_id: str
    text: str
    similarity: float
    document_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    section_heading: str
    block_type: str
    retrieval_date: str


@dataclass(frozen=True)
class DocumentHits:
    """Kept hits for one document, in the order they survived the walk."""

    document_id: str
    hits: tuple[Hit, ...]


@dataclass(frozen=True)
class RetrievalResult:
    """Scope, grouped hits, and the window counted before the threshold.

    ``candidates`` is that window, including neighbours below
    ``MIN_SIMILARITY``. ``groups`` is empty when none survive. An empty
    group list is a result, not an error.
    """

    scope: tuple[ScopeDocument, ...]
    groups: tuple[DocumentHits, ...]
    candidate_count: int
    candidates: tuple[Hit, ...]


def retrieve(
    query: str,
    document_id: str | None = None,
    *,
    embedder: Embedder | None = None,
    collection: VectorCollection | None = None,
) -> RetrievalResult:
    """Search all seven documents, or the one named by ``document_id``."""
    scope = _scope(document_id)
    prefixed = BGE_QUERY_PREFIX + query
    if prefixed.count(BGE_QUERY_PREFIX) != 1:
        raise RetrievalError("query embedding would contain the BGE prefix twice")
    model = embedder or _default_embedder()
    token_count = model.count_tokens(prefixed)
    if token_count > BGE_MAX_SEQUENCE_LENGTH:
        raise RetrievalError(
            f"query is {token_count} tokens, over {BGE_MAX_SEQUENCE_LENGTH}; not truncated"
        )
    vector = model.embed([prefixed])[0]
    store = collection or _default_collection()
    filtered = document_id is not None
    found = store.query(
        query_embeddings=[vector],
        n_results=TOP_K_FILTERED if filtered else CANDIDATE_K,
        where={"document_id": document_id} if filtered else None,
        include=["documents", "metadatas", "distances"],
    )
    candidates = _hits_from_query(found)
    kept = _keep(candidates, filtered=filtered)
    return RetrievalResult(
        scope=scope,
        groups=_group(kept),
        candidate_count=len(candidates),
        candidates=tuple(candidates),
    )


def _scope(document_id: str | None) -> tuple[ScopeDocument, ...]:
    if document_id is None:
        documents = CORPUS
    else:
        document = _REGISTRY.get(document_id)
        if document is None:
            raise RetrievalError(f"document_id {document_id} is not in the registry")
        documents = (document,)
    return tuple(
        ScopeDocument(
            document_id=document.document_id,
            document_name=document.document_name,
            publisher=document.publisher,
            year=document.year,
        )
        for document in documents
    )


def _default_embedder() -> BgeEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = BgeEmbedder()
    return _embedder


def _default_collection() -> VectorCollection:
    global _collection
    if _collection is None:
        _collection = open_collection()
    return _collection


def _hits_from_query(found: chromadb.QueryResult) -> list[Hit]:
    ids = (found.get("ids") or [[]])[0]
    documents = (found.get("documents") or [[]])[0]
    metadatas = (found.get("metadatas") or [[]])[0]
    distances = (found.get("distances") or [[]])[0]
    hits: list[Hit] = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        if not metadata or text is None or distance is None:
            raise RetrievalError(f"{chunk_id} is missing text or metadata")
        document_id = str(metadata["document_id"])
        if document_id not in _REGISTRY_ORDER:
            raise RetrievalError(f"{chunk_id} has document_id {document_id} outside the registry")
        hits.append(
            Hit(
                chunk_id=str(chunk_id),
                text=str(text),
                similarity=1.0 - float(distance),
                document_id=document_id,
                document_name=str(metadata["document_name"]),
                publisher=str(metadata["publisher"]),
                year=int(metadata["year"]),
                source_url=str(metadata["source_url"]),
                section_heading=str(metadata["section_heading"]),
                block_type=str(metadata["block_type"]),
                retrieval_date=str(metadata["retrieval_date"]),
            )
        )
    hits.sort(
        key=lambda hit: (
            -round(hit.similarity, 6),
            _REGISTRY_ORDER[hit.document_id],
            hit.chunk_id,
        )
    )
    return hits


def _keep(candidates: list[Hit], *, filtered: bool) -> list[Hit]:
    eligible = [hit for hit in candidates if hit.similarity >= MIN_SIMILARITY]
    if filtered:
        return eligible[:TOP_K_FILTERED]
    kept: list[Hit] = []
    per_document: dict[str, int] = {}
    for hit in eligible:
        if per_document.get(hit.document_id, 0) >= MAX_CHUNKS_PER_DOCUMENT:
            continue
        kept.append(hit)
        per_document[hit.document_id] = per_document.get(hit.document_id, 0) + 1
        if len(kept) >= TOP_K_ALL:
            break
    return kept


def _group(kept: list[Hit]) -> tuple[DocumentHits, ...]:
    by_document: dict[str, list[Hit]] = {}
    for hit in kept:
        by_document.setdefault(hit.document_id, []).append(hit)
    return tuple(
        DocumentHits(document_id=document.document_id, hits=tuple(by_document[document.document_id]))
        for document in CORPUS
        if document.document_id in by_document
    )
