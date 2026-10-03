"""Retrieval tests. The temp index uses fixed vectors. The real index loads BGE."""

from __future__ import annotations

from pathlib import Path

import chromadb
import pytest

from src.constants import (
    BGE_DIMENSION,
    BGE_MODEL_NAME,
    BGE_QUERY_PREFIX,
    CANDIDATE_K,
    COLLECTION_NAME,
    CORPUS,
    INDEX_DIR,
    MIN_SIMILARITY,
    TOP_K_ALL,
    TOP_K_FILTERED,
)
from src.ingestion.indexer import PrecomputedEmbeddingFunction
from src.rag.retriever import RetrievalError, RetrievalResult, retrieve

REGISTRY = {document.document_id: document for document in CORPUS}
FOOD_SAFETY = frozenset(
    {"cold-food-storage", "kitchen-companion", "fsa-chill", "who-five-keys"}
)
REAL_INDEX = INDEX_DIR / "chroma.sqlite3"


class QueryEmbedder:
    """Returns the unit vector on the first axis. Records the prefixed query."""

    model_name = BGE_MODEL_NAME
    dimension = BGE_DIMENSION
    max_sequence_length = 512

    def __init__(self) -> None:
        self.seen: list[str] = []

    def count_tokens(self, text: str) -> int:
        if "OVER_LIMIT" in text:
            return 600
        return 2 + len(text.split())

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        vector = [0.0] * BGE_DIMENSION
        vector[0] = 1.0
        return [vector]


class RecordingCollection:
    """Stands in for Chroma. ``query`` records its arguments and returns nothing."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def query(self, **kwargs: object) -> dict[str, list[list[object]]]:
        self.calls.append(kwargs)
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def _vector(cosine: float) -> list[float]:
    rest = (1.0 - cosine * cosine) ** 0.5
    vector = [0.0] * BGE_DIMENSION
    vector[0] = cosine
    vector[1] = rest
    return vector


def _collection(tmp_path: Path, rows: list[tuple[str, str, float, str]]) -> chromadb.Collection:
    """Rows are ``(document_id, chunk_id, cosine to the query axis, text)``."""
    client = chromadb.PersistentClient(path=str(tmp_path / "index"))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=PrecomputedEmbeddingFunction(),
    )
    collection.upsert(
        ids=[chunk_id for _, chunk_id, _, _ in rows],
        embeddings=[_vector(cosine) for _, _, cosine, _ in rows],
        documents=[text for _, _, _, text in rows],
        metadatas=[
            {
                "document_id": document_id,
                "document_name": REGISTRY[document_id].document_name,
                "publisher": REGISTRY[document_id].publisher,
                "year": REGISTRY[document_id].year,
                "source_url": REGISTRY[document_id].source_url,
                "section_heading": REGISTRY[document_id].document_name,
                "block_type": "paragraph",
                "retrieval_date": "2026-10-03",
            }
            for document_id, _, _, _ in rows
        ],
    )
    return collection


def _flat(result: RetrievalResult) -> list[str]:
    return [hit.chunk_id for group in result.groups for hit in group.hits]


def test_threshold_constant_is_the_probed_value() -> None:
    assert MIN_SIMILARITY == 0.62
    assert CANDIDATE_K == 24
    assert BGE_MODEL_NAME == "BAAI/bge-small-en-v1.5"


def test_unknown_document_id_does_not_query() -> None:
    embedder = QueryEmbedder()
    collection = RecordingCollection()
    with pytest.raises(RetrievalError, match="not-a-document"):
        retrieve("chicken", "not-a-document", embedder=embedder, collection=collection)
    assert collection.calls == []
    assert embedder.seen == []


def test_query_over_512_tokens_is_not_truncated() -> None:
    embedder = QueryEmbedder()
    collection = RecordingCollection()
    with pytest.raises(RetrievalError, match="not truncated"):
        retrieve("OVER_LIMIT fridge time", embedder=embedder, collection=collection)
    assert collection.calls == []
    assert embedder.seen == []


def test_embedded_query_is_prefixed_once() -> None:
    embedder = QueryEmbedder()
    collection = RecordingCollection()
    question = "How long can I keep a whole chicken in the fridge?"
    retrieve(question, embedder=embedder, collection=collection)
    assert embedder.seen == [BGE_QUERY_PREFIX + question]
    assert embedder.seen[0].count(BGE_QUERY_PREFIX) == 1
    call = collection.calls[0]
    assert call["n_results"] == CANDIDATE_K
    assert call["where"] is None
    assert "query_embeddings" in call
    assert "query_texts" not in call


def test_filtered_search_requests_five_and_ignores_a_nearer_other_document(tmp_path: Path) -> None:
    collection = _collection(
        tmp_path,
        [
            ("kitchen-companion", "kitchen-companion:1", 0.95, "Nearer chicken row in another handbook."),
            ("eatwell-guide", "eatwell-guide:5", 0.80, "Eat at least 5 portions of fruit and vegetables."),
            ("eatwell-guide", "eatwell-guide:6", 0.70, "Fruit and vegetables should make up over a third."),
        ],
    )
    result = retrieve("fruit", "eatwell-guide", embedder=QueryEmbedder(), collection=collection)
    assert [document.document_id for document in result.scope] == ["eatwell-guide"]
    assert _flat(result) == ["eatwell-guide:5", "eatwell-guide:6"]
    assert {hit.document_id for group in result.groups for hit in group.hits} == {"eatwell-guide"}


def test_filtered_search_keeps_five_from_one_document(tmp_path: Path) -> None:
    rows = [
        ("eatwell-guide", f"eatwell-guide:{index}", 0.95 - index * 0.02, f"Fruit line {index}.")
        for index in range(6)
    ]
    collection = _collection(tmp_path, rows)
    result = retrieve("fruit", "eatwell-guide", embedder=QueryEmbedder(), collection=collection)
    assert _flat(result) == [f"eatwell-guide:{index}" for index in range(TOP_K_FILTERED)]
    assert len(result.groups) == 1
    assert len(result.groups[0].hits) == 5


def test_filtered_search_does_not_borrow_when_the_document_is_short(tmp_path: Path) -> None:
    collection = _collection(
        tmp_path,
        [
            ("fsa-chill", "fsa-chill:0", 0.90, "Cool food and refrigerate it within one to two hours."),
            ("kitchen-companion", "kitchen-companion:1", 0.99, "A nearer leftovers rule."),
        ],
    )
    result = retrieve("leftovers", "fsa-chill", embedder=QueryEmbedder(), collection=collection)
    assert _flat(result) == ["fsa-chill:0"]
    assert result.candidate_count == 1


def test_unfiltered_cap_lets_a_second_document_through(tmp_path: Path) -> None:
    rows = [
        ("kitchen-companion", f"kitchen-companion:{index}", 0.95 - index * 0.001, f"Handbook line {index}.")
        for index in range(16)
    ]
    rows.extend(
        [
            ("cold-food-storage", "cold-food-storage:24", 0.80, "Whole chicken, 1 to 2 days, 1 year."),
            ("fsa-chill", "fsa-chill:5", 0.78, "Cool cooked food before the fridge."),
            ("who-five-keys", "who-five-keys:2", 0.76, "Keep food at safe temperatures."),
        ]
    )
    collection = _collection(tmp_path, rows)
    result = retrieve("chicken", embedder=QueryEmbedder(), collection=collection)
    flat = _flat(result)
    assert len(flat) <= TOP_K_ALL
    assert sum(chunk_id.startswith("kitchen-companion:") for chunk_id in flat) == 3
    assert "cold-food-storage:24" in flat
    assert len({hit.document_id for group in result.groups for hit in group.hits}) > 1
    assert [document.document_id for document in result.scope] == [document.document_id for document in CORPUS]


def test_unfiltered_search_stops_at_eight_after_the_cap(tmp_path: Path) -> None:
    rows: list[tuple[str, str, float, str]] = []
    score = 0.99
    for document_id in ("cold-food-storage", "who-healthy-diet", "eatwell-guide", "fsa-chill"):
        for index in range(3):
            rows.append((document_id, f"{document_id}:{index}", score, f"{document_id} line {index}"))
            score -= 0.01
    collection = _collection(tmp_path, rows)
    result = retrieve("guidance", embedder=QueryEmbedder(), collection=collection)
    flat = _flat(result)
    assert len(flat) == TOP_K_ALL
    counts: dict[str, int] = {}
    for group in result.groups:
        counts[group.document_id] = len(group.hits)
        assert len(group.hits) <= 3
    assert sum(counts.values()) == TOP_K_ALL


def test_equal_similarity_uses_registry_order_then_chunk_id(tmp_path: Path) -> None:
    collection = _collection(
        tmp_path,
        [
            ("kitchen-companion", "kitchen-companion:2", 0.90, "Same score, later id."),
            ("kitchen-companion", "kitchen-companion:10", 0.90, "Same score, earlier id."),
            ("cold-food-storage", "cold-food-storage:1", 0.90, "Same score, earlier document."),
        ],
    )
    result = retrieve("storage", embedder=QueryEmbedder(), collection=collection)
    assert _flat(result) == [
        "cold-food-storage:1",
        "kitchen-companion:10",
        "kitchen-companion:2",
    ]


def test_threshold_keeps_an_equal_hit_and_drops_one_below(tmp_path: Path) -> None:
    collection = _collection(
        tmp_path,
        [
            ("eatwell-guide", "eatwell-guide:96", MIN_SIMILARITY, "Choose oils high in unsaturated fat."),
            ("eatwell-guide", "eatwell-guide:14", MIN_SIMILARITY - 0.02, "at home cooking"),
        ],
    )
    result = retrieve("oil", embedder=QueryEmbedder(), collection=collection)
    assert _flat(result) == ["eatwell-guide:96"]
    assert result.candidate_count == 2
    assert result.groups[0].hits[0].similarity >= MIN_SIMILARITY


def test_every_candidate_under_the_threshold_returns_an_empty_group_list(tmp_path: Path) -> None:
    collection = _collection(
        tmp_path,
        [("eatwell-guide", "eatwell-guide:30", 0.50, "Saturated fat, not a tariff.")],
    )
    result = retrieve("tariff", embedder=QueryEmbedder(), collection=collection)
    assert result.groups == ()
    assert result.candidate_count == 1
    assert [document.document_id for document in result.scope] == [document.document_id for document in CORPUS]


@pytest.mark.skipif(not REAL_INDEX.is_file(), reason="data/index has not been built")
def test_real_index_chicken_leftovers_oil_and_fruit() -> None:
    chicken = retrieve("How long can I keep a whole chicken in the fridge?")
    chicken_ids = _flat(chicken)
    assert "cold-food-storage:24" in chicken_ids
    whole = next(
        hit
        for group in chicken.groups
        for hit in group.hits
        if hit.chunk_id == "cold-food-storage:24"
    )
    assert "Refrigerator" in whole.text
    assert "Freezer" in whole.text
    assert "cold-food-storage:25" in chicken_ids
    assert len(chicken_ids) <= TOP_K_ALL
    assert all(len(group.hits) <= 3 for group in chicken.groups)
    assert [document.document_id for document in chicken.scope] == [
        document.document_id for document in CORPUS
    ]

    leftovers = retrieve("How long can I keep cooked leftovers in the fridge?")
    leftover_docs = {group.document_id for group in leftovers.groups}
    assert len(leftover_docs & FOOD_SAFETY) >= 2

    oil = retrieve("What do the documents say about cooking oil?")
    assert any(group.document_id == "eatwell-guide" for group in oil.groups)

    fruit = retrieve(
        "According to the Eatwell Guide, how much of the diet should be fruit and vegetables?",
        "eatwell-guide",
    )
    assert fruit.groups
    assert {group.document_id for group in fruit.groups} == {"eatwell-guide"}
    assert all(hit.document_id == "eatwell-guide" for group in fruit.groups for hit in group.hits)


@pytest.mark.skipif(not REAL_INDEX.is_file(), reason="data/index has not been built")
def test_real_index_misses_are_empty() -> None:
    tariff = retrieve("What is the tariff on imported olive oil?")
    assert tariff.groups == ()
    assert [document.document_id for document in tariff.scope] == [
        document.document_id for document in CORPUS
    ]

    chicken = retrieve("How long can raw chicken stay in the fridge?", "eatwell-guide")
    assert chicken.groups == ()
    assert [document.document_id for document in chicken.scope] == ["eatwell-guide"]
