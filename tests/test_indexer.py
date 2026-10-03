"""Indexer tests. Embeddings are fake. No test here downloads BGE or opens a socket."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.constants import (
    BGE_DIMENSION,
    BGE_MAX_SEQUENCE_LENGTH,
    BGE_MODEL_NAME,
    BGE_QUERY_PREFIX,
    BLOCKS_PATH,
    CORPUS,
    CORPUS_MANIFEST_PATH,
    PROJECT_ROOT,
)
from src.ingestion.chunker import Chunk, chunk_corpus, is_form_blank
from src.ingestion.indexer import (
    IndexBuildError,
    fingerprint_chunks,
    open_collection,
    publish_index,
)

REGISTRY = {document.document_id: document for document in CORPUS}


class FakeEmbedder:
    """Counts tokens by words, except a marked over-long passage."""

    model_name = BGE_MODEL_NAME
    dimension = BGE_DIMENSION
    max_sequence_length = BGE_MAX_SEQUENCE_LENGTH

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.embed_calls = 0

    def count_tokens(self, text: str) -> int:
        if "OVER_LIMIT" in text:
            return 600
        return 2 + len(text.split())

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        self.seen.extend(texts)
        vectors: list[list[float]] = []
        for index, _text in enumerate(texts):
            vector = [0.0] * self.dimension
            vector[0] = 1.0
            vector[1] = (index + 1) / 1000
            vectors.append(vector)
        return vectors


def _chunk(document_id: str, text: str, *, chunk_id: str | None = None, block_type: str = "paragraph") -> Chunk:
    document = REGISTRY[document_id]
    heading = document.document_name
    return Chunk(
        chunk_id=chunk_id or f"{document_id}:0",
        document_id=document_id,
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=document.source_url,
        retrieval_date="2026-10-03",
        section_heading=heading,
        block_type=block_type,
        text=text,
        embed_text=f"{document.document_name} — {heading}\n{text}",
    )


def _seven() -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in CORPUS:
        if document.document_id == "cold-food-storage":
            chunks.append(
                _chunk(
                    document.document_id,
                    "Food: Fresh poultry | Type: Chicken or turkey, whole | "
                    "Refrigerator [40°F (4°C) or below]: 1 to 2 days | "
                    "Freezer [0°F (-18°C) or below]: 1 year",
                    block_type="table_row",
                )
            )
        elif document.document_id == "who-healthy-diet":
            chunks.append(
                _chunk(
                    document.document_id,
                    "Reducing saturated fats to less than 10% of total energy intake "
                    "and trans-fats to less than 1% of total energy intake.",
                    chunk_id="who-healthy-diet:20",
                )
            )
        else:
            chunks.append(_chunk(document.document_id, f"Guidance from {document.document_name}."))
    return chunks


def _write_manifest(path: Path, *, status: str = "ok", missing_file: str | None = None) -> None:
    documents = []
    for document in CORPUS:
        relative = f"data/raw/{document.document_id}/source.{document.kind}"
        if missing_file == document.document_id:
            relative = "data/raw/missing/source.pdf"
        row_status = "stale" if status == "stale" and document.document_id == "fsa-chill" else "ok"
        documents.append(
            {
                "document_id": document.document_id,
                "status": row_status,
                "file": relative,
                "retrieval_date": "2026-10-03",
            }
        )
    path.write_text(json.dumps({"documents": documents}), encoding="utf-8")


def test_metadata_round_trip_and_passage_is_unprefixed(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    index_dir = tmp_path / "index"
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_manifest(manifest_path)
    chunks = _seven()
    manifest = publish_index(
        chunks,
        embedder=embedder,
        index_dir=index_dir,
        manifest_path=manifest_path,
    )
    assert manifest.model_name == BGE_MODEL_NAME
    assert manifest.dimension == BGE_DIMENSION
    assert manifest.max_sequence_length == BGE_MAX_SEQUENCE_LENGTH
    assert manifest.chunk_count == 7
    assert manifest.max_tokens <= BGE_MAX_SEQUENCE_LENGTH
    assert manifest.fingerprint == fingerprint_chunks(chunks)
    assert embedder.seen == [chunk.embed_text for chunk in chunks]
    assert all(not text.startswith(BGE_QUERY_PREFIX) for text in embedder.seen)

    collection = open_collection(index_dir)
    assert collection.count() == 7
    stored = collection.get(ids=["cold-food-storage:0"], include=["documents", "metadatas"])
    assert stored["documents"] == [
        "Food: Fresh poultry | Type: Chicken or turkey, whole | "
        "Refrigerator [40°F (4°C) or below]: 1 to 2 days | "
        "Freezer [0°F (-18°C) or below]: 1 year"
    ]
    metadata = stored["metadatas"][0]
    assert metadata["year"] == 2023
    assert isinstance(metadata["year"], int)
    assert metadata["block_type"] == "table_row"
    assert metadata["source_url"] == REGISTRY["cold-food-storage"].source_url
    assert metadata["document_id"] == "cold-food-storage"
    assert "embed_text" not in metadata
    assert "1 to 2 days" in stored["documents"][0]
    assert "1 year" in stored["documents"][0]


def test_passage_over_512_tokens_is_not_upserted(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    index_dir = tmp_path / "index"
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_manifest(manifest_path)
    chunks = _seven()
    chunks[0] = _chunk("cold-food-storage", "OVER_LIMIT chicken row that must not be truncated")
    with pytest.raises(IndexBuildError, match="cold-food-storage:0"):
        publish_index(chunks, embedder=embedder, index_dir=index_dir, manifest_path=manifest_path)
    assert embedder.embed_calls == 0
    assert not (index_dir / "index_manifest.json").exists()
    assert not list(index_dir.rglob("chroma.sqlite3")) if index_dir.exists() else True


def test_who_five_keys_form_chunk_is_rejected_by_name(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    index_dir = tmp_path / "index"
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_manifest(manifest_path)
    chunks = _seven()
    for index, chunk in enumerate(chunks):
        if chunk.document_id == "who-five-keys":
            chunks[index] = _chunk(
                "who-five-keys",
                "OVER_LIMIT " + ("_" * 40),
                chunk_id="who-five-keys:79",
            )
    with pytest.raises(IndexBuildError, match="who-five-keys:79") as caught:
        publish_index(chunks, embedder=embedder, index_dir=index_dir, manifest_path=manifest_path)
    assert caught.value.chunk_id == "who-five-keys:79"
    assert "not truncated" in str(caught.value)
    assert embedder.embed_calls == 0
    assert not (index_dir / "index_manifest.json").exists()


def test_stale_manifest_leaves_the_existing_index(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    marker = index_dir / "index_manifest.json"
    marker.write_text('{"fingerprint": "old", "model_name": "old"}\n', encoding="utf-8")
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_manifest(manifest_path, status="stale")
    with pytest.raises(IndexBuildError, match="fsa-chill"):
        publish_index(_seven(), embedder=FakeEmbedder(), index_dir=index_dir, manifest_path=manifest_path)
    assert marker.read_text(encoding="utf-8").startswith('{"fingerprint": "old"')


def test_missing_raw_file_does_not_publish(tmp_path: Path) -> None:
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_manifest(manifest_path, missing_file="eatwell-guide")
    with pytest.raises(IndexBuildError, match="eatwell-guide"):
        publish_index(
            _seven(),
            embedder=FakeEmbedder(),
            index_dir=tmp_path / "index",
            manifest_path=manifest_path,
        )
    assert not (tmp_path / "index").exists()


def test_second_run_does_not_reembed(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    index_dir = tmp_path / "index"
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_manifest(manifest_path)
    chunks = _seven()
    first = publish_index(chunks, embedder=embedder, index_dir=index_dir, manifest_path=manifest_path)
    second = publish_index(chunks, embedder=embedder, index_dir=index_dir, manifest_path=manifest_path)
    assert embedder.embed_calls == 1
    assert second.fingerprint == first.fingerprint
    assert second.embedded_at == first.embedded_at


@pytest.mark.skipif(not BLOCKS_PATH.is_file(), reason="parsed blocks have not been written")
@pytest.mark.skipif(not CORPUS_MANIFEST_PATH.is_file(), reason="corpus manifest has not been written")
def test_parsed_corpus_fits_and_keeps_the_fats_passage(tmp_path: Path) -> None:
    chunks = chunk_corpus()
    assert not any(is_form_blank(chunk.text) for chunk in chunks)
    fats = next(chunk for chunk in chunks if chunk.chunk_id == "who-healthy-diet:20")
    assert "less than 10%" in fats.text
    assert "less than 1%" in fats.text
    embedder = FakeEmbedder()
    index_dir = tmp_path / "index"
    manifest = publish_index(chunks, embedder=embedder, index_dir=index_dir)
    assert manifest.max_tokens <= BGE_MAX_SEQUENCE_LENGTH
    assert manifest.chunk_count == len(chunks)
    stored = open_collection(index_dir).get(ids=[fats.chunk_id], include=["documents", "metadatas"])
    assert stored["documents"] == [fats.text]
    assert "less than 10%" in stored["documents"][0]
    assert "less than 1%" in stored["documents"][0]
    assert {chunk.document_id for chunk in chunks} == {document.document_id for document in CORPUS}
    assert PROJECT_ROOT / "data" / "index" != index_dir
