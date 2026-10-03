"""Embed chunk passages with BGE and write one Chroma collection.

Passages are ``embed_text`` with no query prefix. The model window is 512
tokens, special tokens included. A longer passage stops the build before any
upsert. Vectors are L2-normalized and the collection uses cosine distance.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from src.constants import (
    BGE_DIMENSION,
    BGE_MAX_SEQUENCE_LENGTH,
    BGE_MODEL_NAME,
    BGE_QUERY_PREFIX,
    CHUNKS_PATH,
    COLLECTION_NAME,
    CORPUS,
    CORPUS_MANIFEST_PATH,
    INDEX_DIR,
    INDEX_MANIFEST_PATH,
    PROJECT_ROOT,
)
from src.ingestion.chunker import Chunk

_REQUIRED_METADATA = (
    "document_id",
    "document_name",
    "publisher",
    "year",
    "source_url",
    "section_heading",
    "block_type",
    "retrieval_date",
)


class IndexBuildError(RuntimeError):
    """The index was not published. An existing index is left in place."""

    def __init__(
        self,
        message: str,
        *,
        chunk_id: str | None = None,
        document_id: str | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.document_id = document_id
        label = chunk_id or document_id or "index"
        super().__init__(f"{label}: {message}")


class Embedder(Protocol):
    """Token counts and passage vectors. The query prefix is not this type's job."""

    model_name: str
    dimension: int
    max_sequence_length: int

    def count_tokens(self, text: str) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class IndexManifest:
    model_name: str
    dimension: int
    max_sequence_length: int
    max_tokens: int
    chunk_count: int
    fingerprint: str
    embedded_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_sequence_length": self.max_sequence_length,
            "max_tokens": self.max_tokens,
            "chunk_count": self.chunk_count,
            "fingerprint": self.fingerprint,
            "embedded_at": self.embedded_at,
        }


class BgeEmbedder:
    """Local ``BAAI/bge-small-en-v1.5``. Passages are embedded as stored."""

    model_name = BGE_MODEL_NAME
    dimension = BGE_DIMENSION
    max_sequence_length = BGE_MAX_SEQUENCE_LENGTH

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        self._tokenizer = self._model.tokenizer

    def count_tokens(self, text: str) -> int:
        encoded = self._tokenizer(text, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # prompt="" blocks a model-card query prompt. Passages stay unprefixed.
        vectors = self._model.encode(
            list(texts),
            prompt="",
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    """Read ``chunks.jsonl`` in file order."""
    if not path.is_file():
        raise IndexBuildError(f"missing chunks at {path}")
    chunks: list[Chunk] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        chunk_id = str(raw.get("chunk_id") or "")
        if not chunk_id or "embed_text" not in raw or "text" not in raw:
            raise IndexBuildError(f"chunk on line {line_number} is missing fields", chunk_id=chunk_id or None)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=str(raw["document_id"]),
                document_name=str(raw["document_name"]),
                publisher=str(raw["publisher"]),
                year=int(raw["year"]),
                source_url=str(raw["source_url"]),
                retrieval_date=str(raw["retrieval_date"]),
                section_heading=str(raw["section_heading"]),
                block_type=str(raw["block_type"]),
                text=str(raw["text"]),
                embed_text=str(raw["embed_text"]),
            )
        )
    return chunks


def fingerprint_chunks(chunks: Sequence[Chunk]) -> str:
    """sha256 of ``chunk_id + embed_text`` in file order."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk.chunk_id}{chunk.embed_text}".encode())
    return digest.hexdigest()


def publish_index(
    chunks: Sequence[Chunk] | None = None,
    *,
    embedder: Embedder | None = None,
    index_dir: Path = INDEX_DIR,
    chunks_path: Path = CHUNKS_PATH,
    manifest_path: Path = CORPUS_MANIFEST_PATH,
    root: Path = PROJECT_ROOT,
) -> IndexManifest:
    """Write the collection and manifest. Return without embedding when both match.

    A stale or incomplete corpus manifest raises before the index directory
    is replaced. An over-long passage raises before any upsert.
    """
    assert_corpus_publishable(manifest_path, root)
    loaded = list(chunks) if chunks is not None else load_chunks(chunks_path)
    _validate_chunk_set(loaded)
    fingerprint = fingerprint_chunks(loaded)
    current = _read_manifest(index_dir / "index_manifest.json")
    if (
        current is not None
        and current.get("fingerprint") == fingerprint
        and current.get("model_name") == BGE_MODEL_NAME
        and _collection_exists(index_dir)
    ):
        return _manifest_from_dict(current)

    embedder = embedder or BgeEmbedder()
    if embedder.model_name != BGE_MODEL_NAME:
        raise IndexBuildError(
            f"embedder model is {embedder.model_name}, expected {BGE_MODEL_NAME}"
        )
    max_tokens = _assert_within_window(loaded, embedder)
    vectors = l2_normalize(embedder.embed([chunk.embed_text for chunk in loaded]))
    if len(vectors) != len(loaded):
        raise IndexBuildError("embedder returned a different number of vectors than chunks")
    for chunk, vector in zip(loaded, vectors):
        if len(vector) != embedder.dimension:
            raise IndexBuildError(
                f"expected {embedder.dimension} dimensions, got {len(vector)}",
                chunk_id=chunk.chunk_id,
            )

    manifest = IndexManifest(
        model_name=BGE_MODEL_NAME,
        dimension=embedder.dimension,
        max_sequence_length=embedder.max_sequence_length,
        max_tokens=max_tokens,
        chunk_count=len(loaded),
        fingerprint=fingerprint,
        embedded_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_index(index_dir, loaded, vectors, manifest)
    return manifest


def assert_corpus_publishable(manifest_path: Path, root: Path) -> None:
    """Refuse a partial corpus. Does not read or write the index."""
    if not manifest_path.is_file():
        raise IndexBuildError(f"missing corpus manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {row["document_id"]: row for row in manifest.get("documents", []) if row.get("document_id")}
    for document in CORPUS:
        row = rows.get(document.document_id)
        if row is None or row.get("status") != "ok":
            raise IndexBuildError(
                "corpus manifest row is missing or stale",
                document_id=document.document_id,
            )
        relative = row.get("file") or ""
        if not relative or not (root / relative).is_file():
            raise IndexBuildError(
                "corpus manifest row is ok but the raw file is missing",
                document_id=document.document_id,
            )


def l2_normalize(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return unit vectors. A zero vector cannot be stored."""
    normalized: list[list[float]] = []
    for vector in vectors:
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            raise IndexBuildError("embedding norm is zero")
        normalized.append([value / norm for value in vector])
    return normalized


def _assert_within_window(chunks: Sequence[Chunk], embedder: Embedder) -> int:
    maximum = 0
    for chunk in chunks:
        count = embedder.count_tokens(chunk.embed_text)
        if count > embedder.max_sequence_length:
            raise IndexBuildError(
                f"embed_text is {count} tokens, over {embedder.max_sequence_length}; not truncated",
                chunk_id=chunk.chunk_id,
            )
        maximum = max(maximum, count)
    return maximum


def _validate_chunk_set(chunks: Sequence[Chunk]) -> None:
    if not chunks:
        raise IndexBuildError("no chunks to index")
    seen: set[str] = set()
    documents: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            raise IndexBuildError("duplicate chunk_id", chunk_id=chunk.chunk_id)
        seen.add(chunk.chunk_id)
        if chunk.block_type not in {"paragraph", "list_item", "table_row"}:
            raise IndexBuildError(
                f"block_type {chunk.block_type} cannot be indexed",
                chunk_id=chunk.chunk_id,
            )
        if not chunk.text or not chunk.embed_text or not chunk.section_heading:
            raise IndexBuildError("chunk is missing text or a section heading", chunk_id=chunk.chunk_id)
        if chunk.embed_text.startswith(BGE_QUERY_PREFIX):
            raise IndexBuildError(
                "embed_text starts with the query prefix",
                chunk_id=chunk.chunk_id,
            )
        documents.add(chunk.document_id)
    missing = [document.document_id for document in CORPUS if document.document_id not in documents]
    if missing:
        raise IndexBuildError(
            "chunks.jsonl does not cover every registry document",
            document_id=missing[0],
        )


def _write_index(
    index_dir: Path,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    manifest: IndexManifest,
) -> None:
    """Build beside ``index_dir`` and replace it only after the manifest is written.

    Staging is deleted afterwards. The previous index is left in place when
    the new build fails before the replace.
    """
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(dir=index_dir.parent, prefix="index-staging-"))
    published = False
    try:
        staging = staging_parent / "index"
        staging.mkdir()
        _upsert(staging, chunks, vectors)
        (staging / "index_manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        _replace_directory(staging, index_dir)
        if not (index_dir / "chroma.sqlite3").is_file():
            raise IndexBuildError("published index is missing chroma.sqlite3")
        if not (index_dir / "index_manifest.json").is_file():
            raise IndexBuildError("published index is missing index_manifest.json")
        published = True
    finally:
        if staging_parent.exists():
            shutil.rmtree(staging_parent, ignore_errors=True)
        if published and not (index_dir / "chroma.sqlite3").is_file():
            raise IndexBuildError("published index disappeared after staging cleanup")


def _upsert(
    index_dir: Path,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> None:
    client = chromadb.PersistentClient(path=str(index_dir))
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=PrecomputedEmbeddingFunction(),
        )
        batch = 128
        for start in range(0, len(chunks), batch):
            group = chunks[start : start + batch]
            collection.upsert(
                ids=[chunk.chunk_id for chunk in group],
                embeddings=[list(vectors[start + offset]) for offset in range(len(group))],
                documents=[chunk.text for chunk in group],
                metadatas=[_metadata(chunk) for chunk in group],
            )
        stored = collection.count()
    finally:
        _close_client(client)
    if stored != len(chunks):
        raise IndexBuildError(f"collection count is {stored}, expected {len(chunks)}")


def _metadata(chunk: Chunk) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "publisher": chunk.publisher,
        "year": int(chunk.year),
        "source_url": chunk.source_url,
        "section_heading": chunk.section_heading,
        "block_type": chunk.block_type,
        "retrieval_date": chunk.retrieval_date,
    }
    for field in _REQUIRED_METADATA:
        if metadata.get(field) in (None, ""):
            raise IndexBuildError(f"missing {field}", chunk_id=chunk.chunk_id)
    return metadata


def _replace_directory(staging: Path, index_dir: Path) -> None:
    backup = index_dir.with_name(index_dir.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    if index_dir.exists():
        index_dir.rename(backup)
    try:
        staging.rename(index_dir)
    except OSError:
        if backup.exists() and not index_dir.exists():
            backup.rename(index_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _close_client(client: chromadb.ClientAPI) -> None:
    """Release the SQLite handle before the directory is renamed."""
    close = getattr(client, "close", None)
    if close is not None:
        close()
    else:
        system = getattr(client, "_system", None)
        stop = getattr(system, "stop", None)
        if stop is not None:
            stop()
    gc.collect()


def _collection_exists(index_dir: Path) -> bool:
    if not index_dir.is_dir():
        return False
    return any(index_dir.glob("chroma.sqlite3")) or any(index_dir.rglob("chroma.sqlite3"))


def _read_manifest(path: Path) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return None
    return loaded


def _manifest_from_dict(raw: Mapping[str, object]) -> IndexManifest:
    return IndexManifest(
        model_name=str(raw["model_name"]),
        dimension=int(raw["dimension"]),  # type: ignore[arg-type]
        max_sequence_length=int(raw["max_sequence_length"]),  # type: ignore[arg-type]
        max_tokens=int(raw["max_tokens"]),  # type: ignore[arg-type]
        chunk_count=int(raw["chunk_count"]),  # type: ignore[arg-type]
        fingerprint=str(raw["fingerprint"]),
        embedded_at=str(raw["embedded_at"]),
    )


def open_collection(index_dir: Path = INDEX_DIR) -> chromadb.Collection:
    """Open the published collection. The caller passes embeddings on query."""
    client = chromadb.PersistentClient(path=str(index_dir))
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=PrecomputedEmbeddingFunction(),
    )


class PrecomputedEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma must not embed passages or queries. The indexer and retriever do."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "precomputed"

    def __call__(self, input: Documents) -> Embeddings:
        raise IndexBuildError("Chroma tried to embed a passage; pass vectors from the indexer")

    def embed_query(self, input: Documents) -> Embeddings:
        raise IndexBuildError("Chroma tried to embed a query; the retriever embeds queries")

    @staticmethod
    def build_from_config(config: Mapping[str, object]) -> "PrecomputedEmbeddingFunction":
        return PrecomputedEmbeddingFunction()

    def get_config(self) -> dict[str, object]:
        return {}


register_embedding_function(PrecomputedEmbeddingFunction)


def main() -> None:
    chunks = load_chunks()
    manifest = publish_index(chunks)
    print(
        f"indexed {manifest.chunk_count} chunks "
        f"with {manifest.model_name} (max {manifest.max_tokens} tokens) "
        f"at {INDEX_MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()
