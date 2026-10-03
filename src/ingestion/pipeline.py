"""Offline build: fetch, parse, chunk, then index.

A missing or stale corpus row stops the run before the index is replaced.
"""

from __future__ import annotations

from src.ingestion.chunker import chunk_corpus, write_chunks
from src.ingestion.fetcher import fetch_corpus
from src.ingestion.indexer import publish_index
from src.ingestion.parser import parse_corpus, write_blocks


def build_corpus_index() -> None:
    """Publish a complete index, or leave the previous index in place."""
    fetch_corpus()
    write_blocks(parse_corpus())
    chunks = chunk_corpus()
    write_chunks(chunks)
    manifest = publish_index(chunks)
    print(
        f"indexed {manifest.chunk_count} chunks "
        f"with {manifest.model_name} (max {manifest.max_tokens} tokens)"
    )
