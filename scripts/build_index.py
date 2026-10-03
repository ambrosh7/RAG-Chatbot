"""Offline index build: fetch, parse, chunk, and index."""

from __future__ import annotations

from src.ingestion.pipeline import build_corpus_index


def main() -> None:
    build_corpus_index()


if __name__ == "__main__":
    main()
