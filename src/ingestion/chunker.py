"""Turn parsed blocks into block-atomic chunks.

Paragraphs in this corpus are line fragments, so consecutive paragraphs
under one heading pack toward ``CHUNK_TARGET_TOKENS`` (200) and break only
between paragraphs. A list item is one chunk, number included. A table row
is one chunk, with the column headers repeated, which keeps a Cold Food
Storage refrigerator time and freezer time together. A paragraph over the
target splits on sentence boundaries with no overlap. This corpus does not
contain such a paragraph.

Cost: chunk sizes stay uneven, short list items and short rows stay short,
PDF tables and HTML tables each need a parser case, and the parent section
is carried in the heading path and in ``embed_text`` rather than copied
into the chunk body.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.constants import (
    BLOCKS_PATH,
    CHUNKS_PATH,
    CHUNK_TARGET_TOKENS,
    CORPUS,
    CORPUS_MANIFEST_PATH,
    CorpusDocument,
)
from src.ingestion.parser import Block

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")
_SEPARATOR_CELL = re.compile(r"^:?-+:?$")


class ChunkError(RuntimeError):
    """Chunking stopped because a block could not become a valid chunk."""

    def __init__(self, document_id: str, message: str) -> None:
        self.document_id = document_id
        super().__init__(f"{document_id}: {message}")


@dataclass(frozen=True)
class Chunk:
    """One retrieval passage, with the citation fields the answer layer copies."""

    chunk_id: str
    document_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    retrieval_date: str
    section_heading: str
    block_type: str
    text: str
    embed_text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "publisher": self.publisher,
            "year": self.year,
            "source_url": self.source_url,
            "retrieval_date": self.retrieval_date,
            "section_heading": self.section_heading,
            "block_type": self.block_type,
            "text": self.text,
            "embed_text": self.embed_text,
        }


def estimate_tokens(text: str) -> int:
    """Estimate tokens as ``round(word_count × 1.3)`` without loading BGE."""
    return round(len(text.split()) * 1.3)


def is_form_blank(text: str) -> bool:
    """True when ``_`` blank rules outnumber letters and digits.

    The Five Keys evaluation form is this shape. Packed into one paragraph it
    is thousands of wordpiece tokens and cannot be embedded.
    """
    underscores = text.count("_")
    if underscores == 0:
        return False
    alnum = sum(character.isalnum() for character in text)
    return underscores > alnum


def load_blocks(path: Path = BLOCKS_PATH) -> list[Block]:
    """Read ``blocks.jsonl`` back into blocks. Headings stay in the stream."""
    if not path.is_file():
        raise ChunkError("corpus", f"missing parsed blocks at {path}")
    blocks: list[Block] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        document_id = raw.get("document_id") or "corpus"
        if not raw.get("document_id") or "type" not in raw or "text" not in raw:
            raise ChunkError(str(document_id), f"block on line {line_number} is missing fields")
        header = raw.get("header")
        blocks.append(
            Block(
                document_id=raw["document_id"],
                type=raw["type"],
                text=raw["text"],
                heading_path=tuple(raw.get("heading_path") or ()),
                header=tuple(header) if header else None,
            )
        )
    return blocks


def load_retrieval_dates(path: Path = CORPUS_MANIFEST_PATH) -> dict[str, str]:
    """Map ``document_id`` to the manifest retrieval date."""
    if not path.is_file():
        raise ChunkError("corpus", f"missing corpus manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dates: dict[str, str] = {}
    for row in manifest["documents"]:
        document_id = row.get("document_id") or "corpus"
        retrieval_date = row.get("retrieval_date") or ""
        if not row.get("document_id") or not retrieval_date:
            raise ChunkError(str(document_id), "manifest row has no retrieval_date")
        dates[row["document_id"]] = retrieval_date
    return dates


def chunk_blocks(
    blocks: Sequence[Block],
    retrieval_dates: Mapping[str, str],
    documents: Mapping[str, CorpusDocument] | None = None,
) -> list[Chunk]:
    """Pack paragraphs, and emit one chunk per list item and per table row.

    ``chunk_id`` is ``{document_id}:{ordinal}`` over emitted chunks of that
    document, starting at 0. Heading blocks are not chunks and do not consume
    an ordinal. ``source_url`` is the registry URL, never the downloaded asset.
    """
    registry = documents if documents is not None else {item.document_id: item for item in CORPUS}
    chunks: list[Chunk] = []
    ordinals: dict[str, int] = {}
    pack_document: str | None = None
    pack_heading: tuple[str, ...] | None = None
    pack_parts: list[str] = []

    def append(document_id: str, heading_path: tuple[str, ...], block_type: str, text: str) -> None:
        cleaned = " ".join(text.split())
        if not cleaned:
            return
        document = registry.get(document_id)
        retrieval_date = retrieval_dates.get(document_id, "")
        if document is None:
            raise ChunkError(document_id, "not in the corpus registry")
        if not retrieval_date:
            raise ChunkError(document_id, "missing retrieval_date")
        section_heading = _section_heading(heading_path, document.document_name)
        if (
            not document.document_name
            or not document.publisher
            or not isinstance(document.year, int)
            or not document.source_url
            or not section_heading
        ):
            raise ChunkError(document_id, "missing document name, publisher, year, or section heading")
        ordinal = ordinals.get(document_id, 0)
        ordinals[document_id] = ordinal + 1
        chunk_id = f"{document_id}:{ordinal}"
        embed_text = f"{document.document_name} — {section_heading}\n{cleaned}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                document_name=document.document_name,
                publisher=document.publisher,
                year=document.year,
                source_url=document.source_url,
                retrieval_date=retrieval_date,
                section_heading=section_heading,
                block_type=block_type,
                text=cleaned,
                embed_text=embed_text,
            )
        )

    def flush() -> None:
        nonlocal pack_document, pack_heading, pack_parts
        if pack_parts and pack_document is not None and pack_heading is not None:
            append(pack_document, pack_heading, "paragraph", " ".join(pack_parts))
        pack_document = None
        pack_heading = None
        pack_parts = []

    for block in blocks:
        if block.type == "heading":
            flush()
            continue
        if block.type == "paragraph":
            text = block.text.strip()
            # Evaluation-form rules are longer than the 512-token window once
            # wordpiece splits each underscore, and the word estimate does not see them.
            if not text or is_form_blank(text):
                continue
            heading = tuple(block.heading_path)
            if estimate_tokens(text) > CHUNK_TARGET_TOKENS:
                flush()
                for piece in _split_paragraph(block, text):
                    append(block.document_id, heading, "paragraph", piece)
                continue
            fits = (
                bool(pack_parts)
                and pack_document == block.document_id
                and pack_heading == heading
                and estimate_tokens(" ".join([*pack_parts, text])) <= CHUNK_TARGET_TOKENS
            )
            if pack_parts and not fits:
                flush()
            pack_document = block.document_id
            pack_heading = heading
            pack_parts.append(text)
            continue
        flush()
        if block.type == "list_item":
            if block.text.strip():
                append(block.document_id, tuple(block.heading_path), "list_item", block.text)
            continue
        if block.type == "table":
            for row_text in label_table_rows(block):
                append(block.document_id, tuple(block.heading_path), "table_row", row_text)
            continue
        raise ChunkError(block.document_id, f"unknown block type {block.type}")
    flush()
    return chunks


def chunk_corpus(
    blocks_path: Path = BLOCKS_PATH,
    manifest_path: Path = CORPUS_MANIFEST_PATH,
) -> list[Chunk]:
    """Chunk the parsed corpus. Fail when any registry document produced no chunks."""
    chunks = chunk_blocks(load_blocks(blocks_path), load_retrieval_dates(manifest_path))
    produced = {chunk.document_id for chunk in chunks}
    for document in CORPUS:
        if document.document_id not in produced:
            raise ChunkError(document.document_id, "no content chunks")
    return chunks


def write_chunks(chunks: Sequence[Chunk], path: Path = CHUNKS_PATH) -> None:
    """Write one JSON object per chunk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def label_table_rows(block: Block) -> list[str]:
    """One labeled string per data row. The header is repeated. Empty cells are omitted.

    A blank leading cell copies the category from the previous row of this table.
    A continuation row the parser stored with its own product name, such as
    Kitchen Companion's ``Opened package``, is left as stored.
    """
    header, rows = _table_cells(block)
    if not header:
        raise ChunkError(block.document_id, "table has no header")
    labeled: list[str] = []
    carried: tuple[str, ...] | None = None
    for row in rows:
        cells = list(row)
        if len(cells) < len(header):
            cells.extend("" for _ in range(len(header) - len(cells)))
        if carried is not None:
            for index, value in enumerate(cells):
                if value:
                    break
                if index < len(carried):
                    cells[index] = carried[index]
        carried = tuple(cells)
        parts: list[str] = []
        for index, value in enumerate(cells):
            if not value:
                continue
            if index < len(header) and header[index]:
                parts.append(f"{header[index]}: {value}")
            else:
                parts.append(value)
        if parts:
            labeled.append(" | ".join(parts))
    return labeled


def _section_heading(heading_path: Sequence[str], document_name: str) -> str:
    parts = [part.strip() for part in heading_path if part and part.strip()]
    if not parts:
        return document_name
    return " > ".join(parts)


def _split_paragraph(block: Block, text: str) -> list[str]:
    """Split one oversized paragraph on sentence boundaries. Pieces do not overlap."""
    pieces = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    if not pieces:
        pieces = [text.strip()]
    if any(estimate_tokens(piece) > CHUNK_TARGET_TOKENS for piece in pieces):
        heading = _section_heading(block.heading_path, block.document_id)
        preview = text[:80]
        raise ChunkError(
            block.document_id,
            f"paragraph has no sentence boundary under {heading}: {preview!r}",
        )
    packed: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and estimate_tokens(" ".join([*current, piece])) > CHUNK_TARGET_TOKENS:
            packed.append(" ".join(current))
            current = [piece]
        else:
            current.append(piece)
    if current:
        packed.append(" ".join(current))
    return packed


def _table_cells(block: Block) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    header = tuple(cell.strip() for cell in (block.header or ()))
    rows: list[tuple[str, ...]] = []
    for line in block.text.splitlines():
        if not line.strip():
            continue
        cells = _split_markdown_row(line)
        if not cells or _is_separator(cells):
            continue
        if header and _same_row(cells, header):
            continue
        if not header:
            header = cells
            continue
        rows.append(cells)
    return header, rows


def _split_markdown_row(line: str) -> tuple[str, ...]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in inner:
        if escaped:
            current.append(character)
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "|":
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(character)
    cells.append("".join(current).strip())
    return tuple(cells)


def _is_separator(cells: Sequence[str]) -> bool:
    useful = [cell for cell in cells if cell]
    return bool(useful) and all(_SEPARATOR_CELL.match(cell) for cell in useful)


def _same_row(cells: Sequence[str], header: Sequence[str]) -> bool:
    if len(cells) != len(header):
        return False
    return all(left.casefold() == right.casefold() for left, right in zip(cells, header))


def main() -> None:
    chunks = chunk_corpus()
    write_chunks(chunks)
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
    print(f"wrote {len(chunks)} chunks to {CHUNKS_PATH}")
    for document_id, count in counts.items():
        print(f"  {document_id}: {count}")


if __name__ == "__main__":
    main()
