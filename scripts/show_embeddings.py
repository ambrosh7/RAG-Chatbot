"""Print a few passage embeddings stored in the published Chroma collection.

Run from the repo root:

    python -m scripts.show_embeddings
    python -m scripts.show_embeddings cold-food-storage:24 who-healthy-diet:20
"""

from __future__ import annotations

import sys

from src.constants import BGE_DIMENSION, INDEX_DIR
from src.ingestion.indexer import open_collection

DEFAULT_IDS = (
    "cold-food-storage:24",
    "who-healthy-diet:20",
    "who-five-keys:79",
)
VALUES_PER_LINE = 8


def main(argv: list[str] | None = None) -> None:
    chunk_ids = tuple(argv if argv is not None else sys.argv[1:]) or DEFAULT_IDS
    if not (INDEX_DIR / "chroma.sqlite3").is_file():
        raise SystemExit(f"No index at {INDEX_DIR}. Run python -m scripts.build_index first.")

    collection = open_collection()
    found = collection.get(
        ids=list(chunk_ids),
        include=["embeddings", "documents", "metadatas"],
    )
    documents = found["documents"] if found["documents"] is not None else []
    metadatas = found["metadatas"] if found["metadatas"] is not None else []
    embeddings = found["embeddings"] if found["embeddings"] is not None else []
    rows = {
        chunk_id: (document, metadata, embedding)
        for chunk_id, document, metadata, embedding in zip(
            found["ids"],
            documents,
            metadatas,
            embeddings,
        )
    }
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in rows]
    if missing:
        raise SystemExit(f"Chunk ids not in the collection: {', '.join(missing)}")

    print(f"collection count {collection.count()}  dimension {BGE_DIMENSION}")
    for chunk_id in chunk_ids:
        document, metadata, embedding = rows[chunk_id]
        vector = [float(value) for value in embedding]
        norm = sum(value * value for value in vector) ** 0.5
        text = " ".join(document.split())
        if len(text) > 240:
            text = text[:237] + "..."
        print()
        print(f"{chunk_id}  {metadata['document_id']}  {metadata['block_type']}")
        print(text)
        print(f"values {len(vector)}  l2 norm {norm:.6f}")
        for start in range(0, len(vector), VALUES_PER_LINE):
            group = vector[start : start + VALUES_PER_LINE]
            print("  " + " ".join(f"{value:8.4f}" for value in group))


if __name__ == "__main__":
    main()
