"""Print a retrieval probe against the published index.

Run from the repo root:

    python -m scripts.probe_retrieval "How long can I keep a whole chicken in the fridge?"
    python -m scripts.probe_retrieval "How much fruit?" eatwell-guide
"""

from __future__ import annotations

import sys

from src.rag.retriever import RetrievalError, retrieve

PREVIEW = 160


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or len(args) > 2:
        raise SystemExit(
            'usage: python -m scripts.probe_retrieval "question" [document_id]'
        )
    question = args[0]
    document_id = args[1] if len(args) == 2 else None
    try:
        result = retrieve(question, document_id)
    except RetrievalError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"query: {question}")
    print("scope: " + ", ".join(document.document_id for document in result.scope))
    print(f"candidates before threshold: {result.candidate_count}")
    for hit in result.candidates:
        preview = " ".join(hit.text.split())
        if len(preview) > PREVIEW:
            preview = preview[: PREVIEW - 3] + "..."
        print(
            f"  {hit.similarity:.3f}  {hit.chunk_id}  {hit.block_type}  {hit.section_heading}"
        )
        print(f"    {preview}")
    print("grouped:")
    if not result.groups:
        print("  (none)")
        return
    for group in result.groups:
        print(f"  {group.document_id}")
        for hit in group.hits:
            preview = " ".join(hit.text.split())
            if len(preview) > PREVIEW:
                preview = preview[: PREVIEW - 3] + "..."
            print(f"    {hit.similarity:.3f}  {hit.chunk_id}  {preview}")


if __name__ == "__main__":
    main()
