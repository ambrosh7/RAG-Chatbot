"""Corpus registry, chunk limits, and retrieval thresholds.

The registry is the allowlist for fetch, filtered retrieval, and citations.
`retrieval_date` is recorded in the corpus manifest when a file is downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PARSED_DATA_DIR = DATA_DIR / "parsed"
CHUNKS_DATA_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
CORPUS_MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"
BLOCKS_PATH = PARSED_DATA_DIR / "blocks.jsonl"
CHUNKS_PATH = CHUNKS_DATA_DIR / "chunks.jsonl"
INDEX_MANIFEST_PATH = INDEX_DIR / "index_manifest.json"

# Block-atomic chunking. Tokens are estimated as round(word_count × 1.3).
# 200 is the paragraph pack target measured on data/parsed/blocks.jsonl.
# List items and table rows are not packed and are not split.
# A single paragraph over this target is sentence-split with no overlap.
CHUNK_TARGET_TOKENS = 200

TOP_K_ALL = 8
TOP_K_FILTERED = 5
MAX_CHUNKS_PER_DOCUMENT = 3
# Unfiltered retrieval fetches this many neighbours, then applies the
# per-document cap and TOP_K_ALL. 24 is 3 chunks per document times 8 kept.
CANDIDATE_K = 24
# Probed 2026-10-03 against the phase 3 index. Phase 9 may move this
# inside 0.58–0.64. The tariff question topped out at 0.565, and raw
# chicken filtered to the Eatwell Guide topped out at 0.572. The oil-label
# sentence eatwell-guide:96 scored 0.644. The placeholder 0.45 kept the tariff hit.
MIN_SIMILARITY = 0.62

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_DIMENSION = 384
BGE_MAX_SEQUENCE_LENGTH = 512
# BGE v1.5 asymmetric retrieval: queries use this prefix; chunk text does not.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION_NAME = "guidance"

# Phase 6 reads GROQ_MODEL from the environment when it is set.
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0.0
GROQ_MAX_TOKENS = 1024
# A claim is dropped when fewer than this share of its content words appear
# in the cited chunk.
MIN_CLAIM_TERM_OVERLAP = 0.5

HTML_DOCUMENT_IDS = frozenset({"cold-food-storage", "fsa-chill"})
LANDING_PAGE_DOCUMENT_IDS = frozenset({"fao-who-healthy-diets", "who-five-keys"})


@dataclass(frozen=True)
class CorpusDocument:
    """One official guidance document in the corpus allowlist."""

    document_id: str
    document_name: str
    publisher: str
    year: int
    source_url: str
    kind: str
    aliases: tuple[str, ...]
    landing_page: bool = False


CORPUS: tuple[CorpusDocument, ...] = (
    CorpusDocument(
        document_id="cold-food-storage",
        document_name="Cold Food Storage Chart",
        publisher="FoodSafety.gov (U.S. Department of Health and Human Services / FDA / USDA)",
        year=2023,
        source_url="https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts",
        kind="html",
        aliases=("cold food storage", "cold food storage chart"),
    ),
    CorpusDocument(
        document_id="who-healthy-diet",
        document_name="Healthy diet (Fact sheet)",
        publisher="World Health Organization",
        year=2018,
        source_url="https://www.who.int/docs/default-source/healthy-diet/healthy-diet-fact-sheet-394.pdf",
        kind="pdf",
        aliases=("healthy diet", "healthy diet fact sheet", "who fact sheet"),
    ),
    CorpusDocument(
        document_id="fao-who-healthy-diets",
        document_name="What are healthy diets? Joint statement",
        publisher="Food and Agriculture Organization of the United Nations and World Health Organization",
        year=2024,
        source_url="https://www.who.int/publications/i/item/9789240101876",
        kind="pdf",
        aliases=("what are healthy diets", "fao and who"),
        landing_page=True,
    ),
    CorpusDocument(
        document_id="who-five-keys",
        document_name="Five keys to safer food manual",
        publisher="World Health Organization",
        year=2006,
        source_url="https://www.who.int/publications/i/item/9789241594639",
        kind="pdf",
        aliases=("five keys", "five keys to safer food"),
        landing_page=True,
    ),
    CorpusDocument(
        document_id="eatwell-guide",
        document_name="The Eatwell Guide booklet",
        publisher="Public Health England (now Office for Health Improvement and Disparities)",
        year=2018,
        source_url="https://assets.publishing.service.gov.uk/media/5ba8a50540f0b605084c9501/Eatwell_Guide_booklet_2018v4.pdf",
        kind="pdf",
        aliases=("eatwell", "eatwell guide"),
    ),
    CorpusDocument(
        document_id="fsa-chill",
        document_name="How to chill, freeze and defrost food safely",
        publisher="Food Standards Agency (UK)",
        year=2017,
        source_url="https://www.gov.uk/government/publications/how-to-chill-freeze-and-defrost-food-safely/how-to-chill-freeze-and-defrost-food-safely",
        kind="html",
        aliases=("food standards agency", "chill freeze and defrost"),
    ),
    CorpusDocument(
        document_id="kitchen-companion",
        document_name="Kitchen Companion: Your Safe Food Handbook",
        publisher="USDA Food Safety and Inspection Service",
        year=2008,
        source_url="https://www.fsis.usda.gov/sites/default/files/media_file/2020-12/Kitchen-Companion.pdf",
        kind="pdf",
        aliases=("kitchen companion",),
    ),
)

DOCUMENT_IDS: frozenset[str] = frozenset(document.document_id for document in CORPUS)
SOURCE_URLS: frozenset[str] = frozenset(document.source_url for document in CORPUS)

_PROBLEM_STATEMENT_URLS = frozenset(
    {
        "https://www.foodsafety.gov/food-safety-charts/cold-food-storage-charts",
        "https://www.who.int/docs/default-source/healthy-diet/healthy-diet-fact-sheet-394.pdf",
        "https://www.who.int/publications/i/item/9789240101876",
        "https://www.who.int/publications/i/item/9789241594639",
        "https://assets.publishing.service.gov.uk/media/5ba8a50540f0b605084c9501/Eatwell_Guide_booklet_2018v4.pdf",
        "https://www.gov.uk/government/publications/how-to-chill-freeze-and-defrost-food-safely/how-to-chill-freeze-and-defrost-food-safely",
        "https://www.fsis.usda.gov/sites/default/files/media_file/2020-12/Kitchen-Companion.pdf",
    }
)
_REQUIRED_ALIASES = frozenset(
    {
        "eatwell",
        "eatwell guide",
        "kitchen companion",
        "five keys",
        "cold food storage",
        "healthy diet",
        "food standards agency",
    }
)


def _validate_corpus(corpus: tuple[CorpusDocument, ...]) -> None:
    """Fail import when the allowlist drifts from the problem statement."""
    if len(corpus) != 7:
        raise RuntimeError(f"corpus must contain 7 documents, found {len(corpus)}")

    seen_ids: set[str] = set()
    seen_aliases: dict[str, str] = {}
    for document in corpus:
        if not document.document_id or not document.document_name:
            raise RuntimeError("every corpus row needs a document_id and document_name")
        if not document.publisher or not isinstance(document.year, int):
            raise RuntimeError(f"{document.document_id} needs a publisher and an int year")
        if not document.source_url:
            raise RuntimeError(f"{document.document_id} needs a source_url")
        if document.document_id in seen_ids:
            raise RuntimeError(f"duplicate document_id: {document.document_id}")
        seen_ids.add(document.document_id)
        if document.kind not in {"html", "pdf"}:
            raise RuntimeError(f"{document.document_id} kind must be html or pdf")

        expected_kind = "html" if document.document_id in HTML_DOCUMENT_IDS else "pdf"
        if document.kind != expected_kind:
            raise RuntimeError(
                f"{document.document_id} kind must be {expected_kind}"
            )
        if document.landing_page != (document.document_id in LANDING_PAGE_DOCUMENT_IDS):
            raise RuntimeError(
                f"{document.document_id} landing_page must be "
                f"{document.document_id in LANDING_PAGE_DOCUMENT_IDS}"
            )
        for alias in document.aliases:
            key = alias.casefold()
            owner = seen_aliases.get(key)
            if owner is not None:
                raise RuntimeError(
                    f"alias {alias!r} is shared by {owner} and {document.document_id}"
                )
            seen_aliases[key] = document.document_id

    if seen_ids != HTML_DOCUMENT_IDS | {
        "who-healthy-diet",
        "fao-who-healthy-diets",
        "who-five-keys",
        "eatwell-guide",
        "kitchen-companion",
    }:
        raise RuntimeError(f"unexpected document ids: {sorted(seen_ids)}")
    if SOURCE_URLS != _PROBLEM_STATEMENT_URLS:
        raise RuntimeError("a source_url is outside the problem-statement corpus table")
    missing_aliases = _REQUIRED_ALIASES - frozenset(seen_aliases)
    if missing_aliases:
        raise RuntimeError(f"missing aliases: {sorted(missing_aliases)}")


_validate_corpus(CORPUS)
