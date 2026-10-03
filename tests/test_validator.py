"""Claim checks against retrieved chunks. No model and no index."""

from __future__ import annotations

from src.constants import CORPUS, MIN_CLAIM_TERM_OVERLAP
from src.rag.citations import render
from src.rag.models import DraftClaim, DraftDocument
from src.rag.retriever import DocumentHits, Hit
from src.rag.validator import (
    BLENDED_VOICE,
    EMPTY_TEXT,
    FOREIGN_CHUNK,
    LOW_TERM_OVERLAP,
    MISSING_CHUNK,
    UNSUPPORTED_NUMBER,
    validate,
)

REGISTRY = {document.document_id: document for document in CORPUS}
FATS = "Intake of saturated fats should be less than 10% of total energy intake."


def hit(
    document_id: str,
    chunk_id: str,
    text: str,
    *,
    source_url: str | None = None,
    section_heading: str = "Guidance",
) -> Hit:
    document = REGISTRY[document_id]
    return Hit(
        chunk_id=chunk_id,
        text=text,
        similarity=0.8,
        document_id=document_id,
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=source_url if source_url is not None else document.source_url,
        section_heading=section_heading,
        block_type="paragraph",
        retrieval_date="2026-10-03",
    )


def _check(
    document_id: str,
    text: str,
    chunk_ids: list[str],
    hits: list[Hit],
    *,
    extra_hits: list[Hit] | None = None,
):
    groups = [DocumentHits(document_id=document_id, hits=tuple(hits))]
    if extra_hits:
        groups.append(DocumentHits(document_id=extra_hits[0].document_id, hits=tuple(extra_hits)))
    sections = render(
        (DraftDocument(document_id, (DraftClaim(text, tuple(chunk_ids)),)),),
        groups,
    )
    return validate(sections, groups)


def test_population_figure_is_kept_when_the_chunk_states_it() -> None:
    result = _check("who-healthy-diet", FATS, ["who-healthy-diet:12"], [hit("who-healthy-diet", "who-healthy-diet:12", FATS, section_heading="Fats")])
    assert result.dropped == ()
    claim = result.sections[0].claims[0]
    assert claim.text == FATS
    assert claim.chunk_ids == ("who-healthy-diet:12",)
    assert claim.section_heading == "Fats"
    assert "the guidelines say" not in claim.text.casefold()


def test_two_chunks_from_the_same_document_can_support_one_claim() -> None:
    first = hit("cold-food-storage", "cold-food-storage:24", "Refrigerator time is 1 to 2 days.")
    second = hit("cold-food-storage", "cold-food-storage:25", "Freezer time is 9 months.")
    text = "Refrigerator time is 1 to 2 days and freezer time is 9 months."
    result = _check(
        "cold-food-storage",
        text,
        ["cold-food-storage:24", "cold-food-storage:25"],
        [first, second],
    )
    assert result.dropped == ()
    assert result.sections[0].claims[0].chunk_ids == (
        "cold-food-storage:24",
        "cold-food-storage:25",
    )


def test_missing_chunk_is_dropped() -> None:
    stored = hit("cold-food-storage", "cold-food-storage:24", "Refrigerator time is 1 to 2 days.")
    result = _check(
        "cold-food-storage",
        "Refrigerator time is 1 to 2 days.",
        ["cold-food-storage:99"],
        [stored],
    )
    assert result.sections == ()
    assert result.dropped[0].reason == MISSING_CHUNK


def test_chunk_from_another_document_is_dropped() -> None:
    own = hit("eatwell-guide", "eatwell-guide:5", "Fruit and vegetables are at least 5 portions.")
    foreign = hit("kitchen-companion", "kitchen-companion:44", "Leftovers keep 3 to 4 days.")
    result = _check(
        "eatwell-guide",
        "Fruit and vegetables are at least 5 portions.",
        ["eatwell-guide:5", "kitchen-companion:44"],
        [own],
        extra_hits=[foreign],
    )
    assert result.sections == ()
    assert result.dropped[0].reason == FOREIGN_CHUNK


def test_empty_claim_is_dropped() -> None:
    stored = hit("cold-food-storage", "cold-food-storage:24", "Refrigerator time is 1 to 2 days.")
    result = _check("cold-food-storage", "   ", ["cold-food-storage:24"], [stored])
    assert result.dropped[0].reason == EMPTY_TEXT


def test_invented_number_is_dropped() -> None:
    stored = hit("cold-food-storage", "cold-food-storage:24", "Refrigerator time is 1 to 2 days.")
    result = _check(
        "cold-food-storage",
        "Refrigerator time is 3 days.",
        ["cold-food-storage:24"],
        [stored],
    )
    assert result.dropped[0].reason == UNSUPPORTED_NUMBER


def test_overlap_at_the_threshold_is_kept_and_below_it_is_dropped() -> None:
    assert MIN_CLAIM_TERM_OVERLAP == 0.5
    stored = hit("eatwell-guide", "eatwell-guide:1", "alpha beta gamma delta")
    kept = _check(
        "eatwell-guide",
        "alpha beta other words",
        ["eatwell-guide:1"],
        [stored],
    )
    dropped = _check(
        "eatwell-guide",
        "alpha other words extra",
        ["eatwell-guide:1"],
        [stored],
    )
    assert kept.dropped == ()
    assert dropped.dropped[0].reason == LOW_TERM_OVERLAP


def test_bad_citation_url_is_dropped() -> None:
    stored = hit(
        "cold-food-storage",
        "cold-food-storage:24",
        "Refrigerator time is 1 to 2 days.",
        source_url="https://example.com/not-the-registry",
    )
    result = _check(
        "cold-food-storage",
        "Refrigerator time is 1 to 2 days.",
        ["cold-food-storage:24"],
        [stored],
    )
    assert result.sections == ()
    assert result.dropped[0].reason == "bad_citation_url"


def test_personal_weight_and_calorie_claims_are_dropped() -> None:
    weight_hit = hit(
        "who-healthy-diet",
        "who-healthy-diet:3",
        "Adults discuss ideal weight of 70 kg in this energy paragraph.",
    )
    weight = _check(
        "who-healthy-diet",
        "Your ideal weight is 70 kg.",
        ["who-healthy-diet:3"],
        [weight_hit],
    )
    calorie_hit = hit(
        "eatwell-guide",
        "eatwell-guide:2",
        "How many calories should I eat is not a population figure of 1500.",
    )
    calorie = _check(
        "eatwell-guide",
        "How many calories should I eat is 1500.",
        ["eatwell-guide:2"],
        [calorie_hit],
    )
    assert weight.dropped[0].reason == "weight_target"
    assert calorie.dropped[0].reason == "calorie_target"


def test_blended_guidelines_sentence_is_dropped() -> None:
    text = "The guidelines say intake of saturated fats should be less than 10% of total energy intake."
    stored = hit("who-healthy-diet", "who-healthy-diet:12", text)
    result = _check("who-healthy-diet", text, ["who-healthy-diet:12"], [stored])
    assert result.dropped[0].reason == BLENDED_VOICE


def test_renderer_copies_claim_text_and_chunk_citation() -> None:
    stored = hit(
        "who-healthy-diet",
        "who-healthy-diet:12",
        FATS,
        section_heading="Fats",
    )
    result = _check("who-healthy-diet", FATS, ["who-healthy-diet:12"], [stored])
    section = result.sections[0]
    document = REGISTRY["who-healthy-diet"]
    assert section.document_name == document.document_name
    assert section.publisher == document.publisher
    assert section.year == document.year
    assert section.source_url == document.source_url == stored.source_url
    assert section.claims[0].text == FATS
