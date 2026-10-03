"""Chunker tests. Token estimates use round(word_count × 1.3). No network."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src.constants import BLOCKS_PATH, CHUNK_TARGET_TOKENS, CHUNKS_PATH, CORPUS
from src.ingestion.chunker import (
    ChunkError,
    chunk_blocks,
    chunk_corpus,
    estimate_tokens,
    is_form_blank,
    label_table_rows,
    load_retrieval_dates,
)
from src.ingestion.parser import Block, parse_html

FIXTURES = Path(__file__).parent / "fixtures"
GUIDANCE_HTML = (FIXTURES / "guidance_page.html").read_text(encoding="utf-8")

DATES = {document.document_id: "2026-10-03" for document in CORPUS}
REGISTRY = {document.document_id: document for document in CORPUS}


def _words(count: int, token: str = "word") -> str:
    return " ".join(f"{token}{index}" for index in range(count))


def _paragraph(document_id: str, text: str, heading: tuple[str, ...] = ()) -> Block:
    return Block(document_id, "paragraph", text, heading or (REGISTRY[document_id].document_name,))


def _chunk(blocks: list[Block], document_id: str = "fsa-chill"):
    dates = {document_id: DATES[document_id]}
    return chunk_blocks(blocks, dates)


def test_estimate_tokens_matches_the_phase_rule() -> None:
    assert estimate_tokens("one two three") == round(3 * 1.3)
    assert estimate_tokens(_words(154)) <= CHUNK_TARGET_TOKENS
    assert estimate_tokens(_words(155)) > CHUNK_TARGET_TOKENS


def test_guidance_fixture_rows_and_numbered_items_stay_whole() -> None:
    blocks = parse_html(GUIDANCE_HTML, document_id="fsa-chill")
    chunks = chunk_blocks(blocks, {"fsa-chill": "2026-10-03"})
    assert chunks
    assert all(chunk.block_type != "heading" for chunk in chunks)
    assert all(chunk.section_heading for chunk in chunks)
    assert all(chunk.embed_text.startswith(chunk.document_name) for chunk in chunks)
    document = REGISTRY["fsa-chill"]
    assert all(chunk.document_name == document.document_name for chunk in chunks)
    assert all(chunk.publisher == document.publisher for chunk in chunks)
    assert all(chunk.year == document.year and isinstance(chunk.year, int) for chunk in chunks)
    assert all(chunk.source_url == document.source_url for chunk in chunks)
    assert all(chunk.retrieval_date == "2026-10-03" for chunk in chunks)

    whole = next(chunk for chunk in chunks if "Chicken or turkey, whole" in chunk.text)
    pieces = next(chunk for chunk in chunks if "Chicken or turkey, pieces" in chunk.text)
    assert whole.block_type == "table_row"
    assert "1 to 2 days" in whole.text and "1 year" in whole.text
    assert "Poultry" in whole.text
    assert "9 months" not in whole.text
    assert "Pizza" not in whole.text
    assert "1 year" not in pieces.text
    assert "Food:" in whole.text and "Freezer:" in whole.text
    assert whole.embed_text.startswith("How to chill, freeze and defrost food safely — ")
    assert "\n" in whole.embed_text

    numbered = [chunk for chunk in chunks if chunk.text.startswith("1. Wash your hands")]
    assert len(numbered) == 1
    assert numbered[0].block_type == "list_item"
    assert numbered[0].text.startswith("1. Wash your hands before handling food.")
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "Cookies on GOV.UK" not in joined
    assert "Is this page useful?" not in joined


def test_blank_category_cell_is_carried_and_empty_cells_are_omitted() -> None:
    table = Block(
        "cold-food-storage",
        "table",
        "\n".join(
            [
                "| Food | Type | Refrigerator | Freezer |",
                "| --- | --- | --- | --- |",
                "| Poultry | Chicken or turkey, whole | 1 to 2 days | 1 year |",
                "|  | Chicken or turkey, pieces | 1 to 2 days | 9 months |",
                "| Eggs |  | 3 to 5 weeks |  |",
            ]
        ),
        ("Cold Food Storage Chart",),
        header=("Food", "Type", "Refrigerator", "Freezer"),
    )
    rows = label_table_rows(table)
    assert rows[1].startswith("Food: Poultry | Type: Chicken or turkey, pieces")
    assert "9 months" in rows[1]
    assert "1 year" not in rows[1]
    assert rows[2] == "Food: Eggs | Refrigerator: 3 to 5 weeks"
    assert "Type:" not in rows[2]
    assert "Freezer:" not in rows[2]


def test_long_table_row_and_short_list_items_are_not_packed() -> None:
    long_note = _words(180, "leftover")
    blocks = [
        Block(
            "kitchen-companion",
            "table",
            "\n".join(
                [
                    "| Food | Note |",
                    "| --- | --- |",
                    f"| Rice | {long_note} |",
                ]
            ),
            ("Kitchen Companion: Your Safe Food Handbook", "Cold Storage Chart"),
            header=("Food", "Note"),
        ),
        Block(
            "kitchen-companion",
            "list_item",
            "1) Chill leftovers within two hours.",
            ("Kitchen Companion: Your Safe Food Handbook", "Cold Storage Chart"),
        ),
        Block(
            "kitchen-companion",
            "list_item",
            "puddings",
            ("Kitchen Companion: Your Safe Food Handbook", "Cold Storage Chart"),
        ),
        Block(
            "kitchen-companion",
            "list_item",
            "(1) Use a fridge thermometer.",
            ("Kitchen Companion: Your Safe Food Handbook", "Cold Storage Chart"),
        ),
    ]
    chunks = _chunk(blocks, "kitchen-companion")
    assert len(chunks) == 4
    assert chunks[0].block_type == "table_row"
    assert chunks[0].text.startswith("Food: Rice | Note: leftover0")
    assert "leftover179" in chunks[0].text
    assert estimate_tokens(chunks[0].text) > CHUNK_TARGET_TOKENS
    assert [chunk.block_type for chunk in chunks[1:]] == ["list_item", "list_item", "list_item"]
    assert chunks[1].text == "1) Chill leftovers within two hours."
    assert chunks[2].text == "puddings"
    assert chunks[3].text == "(1) Use a fridge thermometer."
    assert chunks[2].chunk_id == "kitchen-companion:2"
    assert [chunk.chunk_id for chunk in chunks] == [
        "kitchen-companion:0",
        "kitchen-companion:1",
        "kitchen-companion:2",
        "kitchen-companion:3",
    ]


def test_paragraphs_pack_to_the_target_and_break_on_a_new_heading() -> None:
    heading = ("Healthy diet (Fact sheet)", "KEY FACTS")
    blocks = [
        Block("who-healthy-diet", "heading", "KEY FACTS", heading),
        _paragraph("who-healthy-diet", _words(50, "alpha"), heading),
        _paragraph("who-healthy-diet", _words(50, "beta"), heading),
        _paragraph("who-healthy-diet", _words(50, "gamma"), heading),
        _paragraph("who-healthy-diet", _words(50, "delta"), heading),
        Block("who-healthy-diet", "heading", "Next", ("Healthy diet (Fact sheet)", "Next")),
        _paragraph("who-healthy-diet", _words(20, "later"), ("Healthy diet (Fact sheet)", "Next")),
    ]
    chunks = _chunk(blocks, "who-healthy-diet")
    assert len(chunks) == 3
    assert chunks[0].block_type == "paragraph"
    assert "alpha0" in chunks[0].text and "gamma49" in chunks[0].text
    assert "delta0" not in chunks[0].text
    assert "beta0" in chunks[0].text
    assert estimate_tokens(chunks[0].text) <= CHUNK_TARGET_TOKENS
    assert chunks[1].text.startswith("delta0")
    assert "gamma49" not in chunks[1].text
    assert chunks[0].section_heading == "Healthy diet (Fact sheet) > KEY FACTS"
    assert chunks[2].section_heading == "Healthy diet (Fact sheet) > Next"
    assert chunks[0].chunk_id == "who-healthy-diet:0"
    assert chunks[2].chunk_id == "who-healthy-diet:2"
    assert all(not chunk.text == "KEY FACTS" for chunk in chunks)


def test_paragraph_at_the_target_is_not_sentence_split() -> None:
    body = "Keep food cold. " + _words(151, "storage")
    assert estimate_tokens(body) <= CHUNK_TARGET_TOKENS
    chunks = _chunk([_paragraph("eatwell-guide", body)], "eatwell-guide")
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Keep food cold.")
    assert "storage150" in chunks[0].text


def test_oversized_paragraph_splits_on_a_sentence_without_overlap() -> None:
    first = "Alpha " + _words(110, "first") + "."
    second = "Beta " + _words(110, "second") + "."
    heading = ("The Eatwell Guide booklet", "8 tips for eating well")
    block = _paragraph("eatwell-guide", f"{first} {second}", heading)
    assert estimate_tokens(block.text) > CHUNK_TARGET_TOKENS
    chunks = _chunk([block], "eatwell-guide")
    assert len(chunks) == 2
    assert chunks[0].section_heading == chunks[1].section_heading
    assert chunks[0].text.endswith("first109.")
    assert chunks[1].text.startswith("Beta ")
    assert "first109" not in chunks[1].text
    assert "Beta" not in chunks[0].text
    assert estimate_tokens(chunks[0].text) <= CHUNK_TARGET_TOKENS
    assert estimate_tokens(chunks[1].text) <= CHUNK_TARGET_TOKENS


def test_oversized_paragraph_without_a_sentence_boundary_stops() -> None:
    block = _paragraph("who-five-keys", _words(160, "clean"), ("Five keys to safer food manual", "KEEP CLEAN"))
    assert estimate_tokens(block.text) > CHUNK_TARGET_TOKENS
    with pytest.raises(ChunkError, match="who-five-keys") as caught:
        _chunk([block], "who-five-keys")
    assert "KEEP CLEAN" in str(caught.value)
    assert "no sentence boundary" in str(caught.value)


def test_missing_heading_uses_the_document_name_and_ids_do_not_skip() -> None:
    blocks = [
        Block("fsa-chill", "heading", "Ignored", ()),
        Block("fsa-chill", "paragraph", "Cool cooked food before it goes in the fridge.", ()),
        Block("cold-food-storage", "paragraph", "Frozen food kept at 0 F stays safe.", ()),
    ]
    chunks = chunk_blocks(
        blocks,
        {"fsa-chill": "2026-10-03", "cold-food-storage": "2026-10-03"},
    )
    assert len(chunks) == 2
    assert chunks[0].section_heading == "How to chill, freeze and defrost food safely"
    assert chunks[0].chunk_id == "fsa-chill:0"
    assert chunks[1].chunk_id == "cold-food-storage:0"
    assert chunks[1].source_url == REGISTRY["cold-food-storage"].source_url
    assert "iris.who.int" not in chunks[1].source_url


def test_list_item_between_paragraphs_closes_the_pack() -> None:
    heading = ("How to chill, freeze and defrost food safely",)
    blocks = [
        _paragraph("fsa-chill", _words(40, "before"), heading),
        Block("fsa-chill", "list_item", "1. Wash your hands.", heading),
        _paragraph("fsa-chill", _words(40, "after"), heading),
    ]
    chunks = _chunk(blocks, "fsa-chill")
    assert [chunk.block_type for chunk in chunks] == ["paragraph", "list_item", "paragraph"]
    assert "after0" not in chunks[0].text
    assert "before0" not in chunks[2].text


def test_form_blank_paragraphs_are_not_chunks() -> None:
    heading = (
        "Five keys to safer food manual",
        "KEEP CLEAN",
        "Adaptation of the Five Keys to Safer Food Manual",
    )
    question = (
        "Do you think the level of language in the Five Keys to Safer Food poster was appropriate?"
    )
    blank = "If no, please explain " + ("_" * 80)
    assert is_form_blank(blank)
    assert not is_form_blank(question)
    chunks = _chunk(
        [
            _paragraph("who-five-keys", question, heading),
            _paragraph("who-five-keys", blank, heading),
            _paragraph("who-five-keys", "_" * 40, heading),
        ],
        "who-five-keys",
    )
    assert len(chunks) == 1
    assert chunks[0].text == question
    assert "_" not in chunks[0].text


def test_unknown_document_is_rejected() -> None:
    block = Block("not-a-document", "paragraph", "Some guidance text here.", ("Title",))
    with pytest.raises(ChunkError, match="not-a-document"):
        chunk_blocks([block], {"not-a-document": "2026-10-03"})


@pytest.mark.skipif(not BLOCKS_PATH.is_file(), reason="parsed blocks have not been written")
def test_parsed_corpus_meets_the_chunk_exit_criteria() -> None:
    chunks = chunk_corpus()
    dates = load_retrieval_dates()
    ids = {chunk.document_id for chunk in chunks}
    assert ids == {document.document_id for document in CORPUS}

    by_document: dict[str, list] = {document.document_id: [] for document in CORPUS}
    for chunk in chunks:
        by_document[chunk.document_id].append(chunk)
        document = REGISTRY[chunk.document_id]
        assert chunk.document_name == document.document_name
        assert chunk.publisher == document.publisher
        assert chunk.year == document.year
        assert chunk.source_url == document.source_url
        assert chunk.retrieval_date == dates[chunk.document_id]
        assert chunk.section_heading
        assert chunk.block_type in {"paragraph", "list_item", "table_row"}
        assert chunk.embed_text.startswith(f"{chunk.document_name} — {chunk.section_heading}\n")
        assert chunk.text
        if chunk.block_type == "paragraph":
            assert estimate_tokens(chunk.text) <= CHUNK_TARGET_TOKENS
            assert not is_form_blank(chunk.text)

    for document_id, group in by_document.items():
        ordinals = [int(chunk.chunk_id.split(":", 1)[1]) for chunk in group]
        assert ordinals == list(range(len(group)))
        assert all(chunk.chunk_id.startswith(f"{document_id}:") for chunk in group)

    blocks = [
        json.loads(line)
        for line in BLOCKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for document_id in ids:
        block_items = [
            row["text"]
            for row in blocks
            if row["document_id"] == document_id and row["type"] == "list_item"
        ]
        chunk_items = [
            chunk.text
            for chunk in by_document[document_id]
            if chunk.block_type == "list_item"
        ]
        assert Counter(block_items) == Counter(chunk_items)

    cold_rows = [
        chunk
        for chunk in by_document["cold-food-storage"]
        if chunk.block_type == "table_row"
    ]
    assert len(cold_rows) == 53
    whole = next(chunk for chunk in cold_rows if "Chicken or turkey, whole" in chunk.text)
    assert "1 to 2 days" in whole.text and "1 year" in whole.text
    assert "9 months" not in whole.text
    assert "Pizza" not in whole.text
    assert all("Refrigerator" in chunk.text and "Freezer" in chunk.text for chunk in cold_rows)

    opened = [
        chunk.text
        for chunk in by_document["kitchen-companion"]
        if chunk.block_type == "table_row" and "Opened package" in chunk.text
    ]
    assert opened
    assert all("Hot dogs" not in text for text in opened)

    heading_texts = {row["text"] for row in blocks if row["type"] == "heading"}
    assert all(chunk.text not in heading_texts for chunk in chunks)
    assert CHUNKS_PATH.name == "chunks.jsonl"
