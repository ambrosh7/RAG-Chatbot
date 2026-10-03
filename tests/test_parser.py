"""Offline tests for HTML and PDF parsing. No test here opens a socket."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from src.constants import BLOCKS_PATH, CORPUS, PROJECT_ROOT
from src.ingestion.parser import (
    Block,
    _merge_list_continuations,
    parse_file,
    parse_html,
    parse_pdf,
    write_blocks,
)

FIXTURES = Path(__file__).parent / "fixtures"
GUIDANCE_HTML = (FIXTURES / "guidance_page.html").read_text(encoding="utf-8")
RAW = PROJECT_ROOT / "data" / "raw"

_REQUIRED = {"document_id", "type", "text", "heading_path"}
_OPTIONAL = {"header"}


def _records(blocks: list[Block]) -> list[dict]:
    return [block.to_dict() for block in blocks]


def test_html_strips_chrome_and_keeps_guidance() -> None:
    blocks = parse_html(GUIDANCE_HTML, document_id="fsa-chill")
    text = "\n".join(block.text for block in blocks)
    assert "Cookies on GOV.UK" not in text
    assert "Is this page useful?" not in text
    assert "Open Government Licence" not in text
    assert "Home" not in text
    assert "Keep chilled food cold" in text
    headings = [block.text for block in blocks if block.type == "heading"]
    assert "1. Chilling food" in headings
    assert "Poultry" in headings
    assert "Contents" not in headings
    assert all(block.heading_path for block in blocks)
    assert all(block.heading_path[0] == "How to chill, freeze and defrost food safely" for block in blocks)


def test_html_list_items_keep_their_numbers() -> None:
    blocks = parse_html(GUIDANCE_HTML, document_id="fsa-chill")
    items = [block.text for block in blocks if block.type == "list_item"]
    assert "your fridge should be between 0 and 5°C" in items
    assert "1. Wash your hands before handling food." in items
    assert "2. Keep raw meat on the bottom shelf." in items
    hands = next(block for block in blocks if block.text.startswith("1. Wash"))
    assert "1. Chilling food" in hands.heading_path
    assert hands.type == "list_item"


def test_html_table_repeats_rowspan_and_stores_the_header() -> None:
    blocks = parse_html(GUIDANCE_HTML, document_id="fsa-chill")
    tables = [block for block in blocks if block.type == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert table.header == ("Food", "Type", "Refrigerator", "Freezer")
    whole = next(line for line in table.text.splitlines() if "Chicken or turkey, whole" in line)
    pieces = next(line for line in table.text.splitlines() if "Chicken or turkey, pieces" in line)
    assert "Poultry" in whole
    assert "1 to 2 days" in whole and "1 year" in whole
    assert "Poultry" in pieces
    assert "9 months" in pieces
    assert "1 year" not in pieces
    payload = table.to_dict()
    assert payload["header"] == ["Food", "Type", "Refrigerator", "Freezer"]
    assert "Poultry" in payload["heading_path"]


def test_block_records_store_only_fields_the_project_uses() -> None:
    blocks = parse_html(GUIDANCE_HTML, document_id="fsa-chill")
    for record in _records(blocks):
        assert set(record) <= _REQUIRED | _OPTIONAL
        assert _REQUIRED <= set(record)
        assert record["type"] in {"heading", "paragraph", "list_item", "table"}
        assert "publisher" not in record
        assert "source_url" not in record
        assert "year" not in record
        if record["type"] == "table":
            assert record["header"]
        else:
            assert "header" not in record


def _write_sample_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 80), "Storage times", fontsize=20, fontname="helv")
    page.insert_text(
        (72, 120),
        "Perishable food belongs in the refrigerator.",
        fontsize=11,
        fontname="helv",
    )
    page.insert_text(
        (72, 150),
        "1. Keep the fridge at 40 F or below.",
        fontsize=11,
        fontname="helv",
    )
    page.insert_text(
        (72, 175),
        "2. Refrigerate leftovers within 2 hours.",
        fontsize=11,
        fontname="helv",
    )
    rows = [
        ["Food", "Refrigerator", "Freezer"],
        ["Chicken, whole", "1 to 2 days", "1 year"],
        ["Pizza", "3 to 4 days", "1 to 2 months"],
    ]
    x0, y0 = 72, 210
    widths = [150, 120, 130]
    row_height = 22
    shape = page.new_shape()
    for row_index in range(len(rows) + 1):
        y = y0 + row_index * row_height
        shape.draw_line((x0, y), (x0 + sum(widths), y))
    for column in range(len(widths) + 1):
        x = x0 + sum(widths[:column])
        shape.draw_line((x, y0), (x, y0 + row_height * len(rows)))
    shape.finish(width=0.8, color=(0, 0, 0))
    shape.commit()
    for row_index, row in enumerate(rows):
        x = x0 + 4
        for column, value in enumerate(row):
            page.insert_text(
                (x, y0 + row_index * row_height + 15),
                value,
                fontsize=10,
                fontname="helv",
            )
            x += widths[column]
    page.insert_text((72, 820), "https://example.test/not-guidance", fontsize=9, fontname="helv")
    page.insert_text((520, 820), "4", fontsize=9, fontname="helv")
    references = document.new_page(width=595, height=842)
    references.insert_text((72, 80), "References", fontsize=18, fontname="helv")
    references.insert_text(
        (72, 120),
        "Smith J. A journal article that is not dietary guidance itself.",
        fontsize=11,
        fontname="helv",
    )
    document.save(path)
    document.close()


def test_pdf_extracts_heading_list_table_and_drops_references(tmp_path: Path) -> None:
    path = tmp_path / "source.pdf"
    _write_sample_pdf(path)
    blocks = parse_pdf(path.read_bytes(), document_id="kitchen-companion")
    text = "\n".join(block.text for block in blocks)
    assert "Perishable food belongs in the refrigerator." in text
    assert "journal article" not in text
    assert "https://example.test/not-guidance" not in text
    items = [block.text for block in blocks if block.type == "list_item"]
    assert any(item.startswith("1. Keep the fridge at 40 F") for item in items)
    assert any(item.startswith("2. Refrigerate leftovers") for item in items)
    tables = [block for block in blocks if block.type == "table"]
    assert len(tables) == 1
    assert tables[0].header == ("Food", "Refrigerator", "Freezer")
    chicken = next(line for line in tables[0].text.splitlines() if "Chicken, whole" in line)
    assert "1 to 2 days" in chicken and "1 year" in chicken
    assert "Pizza" not in chicken
    headings = [block.text for block in blocks if block.type == "heading"]
    assert "Storage times" in headings
    assert "References" not in headings
    prose = next(block for block in blocks if "Perishable food belongs" in block.text)
    assert prose.heading_path[-1] == "Storage times"


def test_write_blocks_jsonl_omits_header_on_prose(tmp_path: Path) -> None:
    blocks = parse_html(GUIDANCE_HTML, document_id="fsa-chill")
    path = tmp_path / "blocks.jsonl"
    write_blocks(blocks, path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all("publisher" not in row for row in rows)
    assert any(row["type"] == "table" and row["header"][0] == "Food" for row in rows)
    assert all("header" not in row for row in rows if row["type"] != "table")


@pytest.mark.skipif(not (RAW / "cold-food-storage" / "source.html").is_file(), reason="corpus not fetched")
def test_cold_food_storage_page_keeps_one_storage_table() -> None:
    blocks = parse_file(RAW / "cold-food-storage" / "source.html", "cold-food-storage")
    text = "\n".join(block.text for block in blocks)
    assert "Return to top" not in text
    assert "FoodKeeper" not in text
    assert "Language Assistance" not in text
    tables = [block for block in blocks if block.type == "table"]
    assert len(tables) == 1
    assert "Refrigerator" in " ".join(tables[0].header)
    assert "Freezer" in " ".join(tables[0].header)
    whole = next(line for line in tables[0].text.splitlines() if "Chicken or turkey, whole" in line)
    assert "1 to 2 days" in whole
    assert "1 year" in whole
    assert "Pizza" not in whole
    assert tables[0].heading_path[0] == "Cold Food Storage Chart"


@pytest.mark.skipif(not (RAW / "fsa-chill" / "source.html").is_file(), reason="corpus not fetched")
def test_fsa_page_keeps_chill_guidance() -> None:
    blocks = parse_file(RAW / "fsa-chill" / "source.html", "fsa-chill")
    text = "\n".join(block.text for block in blocks)
    assert "Cookies on GOV.UK" not in text
    assert "Is this page useful?" not in text
    assert "Open Government Licence" not in text
    assert "between 0 and 5" in text
    headings = [block.text for block in blocks if block.type == "heading"]
    assert any("Chilling food" in heading for heading in headings)
    items = [block for block in blocks if block.type == "list_item"]
    assert any("freezer should be around -18" in block.text for block in items)
    assert all(block.document_id == "fsa-chill" for block in blocks)


def test_wrapped_list_continuation_stays_on_the_same_item() -> None:
    path = ("Five keys to safer food manual", "KEEP CLEAN")
    merged = _merge_list_continuations(
        [
            Block(
                "who-five-keys",
                "list_item",
                "Wash your hands before handling food and often",
                path,
            ),
            Block("who-five-keys", "paragraph", "during food preparation", path),
        ]
    )
    assert len(merged) == 1
    assert merged[0].type == "list_item"
    assert merged[0].text == (
        "Wash your hands before handling food and often during food preparation"
    )


@pytest.mark.skipif(not BLOCKS_PATH.is_file(), reason="parsed blocks have not been written")
def test_stored_blocks_cover_the_corpus_and_keep_storage_rows() -> None:
    rows = [json.loads(line) for line in BLOCKS_PATH.read_text(encoding="utf-8").splitlines()]
    ids = {row["document_id"] for row in rows}
    assert ids == {document.document_id for document in CORPUS}
    for row in rows:
        assert set(row) <= _REQUIRED | _OPTIONAL
        assert row["heading_path"]
        assert row["type"] in {"heading", "paragraph", "list_item", "table"}
        if row["type"] != "table":
            assert "header" not in row
    cold = next(
        row
        for row in rows
        if row["document_id"] == "cold-food-storage" and row["type"] == "table"
    )
    whole = next(line for line in cold["text"].splitlines() if "Chicken or turkey, whole" in line)
    assert "1 to 2 days" in whole and "1 year" in whole and "Pizza" not in whole
    hands = next(
        row["text"]
        for row in rows
        if row["document_id"] == "who-five-keys"
        and row["text"].startswith("Wash your hands before handling food")
    )
    assert "during food preparation" in hands
    joined = "\n".join(row["text"] for row in rows).lower()
    assert "open government licence" not in joined
    assert "is this page useful" not in joined


def test_registry_names_are_the_seven_documents() -> None:
    assert {document.document_id for document in CORPUS} == {
        "cold-food-storage",
        "who-healthy-diet",
        "fao-who-healthy-diets",
        "who-five-keys",
        "eatwell-guide",
        "fsa-chill",
        "kitchen-companion",
    }
    assert BLOCKS_PATH.name == "blocks.jsonl"
