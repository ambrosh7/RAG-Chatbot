"""Refusal templates. Fixed strings, no model call."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.constants import CORPUS
from src.rag.classifier import (
    CALORIE_TARGET,
    MEDICAL,
    NUTRIENT_LOOKUP,
    SCOPE_REASONS,
    WEIGHT_TARGET,
    classify,
)
from src.rag.refusal import (
    GENERATION_UNAVAILABLE_MESSAGE,
    NOT_IN_CORPUS_MESSAGE,
    NUTRIENT_LOOKUP_MESSAGE,
    PROFESSIONAL_REFERRAL,
    generation_unavailable,
    not_in_corpus,
    out_of_scope,
)

REFUSAL_PATH = Path(__file__).resolve().parents[1] / "src" / "rag" / "refusal.py"


def test_builders_do_not_import_groq_or_the_retriever() -> None:
    source = REFUSAL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = " ".join(imported)
    assert "groq" not in joined
    assert "retriever" not in joined
    assert "chromadb" not in joined


@pytest.mark.parametrize("reason", [MEDICAL, CALORIE_TARGET, WEIGHT_TARGET])
def test_personal_refusal_refers_to_a_professional(reason: str) -> None:
    body = out_of_scope(reason)
    assert body == {
        "type": "out_of_scope",
        "reason": reason,
        "message": PROFESSIONAL_REFERRAL,
        "sections": [],
        "searched": [],
    }
    assert "qualified health professional" in str(body["message"])


def test_nutrient_lookup_has_its_own_sentence() -> None:
    body = out_of_scope(NUTRIENT_LOOKUP)
    assert body["type"] == "out_of_scope"
    assert body["reason"] == NUTRIENT_LOOKUP
    assert body["message"] == NUTRIENT_LOOKUP_MESSAGE
    assert body["sections"] == []
    assert body["searched"] == []
    message = str(body["message"]).casefold()
    assert "dietary guidance" in message
    assert "nutrient numbers" in message
    assert "individual foods" in message
    assert "qualified" not in message


def test_every_guard_reason_builds_an_out_of_scope_response() -> None:
    for reason in SCOPE_REASONS:
        body = out_of_scope(reason)
        assert body["type"] == "out_of_scope"
        assert body["reason"] == reason
        assert body["searched"] == []
        assert body["sections"] == []


def test_unknown_reason_is_rejected() -> None:
    with pytest.raises(ValueError, match="no_supporting_chunks"):
        out_of_scope("no_supporting_chunks")


def test_not_in_corpus_names_all_seven_in_registry_order() -> None:
    body = not_in_corpus()
    assert body["type"] == "not_in_corpus"
    assert body["reason"] == "no_supporting_chunks"
    assert body["message"] == NOT_IN_CORPUS_MESSAGE
    assert body["sections"] == []
    assert body["searched"] == [
        {
            "document_id": document.document_id,
            "document_name": document.document_name,
            "publisher": document.publisher,
            "year": document.year,
        }
        for document in CORPUS
    ]
    assert len(body["searched"]) == 7


def test_filtered_not_in_corpus_names_only_that_document() -> None:
    body = not_in_corpus(["eatwell-guide"])
    assert body["message"] == NOT_IN_CORPUS_MESSAGE
    assert body["searched"] == [
        {
            "document_id": "eatwell-guide",
            "document_name": "The Eatwell Guide booklet",
            "publisher": "Public Health England (now Office for Health Improvement and Disparities)",
            "year": 2018,
        }
    ]


def test_subset_follows_registry_order() -> None:
    body = not_in_corpus(["kitchen-companion", "cold-food-storage"])
    searched = body["searched"]
    assert isinstance(searched, list)
    assert [row["document_id"] for row in searched] == [
        "cold-food-storage",
        "kitchen-companion",
    ]


def test_not_in_corpus_rejects_an_unknown_document() -> None:
    with pytest.raises(ValueError, match="not-a-document"):
        not_in_corpus(["not-a-document"])


def test_not_in_corpus_rejects_an_empty_scope() -> None:
    with pytest.raises(ValueError, match="searched scope"):
        not_in_corpus([])


def test_generation_unavailable_is_not_a_corpus_miss() -> None:
    body = generation_unavailable()
    assert body == {
        "type": "generation_unavailable",
        "reason": "model_error",
        "message": GENERATION_UNAVAILABLE_MESSAGE,
        "sections": [],
        "searched": [],
    }
    message = str(body["message"]).casefold()
    assert "generate" in message
    assert "does not cover" not in message
    assert "i searched" not in message
    assert "qualified" not in message


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("What should I weigh?", WEIGHT_TARGET),
        ("How many calories should I eat to lose weight?", CALORIE_TARGET),
        ("Is this chest pain from something I ate?", MEDICAL),
        ("How much protein is in 100 g of chicken?", NUTRIENT_LOOKUP),
    ],
)
def test_guarded_question_builds_an_empty_search(message: str, reason: str) -> None:
    result = classify(message)
    assert result.reason == reason
    body = out_of_scope(result.reason or "")
    assert body["type"] == "out_of_scope"
    assert body["reason"] == reason
    assert body["searched"] == []
    assert body["sections"] == []
