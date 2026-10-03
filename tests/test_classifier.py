"""Scope guard. No index and no model."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.rag.classifier import (
    CALORIE_TARGET,
    MEDICAL,
    NUTRIENT_LOOKUP,
    WEIGHT_TARGET,
    Classification,
    EmptyMessageError,
    UnknownDocumentError,
    classify,
)

CLASSIFIER_PATH = Path(__file__).resolve().parents[1] / "src" / "rag" / "classifier.py"


def test_classifier_does_not_import_retriever_or_groq() -> None:
    source = CLASSIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
            imported.extend(alias.name for alias in node.names)
    joined = " ".join(imported)
    assert "retriever" not in joined
    assert "groq" not in joined
    assert "chromadb" not in joined


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        pytest.param("What should I weigh?", WEIGHT_TARGET, id="P5-01"),
        pytest.param("What is my ideal weight?", WEIGHT_TARGET, id="P5-02-ideal"),
        pytest.param("How much should I weigh at 180 cm?", WEIGHT_TARGET, id="P5-02-height"),
        pytest.param(
            "How many calories should I eat to lose weight?",
            CALORIE_TARGET,
            id="P5-03",
        ),
        pytest.param("What calorie deficit should I run?", CALORIE_TARGET, id="P5-04"),
        pytest.param("Is this chest pain from something I ate?", MEDICAL, id="P5-05"),
        pytest.param("I have diabetes. What should I eat?", MEDICAL, id="P5-06"),
        pytest.param(
            "Which medicine should I take for food poisoning?",
            MEDICAL,
            id="P5-07",
        ),
        pytest.param("Can I eat this cheese if I am pregnant?", MEDICAL, id="P5-08"),
        pytest.param(
            "How much protein is in 100 g of chicken?",
            NUTRIENT_LOOKUP,
            id="P5-10",
        ),
        pytest.param("How many calories are in a banana?", NUTRIENT_LOOKUP, id="P5-11"),
        pytest.param("How many grams of fat are in olive oil?", NUTRIENT_LOOKUP, id="P5-12"),
        pytest.param(
            "According to the Eatwell Guide, how many calories should I eat?",
            CALORIE_TARGET,
            id="P5-15",
        ),
        pytest.param(
            "I weigh 90 kg. Is that too much, and how much protein is in chicken?",
            WEIGHT_TARGET,
            id="P5-16",
        ),
        pytest.param("Protein in 100 g of chicken", NUTRIENT_LOOKUP, id="X-04"),
    ],
)
def test_out_of_scope_reason(message: str, reason: str) -> None:
    assert classify(message).reason == reason


@pytest.mark.parametrize(
    ("message", "document_id"),
    [
        pytest.param("Is it safe to eat raw chicken?", None, id="P5-09"),
        pytest.param(
            "What does WHO say about limiting free sugars?",
            None,
            id="P5-13",
        ),
        pytest.param("What does the guidance say about salt?", None, id="P5-14"),
        pytest.param(
            "What does WHO say about limiting free sugars to less than 10%?",
            None,
            id="P5-13-percent",
        ),
        pytest.param("What is a healthy diet?", None, id="P5-22"),
        pytest.param("What does WHO say about free sugars?", None, id="P5-23"),
        pytest.param(
            "According to the Eatwell Guide, how much of the diet should be fruit and vegetables?",
            "eatwell-guide",
            id="P5-18",
        ),
        pytest.param("How long can I keep a whole chicken in the fridge?", None, id="fridge"),
        pytest.param("What do the documents say about cooking oil?", None, id="oil"),
        pytest.param("What is the tariff on imported olive oil?", None, id="tariff"),
        pytest.param(
            "What does WHO say about saturated fat and trans fat?",
            None,
            id="fats",
        ),
        pytest.param("How long should I cook a 20 pound turkey?", None, id="turkey"),
        pytest.param("How much does a whole chicken weigh?", None, id="bird-weight"),
        pytest.param("I weigh the chicken before roasting", None, id="weigh-the-food"),
    ],
)
def test_in_scope(message: str, document_id: str | None) -> None:
    result = classify(message)
    assert result.reason is None
    assert result.document_id == document_id


def test_personal_calorie_goal_still_records_the_named_document() -> None:
    # The reason is the decision. The alias remains visible so a caller can
    # see what was named, and must not retrieve while reason is set.
    result = classify("According to the Eatwell Guide, how many calories should I eat?")
    assert result == Classification(CALORIE_TARGET, "eatwell-guide")


def test_weight_beats_a_nutrient_lookup_in_the_same_message() -> None:
    result = classify(
        "I weigh 90 kg. Is that too much, and how much protein is in chicken?"
    )
    assert result.reason == WEIGHT_TARGET


def test_medical_beats_a_later_reason() -> None:
    result = classify("I have diabetes. What should I weigh?")
    assert result.reason == MEDICAL


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_empty_message_is_rejected_before_the_guard(message: str) -> None:
    with pytest.raises(EmptyMessageError):
        classify(message)


@pytest.mark.parametrize(
    "message",
    [
        "eatwell",
        "Eatwell Guide",
        "the Eatwell Guide booklet",
    ],
)
def test_eatwell_aliases(message: str) -> None:
    result = classify(f"According to {message}, how much fruit?")
    assert result.reason is None
    assert result.document_id == "eatwell-guide"


@pytest.mark.parametrize(
    "message",
    [
        "Five keys to safer food",
        "the five keys",
    ],
)
def test_five_keys_aliases(message: str) -> None:
    result = classify(f"What does {message} say about leftovers?")
    assert result.document_id == "who-five-keys"


def test_two_named_documents_stay_unfiltered() -> None:
    result = classify("What do the Eatwell Guide and Kitchen Companion say about oil?")
    assert result.reason is None
    assert result.document_id is None


def test_explicit_document_id_overrides_an_alias() -> None:
    result = classify(
        "According to the Eatwell Guide, how much of the diet should be fruit and vegetables?",
        document_id="kitchen-companion",
    )
    assert result.reason is None
    assert result.document_id == "kitchen-companion"


def test_explicit_document_id_when_the_message_names_none() -> None:
    result = classify(
        "How long can I keep cooked leftovers?",
        document_id="fsa-chill",
    )
    assert result.reason is None
    assert result.document_id == "fsa-chill"


def test_unknown_document_id_is_rejected() -> None:
    with pytest.raises(UnknownDocumentError, match="not-a-document"):
        classify("How long can I keep chicken?", document_id="not-a-document")


def test_alias_must_be_its_own_span() -> None:
    assert classify("The greatwell index lists five keystones").document_id is None
    assert classify("I want to eat chicken tonight").document_id is None
    assert classify("I need an eatwellness plan").document_id is None


def test_fact_sheet_alias_selects_the_who_document() -> None:
    result = classify("According to the Healthy diet (Fact sheet), what about fruit?")
    assert result.reason is None
    assert result.document_id == "who-healthy-diet"


def test_bare_who_does_not_select_a_document() -> None:
    assert classify("What does WHO say about free sugars?").document_id is None


def test_cold_storage_and_kitchen_companion_aliases() -> None:
    assert (
        classify("According to the Cold Food Storage Chart, how long is raw chicken?").document_id
        == "cold-food-storage"
    )
    assert (
        classify("What does Kitchen Companion say about leftovers?").document_id
        == "kitchen-companion"
    )
