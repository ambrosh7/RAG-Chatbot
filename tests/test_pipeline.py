"""Answer pipeline with a fake retriever and a fake Groq client. No network."""

from __future__ import annotations

import json

import pytest

from src.constants import CORPUS
from src.rag.classifier import EmptyMessageError, UnknownDocumentError
from src.rag.generator import GenerationError
from src.rag.pipeline import answer
from src.rag.refusal import (
    GENERATION_UNAVAILABLE_MESSAGE,
    NOT_IN_CORPUS_MESSAGE,
    NO_SURVIVING_CLAIMS,
    searched_entries,
)
from src.rag.retriever import DocumentHits, Hit, RetrievalResult, ScopeDocument

REGISTRY = {document.document_id: document for document in CORPUS}
CHICKEN = "Whole chicken keeps 1 to 2 days in the refrigerator and 1 year in the freezer."
FATS = "Intake of saturated fats should be less than 10% of total energy intake."
OIL = "Choose unsaturated oils such as rapeseed olive and sunflower."
LEFTOVERS = "Store cooking oil with leftovers in the refrigerator for 3 to 4 days."


def hit(
    document_id: str,
    chunk_id: str,
    text: str,
    *,
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
        source_url=document.source_url,
        section_heading=section_heading,
        block_type="paragraph",
        retrieval_date="2026-10-03",
    )


def retrieval(groups: list[DocumentHits], *, document_id: str | None = None) -> RetrievalResult:
    if document_id is None:
        scope = tuple(
            ScopeDocument(document.document_id, document.document_name, document.publisher, document.year)
            for document in CORPUS
        )
    else:
        document = REGISTRY[document_id]
        scope = (
            ScopeDocument(document.document_id, document.document_name, document.publisher, document.year),
        )
    flat = tuple(item for group in groups for item in group.hits)
    return RetrievalResult(
        scope=scope,
        groups=tuple(groups),
        candidate_count=len(flat),
        candidates=flat,
    )


def payload(documents: list[tuple[str, list[tuple[str, list[str]]]]]) -> str:
    return json.dumps(
        {
            "documents": [
                {
                    "document_id": document_id,
                    "claims": [
                        {"text": text, "chunk_ids": chunk_ids} for text, chunk_ids in claims
                    ],
                }
                for document_id, claims in documents
            ]
        }
    )


class RecordingRetriever:
    def __init__(self, result: RetrievalResult | None = None) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.result = result

    def __call__(self, query: str, document_id: str | None = None) -> RetrievalResult:
        self.calls.append((query, document_id))
        if self.result is None:
            raise AssertionError("retriever should not be called")
        return self.result


class RecordingClient:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, str]] = []
        self.content = content
        self.error = error

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if self.error is not None:
            raise self.error
        if self.content is None:
            raise AssertionError("client should not be called")
        return self.content


def test_empty_message_is_rejected_before_retrieval() -> None:
    retriever = RecordingRetriever()
    client = RecordingClient()
    with pytest.raises(EmptyMessageError):
        answer("   ", retriever=retriever, client=client)
    assert retriever.calls == []
    assert client.calls == []


def test_unknown_document_id_does_not_retrieve() -> None:
    retriever = RecordingRetriever()
    with pytest.raises(UnknownDocumentError):
        answer("How long can I keep chicken?", "not-a-document", retriever=retriever)
    assert retriever.calls == []


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("What should I weigh?", "weight_target"),
        ("How many calories should I eat to lose weight?", "calorie_target"),
        ("Is this chest pain from something I ate?", "medical"),
        ("How much protein is in 100 g of chicken?", "nutrient_lookup"),
    ],
)
def test_guard_match_does_not_retrieve_or_call_groq(message: str, reason: str) -> None:
    retriever = RecordingRetriever()
    client = RecordingClient()
    body = answer(message, retriever=retriever, client=client)
    assert body["type"] == "out_of_scope"
    assert body["reason"] == reason
    assert body["sections"] == []
    assert body["searched"] == []
    assert retriever.calls == []
    assert client.calls == []
    if reason == "nutrient_lookup":
        assert "nutrient numbers" in str(body["message"])
    else:
        assert "qualified health professional" in str(body["message"])


def test_in_scope_miss_does_not_call_groq_and_names_all_seven() -> None:
    retriever = RecordingRetriever(retrieval([]))
    client = RecordingClient()
    body = answer("What is the tariff on imported olive oil?", retriever=retriever, client=client)
    assert body["type"] == "not_in_corpus"
    assert body["reason"] == "no_supporting_chunks"
    assert body["message"] == NOT_IN_CORPUS_MESSAGE
    assert body["sections"] == []
    assert body["searched"] == searched_entries()
    assert client.calls == []
    assert retriever.calls == [("What is the tariff on imported olive oil?", None)]


def test_filtered_miss_names_only_that_document() -> None:
    retriever = RecordingRetriever(retrieval([], document_id="eatwell-guide"))
    client = RecordingClient()
    body = answer(
        "How long can raw chicken stay in the fridge?",
        "eatwell-guide",
        retriever=retriever,
        client=client,
    )
    assert body["type"] == "not_in_corpus"
    assert body["searched"] == searched_entries(["eatwell-guide"])
    document = REGISTRY["eatwell-guide"]
    assert body["searched"] == [
        {
            "document_id": "eatwell-guide",
            "document_name": document.document_name,
            "publisher": document.publisher,
            "year": document.year,
        }
    ]
    assert client.calls == []
    assert retriever.calls[0][1] == "eatwell-guide"


def test_alias_sets_the_document_scope() -> None:
    message = "According to the Eatwell Guide, how much of the diet should be fruit and vegetables?"
    retriever = RecordingRetriever(retrieval([], document_id="eatwell-guide"))
    client = RecordingClient()
    body = answer(message, retriever=retriever, client=client)
    assert retriever.calls == [(message, "eatwell-guide")]
    assert [row["document_id"] for row in body["searched"]] == ["eatwell-guide"]
    assert client.calls == []


def test_explicit_document_id_overrides_the_alias() -> None:
    retriever = RecordingRetriever(retrieval([], document_id="kitchen-companion"))
    answer(
        "According to the Eatwell Guide, how much fruit?",
        "kitchen-companion",
        retriever=retriever,
        client=RecordingClient(),
    )
    assert retriever.calls[0][1] == "kitchen-companion"


def test_two_documents_stay_separate_and_follow_registry_order() -> None:
    groups = [
        DocumentHits("kitchen-companion", (hit("kitchen-companion", "kitchen-companion:44", LEFTOVERS, section_heading="Leftovers"),)),
        DocumentHits("eatwell-guide", (hit("eatwell-guide", "eatwell-guide:71", OIL, section_heading="Oils"),)),
    ]
    # The model returns the later registry document first, and invents a link.
    raw = json.dumps(
        {
            "documents": [
                {
                    "document_id": "kitchen-companion",
                    "publisher": "Invented",
                    "url": "https://evil.example",
                    "claims": [
                        {
                            "text": LEFTOVERS,
                            "chunk_ids": ["kitchen-companion:44"],
                            "source_url": "https://evil.example",
                        }
                    ],
                },
                {
                    "document_id": "eatwell-guide",
                    "claims": [{"text": OIL, "chunk_ids": ["eatwell-guide:71"]}],
                },
            ]
        }
    )
    client = RecordingClient(raw)
    body = answer(
        "What do the documents say about cooking oil?",
        retriever=RecordingRetriever(retrieval(groups)),
        client=client,
    )
    assert body["type"] == "answer"
    assert body["searched"] == searched_entries()
    assert [section["document_id"] for section in body["sections"]] == [
        "eatwell-guide",
        "kitchen-companion",
    ]
    eatwell, kitchen = body["sections"]
    assert eatwell["source_url"] == REGISTRY["eatwell-guide"].source_url
    assert eatwell["document_name"] == REGISTRY["eatwell-guide"].document_name
    assert eatwell["publisher"] == REGISTRY["eatwell-guide"].publisher
    assert eatwell["year"] == 2018
    assert eatwell["claims"][0]["text"] == OIL
    assert eatwell["claims"][0]["chunk_ids"] == ["eatwell-guide:71"]
    assert eatwell["claims"][0]["section_heading"] == "Oils"
    assert kitchen["source_url"] == REGISTRY["kitchen-companion"].source_url
    assert kitchen["publisher"] == REGISTRY["kitchen-companion"].publisher
    assert kitchen["year"] == 2008
    assert kitchen["claims"][0]["chunk_ids"] == ["kitchen-companion:44"]
    assert "rapeseed" not in kitchen["claims"][0]["text"]
    assert "leftovers" not in eatwell["claims"][0]["text"]
    blob = json.dumps(body["sections"]).casefold()
    assert "the guidelines say" not in blob
    assert "evil.example" not in blob
    assert "invented" not in blob
    user = client.calls[0]["user"]
    assert "What do the documents say about cooking oil?" in user
    assert OIL in user
    assert LEFTOVERS in user
    assert "DECOY CHUNK" not in user
    assert "DECOY CHUNK" not in client.calls[0]["system"]


def test_prompt_does_not_include_a_chunk_that_was_not_retrieved() -> None:
    groups = [DocumentHits("eatwell-guide", (hit("eatwell-guide", "eatwell-guide:71", OIL),))]
    client = RecordingClient(payload([("eatwell-guide", [(OIL, ["eatwell-guide:71"])])]))
    answer(
        "What do the documents say about cooking oil?",
        retriever=RecordingRetriever(retrieval(groups)),
        client=client,
    )
    assert LEFTOVERS not in client.calls[0]["user"]
    assert "[eatwell-guide:71]" in client.calls[0]["user"]


def test_invented_number_is_dropped_and_a_supported_claim_remains() -> None:
    groups = [
        DocumentHits(
            "cold-food-storage",
            (hit("cold-food-storage", "cold-food-storage:24", CHICKEN, section_heading="Poultry"),),
        )
    ]
    raw = payload(
        [
            (
                "cold-food-storage",
                [
                    ("Whole chicken keeps 3 days in the refrigerator.", ["cold-food-storage:24"]),
                    (CHICKEN, ["cold-food-storage:24"]),
                ],
            )
        ]
    )
    body = answer("How long can I keep a whole chicken?", retriever=RecordingRetriever(retrieval(groups)), client=RecordingClient(raw))
    claims = body["sections"][0]["claims"]
    assert [claim["text"] for claim in claims] == [CHICKEN]
    assert "3 days" not in claims[0]["text"]


def test_every_dropped_claim_names_the_searched_scope() -> None:
    groups = [
        DocumentHits(
            "cold-food-storage",
            (hit("cold-food-storage", "cold-food-storage:24", CHICKEN),),
        )
    ]
    raw = payload(
        [("cold-food-storage", [("Whole chicken keeps 3 days in the refrigerator.", ["cold-food-storage:24"])])]
    )
    body = answer(
        "How long can I keep a whole chicken?",
        retriever=RecordingRetriever(retrieval(groups)),
        client=RecordingClient(raw),
    )
    assert body["type"] == "not_in_corpus"
    assert body["reason"] == NO_SURVIVING_CLAIMS
    assert body["message"] == NOT_IN_CORPUS_MESSAGE
    assert body["sections"] == []
    assert body["searched"] == searched_entries()


def test_empty_model_documents_are_not_in_corpus() -> None:
    groups = [DocumentHits("eatwell-guide", (hit("eatwell-guide", "eatwell-guide:71", OIL),))]
    body = answer(
        "What do the documents say about cooking oil?",
        retriever=RecordingRetriever(retrieval(groups)),
        client=RecordingClient('{"documents": []}'),
    )
    assert body["type"] == "not_in_corpus"
    assert body["reason"] == NO_SURVIVING_CLAIMS
    assert [row["document_id"] for row in body["searched"]] == [document.document_id for document in CORPUS]


def test_a_document_with_no_surviving_claim_is_omitted() -> None:
    groups = [
        DocumentHits("who-healthy-diet", (hit("who-healthy-diet", "who-healthy-diet:12", FATS, section_heading="Fats"),)),
        DocumentHits("kitchen-companion", (hit("kitchen-companion", "kitchen-companion:44", LEFTOVERS),)),
    ]
    raw = payload(
        [
            ("kitchen-companion", [("Store cooking oil for 9 days.", ["kitchen-companion:44"])]),
            ("who-healthy-diet", [(FATS, ["who-healthy-diet:12"])]),
        ]
    )
    body = answer(
        "What do the documents say about fat?",
        retriever=RecordingRetriever(retrieval(groups)),
        client=RecordingClient(raw),
    )
    assert body["type"] == "answer"
    assert [section["document_id"] for section in body["sections"]] == ["who-healthy-diet"]
    assert body["searched"] == searched_entries()
    section = body["sections"][0]
    assert section["source_url"] == REGISTRY["who-healthy-diet"].source_url
    assert section["document_name"] == REGISTRY["who-healthy-diet"].document_name
    assert section["publisher"] == REGISTRY["who-healthy-diet"].publisher
    assert section["year"] == 2018
    assert section["claims"][0]["section_heading"] == "Fats"


@pytest.mark.parametrize(
    "content",
    [
        "The chicken keeps for 1 to 2 days.",
        '```json\n{"documents": []}\n```',
        '{"documents": [{"document_id": "eatwell-guide", "claims": [],},]}',
    ],
)
def test_invalid_model_json_is_generation_unavailable(content: str) -> None:
    groups = [DocumentHits("eatwell-guide", (hit("eatwell-guide", "eatwell-guide:71", OIL),))]
    body = answer(
        "What do the documents say about cooking oil?",
        retriever=RecordingRetriever(retrieval(groups)),
        client=RecordingClient(content),
    )
    assert body["type"] == "generation_unavailable"
    assert body["type"] != "not_in_corpus"
    assert body["reason"] == "model_error"
    assert body["message"] == GENERATION_UNAVAILABLE_MESSAGE
    assert body["searched"] == []
    assert body["sections"] == []
    assert "does not cover" not in str(body["message"]).casefold()


def test_weight_target_logs_the_reason_without_a_hit_count(caplog: pytest.LogCaptureFixture) -> None:
    retriever = RecordingRetriever()
    with caplog.at_level("INFO", logger="src.rag.pipeline"):
        answer("What should I weigh?", retriever=retriever, client=RecordingClient())
    assert retriever.calls == []
    assert "document_scope=none" in caplog.text
    assert "hit_count=0" in caplog.text
    assert "type=out_of_scope" in caplog.text
    assert "reason=weight_target" in caplog.text


def test_filtered_miss_logs_that_document(caplog: pytest.LogCaptureFixture) -> None:
    retriever = RecordingRetriever(retrieval([], document_id="eatwell-guide"))
    with caplog.at_level("INFO", logger="src.rag.pipeline"):
        answer(
            "How long can raw chicken stay in the fridge?",
            "eatwell-guide",
            retriever=retriever,
            client=RecordingClient(),
        )
    assert "document_scope=eatwell-guide" in caplog.text
    assert "hit_count=0" in caplog.text
    assert "type=not_in_corpus" in caplog.text
    assert "reason=no_supporting_chunks" in caplog.text


def test_answer_logs_scope_hits_and_an_empty_drop_list(caplog: pytest.LogCaptureFixture) -> None:
    groups = [
        DocumentHits(
            "cold-food-storage",
            (hit("cold-food-storage", "cold-food-storage:24", CHICKEN),),
        )
    ]
    with caplog.at_level("INFO", logger="src.rag.pipeline"):
        body = answer(
            "How long can I keep a whole chicken?",
            retriever=RecordingRetriever(retrieval(groups)),
            client=RecordingClient(payload([("cold-food-storage", [(CHICKEN, ["cold-food-storage:24"])])])),
        )
    assert body["type"] == "answer"
    assert "document_scope=all" in caplog.text
    assert "hit_count=1" in caplog.text
    assert "dropped_claim_reasons=" in caplog.text
    assert "type=answer" in caplog.text


def test_dropped_claim_logs_the_reason_code_and_not_the_model_json(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "RAW-MODEL-JSON-MARKER"
    groups = [
        DocumentHits(
            "cold-food-storage",
            (hit("cold-food-storage", "cold-food-storage:24", CHICKEN),),
        )
    ]
    raw = json.dumps(
        {
            "trace": marker,
            "documents": [
                {
                    "document_id": "cold-food-storage",
                    "claims": [
                        {
                            "text": "Whole chicken keeps 9 days in the refrigerator.",
                            "chunk_ids": ["cold-food-storage:24"],
                        }
                    ],
                }
            ],
        }
    )
    with caplog.at_level("INFO", logger="src.rag.pipeline"):
        body = answer(
            "How long can I keep a whole chicken?",
            retriever=RecordingRetriever(retrieval(groups)),
            client=RecordingClient(raw),
        )
    assert body["type"] == "not_in_corpus"
    assert "dropped_claim_reasons=unsupported_number" in caplog.text
    assert "reason=no_surviving_claims" in caplog.text
    assert marker not in caplog.text
    assert "9 days" not in caplog.text


def test_groq_failure_is_not_an_empty_corpus() -> None:
    groups = [DocumentHits("eatwell-guide", (hit("eatwell-guide", "eatwell-guide:71", OIL),))]
    body = answer(
        "What do the documents say about cooking oil?",
        retriever=RecordingRetriever(retrieval(groups)),
        client=RecordingClient(error=GenerationError("quota")),
    )
    assert body["type"] == "generation_unavailable"
    assert body["searched"] == []
    assert "does not cover" not in str(body["message"]).casefold()
