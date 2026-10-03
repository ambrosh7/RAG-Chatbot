"""Claim generation. Parsing and the Groq client stay offline."""

from __future__ import annotations

import json

import httpx
import pytest
from groq import APITimeoutError, RateLimitError

from src.constants import CORPUS, GROQ_DEFAULT_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE
from src.rag.generator import (
    RULES,
    SYSTEM_PROMPT,
    GenerationError,
    GroqChatClient,
    build_user_prompt,
    generate,
    groq_model_name,
    parse_draft,
)
from src.rag.retriever import DocumentHits, Hit


def _hit(text: str) -> Hit:
    document = CORPUS[0]
    return Hit(
        chunk_id="cold-food-storage:24",
        text=text,
        similarity=0.8,
        document_id=document.document_id,
        document_name=document.document_name,
        publisher=document.publisher,
        year=document.year,
        source_url=document.source_url,
        section_heading="Poultry",
        block_type="table_row",
        retrieval_date="2026-10-03",
    )


def test_system_prompt_states_the_three_rules() -> None:
    for rule in RULES:
        assert rule in SYSTEM_PROMPT
    assert "publisher" in SYSTEM_PROMPT.casefold()
    assert "url" in SYSTEM_PROMPT.casefold()


def test_user_prompt_contains_only_the_question_and_retrieved_chunks() -> None:
    decoy = "DECOY CHUNK FROM OUTSIDE THE RETRIEVED SET"
    groups = (
        DocumentHits(
            document_id="cold-food-storage",
            hits=(_hit("Whole chicken keeps 1 to 2 days in the refrigerator."),),
        ),
    )
    prompt = build_user_prompt("How long can I keep a whole chicken in the fridge?", groups)
    assert "How long can I keep a whole chicken in the fridge?" in prompt
    assert "[cold-food-storage:24]" in prompt
    assert "1 to 2 days" in prompt
    assert decoy not in prompt
    assert decoy not in SYSTEM_PROMPT


def test_parse_ignores_publisher_and_url_fields() -> None:
    raw = json.dumps(
        {
            "documents": [
                {
                    "document_id": "cold-food-storage",
                    "publisher": "Invented publisher",
                    "url": "https://evil.example/not-the-registry",
                    "claims": [
                        {
                            "text": "Whole chicken keeps 1 to 2 days.",
                            "chunk_ids": ["cold-food-storage:24"],
                            "source_url": "https://evil.example/not-the-registry",
                        }
                    ],
                }
            ]
        }
    )
    draft = parse_draft(raw)
    assert draft[0].document_id == "cold-food-storage"
    assert draft[0].claims[0].text == "Whole chicken keeps 1 to 2 days."
    assert draft[0].claims[0].chunk_ids == ("cold-food-storage:24",)
    assert not hasattr(draft[0].claims[0], "source_url")


def test_empty_documents_array_is_valid_json() -> None:
    assert parse_draft('{"documents": []}') == ()
    assert parse_draft('{"documents": [{"document_id": "eatwell-guide", "claims": []}]}')[
        0
    ].claims == ()


def test_generate_asks_once_more_when_the_first_reply_is_not_json() -> None:
    replies = iter(
        (
            "[",
            json.dumps(
                {
                    "documents": [
                        {
                            "document_id": "cold-food-storage",
                            "claims": [
                                {
                                    "text": "Whole chicken keeps 1 to 2 days.",
                                    "chunk_ids": ["cold-food-storage:24"],
                                }
                            ],
                        }
                    ]
                }
            ),
        )
    )
    calls: list[str] = []

    class Client:
        def complete(self, *, system: str, user: str) -> str:
            calls.append(user)
            return next(replies)

    groups = (
        DocumentHits(
            document_id="cold-food-storage",
            hits=(_hit("Whole chicken keeps 1 to 2 days in the refrigerator."),),
        ),
    )
    draft = generate("How long can I keep a whole chicken in the fridge?", groups, client=Client())
    assert len(calls) == 2
    assert draft[0].claims[0].text == "Whole chicken keeps 1 to 2 days."


@pytest.mark.parametrize(
    "raw",
    [
        "The chicken keeps for 1 to 2 days.",
        '```json\n{"documents": []}\n```',
        '{"documents": [{"document_id": "eatwell-guide", "claims": [],},]}',
    ],
)
def test_strict_parse_rejects_prose_fence_and_trailing_comma(raw: str) -> None:
    with pytest.raises(GenerationError):
        parse_draft(raw)


def test_groq_client_uses_temperature_zero_and_the_env_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_MODEL", "test-model")
    seen: dict[str, object] = {}

    def create(**kwargs: object) -> object:
        seen.update(kwargs)
        return _completion('{"documents": []}')

    text = GroqChatClient(create=create).complete(system="rules", user="question")
    assert text == '{"documents": []}'
    assert seen["model"] == "test-model"
    assert seen["temperature"] == GROQ_TEMPERATURE == 0.0
    assert seen["max_tokens"] == GROQ_MAX_TOKENS
    assert seen["response_format"] == {"type": "json_object"}


def test_unset_model_uses_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert groq_model_name() == GROQ_DEFAULT_MODEL


@pytest.mark.parametrize(
    "error",
    [
        RateLimitError(
            "rate limit",
            response=httpx.Response(
                429,
                request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
            ),
            body=None,
        ),
        APITimeoutError(httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")),
    ],
)
def test_quota_and_timeout_become_generation_errors(error: Exception) -> None:
    def create(**kwargs: object) -> object:
        raise error

    with pytest.raises(GenerationError, match="groq request failed"):
        GroqChatClient(create=create).complete(system="rules", user="question")


def _completion(content: str) -> object:
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message})()
    return type("Completion", (), {"choices": [choice]})()
