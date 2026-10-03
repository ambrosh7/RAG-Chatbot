"""Chat page: request shape, citation blocks, and the two refusal styles."""

from __future__ import annotations

import os

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from src.constants import CORPUS, PROJECT_ROOT
from src.rag.refusal import (
    GENERATION_UNAVAILABLE_MESSAGE,
    NOT_IN_CORPUS_MESSAGE,
    generation_unavailable,
    not_in_corpus,
    out_of_scope,
    searched_entries,
)
from src.ui.app import (
    ALL_DOCUMENTS,
    DISCLAIMER,
    EXAMPLE_PROMPTS,
    LOCAL_API_BASE_URL,
    apply_streamlit_secrets,
    ask,
    default_api_base_url,
    document_id_for_label,
    picker_labels,
)
from src.ui.client import ChatApiError, post_chat
from src.ui.present import present

APP = PROJECT_ROOT / "src" / "ui" / "app.py"
REGISTRY = {document.document_id: document for document in CORPUS}


def _section(document_id: str, text: str) -> dict[str, object]:
    document = REGISTRY[document_id]
    return {
        "document_id": document.document_id,
        "document_name": document.document_name,
        "publisher": document.publisher,
        "year": document.year,
        "source_url": document.source_url,
        "claims": [
            {
                "text": text,
                "chunk_ids": [f"{document_id}:1"],
                "section_heading": "Guidance",
            }
        ],
    }


def _answer(*document_ids: str) -> dict[str, object]:
    return {
        "type": "answer",
        "sections": [_section(document_id, f"Claim from {document_id}.") for document_id in document_ids],
        "searched": searched_entries(),
    }


def test_picker_labels_are_all_documents_then_the_registry_names() -> None:
    assert picker_labels() == (ALL_DOCUMENTS, *(document.document_name for document in CORPUS))
    assert document_id_for_label(ALL_DOCUMENTS) is None
    assert document_id_for_label(REGISTRY["eatwell-guide"].document_name) == "eatwell-guide"


def test_two_sections_keep_each_citation_link_in_registry_order() -> None:
    kitchen = REGISTRY["kitchen-companion"]
    eatwell = REGISTRY["eatwell-guide"]
    view = present(_answer("kitchen-companion", "eatwell-guide"))
    assert [section.document_id for section in view.sections] == ["eatwell-guide", "kitchen-companion"]
    assert view.sections[0].claims[0].source_url == eatwell.source_url
    assert view.sections[1].claims[0].source_url == kitchen.source_url
    eatwell_claim = view.sections[0].claims[0]
    assert eatwell_claim.document_name == eatwell.document_name
    assert eatwell_claim.publisher == eatwell.publisher
    assert eatwell_claim.year == eatwell.year
    assert eatwell_claim.section_heading == "Guidance"


def test_weight_target_view_has_no_searched_list() -> None:
    body = out_of_scope("weight_target")
    body["searched"] = searched_entries()
    view = present(body)
    assert view.kind == "out_of_scope"
    assert view.searched == ()
    assert view.sections == ()
    assert "qualified health professional" in view.message


def test_nutrient_lookup_uses_its_own_sentence() -> None:
    view = present(out_of_scope("nutrient_lookup"))
    assert view.kind == "out_of_scope"
    assert "nutrient numbers" in view.message
    assert "qualified health professional" not in view.message
    assert view.searched == ()


def test_not_in_corpus_lists_the_searched_documents() -> None:
    view = present(not_in_corpus())
    assert view.kind == "not_in_corpus"
    assert view.message == NOT_IN_CORPUS_MESSAGE
    assert [(row.document_name, row.publisher, row.year) for row in view.searched] == [
        (document.document_name, document.publisher, document.year) for document in CORPUS
    ]


def test_filtered_not_in_corpus_lists_only_that_document() -> None:
    view = present(not_in_corpus(["eatwell-guide"]))
    document = REGISTRY["eatwell-guide"]
    assert [(row.document_name, row.publisher, row.year) for row in view.searched] == [
        (document.document_name, document.publisher, document.year)
    ]


def test_generation_unavailable_is_not_a_corpus_miss() -> None:
    view = present(generation_unavailable())
    assert view.kind == "generation_unavailable"
    assert view.kind != "not_in_corpus"
    assert view.message == GENERATION_UNAVAILABLE_MESSAGE
    assert "does not cover" not in view.message.casefold()
    assert view.searched == ()


def test_post_chat_sends_null_for_all_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return httpx.Response(200, json=out_of_scope("weight_target"))

    monkeypatch.setattr("src.ui.client.httpx.post", fake_post)
    body = post_chat("http://127.0.0.1:8000/", "What should I weigh?", None)
    assert captured["url"] == "http://127.0.0.1:8000/chat"
    assert captured["json"] == {"message": "What should I weigh?", "document_id": None}
    assert body["type"] == "out_of_scope"


def test_post_chat_sends_the_selected_document_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        captured["json"] = json
        return httpx.Response(200, json=_answer("eatwell-guide"))

    monkeypatch.setattr("src.ui.client.httpx.post", fake_post)
    post_chat("http://127.0.0.1:8000", "How much fruit?", "eatwell-guide")
    assert captured["json"] == {"message": "How much fruit?", "document_id": "eatwell-guide"}


def test_post_chat_reports_a_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> httpx.Response:
        raise httpx.ConnectError("down")

    monkeypatch.setattr("src.ui.client.httpx.post", fake_post)
    with pytest.raises(ChatApiError, match="could not be reached"):
        post_chat("http://127.0.0.1:8000", "How much fruit?", None)


def test_blank_url_is_the_cloud_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHAT_API_BASE_URL", raising=False)
    assert default_api_base_url() == ""


def test_blank_url_calls_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_answer(message: str, document_id: str | None = None) -> dict[str, object]:
        calls.append((message, document_id))
        return out_of_scope("weight_target")

    def fail_post(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("HTTP client should not run")

    monkeypatch.setattr("src.ui.app.answer", fake_answer)
    monkeypatch.setattr("src.ui.app.post_chat", fail_post)
    monkeypatch.setattr("src.ui.app.apply_streamlit_secrets", lambda: None)
    body = ask("What should I weigh?", None, "  ")
    assert body["type"] == "out_of_scope"
    assert calls == [("What should I weigh?", None)]


def test_set_url_posts_to_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_post(
        base_url: str,
        message: str,
        document_id: str | None,
        *,
        timeout: float = 120.0,
    ) -> dict[str, object]:
        seen["call"] = (base_url, message, document_id)
        return out_of_scope("weight_target")

    def fail_answer(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("pipeline should not run")

    monkeypatch.setattr("src.ui.app.post_chat", fake_post)
    monkeypatch.setattr("src.ui.app.answer", fail_answer)
    monkeypatch.setattr("src.ui.app.apply_streamlit_secrets", lambda: None)
    body = ask("What should I weigh?", "eatwell-guide", "http://127.0.0.1:8000/")
    assert body["reason"] == "weight_target"
    assert seen["call"] == ("http://127.0.0.1:8000/", "What should I weigh?", "eatwell-guide")


def test_secrets_fill_only_missing_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "already-set")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    class _Secrets(dict[str, str]):
        def __getitem__(self, key: str) -> str:
            return dict.__getitem__(self, key)

    monkeypatch.setattr(
        "streamlit.secrets",
        _Secrets({"GROQ_API_KEY": "from-secrets", "GROQ_MODEL": "from-secrets-model"}),
    )
    apply_streamlit_secrets()
    assert os.environ["GROQ_API_KEY"] == "already-set"
    assert os.environ["GROQ_MODEL"] == "from-secrets-model"


def test_blank_page_answers_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHAT_API_BASE_URL", raising=False)

    def fail_post(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("HTTP client should not run")

    monkeypatch.setattr("src.ui.client.post_chat", fail_post)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert app.text_input(key="api_base_url").value == ""
    button = next(item for item in app.button if item.label == "What should I weigh?")
    app = button.click().run()
    assert not app.exception
    reply = app.chat_message[-1]
    assert reply.error
    assert "qualified health professional" in reply.error[0].value
    assert not reply.warning


class _Page:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, replies: dict[str, dict[str, object]]) -> None:
        self.calls: list[dict[str, object]] = []
        monkeypatch.setenv("CHAT_API_BASE_URL", LOCAL_API_BASE_URL)

        def fake_post_chat(
            base_url: str,
            message: str,
            document_id: str | None,
            *,
            timeout: float = 120.0,
        ) -> dict[str, object]:
            self.calls.append(
                {"base_url": base_url, "message": message, "document_id": document_id}
            )
            return replies[message]

        monkeypatch.setattr("src.ui.client.post_chat", fake_post_chat)
        self.app = AppTest.from_file(str(APP), default_timeout=30).run()
        assert not self.app.exception

    def click(self, label: str) -> None:
        button = next(item for item in self.app.button if item.label == label)
        self.app = button.click().run()
        assert not self.app.exception


def test_page_shows_the_disclaimer_and_example_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _Page(monkeypatch, {})
    assert any(DISCLAIMER in item.value for item in page.app.caption)
    assert [button.label for button in page.app.button] == list(EXAMPLE_PROMPTS)
    assert list(page.app.selectbox[0].options) == list(picker_labels())


def test_example_prompts_send_those_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = {
        EXAMPLE_PROMPTS[0]: _answer("kitchen-companion", "cold-food-storage"),
        EXAMPLE_PROMPTS[1]: _answer("eatwell-guide"),
        EXAMPLE_PROMPTS[2]: _answer("eatwell-guide"),
        EXAMPLE_PROMPTS[3]: out_of_scope("weight_target"),
    }
    page = _Page(monkeypatch, replies)
    for prompt in EXAMPLE_PROMPTS:
        page.click(prompt)
    assert [call["message"] for call in page.calls] == list(EXAMPLE_PROMPTS)
    assert {call["document_id"] for call in page.calls} == {None}


def test_two_section_answer_shows_each_source_url(monkeypatch: pytest.MonkeyPatch) -> None:
    eatwell = REGISTRY["eatwell-guide"]
    kitchen = REGISTRY["kitchen-companion"]
    page = _Page(
        monkeypatch,
        {EXAMPLE_PROMPTS[1]: _answer("kitchen-companion", "eatwell-guide")},
    )
    page.click(EXAMPLE_PROMPTS[1])
    reply = page.app.chat_message[-1]
    blocks = [
        "\n".join(item.value for item in block.markdown)
        for block in reply.container
    ]
    assert len(blocks) == 2
    assert eatwell.source_url in blocks[0]
    assert kitchen.source_url not in blocks[0]
    assert kitchen.source_url in blocks[1]
    assert eatwell.source_url not in blocks[1]
    assert eatwell.document_name in blocks[0]
    assert str(eatwell.year) in blocks[0]
    assert eatwell.publisher in blocks[0]


def test_weight_prompt_is_out_of_scope_without_a_document_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _Page(monkeypatch, {EXAMPLE_PROMPTS[3]: out_of_scope("weight_target")})
    page.click(EXAMPLE_PROMPTS[3])
    reply = page.app.chat_message[-1]
    assert reply.error
    assert "qualified health professional" in reply.error[0].value
    assert not reply.warning
    text = "\n".join(item.value for item in reply.markdown)
    assert "Outside this assistant" in text
    assert "Guidance searched" not in text
    assert "Cold Food Storage Chart" not in text
    assert page.calls[0]["document_id"] is None


def test_not_in_corpus_lists_documents_and_differs_from_out_of_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "What is the tariff on imported olive oil?"
    page = _Page(monkeypatch, {question: not_in_corpus()})
    page.app.chat_input[0].set_value(question).run()
    assert not page.app.exception
    reply = page.app.chat_message[-1]
    assert reply.warning
    assert NOT_IN_CORPUS_MESSAGE in reply.warning[0].value
    assert not reply.error
    text = "\n".join(item.value for item in reply.markdown)
    assert "Not covered by the guidance" in text
    assert "Guidance searched" in text
    assert "Outside this assistant" not in text
    for document in CORPUS:
        assert document.document_name in text
        assert document.publisher in text
        assert str(document.year) in text
    assert page.calls == [
        {"base_url": "http://127.0.0.1:8000", "message": question, "document_id": None}
    ]


def test_document_picker_sends_that_document_id(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "How much of the diet should be fruit and vegetables?"
    page = _Page(monkeypatch, {question: _answer("eatwell-guide")})
    page.app.selectbox[0].select(REGISTRY["eatwell-guide"].document_name).run()
    assert not page.app.exception
    page.app.chat_input[0].set_value(question).run()
    assert not page.app.exception
    assert page.calls == [
        {
            "base_url": "http://127.0.0.1:8000",
            "message": question,
            "document_id": "eatwell-guide",
        }
    ]
    reply = page.app.chat_message[-1]
    text = "\n".join(item.value for item in reply.markdown)
    assert REGISTRY["eatwell-guide"].source_url in text
    assert REGISTRY["kitchen-companion"].document_name not in text


def test_generation_unavailable_is_its_own_state(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "How long can I keep cooked leftovers in the fridge?"
    page = _Page(monkeypatch, {question: generation_unavailable()})
    page.app.chat_input[0].set_value(question).run()
    assert not page.app.exception
    reply = page.app.chat_message[-1]
    assert reply.info
    assert GENERATION_UNAVAILABLE_MESSAGE in reply.info[0].value
    assert not reply.warning
    assert not reply.error
    text = "\n".join(item.value for item in reply.markdown)
    assert "Answer unavailable" in text
    assert "Not covered by the guidance" not in text
    assert "Guidance searched" not in text
