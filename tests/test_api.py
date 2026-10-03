"""POST /chat and GET /health with a pipeline stub. No retrieval and no Groq."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, cors_origins, index_is_ready
from src.constants import CORPUS
from src.rag.refusal import generation_unavailable, not_in_corpus, out_of_scope


def _client() -> TestClient:
    return TestClient(app)


def _searched() -> list[dict[str, object]]:
    return [
        {
            "document_id": document.document_id,
            "document_name": document.document_name,
            "publisher": document.publisher,
            "year": document.year,
        }
        for document in CORPUS
    ]


def _section(document_id: str) -> dict[str, object]:
    document = next(item for item in CORPUS if item.document_id == document_id)
    return {
        "document_id": document.document_id,
        "document_name": document.document_name,
        "publisher": document.publisher,
        "year": document.year,
        "source_url": document.source_url,
        "claims": [
            {
                "text": "A claim kept from this document.",
                "chunk_ids": [f"{document.document_id}:1"],
                "section_heading": "Guidance",
            }
        ],
    }


def test_unfiltered_answer_round_trips_seven_searched_and_surviving_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "type": "answer",
        "sections": [_section("cold-food-storage")],
        "searched": _searched(),
    }
    calls: list[tuple[str, str | None]] = []

    def fake_answer(message: str, document_id: str | None = None) -> dict[str, object]:
        calls.append((message, document_id))
        return payload

    monkeypatch.setattr("src.api.main.answer", fake_answer)
    with _client() as client:
        response = client.post(
            "/chat",
            json={"message": "How long can I keep a whole chicken in the fridge?", "document_id": None},
        )
    assert response.status_code == 200
    assert response.json() == payload
    assert response.json()["type"] == "answer"
    assert len(response.json()["searched"]) == 7
    assert [row["document_id"] for row in response.json()["searched"]] == [
        document.document_id for document in CORPUS
    ]
    assert [section["document_id"] for section in response.json()["sections"]] == ["cold-food-storage"]
    assert calls == [("How long can I keep a whole chicken in the fridge?", None)]


def test_not_in_corpus_round_trips_with_its_type(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = not_in_corpus()
    monkeypatch.setattr("src.api.main.answer", lambda message, document_id=None: payload)
    with _client() as client:
        response = client.post("/chat", json={"message": "What is the tariff on imported olive oil?"})
    assert response.status_code == 200
    assert response.json() == payload
    assert response.json()["type"] == "not_in_corpus"
    assert len(response.json()["searched"]) == 7


def test_out_of_scope_round_trips_with_its_type(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = out_of_scope("weight_target")
    monkeypatch.setattr("src.api.main.answer", lambda message, document_id=None: payload)
    with _client() as client:
        response = client.post("/chat", json={"message": "What should I weigh?"})
    assert response.status_code == 200
    assert response.json() == payload
    assert response.json()["type"] == "out_of_scope"
    assert response.json()["reason"] == "weight_target"
    assert response.json()["searched"] == []


def test_generation_unavailable_round_trips_with_its_type(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = generation_unavailable()
    monkeypatch.setattr("src.api.main.answer", lambda message, document_id=None: payload)
    with _client() as client:
        response = client.post("/chat", json={"message": "How long can I keep a whole chicken?"})
    assert response.status_code == 200
    assert response.json()["type"] == "generation_unavailable"
    assert response.json()["type"] != "not_in_corpus"
    assert response.json() == payload


def test_explicit_document_id_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str | None] = []

    def fake_answer(message: str, document_id: str | None = None) -> dict[str, object]:
        calls.append(document_id)
        return out_of_scope("nutrient_lookup")

    monkeypatch.setattr("src.api.main.answer", fake_answer)
    with _client() as client:
        response = client.post(
            "/chat",
            json={"message": "How much fruit?", "document_id": "eatwell-guide"},
        )
    assert response.status_code == 200
    assert calls == ["eatwell-guide"]


@pytest.mark.parametrize(
    "body",
    [
        {"document_id": "eatwell-guide"},
        {"message": "   ", "document_id": "eatwell-guide"},
        {"message": "", "document_id": None},
    ],
)
def test_empty_message_is_422_and_does_not_call_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, object],
) -> None:
    def fake_answer(message: str, document_id: str | None = None) -> dict[str, object]:
        raise AssertionError("pipeline should not run")

    monkeypatch.setattr("src.api.main.answer", fake_answer)
    with _client() as client:
        response = client.post("/chat", json=body)
    assert response.status_code == 422


def test_unknown_document_id_is_422_and_does_not_call_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_answer(message: str, document_id: str | None = None) -> dict[str, object]:
        raise AssertionError("pipeline should not run")

    monkeypatch.setattr("src.api.main.answer", fake_answer)
    with _client() as client:
        response = client.post(
            "/chat",
            json={"message": "How long can I keep chicken?", "document_id": "not-a-document"},
        )
    assert response.status_code == 422


def _publish(directory: Path, *, sqlite: bool, manifest: bool) -> tuple[Path, Path]:
    directory.mkdir()
    if sqlite:
        (directory / "chroma.sqlite3").write_bytes(b"")
    manifest_path = directory.parent / "index_manifest.json"
    if manifest:
        manifest_path.write_text("{}\n", encoding="utf-8")
    return directory, manifest_path


def test_health_is_ok_when_the_index_is_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_dir, manifest_path = _publish(tmp_path / "index", sqlite=True, manifest=True)
    monkeypatch.setattr("src.api.main.INDEX_DIR", index_dir)
    monkeypatch.setattr("src.api.main.INDEX_MANIFEST_PATH", manifest_path)
    with _client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("sqlite", "manifest"),
    [(False, True), (True, False), (False, False)],
)
def test_health_is_unavailable_when_the_index_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sqlite: bool,
    manifest: bool,
) -> None:
    index_dir, manifest_path = _publish(tmp_path / "index", sqlite=sqlite, manifest=manifest)
    monkeypatch.setattr("src.api.main.INDEX_DIR", index_dir)
    monkeypatch.setattr("src.api.main.INDEX_MANIFEST_PATH", manifest_path)
    with _client() as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


def test_documents_lists_the_registry_in_order() -> None:
    with _client() as client:
        response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert [row["document_id"] for row in body] == [document.document_id for document in CORPUS]
    assert [row["document_name"] for row in body] == [document.document_name for document in CORPUS]
    assert body[0]["publisher"] == CORPUS[0].publisher
    assert body[0]["year"] == CORPUS[0].year


def test_browser_origin_is_allowed() -> None:
    with _client() as client:
        response = client.get("/documents", headers={"Origin": "https://dietary.vercel.app"})
    assert response.headers["access-control-allow-origin"] == "*"


def test_chat_preflight_allows_a_browser_post() -> None:
    with _client() as client:
        response = client.options(
            "/chat",
            headers={
                "Origin": "https://dietary.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert response.status_code == 200
    assert "POST" in response.headers["access-control-allow-methods"]


def test_cors_origins_splits_a_host_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "https://a.vercel.app, https://b.vercel.app")
    assert cors_origins() == ["https://a.vercel.app", "https://b.vercel.app"]


def test_published_index_health_when_present() -> None:
    if not index_is_ready():
        pytest.skip("index is not published")
    with _client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
