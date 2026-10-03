"""Offline tests for the corpus fetcher. No test here opens a socket."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from src.constants import CORPUS
from src.ingestion.fetcher import (
    BROWSER_USER_AGENT,
    FetchError,
    build_client,
    fetch_corpus,
    find_english_pdf_url,
)

FIXTURES = Path(__file__).parent / "fixtures"
PDF_BYTES = (FIXTURES / "sample.pdf").read_bytes()
HTML_BYTES = (FIXTURES / "cold_food_storage.html").read_bytes()
LANDING_BYTES = (FIXTURES / "who_landing_page.html").read_bytes()
CHALLENGE_BYTES = (FIXTURES / "akamai_denied.html").read_bytes()
ENGLISH_PDF_URL = "https://iris.who.int/server/api/core/bitstreams/english-id/content"
FRENCH_PDF_URL = "https://apps.who.int/iris/bitstream/handle/10665/1/manual_fre.pdf"

BY_ID = {document.document_id: document for document in CORPUS}


class FakeClient:
    def __init__(self, routes: dict[str, httpx.Response | Exception]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str) -> httpx.Response:
        self.calls.append(url)
        spec = self.routes[url]
        if isinstance(spec, Exception):
            raise spec
        return spec


def response(
    url: str,
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "application/octet-stream",
    final_url: str | None = None,
) -> httpx.Response:
    return httpx.Response(
        status,
        content=body,
        headers={"content-type": content_type},
        request=httpx.Request("GET", final_url or url),
    )


def test_client_sends_a_browser_user_agent() -> None:
    with build_client() as client:
        assert client.headers["user-agent"] == BROWSER_USER_AGENT
        assert "Mozilla" in client.headers["user-agent"]


def test_direct_pdf_records_provenance(tmp_path: Path) -> None:
    document = BY_ID["who-healthy-diet"]
    final_url = "https://cdn.example.test/healthy-diet.pdf"
    client = FakeClient(
        {
            document.source_url: response(
                document.source_url,
                PDF_BYTES,
                content_type="application/pdf",
                final_url=final_url,
            )
        }
    )
    when = datetime(2026, 10, 4, 2, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    manifest = fetch_corpus([document], client, root=tmp_path, now=when)

    row = manifest["documents"][0]
    stored = tmp_path / "data" / "raw" / "who-healthy-diet" / "source.pdf"
    assert stored.read_bytes() == PDF_BYTES
    assert row["status"] == "ok"
    assert row["publisher"] == document.publisher
    assert row["year"] == 2018
    assert row["source_url"] == document.source_url
    assert row["downloaded_url"] == final_url
    assert row["retrieval_date"] == "2026-10-03"
    assert row["file"] == "data/raw/who-healthy-diet/source.pdf"
    assert row["http_status"] == 200
    assert row["content_type"] == "application/pdf"
    assert row["error"] is None
    assert row["sha256"]


def test_html_page_is_stored_without_following_links(tmp_path: Path) -> None:
    document = BY_ID["cold-food-storage"]
    client = FakeClient(
        {
            document.source_url: response(
                document.source_url,
                HTML_BYTES,
                content_type="text/html; charset=utf-8",
            )
        }
    )

    manifest = fetch_corpus([document], client, root=tmp_path)

    stored = tmp_path / "data" / "raw" / "cold-food-storage" / "source.html"
    assert stored.read_bytes() == HTML_BYTES
    assert not (tmp_path / "data" / "raw" / "cold-food-storage" / "source.pdf").exists()
    assert client.calls == [document.source_url]
    assert manifest["documents"][0]["status"] == "ok"
    assert manifest["documents"][0]["source_url"] == document.source_url


def test_landing_page_stores_the_english_pdf_and_keeps_the_citation_url(tmp_path: Path) -> None:
    document = BY_ID["fao-who-healthy-diets"]
    client = FakeClient(
        {
            document.source_url: response(
                document.source_url,
                LANDING_BYTES,
                content_type="text/html; charset=utf-8",
            ),
            ENGLISH_PDF_URL: response(
                ENGLISH_PDF_URL,
                PDF_BYTES,
                content_type="application/pdf",
            ),
        }
    )

    manifest = fetch_corpus([document], client, root=tmp_path)

    row = manifest["documents"][0]
    stored = tmp_path / "data" / "raw" / "fao-who-healthy-diets" / "source.pdf"
    assert stored.read_bytes().startswith(b"%PDF")
    assert row["source_url"] == document.source_url
    assert row["downloaded_url"] == ENGLISH_PDF_URL
    assert row["status"] == "ok"
    assert FRENCH_PDF_URL not in client.calls
    assert client.calls == [document.source_url, ENGLISH_PDF_URL]


def test_english_pdf_link_prefers_the_unlabelled_download_over_other_languages() -> None:
    chosen = find_english_pdf_url(LANDING_BYTES, "https://www.who.int/publications/i/item/9789240101876")
    assert chosen == ENGLISH_PDF_URL


def test_failed_download_keeps_the_previous_file_and_marks_the_row_stale(tmp_path: Path) -> None:
    document = BY_ID["kitchen-companion"]
    first = FakeClient(
        {document.source_url: response(document.source_url, PDF_BYTES, content_type="application/pdf")}
    )
    fetch_corpus(
        [document],
        first,
        root=tmp_path,
        now=datetime(2024, 5, 1, tzinfo=timezone.utc),
    )
    stored = tmp_path / "data" / "raw" / "kitchen-companion" / "source.pdf"
    os.utime(stored, (1_700_000_000, 1_700_000_000))
    previous_hash = json.loads((tmp_path / "data" / "corpus_manifest.json").read_text())["documents"][0]["sha256"]

    failed = FakeClient({document.source_url: response(document.source_url, b"nope", status=500)})
    with pytest.raises(FetchError) as caught:
        fetch_corpus(
            [document],
            failed,
            root=tmp_path,
            now=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )

    row = json.loads((tmp_path / "data" / "corpus_manifest.json").read_text())["documents"][0]
    assert caught.value.document_id == "kitchen-companion"
    assert stored.read_bytes() == PDF_BYTES
    assert stored.stat().st_mtime == 1_700_000_000
    assert row["status"] == "stale"
    assert row["sha256"] == previous_hash
    assert row["retrieval_date"] == "2024-05-01"
    assert "kitchen-companion" in str(caught.value)


def test_unchanged_hash_does_not_rewrite_the_file(tmp_path: Path) -> None:
    document = BY_ID["eatwell-guide"]
    client = FakeClient(
        {document.source_url: response(document.source_url, PDF_BYTES, content_type="application/pdf")}
    )
    fetch_corpus(
        [document],
        client,
        root=tmp_path,
        now=datetime(2024, 5, 1, tzinfo=timezone.utc),
    )
    stored = tmp_path / "data" / "raw" / "eatwell-guide" / "source.pdf"
    os.utime(stored, (1_700_000_000, 1_700_000_000))

    again = FakeClient(
        {document.source_url: response(document.source_url, PDF_BYTES, content_type="application/pdf")}
    )
    manifest = fetch_corpus(
        [document],
        again,
        root=tmp_path,
        now=datetime(2026, 10, 3, tzinfo=timezone.utc),
    )

    row = manifest["documents"][0]
    assert stored.stat().st_mtime == 1_700_000_000
    assert stored.read_bytes() == PDF_BYTES
    assert row["status"] == "ok"
    assert row["retrieval_date"] == "2024-05-01"


def test_first_failure_names_the_document_and_does_not_write_an_empty_file(tmp_path: Path) -> None:
    document = BY_ID["fsa-chill"]
    client = FakeClient({document.source_url: response(document.source_url, b"", status=200)})

    with pytest.raises(FetchError) as caught:
        fetch_corpus([document], client, root=tmp_path)

    stored = tmp_path / "data" / "raw" / "fsa-chill" / "source.html"
    row = json.loads((tmp_path / "data" / "corpus_manifest.json").read_text())["documents"][0]
    assert caught.value.document_id == "fsa-chill"
    assert not stored.exists()
    assert row["status"] == "error"
    assert row["file"] is None
    assert row["sha256"] is None


def test_http_403_body_is_not_saved(tmp_path: Path) -> None:
    document = BY_ID["fsa-chill"]
    client = FakeClient(
        {document.source_url: response(document.source_url, CHALLENGE_BYTES, status=403, content_type="text/html")}
    )

    with pytest.raises(FetchError) as caught:
        fetch_corpus([document], client, root=tmp_path)

    assert caught.value.document_id == "fsa-chill"
    assert not (tmp_path / "data" / "raw" / "fsa-chill" / "source.html").exists()
    written = list((tmp_path / "data" / "raw").rglob("*")) if (tmp_path / "data" / "raw").exists() else []
    assert written == []


def test_bot_challenge_with_http_200_is_not_ok(tmp_path: Path) -> None:
    document = BY_ID["cold-food-storage"]
    client = FakeClient(
        {
            document.source_url: response(
                document.source_url,
                CHALLENGE_BYTES,
                content_type="text/html",
            )
        }
    )

    with pytest.raises(FetchError):
        fetch_corpus([document], client, root=tmp_path)

    row = json.loads((tmp_path / "data" / "corpus_manifest.json").read_text())["documents"][0]
    assert row["status"] == "error"
    assert not (tmp_path / "data" / "raw" / "cold-food-storage" / "source.html").exists()


def test_landing_page_html_is_not_saved_as_the_pdf(tmp_path: Path) -> None:
    document = BY_ID["who-five-keys"]
    client = FakeClient(
        {
            document.source_url: response(document.source_url, LANDING_BYTES, content_type="text/html"),
            ENGLISH_PDF_URL: response(ENGLISH_PDF_URL, b"<html>not a pdf</html>", content_type="text/html"),
        }
    )

    with pytest.raises(FetchError) as caught:
        fetch_corpus([document], client, root=tmp_path)

    assert caught.value.document_id == "who-five-keys"
    assert not (tmp_path / "data" / "raw" / "who-five-keys" / "source.pdf").exists()
    row = json.loads((tmp_path / "data" / "corpus_manifest.json").read_text())["documents"][0]
    assert row["status"] == "error"
    assert row["source_url"] == document.source_url


def test_timeout_discards_a_partial_write_and_keeps_the_previous_file(tmp_path: Path) -> None:
    document = BY_ID["eatwell-guide"]
    fetch_corpus(
        [document],
        FakeClient({document.source_url: response(document.source_url, PDF_BYTES, content_type="application/pdf")}),
        root=tmp_path,
    )
    stored = tmp_path / "data" / "raw" / "eatwell-guide" / "source.pdf"
    partial = stored.with_name(stored.name + ".partial")
    partial.write_bytes(b"truncated")

    with pytest.raises(FetchError):
        fetch_corpus(
            [document],
            FakeClient({document.source_url: httpx.TimeoutException("timed out")}),
            root=tmp_path,
        )

    assert stored.read_bytes() == PDF_BYTES
    assert not partial.exists()
    row = json.loads((tmp_path / "data" / "corpus_manifest.json").read_text())["documents"][0]
    assert row["status"] == "stale"


def test_a_failed_document_stops_the_run_before_the_next_download(tmp_path: Path) -> None:
    first, second, third = CORPUS[0], CORPUS[1], CORPUS[2]
    routes: dict[str, httpx.Response | Exception] = {
        first.source_url: response(first.source_url, HTML_BYTES, content_type="text/html"),
        second.source_url: response(second.source_url, b"", status=200),
        third.source_url: response(third.source_url, PDF_BYTES, content_type="application/pdf"),
    }
    client = FakeClient(routes)

    with pytest.raises(FetchError) as caught:
        fetch_corpus([first, second, third], client, root=tmp_path)

    assert caught.value.document_id == second.document_id
    assert third.source_url not in client.calls
    rows = json.loads((tmp_path / "data" / "corpus_manifest.json").read_text())["documents"]
    assert [row["document_id"] for row in rows] == [first.document_id, second.document_id]
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "error"


def test_manifest_write_leaves_a_complete_json_document(tmp_path: Path) -> None:
    document = BY_ID["who-healthy-diet"]
    fetch_corpus(
        [document],
        FakeClient({document.source_url: response(document.source_url, PDF_BYTES, content_type="application/pdf")}),
        root=tmp_path,
    )

    manifest_path = tmp_path / "data" / "corpus_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["documents"][0]["document_id"] == document.document_id
    assert list((tmp_path / "data").glob("*.partial")) == []


def test_body_that_is_html_is_not_stored_as_a_pdf(tmp_path: Path) -> None:
    document = BY_ID["kitchen-companion"]
    client = FakeClient(
        {
            document.source_url: response(
                document.source_url,
                HTML_BYTES,
                content_type="application/pdf",
            )
        }
    )

    with pytest.raises(FetchError):
        fetch_corpus([document], client, root=tmp_path)

    assert not (tmp_path / "data" / "raw" / "kitchen-companion" / "source.pdf").exists()
