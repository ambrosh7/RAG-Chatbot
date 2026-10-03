"""Download the corpus allowlist and write a provenance manifest.

Citation links stay on the registry `source_url`. Landing pages are publication
HTML; the file stored for those rows is the English PDF linked from the page.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol, Sequence
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from src.constants import CORPUS, PROJECT_ROOT, CorpusDocument

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)

_NON_ENGLISH_LABELS = (
    "arabic",
    "chinese",
    "français",
    "french",
    "spanish",
    "español",
    "russian",
    "italiano",
    "italian",
    "japanese",
    "persian",
    "portuguese",
    "german",
    "deutsch",
    "korean",
)
_NON_ENGLISH_CODES = (
    "_ara",
    "_chi",
    "_fre",
    "_spa",
    "_rus",
    "_ita",
    "_jpn",
    "_per",
    "_por",
    "_deu",
    "_ger",
    "_kor",
    "-ara",
    "-chi",
    "-fre",
    "-spa",
    "-rus",
    "-per",
    "-por",
    "-jpn",
    "-ita",
)
_CHALLENGE_MARKERS = (
    "access denied",
    "errors.edgesuite.net",
    "request blocked",
    "pardon our interruption",
    "just a moment",
    "cf-browser-verification",
    "are you a robot",
)


class FetchError(RuntimeError):
    """A corpus download failed. `document_id` names the row that stopped the run."""

    def __init__(self, document_id: str, message: str) -> None:
        self.document_id = document_id
        super().__init__(f"{document_id}: {message}")


class HttpClient(Protocol):
    def get(self, url: str) -> httpx.Response: ...


@dataclass(frozen=True)
class _Download:
    body: bytes
    status_code: int
    content_type: str
    final_url: str


class BrowserClient:
    """HTTP client whose TLS handshake matches a browser.

    Several corpus hosts answer 403 to clients that only send a browser
    User-Agent. Chrome impersonation gets the document; a 403 body is still
    rejected by the fetcher and is not stored.
    """

    def __init__(self) -> None:
        from curl_cffi.requests import Session

        self.headers = httpx.Headers(
            {
                "User-Agent": BROWSER_USER_AGENT,
                "Accept-Language": "en",
            }
        )
        self._session = Session(impersonate="chrome")

    def __enter__(self) -> BrowserClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    def get(self, url: str) -> httpx.Response:
        try:
            downloaded = self._session.get(
                url,
                headers=dict(self.headers),
                timeout=90,
                allow_redirects=True,
            )
        except OSError:
            raise
        except Exception as exc:
            raise httpx.TransportError(str(exc)) from exc
        # curl_cffi already decompressed the body. Passing content-encoding
        # through makes httpx decode those bytes a second time.
        headers = {
            key: value
            for key, value in dict(downloaded.headers).items()
            if key.lower() not in {"content-encoding", "transfer-encoding", "content-length"}
        }
        return httpx.Response(
            downloaded.status_code,
            content=downloaded.content,
            headers=headers,
            request=httpx.Request("GET", str(downloaded.url)),
        )


def build_client() -> BrowserClient:
    """HTTP client that sends a browser User-Agent and follows redirects."""
    return BrowserClient()


def fetch_corpus(
    documents: Sequence[CorpusDocument] | None = None,
    client: HttpClient | None = None,
    *,
    root: Path | None = None,
    now: datetime | None = None,
) -> dict:
    """Download each document and write `data/corpus_manifest.json`.

    Raises FetchError at the first row that is not `ok`, after that row is
    recorded. Later documents in this run are left untouched.
    """
    documents = CORPUS if documents is None else documents
    root = PROJECT_ROOT if root is None else root
    owns_client = client is None
    if client is None:
        client = build_client()
    try:
        return _fetch_all(documents, client, root=root, now=now)
    finally:
        if owns_client:
            client.close()


def find_english_pdf_url(html: bytes, base_url: str) -> str:
    """Return the English PDF href on a publication page."""
    soup = BeautifulSoup(html, "html.parser")
    page_lang = ""
    if soup.html is not None:
        page_lang = (soup.html.get("lang") or "").casefold()

    candidates: list[tuple[int, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = urljoin(base_url, anchor["href"].strip())
        if href.startswith(("javascript:", "mailto:")):
            continue
        text = " ".join(anchor.get_text(" ", strip=True).split())
        hreflang = (anchor.get("hreflang") or "").casefold()
        if not _is_download_link(href, text):
            continue
        if _is_non_english(href, text):
            continue
        score = 0
        if _is_english_marked(href, text, hreflang):
            score += 5
        elif page_lang.startswith("en") and text.casefold().startswith("download"):
            score += 4
        if text.casefold().startswith("download"):
            score += 2
        if "bitstream" in href.casefold():
            score += 1
        candidates.append((score, href))

    if not candidates:
        raise ValueError("no English PDF link on the publication page")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _fetch_all(
    documents: Sequence[CorpusDocument],
    client: HttpClient,
    *,
    root: Path,
    now: datetime | None,
) -> dict:
    manifest_path = root / "data" / "corpus_manifest.json"
    rows = _load_rows(manifest_path)
    for document in documents:
        row = _fetch_document(document, client, root=root, now=now, previous=rows.get(document.document_id))
        rows[document.document_id] = row
        _write_manifest(manifest_path, documents, rows, now=now)
        if row["status"] != "ok":
            raise FetchError(document.document_id, row["error"] or "download failed")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _fetch_document(
    document: CorpusDocument,
    client: HttpClient,
    *,
    root: Path,
    now: datetime | None,
    previous: Mapping[str, object] | None,
) -> dict:
    destination = root / "data" / "raw" / document.document_id / f"source.{document.kind}"
    try:
        download = _download_document(document, client)
    except (httpx.HTTPError, ValueError, OSError) as exc:
        message = str(exc).strip() or exc.__class__.__name__
        return _failure_row(
            document,
            previous,
            destination,
            root=root,
            error=message,
            http_status=None,
        )

    problem = _validate_body(download.body, expected="pdf" if document.kind == "pdf" else "html")
    if problem is not None:
        return _failure_row(
            document,
            previous,
            destination,
            root=root,
            error=problem,
            http_status=download.status_code,
        )

    new_hash = _sha256(download.body)
    previous_hash = _file_sha256(destination)
    if previous_hash == new_hash:
        retrieval_date = _previous_text(previous, "retrieval_date") or _utc_date(now)
        downloaded_url = download.final_url
    else:
        _replace_file(destination, download.body)
        retrieval_date = _utc_date(now)
        downloaded_url = download.final_url

    return _row(
        document,
        status="ok",
        destination=destination,
        root=root,
        sha256=new_hash,
        http_status=download.status_code,
        content_type=download.content_type,
        downloaded_url=downloaded_url,
        retrieval_date=retrieval_date,
        error=None,
    )


def _download_document(document: CorpusDocument, client: HttpClient) -> _Download:
    page = _get(client, document.source_url)
    if document.landing_page:
        problem = _validate_body(page.body, expected="html")
        if problem is not None:
            raise ValueError(problem)
        pdf_url = find_english_pdf_url(page.body, page.final_url)
        pdf = _get(client, pdf_url)
        if _sniff(pdf.body) != "pdf":
            raise ValueError("English PDF link did not return a PDF")
        return pdf
    return page


def _get(client: HttpClient, url: str) -> _Download:
    response = client.get(url)
    content_type = response.headers.get("content-type", "")
    if response.status_code >= 400:
        raise ValueError(f"HTTP {response.status_code}")
    return _Download(
        body=response.content,
        status_code=response.status_code,
        content_type=content_type,
        final_url=str(response.url),
    )


def _validate_body(body: bytes, *, expected: str) -> str | None:
    if not body.strip():
        return "empty body"
    sniffed = _sniff(body)
    if expected == "html" and _is_challenge(body):
        return "response is a bot-challenge page"
    if sniffed != expected:
        return f"body is {sniffed}, expected {expected}"
    return None


def _sniff(body: bytes) -> str:
    stripped = body.lstrip()
    if stripped.startswith(b"%PDF"):
        return "pdf"
    head = stripped[:800].lower()
    if head.startswith((b"<!doctype html", b"<html", b"<head")) or b"<html" in head:
        return "html"
    return "unknown"


def _is_challenge(body: bytes) -> bool:
    if len(body) > 80_000:
        return False
    text = body[:12_000].decode("utf-8", errors="ignore").casefold()
    return any(marker in text for marker in _CHALLENGE_MARKERS)


def _is_download_link(href: str, text: str) -> bool:
    lowered = href.casefold()
    label = text.casefold()
    return (
        ".pdf" in lowered
        or "bitstream" in lowered
        or label.startswith("download")
        or ("iris.who.int" in lowered and lowered.rstrip("/").endswith("/content"))
    )


def _is_non_english(href: str, text: str) -> bool:
    label = text.casefold()
    lowered = href.casefold()
    if any(language in label for language in _NON_ENGLISH_LABELS):
        return True
    return any(code in lowered for code in _NON_ENGLISH_CODES)


def _is_english_marked(href: str, text: str, hreflang: str) -> bool:
    if hreflang.startswith("en"):
        return True
    blob = f"{text} {href}".casefold()
    return "english" in blob or "-eng" in blob or "_eng" in blob or "eng.pdf" in blob


def _failure_row(
    document: CorpusDocument,
    previous: Mapping[str, object] | None,
    destination: Path,
    *,
    root: Path,
    error: str,
    http_status: int | None,
) -> dict:
    _discard_partial(destination)
    if destination.is_file() and destination.stat().st_size > 0:
        return _row(
            document,
            status="stale",
            destination=destination,
            root=root,
            sha256=_file_sha256(destination),
            http_status=http_status if http_status is not None else _previous_int(previous, "http_status"),
            content_type=_previous_text(previous, "content_type"),
            downloaded_url=_previous_text(previous, "downloaded_url"),
            retrieval_date=_previous_text(previous, "retrieval_date"),
            error=error,
        )
    return _row(
        document,
        status="error",
        destination=None,
        root=None,
        sha256=None,
        http_status=http_status,
        content_type=None,
        downloaded_url=None,
        retrieval_date=None,
        error=error,
    )


def _row(
    document: CorpusDocument,
    *,
    status: str,
    destination: Path | None,
    root: Path | None,
    sha256: str | None,
    http_status: int | None,
    content_type: str | None,
    downloaded_url: str | None,
    retrieval_date: str | None,
    error: str | None,
) -> dict:
    relative = None
    if destination is not None and root is not None:
        relative = destination.relative_to(root).as_posix()
    return {
        "document_id": document.document_id,
        "document_name": document.document_name,
        "publisher": document.publisher,
        "year": document.year,
        "source_url": document.source_url,
        "downloaded_url": downloaded_url,
        "retrieval_date": retrieval_date,
        "sha256": sha256,
        "http_status": http_status,
        "content_type": content_type,
        "status": status,
        "file": relative,
        "error": error,
    }


def _replace_file(destination: Path, body: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.write_bytes(body)
    os.replace(partial, destination)


def _discard_partial(destination: Path) -> None:
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)


def _load_rows(manifest_path: Path) -> dict[str, dict]:
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    if not isinstance(documents, list):
        raise FetchError("corpus", "corpus manifest documents field is not a list")
    return {row["document_id"]: row for row in documents}


def _write_manifest(
    manifest_path: Path,
    documents: Sequence[CorpusDocument],
    rows: Mapping[str, dict],
    *,
    now: datetime | None,
) -> None:
    ordered = [rows[document.document_id] for document in documents if document.document_id in rows]
    payload = {
        "generated_at": _utc_now(now).isoformat(),
        "documents": ordered,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial = manifest_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, manifest_path)


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _utc_date(now: datetime | None) -> str:
    return _utc_now(now).date().isoformat()


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return _sha256(path.read_bytes())


def _previous_text(previous: Mapping[str, object] | None, key: str) -> str | None:
    if previous is None:
        return None
    value = previous.get(key)
    return value if isinstance(value, str) else None


def _previous_int(previous: Mapping[str, object] | None, key: str) -> int | None:
    if previous is None:
        return None
    value = previous.get(key)
    return value if isinstance(value, int) else None


def main() -> None:
    try:
        manifest = fetch_corpus()
    except FetchError as exc:
        raise SystemExit(str(exc)) from exc
    ok = sum(1 for row in manifest["documents"] if row["status"] == "ok")
    print(f"fetched {ok} documents")


if __name__ == "__main__":
    main()
