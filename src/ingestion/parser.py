"""Parse corpus HTML and PDF files into typed guidance blocks.

A block is a heading, a paragraph, a list item, or a table. Headings update
the heading path and are not guidance by themselves. A table keeps its header
row in `header` so each data row can be repeated with those column names.

Stored fields are only what later chunking and citation need: `document_id`,
`type`, `text`, `heading_path`, and `header` on tables. Publisher, year,
source URL, and retrieval date stay on the corpus manifest.

Navigation, cookie banners, feedback widgets, contents lists, page numbers,
running headers, bibliographies, indexes, and per-product nutrition labels
are dropped. Those are not dietary or food-safety guidance.

PDF prose comes from the text layer. Tables are extracted on a second pass,
and reading-order text that only repeats an extracted table is dropped.
HTML uses one case (`h1`–`h3`, paragraphs, lists, tables). PDF uses another
(font size or outline for headings, a table extractor for grids). That split
is the parser cost: a new format needs its own case, and parent-section
context lives in the heading path rather than being copied into every block.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    import pymupdf
    from bs4 import Tag

from src.constants import BLOCKS_PATH, CORPUS, CORPUS_MANIFEST_PATH, PROJECT_ROOT

BLOCK_TYPES = frozenset({"heading", "paragraph", "list_item", "table"})

_LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2011": "-",
        "\u00a0": " ",
        "\u200b": "",
    }
)
_BULLET_CHARS = (
    "\u2022\u2023\u25a0\u25cf\u25e6\u2043\u2219\u00b7\u0099"
    "\uf0a7\uf0b7\u2610\u2611\u25aa\u25ab"
)
_BULLET_SPLIT = re.compile(rf"\s*[{re.escape(_BULLET_CHARS)}]\s+")
_NUMBERED_ITEM = re.compile(r"^(\d{1,2})[.)]\s+(\S.*)$")
_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\s+\S")
_SKIP_HEADING = re.compile(
    r"^(references|bibliography|index|corrigenda|evaluation forms?|"
    r"food safety contacts|acknowledgements|acknowledgments|further reading)$",
    re.IGNORECASE,
)
_DROP_TEXT = re.compile(
    r"^(https?://\S+|©|\(c\)|copyright\b|all rights reserved\b|isbn\b|"
    r"photography credit\b|published \d|date last reviewed\b|"
    r"fact sheet n\b.*|updated [a-z]+ \d{4}|page \d+ of \d+|"
    r"[ivx]+|\d{1,3})$",
    re.IGNORECASE,
)
_LICENCE = re.compile(r"open government licence|all rights reserved|isbn\s*\d", re.IGNORECASE)
_PROMO = re.compile(r"\b(check out foodkeeper|download .{0,40} as pdf)\b", re.IGNORECASE)
_DOT_LEADER = re.compile(r"(?:\.\s*){4,}")
_CHROME_CLASS_MARKERS = (
    "cookie-banner",
    "gem-c-feedback",
    "gem-c-contents-list",
    "gem-c-print-link",
    "print-metadata",
    "organisation-logos",
    "gem-c-devolved-nations",
    "social-media-sharing",
    "usa-banner",
    "gem-c-breadcrumbs",
    "govuk-breadcrumbs",
    "govuk-cookie",
)
_SKIP_TAGS = {"script", "style", "noscript", "svg", "form", "button", "nav", "footer"}


class ParseError(RuntimeError):
    """A corpus file could not be turned into guidance blocks."""

    def __init__(self, document_id: str, message: str) -> None:
        self.document_id = document_id
        super().__init__(f"{document_id}: {message}")


@dataclass(frozen=True)
class Block:
    """One typed span of guidance, with the heading path above it."""

    document_id: str
    type: str
    text: str
    heading_path: tuple[str, ...]
    header: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "document_id": self.document_id,
            "type": self.type,
            "text": self.text,
            "heading_path": list(self.heading_path),
        }
        if self.type == "table":
            payload["header"] = list(self.header or ())
        return payload


@dataclass(frozen=True)
class _Style:
    size: float
    bold: bool
    dingbat: bool


@dataclass
class _PdfBlock:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    style: _Style


@dataclass(frozen=True)
class _PdfTable:
    page_index: int
    y0: float
    x0: float
    x1: float
    y1: float
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    caption: str


def document_name(document_id: str) -> str:
    for document in CORPUS:
        if document.document_id == document_id:
            return document.document_name
    raise ParseError(document_id, "not in the corpus registry")


def parse_file(path: Path, document_id: str) -> list[Block]:
    """Parse one raw corpus file. `kind` comes from the file suffix."""
    if not path.is_file():
        raise ParseError(document_id, f"missing source file {path}")
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".html":
        blocks = parse_html(data, document_id=document_id)
    elif suffix == ".pdf":
        blocks = parse_pdf(data, document_id=document_id)
    else:
        raise ParseError(document_id, f"unsupported file type {suffix}")
    if not any(block.type != "heading" for block in blocks):
        raise ParseError(document_id, "no guidance text extracted")
    return blocks


def parse_html(html: bytes | str, *, document_id: str) -> list[Block]:
    """Map a guidance page to blocks. Site chrome is removed first."""
    from bs4 import BeautifulSoup

    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("main") or soup.body or soup
    _strip_chrome(root)

    title = document_name(document_id)
    path: list[str] = [title]
    blocks: list[Block] = []

    def emit(block_type: str, text: str, header: Sequence[str] | None = None) -> None:
        cleaned = _normalize(text)
        if not cleaned:
            return
        if block_type != "table" and _DROP_TEXT.match(cleaned):
            return
        if block_type == "paragraph" and (
            _PROMO.search(cleaned) or _LICENCE.search(cleaned) or len(cleaned) < 40
        ):
            return
        blocks.append(
            Block(
                document_id=document_id,
                type=block_type,
                text=cleaned if block_type != "table" else text.strip(),
                heading_path=tuple(part for part in path if part),
                header=tuple(header) if header is not None else None,
            )
        )

    def add_heading(level: int, text: str) -> None:
        cleaned = _normalize(text)
        if not cleaned or _DROP_TEXT.match(cleaned):
            return
        if _same_title(cleaned, title):
            del path[1:]
            return
        del path[max(level, 1) :]
        path.append(cleaned)
        emit("heading", cleaned)

    _walk_html(root, add_heading, emit)
    return blocks


def parse_pdf(pdf: bytes, *, document_id: str) -> list[Block]:
    """Read the text layer and a separate table pass. Grids are indexed once."""
    title = document_name(document_id)
    pages = _pdf_pages(pdf)
    repeated = _repeated_margin_text(pages)
    tables = _pdf_tables(pdf)
    path: list[str] = [title]
    skipping = False
    blocks: list[Block] = []

    def emit(block_type: str, text: str, header: Sequence[str] | None = None) -> None:
        if skipping and block_type != "heading":
            return
        cleaned = text.strip() if block_type == "table" else _normalize(text)
        if not cleaned:
            return
        if block_type == "paragraph" and _LICENCE.search(cleaned):
            return
        # Short lowercase lines are often the tail of a wrapped list item.
        if block_type == "paragraph" and len(cleaned) < 40 and not cleaned[:1].islower():
            return
        if block_type != "table" and (_DROP_TEXT.match(cleaned) or _PROMO.search(cleaned)):
            return
        if block_type != "heading" and _letterspaced(cleaned):
            return
        blocks.append(
            Block(
                document_id=document_id,
                type=block_type,
                text=cleaned,
                heading_path=tuple(part for part in path if part),
                header=tuple(header) if header is not None else None,
            )
        )

    def add_heading(level: int, text: str) -> None:
        nonlocal skipping
        cleaned = _normalize(text)
        if not cleaned or _letterspaced(cleaned) or _DROP_TEXT.match(cleaned):
            return
        if _SKIP_HEADING.match(cleaned):
            skipping = True
            return
        skipping = False
        if _same_title(cleaned, title):
            del path[1:]
            return
        del path[max(level, 1) :]
        path.append(cleaned)
        emit("heading", cleaned)

    body_size = _body_size(pages)
    for page in pages:
        if _skip_pdf_page(page):
            continue
        page_tables = [table for table in tables if table.page_index == page.index]
        flow = _reading_order(page.blocks, page.width, page_tables)
        drop_cap = ""
        for item in flow:
            if isinstance(item, _PdfTable):
                drop_cap = ""
                if skipping or _is_product_nutrition_panel(item.header, item.rows):
                    continue
                if item.caption and len(item.caption) <= 80:
                    add_heading(1, item.caption)
                emit("table", _markdown_table(item.header, item.rows), item.header)
                continue
            if _is_furniture(item, page.height, repeated, body_size):
                continue
            if _overlaps_table(item, page_tables):
                continue
            lone = _normalize(item.text)
            if re.fullmatch(r"[A-Z]", lone):
                drop_cap = lone
                continue
            for block_type, text in _classify_pdf_block(item, body_size):
                if drop_cap and text[:1].islower():
                    text = drop_cap + text
                drop_cap = ""
                if block_type == "heading":
                    add_heading(_heading_level(item.style.size, body_size), text)
                else:
                    emit(block_type, text)
    return _drop_tiny_paragraphs(_merge_heading_fragments(_merge_list_continuations(blocks)))


def parse_corpus(
    manifest_path: Path = CORPUS_MANIFEST_PATH,
    root: Path = PROJECT_ROOT,
) -> list[Block]:
    """Parse every `ok` row in the corpus manifest, in registry order."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {row["document_id"]: row for row in manifest["documents"]}
    blocks: list[Block] = []
    for document in CORPUS:
        row = rows.get(document.document_id)
        if row is None or row.get("status") != "ok" or not row.get("file"):
            raise ParseError(document.document_id, "corpus manifest row is missing or not ok")
        blocks.extend(parse_file(root / row["file"], document.document_id))
    return blocks


def write_blocks(blocks: Sequence[Block], path: Path = BLOCKS_PATH) -> None:
    """Write one JSON object per block. Tables include `header`; other blocks do not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for block in blocks:
            handle.write(json.dumps(block.to_dict(), ensure_ascii=False) + "\n")


def _is_html_tag(node: object) -> bool:
    from bs4 import Tag

    return isinstance(node, Tag)


def _strip_chrome(root: Tag) -> None:
    for tag in list(root.find_all(True)):
        if getattr(tag, "attrs", None) is None:
            continue
        if tag.name in _SKIP_TAGS or _is_chrome(tag):
            tag.decompose()


def _is_chrome(tag: Tag) -> bool:
    if tag.get("id") in {"block-social-sharing"}:
        return True
    classes = " ".join(tag.get("class") or [])
    return any(marker in classes for marker in _CHROME_CLASS_MARKERS)


def _walk_html(node: Tag, add_heading, emit) -> None:
    for child in list(node.children):
        if not _is_html_tag(child):
            continue
        name = child.name
        if name in {"h1", "h2", "h3"}:
            add_heading(int(name[1]), child.get_text(" ", strip=True))
            continue
        if name == "p":
            emit("paragraph", child.get_text(" ", strip=True))
            continue
        if name == "li":
            _emit_list_item(child, emit)
            for nested in child.find_all(["ul", "ol"], recursive=False):
                _walk_html(nested, add_heading, emit)
            continue
        if name == "table":
            parsed = _html_table(child)
            if parsed is not None:
                header, rows = parsed
                emit("table", _markdown_table(header, rows), header)
            continue
        if name in {"ul", "ol"}:
            _walk_html(child, add_heading, emit)
            continue
        _walk_html(child, add_heading, emit)


def _emit_list_item(item: Tag, emit) -> None:
    text = _own_text(item)
    if not text:
        return
    parent = item.find_parent(["ol", "ul"])
    if parent is not None and parent.name == "ol":
        siblings = parent.find_all("li", recursive=False)
        if item in siblings:
            start = int(parent.get("start") or 1)
            number = start + siblings.index(item)
            if not _NUMBERED_ITEM.match(text):
                text = f"{number}. {text}"
    emit("list_item", text)


def _own_text(item: Tag) -> str:
    parts: list[str] = []
    for child in item.children:
        if _is_html_tag(child) and child.name in {"ul", "ol", "table"}:
            continue
        if isinstance(child, str):
            parts.append(child)
        else:
            parts.append(child.get_text(" ", strip=True))
    return _normalize(" ".join(parts))


def _html_table(table: Tag) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None:
    grid = _expand_html_table(table)
    if len(grid) < 2:
        return None
    thead = table.find("thead")
    header_count = len(thead.find_all("tr")) if thead is not None else 1
    header_count = max(1, min(header_count, len(grid) - 1))
    width = max(len(row) for row in grid)
    header_rows = grid[:header_count]
    header_cells: list[str] = []
    for column in range(width):
        parts: list[str] = []
        for row in header_rows:
            if column < len(row) and row[column] and row[column] not in parts:
                parts.append(row[column])
        header_cells.append(" ".join(parts))
    while header_cells and not header_cells[-1]:
        header_cells.pop()
    if len(header_cells) < 2:
        return None
    data: list[tuple[str, ...]] = []
    for row in grid[header_count:]:
        cells = list(row[: len(header_cells)])
        cells.extend("" for _ in range(len(header_cells) - len(cells)))
        if any(cells):
            data.append(tuple(cells))
    if not data:
        return None
    return tuple(header_cells), tuple(data)


def _expand_html_table(table: Tag) -> list[list[str]]:
    """Fill rowspan and colspan so each data row stands on its own."""
    occupied: dict[tuple[int, int], str] = {}
    grid: list[list[str]] = []
    for row_index, tr in enumerate(table.find_all("tr")):
        row: list[str] = []
        column = 0
        cells = tr.find_all(["th", "td"], recursive=False)

        def take_occupied() -> None:
            nonlocal column
            while (row_index, column) in occupied:
                row.append(occupied[(row_index, column)])
                column += 1

        for cell in cells:
            take_occupied()
            text = _normalize(cell.get_text(" ", strip=True))
            rowspan = max(1, int(cell.get("rowspan") or 1))
            colspan = max(1, int(cell.get("colspan") or 1))
            for offset in range(colspan):
                row.append(text)
                for extra in range(1, rowspan):
                    occupied[(row_index + extra, column + offset)] = text
            column += colspan
        take_occupied()
        if any(row):
            grid.append(row)
    return grid


def _pdf_pages(pdf: bytes) -> list["_PdfPage"]:
    import pymupdf

    document = pymupdf.open(stream=pdf, filetype="pdf")
    pages: list[_PdfPage] = []
    try:
        for index, page in enumerate(document):
            styles = _line_styles(page)
            blocks: list[_PdfBlock] = []
            for raw in page.get_text("blocks"):
                if len(raw) < 7 or raw[6] != 0:
                    continue
                text = raw[4] or ""
                if not text.strip():
                    continue
                x0, y0, x1, y1 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
                blocks.append(
                    _PdfBlock(
                        text=text,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        style=_style_for_box(x0, y0, x1, y1, styles),
                    )
                )
            pages.append(
                _PdfPage(
                    index=index,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    blocks=blocks,
                )
            )
    finally:
        document.close()
    return pages


@dataclass
class _PdfPage:
    index: int
    width: float
    height: float
    blocks: list[_PdfBlock]

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks)


@dataclass
class _LineStyle:
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    chars: int
    bold_chars: int
    dingbat: bool


def _line_styles(page: pymupdf.Page) -> list[_LineStyle]:
    styles: list[_LineStyle] = []
    payload = page.get_text("dict")
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            chars = 0
            bold_chars = 0
            dingbat = False
            weighted = 0.0
            for span in line.get("spans", []):
                text = span.get("text") or ""
                count = len(text.strip())
                if not count:
                    continue
                font = (span.get("font") or "").lower()
                if "dingbat" in font or "wingding" in font or "symbol" in font:
                    dingbat = True
                chars += count
                weighted += float(span.get("size") or 0) * count
                if int(span.get("flags") or 0) & 16:
                    bold_chars += count
            if not chars:
                continue
            x0, y0, x1, y1 = line["bbox"]
            styles.append(
                _LineStyle(
                    x0=float(x0),
                    y0=float(y0),
                    x1=float(x1),
                    y1=float(y1),
                    size=weighted / chars,
                    chars=chars,
                    bold_chars=bold_chars,
                    dingbat=dingbat,
                )
            )
    return styles


def _style_for_box(
    x0: float, y0: float, x1: float, y1: float, lines: Sequence[_LineStyle]
) -> _Style:
    chars = 0
    bold = 0
    weighted = 0.0
    dingbat = False
    for line in lines:
        if line.y1 < y0 - 2 or line.y0 > y1 + 2:
            continue
        if line.x1 < x0 - 2 or line.x0 > x1 + 2:
            continue
        chars += line.chars
        bold += line.bold_chars
        weighted += line.size * line.chars
        dingbat = dingbat or line.dingbat
    if not chars:
        return _Style(size=11.0, bold=False, dingbat=False)
    return _Style(size=weighted / chars, bold=bold > chars * 0.55, dingbat=dingbat)


def _body_size(pages: Sequence[_PdfPage]) -> float:
    totals: dict[float, int] = {}
    for page in pages:
        for block in page.blocks:
            if block.y0 < 36 or block.y0 > page.height - 72:
                continue
            key = round(block.style.size, 1)
            totals[key] = totals.get(key, 0) + len(block.text)
    if not totals:
        return 11.0
    return max(totals, key=totals.get)


def _repeated_margin_text(pages: Sequence[_PdfPage]) -> set[str]:
    counts: dict[str, int] = {}
    for page in pages:
        seen: set[str] = set()
        for block in page.blocks:
            in_margin = block.y1 < 48 or block.y0 > page.height - 80
            if not in_margin:
                continue
            text = _normalize(block.text)
            if len(text) < 8 or text in seen:
                continue
            seen.add(text)
            counts[text] = counts.get(text, 0) + 1
    return {text for text, count in counts.items() if count >= 3}


def _skip_pdf_page(page: _PdfPage) -> bool:
    text = page.text
    words = _normalize(text).split()
    if len(words) < 40:
        return True
    if len(_DOT_LEADER.findall(text)) >= 5:
        return True
    if re.search(r"^\s*contents\s*$", text, re.IGNORECASE | re.MULTILINE):
        return True
    head = _normalize(text[:240]).lower()
    if head.startswith("corrigenda") or head.startswith("contents"):
        return True
    lowered = text.lower()
    squashed = re.sub(r"\s+", "", text[:500].lower())
    if "evaluationform" in squashed or "evaluationforms" in squashed:
        return True
    if (
        "this form evaluates" in lowered
        or "tick all appropriate" in lowered
        or "for the organizer and/or trainer" in lowered
        or "for the participants" in lowered
        or "self-reported behaviour" in lowered
    ):
        return True
    return False


def _reading_order(
    blocks: Sequence[_PdfBlock], width: float, tables: Sequence[_PdfTable]
) -> list[_PdfBlock | _PdfTable]:
    content = [block for block in blocks if not _overlaps_table(block, tables)]
    ordered = _order_columns(content, width)
    flow: list[_PdfBlock | _PdfTable] = []
    table_queue = sorted(tables, key=lambda table: table.y0)
    table_index = 0
    for block in ordered:
        while table_index < len(table_queue) and table_queue[table_index].y0 <= block.y0:
            flow.append(table_queue[table_index])
            table_index += 1
        flow.append(block)
    flow.extend(table_queue[table_index:])
    return flow


def _order_columns(blocks: Sequence[_PdfBlock], width: float) -> list[_PdfBlock]:
    if len(blocks) < 6:
        return sorted(blocks, key=lambda block: (block.y0, block.x0))
    mid = width / 2
    left: list[_PdfBlock] = []
    right: list[_PdfBlock] = []
    spanning: list[_PdfBlock] = []
    for block in blocks:
        if block.x1 <= mid + 10:
            left.append(block)
        elif block.x0 >= mid - 10:
            right.append(block)
        else:
            spanning.append(block)
    if len(left) < 3 or len(right) < 3:
        return sorted(blocks, key=lambda block: (block.y0, block.x0))
    if min(block.x0 for block in right) - max(block.x1 for block in left) < 6:
        return sorted(blocks, key=lambda block: (block.y0, block.x0))
    ordered: list[_PdfBlock] = []
    remaining_left = sorted(left, key=lambda block: block.y0)
    remaining_right = sorted(right, key=lambda block: block.y0)
    for span in sorted(spanning, key=lambda block: block.y0):
        ordered.extend(block for block in remaining_left if block.y0 < span.y0)
        ordered.extend(block for block in remaining_right if block.y0 < span.y0)
        remaining_left = [block for block in remaining_left if block.y0 >= span.y0]
        remaining_right = [block for block in remaining_right if block.y0 >= span.y0]
        ordered.append(span)
    ordered.extend(remaining_left)
    ordered.extend(remaining_right)
    return ordered


def _overlaps_table(block: _PdfBlock, tables: Sequence[_PdfTable]) -> bool:
    area = max(1.0, (block.x1 - block.x0) * (block.y1 - block.y0))
    for table in tables:
        overlap_x = min(block.x1, table.x1) - max(block.x0, table.x0)
        overlap_y = min(block.y1, table.y1) - max(block.y0, table.y0)
        if overlap_x > 0 and overlap_y > 0 and (overlap_x * overlap_y) / area >= 0.45:
            return True
    return False


def _is_furniture(
    block: _PdfBlock, page_height: float, repeated: set[str], body_size: float
) -> bool:
    text = _normalize(block.text)
    if not text:
        return True
    if re.fullmatch(r"\d{1,3}", text):
        return True
    if text.startswith("http://") or text.startswith("https://"):
        return True
    if _letterspaced(text) and len(text.split()) >= 4:
        return True
    in_margin = block.y1 < 34 or block.y0 > page_height - 78
    if text in repeated and in_margin:
        return True
    if in_margin and len(text) < 80 and block.style.size <= body_size + 0.4:
        return True
    return False


def _classify_pdf_block(block: _PdfBlock, body_size: float) -> list[tuple[str, str]]:
    raw_lines = [line.strip() for line in block.text.splitlines() if line.strip()]
    if not raw_lines:
        return []
    if len(raw_lines) == 1 and re.fullmatch(r"[A-Z]", raw_lines[0]):
        return []
    pieces = _split_bullets(raw_lines, block.style.dingbat)
    classified: list[tuple[str, str]] = []
    for is_item, text in pieces:
        cleaned = _normalize(text)
        if not cleaned or _DROP_TEXT.match(cleaned):
            continue
        kind = "list_item" if is_item else _pdf_piece_kind(cleaned, block.style, body_size)
        if kind == "ignore":
            continue
        if kind == "list_item" and len(cleaned) < 8:
            continue
        classified.append((kind, cleaned))
    return classified


def _split_bullets(lines: Sequence[str], dingbat: bool) -> list[tuple[bool, str]]:
    items: list[tuple[bool, list[str]]] = []
    current_item = False
    current: list[str] = []

    def flush() -> None:
        nonlocal current_item
        if current:
            items.append((current_item, list(current)))
            current.clear()
        current_item = False

    pending_bullet = False
    for line in lines:
        if _is_bullet_marker(line):
            pending_bullet = True
            continue
        bullet = line.strip() if pending_bullet else _bullet_line(line, dingbat)
        pending_bullet = False
        if bullet:
            flush()
            current_item = True
            current.append(bullet)
            continue
        if current and current_item and not _looks_like_new_paragraph(line):
            current.append(line)
            continue
        flush()
        current.append(line)
    flush()
    expanded: list[tuple[bool, str]] = []
    for is_item, parts in items:
        text = " ".join(parts)
        split = [part.strip() for part in _BULLET_SPLIT.split(text) if part and part.strip()]
        if len(split) > 1:
            lead, *rest = split
            if lead:
                expanded.append((False, lead))
            expanded.extend((True, part) for part in rest)
        else:
            expanded.append((is_item, text))
    return expanded


def _is_bullet_marker(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and all(character in _BULLET_CHARS for character in stripped)


def _bullet_line(line: str, dingbat: bool) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped[0] in _BULLET_CHARS:
        return stripped[1:].strip(" \t-–—") or None
    if re.match(r"^[-–—]\s+\S", stripped) and not re.match(r"^-\d", stripped):
        return re.sub(r"^[-–—]\s+", "", stripped)
    if dingbat and re.match(r"^n\s+[A-Z“\"(]", stripped):
        return re.sub(r"^n\s+", "", stripped)
    numbered = _NUMBERED_ITEM.match(stripped)
    if numbered and _normalize(stripped).endswith("."):
        return stripped
    return None


def _looks_like_new_paragraph(line: str) -> bool:
    return bool(re.match(r"^[A-Z0-9“\"]", line.strip()))


def _pdf_piece_kind(text: str, style: _Style, body_size: float) -> str:
    if not text or _letterspaced(text):
        return "ignore"
    words = text.split()
    numbered = _NUMBERED_ITEM.match(text)
    headingish = _is_heading_text(text, style, body_size)
    if numbered and (text.endswith(".") or len(words) > 14 or not headingish):
        return "list_item"
    if text[0] in _BULLET_CHARS or text.startswith("- "):
        return "list_item"
    if headingish:
        return "heading"
    if len(text) < 25 and not text[:1].islower():
        return "ignore"
    return "paragraph"


def _is_heading_text(text: str, style: _Style, body_size: float) -> bool:
    words = text.split()
    if not words or len(words) > 14 or len(text) > 120:
        return False
    if text[0].islower():
        return False
    if re.fullmatch(r"[\dIVXivx.]+", text):
        return False
    if re.fullmatch(r"\d[\d\s./%\-–]*", text):
        return False
    if re.fullmatch(r"\d{1,2}\s+[A-Z]{3}\s+\d{2,4}", text):
        return False
    if text.endswith(".") or re.search(r"\.\d+$", text):
        return False
    if re.search(r"\.\s+\S", text) and not _NUMBERED_HEADING.match(text):
        return False
    if re.match(r"^(fig\.|figure|table)\s+\d", text, re.IGNORECASE):
        return False
    larger = style.size >= body_size + 0.9
    if len(words) == 1:
        return larger and not any(character.isdigit() for character in text)
    bold_short = style.bold and style.size >= body_size - 0.7 and len(words) <= 10
    numbered = bool(_NUMBERED_HEADING.match(text)) and (larger or style.bold) and len(words) <= 12
    return larger or bold_short or numbered


def _heading_level(size: float, body_size: float) -> int:
    if size >= body_size + 4:
        return 1
    if size >= body_size + 1.3:
        return 2
    return 3


def _pdf_tables(pdf: bytes) -> list[_PdfTable]:
    import pdfplumber

    found: list[_PdfTable] = []
    with pdfplumber.open(pdf if isinstance(pdf, (str, Path)) else _as_stream(pdf)) as document:
        for index, page in enumerate(document.pages):
            for table in page.find_tables() or []:
                parsed = _accept_pdf_table(table.extract() or [], page.width, page.height, table.bbox)
                if parsed is None:
                    continue
                caption, header, rows = parsed
                x0, y0, x1, y1 = table.bbox
                found.append(
                    _PdfTable(
                        page_index=index,
                        y0=float(y0),
                        x0=float(x0),
                        x1=float(x1),
                        y1=float(y1),
                        header=header,
                        rows=rows,
                        caption=caption,
                    )
                )
    return found


def _as_stream(pdf: bytes):
    import io

    return io.BytesIO(pdf)


def _accept_pdf_table(
    raw_rows: Sequence[Sequence[str | None]],
    page_width: float,
    page_height: float,
    bbox: Sequence[float],
) -> tuple[str, tuple[str, ...], tuple[tuple[str, ...], ...]] | None:
    x0, y0, x1, y1 = bbox
    if x0 < -5 or y0 < -5 or x1 > page_width + 5 or y1 > page_height + 5:
        return None
    cleaned: list[list[str]] = []
    for row in raw_rows:
        cells = [_normalize(cell or "") for cell in row]
        if any(cells):
            cleaned.append(cells)
    if len(cleaned) < 3:
        return None
    header_index = next((i for i, row in enumerate(cleaned) if sum(bool(cell) for cell in row) >= 2), None)
    if header_index is None:
        return None
    header = list(cleaned[header_index])
    while header and not header[-1]:
        header.pop()
    if len(header) < 2 or any(len(cell) > 80 for cell in header):
        return None
    caption = " ".join(cell for row in cleaned[:header_index] for cell in row if cell)
    data: list[tuple[str, ...]] = []
    for row in cleaned[header_index + 1 :]:
        cells = list(row[: len(header)])
        cells.extend("" for _ in range(len(header) - len(cells)))
        # A label with no values is a category band, not a storage-time row.
        if sum(bool(cell) for cell in cells) >= 2:
            data.append(tuple(cells))
    if len(data) < 2:
        return None
    lengths = sorted(len(cell) for row in data for cell in row if cell)
    if not lengths or lengths[len(lengths) // 2] > 80:
        return None
    filled = sum(1 for row in data if sum(bool(cell) for cell in row) >= 2)
    if filled < 2:
        return None
    blob = " ".join(header)
    if _letterspaced(blob) or _letterspaced(" ".join(data[0])):
        return None
    return _normalize(caption), tuple(header), tuple(data)


def _is_product_nutrition_panel(
    header: Sequence[str], rows: Sequence[Sequence[str]]
) -> bool:
    """A packaged-food label is per-food composition, which this corpus does not keep."""
    blob = " ".join([*header, *(cell for row in rows for cell in row)]).lower()
    return "saturates" in blob and "kcal" in blob and len(rows) <= 8


def _markdown_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    width = len(header)

    def cell(value: str) -> str:
        return _normalize(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(value) for value in header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in rows:
        padded = list(row[:width]) + [""] * (width - len(row))
        lines.append("| " + " | ".join(cell(value) for value in padded) + " |")
    return "\n".join(lines)


def _merge_heading_fragments(blocks: Sequence[Block]) -> list[Block]:
    """Join a display heading whose lines were split, without swallowing prose."""
    merged: list[Block] = []
    for block in blocks:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.type == "heading"
            and block.type == "heading"
            and block.document_id == previous.document_id
            and block.text[:1].islower()
            and len(previous.text.split()) <= 8
            and len(block.text) <= 40
        ):
            text = f"{previous.text} {block.text}"
            merged[-1] = Block(
                document_id=previous.document_id,
                type="heading",
                text=text,
                heading_path=previous.heading_path[:-1] + (text,),
            )
            continue
        merged.append(block)
    return merged


def _drop_tiny_paragraphs(blocks: Sequence[Block]) -> list[Block]:
    return [
        block
        for block in blocks
        if block.type != "paragraph" or len(block.text) >= 40
    ]


def _merge_list_continuations(blocks: Sequence[Block]) -> list[Block]:
    merged: list[Block] = []
    for block in blocks:
        if (
            merged
            and merged[-1].type == "list_item"
            and block.type == "paragraph"
            and merged[-1].heading_path == block.heading_path
            and (block.text[:1].islower() or block.text[:1] == "(")
        ):
            previous = merged[-1]
            merged[-1] = Block(
                document_id=previous.document_id,
                type="list_item",
                text=f"{previous.text} {block.text}",
                heading_path=previous.heading_path,
            )
            continue
        merged.append(block)
    return merged


def _normalize(text: str) -> str:
    text = text.translate(_LIGATURES).replace("\u00ad", "")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _same_title(heading: str, document_name_text: str) -> bool:
    left = heading.casefold().strip(" .")
    right = document_name_text.casefold().strip(" .")
    if left == right:
        return True
    # "Healthy diet" is the title line of "Healthy diet (Fact sheet)".
    return len(left) >= 12 and (right.startswith(left) or left.startswith(right))


def _letterspaced(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+", text)
    if len(tokens) < 8:
        return False
    short = sum(1 for token in tokens if len(token) <= 2)
    return short / len(tokens) > 0.45


def main() -> None:
    blocks = parse_corpus()
    write_blocks(blocks)
    counts: dict[str, int] = {}
    for block in blocks:
        counts[block.document_id] = counts.get(block.document_id, 0) + 1
    print(f"wrote {len(blocks)} blocks to {BLOCKS_PATH}")
    for document_id, count in counts.items():
        print(f"  {document_id}: {count}")


if __name__ == "__main__":
    main()
