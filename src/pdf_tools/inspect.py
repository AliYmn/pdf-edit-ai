"""`pdf-tool inspect`: structural analysis of a PDF with PyMuPDF."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import pymupdf

from .common import floats_to_hex, int_to_hex, open_pdf, parse_pages, rnd, strip_subset
from .fonts import embedded_fonts, norm_name
from .models import (
    Document,
    DocumentMeta,
    Element,
    FontInfo,
    Line,
    Link,
    Page,
    Span,
    Table,
    Typography,
)

TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES & ~pymupdf.TEXT_PRESERVE_LIGATURES
MAX_SHAPES_PER_PAGE = 400
MIN_TEXT_CHARS = 20  # below this a non-blank page is considered to need OCR


def span_style(span: dict) -> dict:
    flags = span["flags"]
    name = strip_subset(span["font"])
    lower = name.lower()
    return {
        "font": name,
        "size": round(span["size"], 2),
        "color": int_to_hex(span["color"]),
        "bold": bool(flags & 16) or any(w in lower for w in ("bold", "black", "heavy", "semibold")),
        "italic": bool(flags & 2) or "italic" in lower or "oblique" in lower,
        "serif": bool(flags & 4),
        "monospace": bool(flags & 8),
    }


def dominant_style(spans: list[dict]) -> dict | None:
    weights: Counter = Counter()
    for s in spans:
        st = span_style(s)
        key = (st["font"], st["size"], st["color"], st["bold"], st["italic"])
        weights[key] += len(s["text"].strip()) or 0
    if not weights:
        return None
    font, size, color, bold, italic = weights.most_common(1)[0][0]
    return {"font": font, "size": size, "color": color, "bold": bold, "italic": italic}


def line_pitch(lines: list[dict]) -> float | None:
    """Median baseline distance of a block's lines (needs >= 2 lines)."""
    ys = [ln["spans"][0]["origin"][1] for ln in lines if ln["spans"]]
    gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 0.5]
    return round(median(gaps), 2) if gaps else None


def text_stats(page: pymupdf.Page) -> tuple[int, float]:
    """(usable character count, fraction of unmapped glyphs)."""
    text = page.get_text("text")
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0, 0.0
    bad = sum(1 for c in chars if c == "�")
    return len(chars) - bad, bad / len(chars)


def page_needs_ocr(page: pymupdf.Page) -> tuple[bool, int, float, list[str]]:
    """Return (needs_ocr, text_chars, image_coverage, warnings) for one page."""
    warnings: list[str] = []
    chars, bad_ratio = text_stats(page)
    coverage = image_coverage(page)
    if chars == 0:  # scanned page, or text drawn as outlines
        needs = coverage > 0 or bool(page.get_drawings())
    else:  # a few stray characters (page numbers, stamps) on top of a large image
        needs = chars < MIN_TEXT_CHARS and coverage > 0.3
    if bad_ratio > 0.3:
        needs = True
        warnings.append(f"{bad_ratio:.0%} of glyphs have no Unicode mapping; text layer unusable")
    return needs, chars, coverage, warnings


def image_coverage(page: pymupdf.Page) -> float:
    area = abs(page.rect)
    if not area:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        covered += abs(r)
    return round(min(1.0, covered / area), 3)


def page_background(page: pymupdf.Page) -> str | None:
    """Most common color along the page edges of a low-res render (inferred)."""
    pix = page.get_pixmap(dpi=36, alpha=False, colorspace=pymupdf.csRGB)
    w, h, n = pix.width, pix.height, pix.n
    if w < 2 or h < 2:
        return None
    s = pix.samples
    edge = [(x, 0) for x in range(w)] + [(x, h - 1) for x in range(w)]
    edge += [(0, y) for y in range(h)] + [(w - 1, y) for y in range(h)]
    counts = Counter(bytes(s[(y * w + x) * n : (y * w + x) * n + 3]) for x, y in edge)
    return "#" + counts.most_common(1)[0][0].hex()


def text_elements(page: pymupdf.Page, pno: int) -> list[Element]:
    data = page.get_text("dict", flags=TEXT_FLAGS, sort=True)
    out = []
    for block in data["blocks"]:
        if block["type"] != 0:
            continue
        lines, all_spans, texts = [], [], []
        for ln in block["lines"]:
            spans = [s for s in ln["spans"] if s["text"]]
            if not spans:
                continue
            all_spans += spans
            texts.append("".join(s["text"] for s in spans))
            lines.append(
                Line(
                    bbox=rnd(ln["bbox"]),
                    spans=[
                        Span(text=s["text"], bbox=rnd(s["bbox"]), origin=rnd(s["origin"]), **span_style(s))
                        for s in spans
                    ],
                )
            )
        text = "\n".join(texts)
        if not text.strip():
            continue
        style = dominant_style(all_spans)
        out.append(
            Element(
                id=f"p{pno}-t{len(out) + 1}",
                type="text",
                page=pno,
                bbox=rnd(block["bbox"]),
                provenance="extracted",
                source="pymupdf.get_text(dict)",
                text=text,
                typography=Typography(**style, line_height=line_pitch(block["lines"])) if style else None,
                lines=lines,
            )
        )
    return out


def image_elements(page: pymupdf.Page, pno: int) -> list[Element]:
    out = []
    for info in page.get_image_info(xrefs=True):
        bbox = pymupdf.Rect(info["bbox"]) & page.rect
        if bbox.is_empty:
            continue
        xref = info.get("xref", 0)
        out.append(
            Element(
                id=f"p{pno}-i{len(out) + 1}",
                type="image",
                page=pno,
                bbox=rnd(bbox),
                provenance="extracted",
                source="pymupdf.get_image_info",
                asset_id=f"img-{xref}" if xref else f"img-p{pno}-inline{info['number']}",
                image_size=(info["width"], info["height"]),
            )
        )
    return out


def table_elements(page: pymupdf.Page, pno: int) -> tuple[list[Element], list[str]]:
    try:
        tables = page.find_tables().tables
    except Exception as exc:  # table detection is best-effort
        return [], [f"table detection failed: {exc}"]
    out = []
    for t in tables:
        cells = t.extract()
        bboxes = [[rnd(c) if c else None for c in row.cells] for row in t.rows]
        out.append(
            Element(
                id=f"p{pno}-tb{len(out) + 1}",
                type="table",
                page=pno,
                bbox=rnd(t.bbox),
                provenance="inferred",
                source="pymupdf.find_tables",
                table=Table(
                    rows=t.row_count,
                    cols=t.col_count,
                    header=list(t.header.names) if t.header else [],
                    cells=cells,
                    cell_bboxes=bboxes,
                ),
            )
        )
    return out, []


def shape_elements(page: pymupdf.Page, pno: int) -> tuple[list[Element], list[str]]:
    out, warnings = [], []
    drawings = [d for d in page.get_drawings() if d.get("fill") or d.get("color")]
    if len(drawings) > MAX_SHAPES_PER_PAGE:
        warnings.append(f"{len(drawings)} vector paths; only the {MAX_SHAPES_PER_PAGE} largest are listed")
        drawings.sort(key=lambda d: abs(d["rect"]), reverse=True)
        drawings = drawings[:MAX_SHAPES_PER_PAGE]
    for d in drawings:
        r = d["rect"]
        if r.x1 < 0 or r.y1 < 0 or r.x0 > page.rect.x1 or r.y0 > page.rect.y1:
            continue
        out.append(
            Element(
                id=f"p{pno}-s{len(out) + 1}",
                type="shape",
                page=pno,
                bbox=rnd(d["rect"]),
                provenance="extracted",
                source="pymupdf.get_drawings",
                fill=floats_to_hex(d.get("fill")),
                stroke=floats_to_hex(d.get("color")),
                stroke_width=round(d["width"], 2) if d.get("color") and d.get("width") else None,
            )
        )
    return out, warnings


def infer_align(lines: list[pymupdf.Rect], box: pymupdf.Rect | None = None) -> str:
    """Rough alignment guess from line edges (inferred)."""
    if len(lines) < 2:
        return "left"
    tol = 1.5
    lefts = [r.x0 for r in lines]
    rights = [r.x1 for r in lines]
    same_left = max(lefts) - min(lefts) < tol
    same_right_body = max(rights[:-1]) - min(rights[:-1]) < tol if len(lines) > 2 else abs(rights[0] - rights[1]) < tol
    if same_left and same_right_body and len(lines) > 2:
        return "justify"
    if same_left or max(lefts[1:]) - min(lefts[1:]) < tol:  # left, or left with a hanging/first-line indent
        return "left"
    if max(rights) - min(rights) < tol:
        return "right"
    centers = [(r.x0 + r.x1) / 2 for r in lines]
    if max(centers) - min(centers) < tol * 2:
        return "center"
    return "left"


def page_links(page: pymupdf.Page) -> list[Link]:
    """Hyperlinks. Producers often add one link per word piece inside a larger link
    box; areas contained in another area with the same target are dropped."""
    raw = []
    for ln in page.get_links():
        uri = ln.get("uri") or None
        target = ln["page"] + 1 if ln.get("kind") == pymupdf.LINK_GOTO and ln.get("page", -1) >= 0 else None
        if uri is not None or target is not None:
            raw.append((pymupdf.Rect(ln["from"]), uri, target))
    out = []
    for i, (r, uri, target) in enumerate(raw):
        covered = any(
            (u, t) == (uri, target) and (o + (-0.5, -0.5, 0.5, 0.5)).contains(r) and (o != r or j < i)
            for j, (o, u, t) in enumerate(raw) if j != i
        )  # fmt: skip
        if not covered:
            out.append(Link(bbox=rnd(r), uri=uri, target_page=target))
    return out


def assign_reading_order(elements: list[Element]) -> None:
    """Top-to-bottom, left-to-right. Single-column heuristic (inferred)."""
    content = [e for e in elements if e.type != "shape"]
    content.sort(key=lambda e: (round(e.bbox[1] / 3), e.bbox[0]))
    for i, e in enumerate(content, 1):
        e.reading_order = i


def collect_fonts(doc: pymupdf.Document, pages: list[int], usage: dict) -> list[FontInfo]:
    fonts: dict[str, FontInfo] = {}
    alias: dict[str, str] = {}  # font program name (as reported by spans) -> BaseFont key
    for pno in pages:
        for xref, ext, ftype, basefont, *_ in doc[pno].get_fonts(full=True):
            name = strip_subset(basefont) or f"font-{xref}"
            fi = fonts.get(name)
            if fi is None:
                fi = fonts[name] = FontInfo(
                    name=name,
                    basefont=basefont,
                    type=ftype,
                    embedded=ext != "n/a",
                    subset=basefont != name,
                    xrefs=[],
                    pages=[],
                )
            if xref not in fi.xrefs:
                fi.xrefs.append(xref)
            if pno + 1 not in fi.pages:
                fi.pages.append(pno + 1)
        for prog, (xref, _ext, _buf) in embedded_fonts(doc, doc[pno]).items():
            owner = next((f.name for f in fonts.values() if xref in f.xrefs), None)
            if owner:
                alias.setdefault(prog, owner)
    for span_name, (sizes, chars) in usage.items():
        key = span_name if span_name in fonts else alias.get(norm_name(span_name)) or span_name
        fi = fonts.setdefault(
            key, FontInfo(name=key, basefont=key, type="unknown", embedded=False, subset=False, xrefs=[], pages=[])
        )
        fi.sizes = sorted(set(fi.sizes) | sizes)
        fi.chars += chars
        if span_name != key and span_name not in fi.span_names:
            fi.span_names.append(span_name)
    return sorted(fonts.values(), key=lambda f: -f.chars)


def inspect_pdf(path: str | Path, pages: str | None = None, shapes: bool = True, tables: bool = True) -> Document:
    doc = open_pdf(path)
    try:
        selected = parse_pages(pages, doc.page_count)
        out_pages: list[Page] = []
        usage: dict[str, tuple[set, int]] = defaultdict(lambda: (set(), 0))
        for i in selected:
            page = doc[i]
            pno = i + 1
            needs_ocr, chars, coverage, warnings = page_needs_ocr(page)
            elements = text_elements(page, pno) + image_elements(page, pno)
            if tables:
                tbls, w = table_elements(page, pno)
                elements += tbls
                warnings += w
            if shapes:
                shp, w = shape_elements(page, pno)
                elements += shp
                warnings += w
            assign_reading_order(elements)
            for e in elements:
                for ln in e.lines or []:
                    for s in ln.spans:
                        sizes, n = usage[s.font]
                        sizes.add(s.size)
                        usage[s.font] = (sizes, n + len(s.text.strip()))
            if needs_ocr:
                warnings.append("little or no text layer; use `pdf-tool extract --ocr auto` (Marker OCR)")
            out_pages.append(
                Page(
                    number=pno,
                    width=round(page.rect.width, 2),
                    height=round(page.rect.height, 2),
                    rotation=page.rotation,
                    background=page_background(page),
                    text_chars=chars,
                    image_coverage=coverage,
                    needs_ocr=needs_ocr,
                    elements=elements,
                    links=page_links(page),
                    warnings=warnings,
                )
            )
        meta = {k: v for k, v in (doc.metadata or {}).items() if v}
        pdf_format = meta.pop("format", None)
        ocr_pages = sum(p.needs_ocr for p in out_pages)
        text_layer = "full" if ocr_pages == 0 else "none" if ocr_pages == len(out_pages) else "partial"
        doc_warnings = []
        if doc.is_form_pdf:
            doc_warnings.append("document has form fields; field values are not part of the text layer analysis")
        return Document(
            meta=DocumentMeta(
                source=str(Path(path).resolve()),
                page_count=doc.page_count,
                metadata=meta,
                pdf_format=pdf_format,
                text_layer=text_layer,
            ),
            fonts=collect_fonts(doc, selected, usage),
            pages=out_pages,
            warnings=doc_warnings,
        )
    finally:
        doc.close()
