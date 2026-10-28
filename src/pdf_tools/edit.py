"""`pdf-tool edit`: structured in-place edits of an existing PDF (Strategy A).

Text is removed with real PDF redaction (content-stream removal, not a white
box) and rewritten with the original font when it has the needed glyphs,
otherwise with an explicitly reported substitute. All operations are applied
to an in-memory copy; nothing is written unless every operation succeeds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .common import (
    InputError,
    PdfToolError,
    atomic_output,
    ensure_writable_output,
    hex_to_floats,
    open_pdf,
    rnd,
)
from .fonts import ResolvedFont, resolve_font
from .inspect import dominant_style, infer_align, span_style
from .richtext import Piece, Placement, Style, build_pieces, describe, layout, parse_html
from .models import BBox

RAW_FLAGS = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES & ~pymupdf.TEXT_PRESERVE_LIGATURES
BASELINE_RATIO = 0.8  # first baseline sits 0.8em below the top of a CSS-like line box
Overflow = Literal["error", "shrink", "allow"]


# ----------------------------------------------------------------- schema


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int = Field(ge=1, description="1-based page number")


class _Style(BaseModel):
    font: str | None = Field(None, description="Font family override, e.g. 'Arial', 'Times New Roman'")
    font_file: str | None = Field(None, description="Path to a .ttf/.otf to use (relative to the changes file)")
    size: float | None = Field(None, gt=0)
    color: str | None = Field(None, description="#rrggbb")
    bold: bool | None = None
    italic: bool | None = None
    align: Literal["left", "center", "right", "justify"] | None = None
    line_height: float | None = Field(None, gt=0, description="Baseline-to-baseline distance in points")
    overflow: Overflow = Field("error", description="error: fail; shrink: scale text down to min_scale; allow: write anyway")
    min_scale: float = Field(0.7, gt=0.2, le=1)


class Run(BaseModel):
    """A piece of text with explicit style; unset fields follow the replaced text."""

    model_config = ConfigDict(extra="forbid")
    text: str
    bold: bool | None = None
    italic: bool | None = None
    color: str | None = None
    size: float | None = Field(None, gt=0)


class _Content(BaseModel):
    text: str | None = Field(None, description="Plain text; styles are transferred word by word from the old text")
    html: str | None = Field(None, description="Simple HTML: <b>/<strong>, <i>/<em>, <br>, <p>, <span style='color:#..'>")
    runs: list[Run] | None = Field(None, description="Explicitly styled pieces")

    def _check_content(self, single_line: bool, what: str) -> None:
        if sum(x is not None for x in (self.text, self.html, self.runs)) != 1:
            raise ValueError(f"{what} needs exactly one of 'text', 'html' or 'runs'")
        if single_line and "\n" in "".join(t for t, _ in styled_input(self)):
            raise ValueError(f"{what}: this mode writes a single line; use 'rect' for line breaks")


class ReplaceText(_Op, _Style, _Content):
    op: Literal["replace_text"]
    search: str | None = Field(None, description="Exact text to replace (within one line)")
    rect: BBox | None = Field(None, description="Replace all text in this area (paragraph mode)")
    occurrence: int | Literal["all"] = Field("all", description="1-based occurrence for search mode")
    fill: str | None = Field(None, description="Paint the cleared area with this color (default: transparent)")

    @model_validator(mode="after")
    def _check(self):
        if (self.search is None) == (self.rect is None):
            raise ValueError("replace_text needs exactly one of 'search' or 'rect'")
        if self.search is not None and not self.search.strip():
            raise ValueError("'search' must not be empty")
        self._check_content(self.search is not None, "replace_text")
        return self


class InsertText(_Op, _Style, _Content):
    op: Literal["insert_text"]
    rect: BBox | None = Field(None, description="Text box; text wraps inside it")
    point: tuple[float, float] | None = Field(None, description="Baseline start point for a single line")

    @model_validator(mode="after")
    def _check(self):
        if (self.rect is None) == (self.point is None):
            raise ValueError("insert_text needs exactly one of 'rect' or 'point'")
        self._check_content(self.point is not None, "insert_text")
        return self


def styled_input(op: _Content) -> list[tuple[str, dict]]:
    """The new content as (text, style overrides) chunks."""
    if op.runs is not None:
        return [(r.text, r.model_dump(exclude={"text"}, exclude_none=True)) for r in op.runs]
    if op.html is not None:
        return parse_html(op.html)
    return [(op.text or "", {})]


class Redact(_Op):
    op: Literal["redact"]
    search: str | None = None
    rect: BBox | None = None
    occurrence: int | Literal["all"] = "all"
    fill: str | None = Field(None, description="Box color after removal, e.g. '#000000' (default: transparent)")
    remove_images: bool = Field(False, description="Also blank image pixels under the area")
    remove_graphics: bool = Field(False, description="Also remove vector graphics fully inside the area")

    @model_validator(mode="after")
    def _check(self):
        if (self.search is None) == (self.rect is None):
            raise ValueError("redact needs exactly one of 'search' or 'rect'")
        return self


class ReplaceImage(_Op):
    op: Literal["replace_image"]
    image: str = Field(description="New image file (relative to the changes file)")
    xref: int | None = None
    asset_id: str | None = Field(None, description="Asset id from inspect/assets, e.g. 'img-12'")
    rect: BBox | None = Field(None, description="Pick the image overlapping this area most")
    keep_proportion: bool = True

    @model_validator(mode="after")
    def _check(self):
        if sum(x is not None for x in (self.xref, self.asset_id, self.rect)) != 1:
            raise ValueError("replace_image needs exactly one of 'xref', 'asset_id' or 'rect'")
        return self


class RemoveImage(_Op):
    op: Literal["remove_image"]
    xref: int | None = None
    asset_id: str | None = Field(None, description="Asset id from inspect/assets, e.g. 'img-12'")
    rect: BBox | None = Field(None, description="Pick the image overlapping this area most")

    @model_validator(mode="after")
    def _check(self):
        if sum(x is not None for x in (self.xref, self.asset_id, self.rect)) != 1:
            raise ValueError("remove_image needs exactly one of 'xref', 'asset_id' or 'rect'")
        return self


class InsertImage(_Op):
    op: Literal["insert_image"]
    rect: BBox = Field(description="Target area; the image is centred in it when keeping proportions")
    image: str = Field(description="Image file (relative to the changes file)")
    keep_proportion: bool = True


class ShiftContent(_Op):
    """Move everything at or below ``y`` by ``dy`` points. Runs after all other
    operations; coordinates always refer to the original page."""

    op: Literal["shift_content"]
    y: float = Field(gt=0, description="Content whose top is at or below this y moves")
    dy: float = Field(description="Negative moves content up (closes space), positive moves it down")

    @model_validator(mode="after")
    def _check(self):
        if abs(self.dy) < 0.01:
            raise ValueError("dy must not be 0")
        return self


Operation = Annotated[
    ReplaceText | InsertText | Redact | ReplaceImage | RemoveImage | InsertImage | ShiftContent, Field(discriminator="op")
]


class ChangeSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    font_dirs: list[str] = Field([], description="Extra directories searched for fonts")
    operations: list[Operation] = Field(min_length=1)


class OpResult(BaseModel):
    index: int
    op: str
    page: int
    status: Literal["applied", "failed"]
    message: str = ""
    font: str | None = None
    font_source: str | None = None
    scale: float | None = None
    styled: str | None = Field(None, description="Written text with **bold**/_italic_ markers (mixed styles only)")
    height_change: float | None = Field(
        None, description="rect replace: new minus old text height in pt (pair with shift_content)"
    )
    rects: list[BBox] = Field([], description="Changed areas in the output")
    reference_rects: list[BBox] = Field([], description="Matching areas in the input when they differ (rect replace)")
    other_pages: list[dict] = Field([], description="Areas changed on other pages (shared image objects)")
    verified: bool | None = None
    warnings: list[str] = []


class EditReport(BaseModel):
    source: str
    output: str | None
    ok: bool
    results: list[OpResult]
    masks: list[dict] = Field([], description="Changed areas; pass to `compare --masks`")
    shifts: list[dict] = Field([], description="Intentional content moves; `compare --masks` expects them")
    warnings: list[str] = []


class EditError(PdfToolError):
    def __init__(self, report: EditReport):
        failed = [r for r in report.results if r.status == "failed"]
        super().__init__("; ".join(f"op {r.index} ({r.op}): {r.message}" for r in failed) or "edit failed")
        self.report = report


def load_changes(path: str | Path) -> tuple[ChangeSet, Path]:
    p = Path(path)
    if not p.is_file():
        raise InputError(f"changes file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{p} is not valid JSON: {exc}") from exc
    if isinstance(data, list):
        data = {"operations": data}
    try:
        return ChangeSet.model_validate(data), p.parent
    except ValidationError as exc:
        raise InputError(f"invalid changes file {p}:\n{exc}") from exc


# ----------------------------------------------------------------- helpers


def _rgb(color: str | None, default: str = "#000000") -> tuple[float, float, float]:
    return hex_to_floats(color or default)


def _squash(s: str) -> str:
    return "".join(s.split())


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _union(rects) -> pymupdf.Rect:
    out = pymupdf.Rect()
    for r in rects:
        out |= pymupdf.Rect(r)
    return out


def _redact(page: pymupdf.Page, rects, fill: str | None = None, images: bool = False, graphics: bool = False) -> None:
    for r in rects:
        page.add_redact_annot(pymupdf.Rect(r), fill=_rgb(fill) if fill else False, cross_out=False)
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_PIXELS if images else pymupdf.PDF_REDACT_IMAGE_NONE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED if graphics else pymupdf.PDF_REDACT_LINE_ART_NONE,
        text=pymupdf.PDF_REDACT_TEXT_REMOVE,
    )


class _Hit:
    """One occurrence of a search string inside a single text line."""

    def __init__(self, block: dict, line: dict, chars: list[tuple[dict, dict]], start: int, end: int):
        self.block, self.line, self.chars, self.start, self.end = block, line, chars, start, end

    @property
    def rect(self) -> pymupdf.Rect:
        return _union(c["bbox"] for c, _ in self.chars[self.start : self.end])


def logical_lines(page: pymupdf.Page) -> list[dict]:
    """Text lines on one baseline that (nearly) touch form one visual line.

    PDF producers - and earlier edits - often split a visual line into several
    text objects. Large gaps (table columns) keep lines separate.
    """
    items = []
    for block in page.get_text("rawdict", flags=RAW_FLAGS)["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            chars = [(c, s) for s in line["spans"] for c in s["chars"]]
            if chars:
                r = pymupdf.Rect(line["bbox"])
                items.append({"block": block, "bbox": r, "rects": [r], "chars": chars,
                              "y": chars[0][0]["origin"][1], "size": chars[0][1]["size"]})  # fmt: skip
    items.sort(key=lambda it: it["y"])
    rows: list[list[dict]] = []
    for it in items:
        if rows and abs(it["y"] - rows[-1][0]["y"]) < 0.5:
            rows[-1].append(it)
        else:
            rows.append([it])
    merged = []
    for row in rows:
        row.sort(key=lambda it: it["bbox"].x0)
        cur = row[0]
        for it in row[1:]:
            gap = it["bbox"].x0 - cur["bbox"].x1
            if -0.5 <= gap < 0.6 * cur["size"]:  # wider gaps are columns or table cells
                last_c, last_s = cur["chars"][-1]
                if gap > 0.15 * cur["size"] and not last_c["c"].isspace() and not it["chars"][0][0]["c"].isspace():
                    gap_box = (cur["bbox"].x1, last_c["bbox"][1], it["bbox"].x0, last_c["bbox"][3])
                    cur["chars"].append(({"c": " ", "bbox": gap_box, "origin": (cur["bbox"].x1, last_c["origin"][1])}, last_s))
                cur["chars"] += it["chars"]
                cur["bbox"] = cur["bbox"] | it["bbox"]
                cur["rects"].append(it["bbox"])
            else:
                merged.append(cur)
                cur = it
        merged.append(cur)
    return merged


def find_line_hits(page: pymupdf.Page, needle: str) -> list[_Hit]:
    """Whitespace-insensitive search within visual lines, keeping per-character geometry."""
    target = _norm_ws(needle)
    hits = []
    for line in logical_lines(page):
        chars = line["chars"]
        norm, idx = [], []
        for k, (c, _) in enumerate(chars):
            ch = " " if c["c"].isspace() else c["c"]
            if ch == " " and (not norm or norm[-1] == " "):
                continue
            norm.append(ch)
            idx.append(k)
        text = "".join(norm)
        pos = text.find(target)
        while pos >= 0:
            hits.append(_Hit(line["block"], line, chars, idx[pos], idx[pos + len(target) - 1] + 1))
            pos = text.find(target, pos + len(target))
    return hits


def _select(hits: list, occurrence: int | str, what: str) -> list:
    if not hits:
        raise PdfToolError(f"{what} not found on page")
    if occurrence == "all":
        return hits
    if occurrence > len(hits):
        raise PdfToolError(f"occurrence {occurrence} requested but only {len(hits)} found")
    return [hits[occurrence - 1]]


class _PageGeometry:
    """Obstacles used to decide how far a rewritten line may extend."""

    def __init__(self, page: pymupdf.Page):
        self.page = page
        d = page.get_text("dict", flags=RAW_FLAGS)
        self.lines = [
            (pymupdf.Rect(ln["bbox"]), bi) for bi, b in enumerate(d["blocks"]) if b["type"] == 0 for ln in b["lines"]
        ]
        self.images = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
        self.drawings = [dr["rect"] for dr in page.get_drawings()]
        text_rects = [r for r, _ in self.lines]
        content = _union(text_rects) if text_rects else page.rect
        width = page.rect.width
        self.margin = max(0.0, min(content.x0, width - content.x1))

    def limits(self, line_rect: pymupdf.Rect, block: dict, own: list[pymupdf.Rect]) -> tuple[float, float]:
        width = self.page.rect.width
        left, right = self.margin, width - self.margin
        if len(block["lines"]) > 1:
            left, right = max(left, block["bbox"][0] - 1), min(right, block["bbox"][2] + 1)
        h = line_rect.height or 1

        def overlaps(r: pymupdf.Rect) -> bool:
            return min(r.y1, line_rect.y1) - max(r.y0, line_rect.y0) > 0.3 * h

        obstacles = [r for r, _ in self.lines if r not in own and r != line_rect] + self.images
        obstacles += [r for r in self.drawings if r.width < 3 and r.height > 0.5 * h]  # vertical rules
        for r in obstacles:
            if not overlaps(r):
                continue
            if r.x0 >= line_rect.x1 - 0.5:
                right = min(right, r.x0 - 2)
            elif r.x1 <= line_rect.x0 + 0.5:
                left = max(left, r.x1 + 2)
        for r in self.drawings:  # enclosing boxes such as table cells
            if r.width < 0.9 * width and r.contains(line_rect + (-0.5, -0.5, 0.5, 0.5)):
                left, right = max(left, r.x0 + 1), min(right, r.x1 - 1)
        return left, right


# ------------------------------------------------------------ operations


class _Ctx:
    def __init__(self, doc: pymupdf.Document, base_dir: Path, font_dirs: tuple[str, ...]):
        self.doc, self.base_dir, self.font_dirs = doc, base_dir, font_dirs
        self.inserted_fonts = False

    def path(self, p: str | None) -> str | None:
        if p is None:
            return None
        q = Path(p).expanduser()
        return str(q if q.is_absolute() else self.base_dir / q)


class _Fonts:
    """Resolve and cache one font per text style (glyph coverage checked per style)."""

    def __init__(self, ctx: _Ctx, page, op: _Style, res: OpResult):
        self.ctx, self.page, self.op, self.res = ctx, page, op, res
        self.cache: dict[Style, ResolvedFont] = {}
        self.embolden: dict[Style, float] = {}  # stroke width (x font size) for faux bold
        self.page_fonts: dict[int, str] = {}

    def prepare(self, pieces: list[Piece]) -> None:
        texts: dict[Style, str] = {}
        for text, st in pieces:
            texts[st] = texts.get(st, "") + text
        for st, text in texts.items():
            if st not in self.cache:
                rf = resolve_font(
                    self.ctx.doc, self.page, text, original=st.font, family=self.op.font,
                    font_file=self.ctx.path(self.op.font_file), bold=st.bold, italic=st.italic,
                    serif=st.serif, mono=st.monospace, font_dirs=self.ctx.font_dirs,
                )  # fmt: skip
                self.cache[st] = rf
                self.res.warnings += [w for w in rf.warnings if w not in self.res.warnings]
                heavy = st.font and any(w in st.font.lower() for w in ("black", "heavy", "ultra", "extrabold"))
                if heavy and rf.source != "original-embedded" and rf.buffer:
                    # a black/heavy face replaced by a lighter bold: thicken it (measured ink density match)
                    self.embolden[st] = 0.04
                    self.res.warnings.append(f"'{st.font}' substituted by {rf.label}, emboldened to match its weight")
        self.ctx.inserted_fonts = True  # every rewrite embeds a font program; subset on save

    def width(self, text: str, st: Style, scale: float = 1.0) -> float:
        return self.cache[st].font.text_length(text, st.size * scale)

    def summary(self, res: OpResult) -> None:
        main = self.cache[max(self.cache, key=lambda s: s.size)] if self.cache else None
        if main:
            res.font, res.font_source = main.label, main.source


def _write_placements(page, placements: list[Placement], fonts: _Fonts, scale: float) -> pymupdf.Rect:
    """Write in reading order (a new writer only when the colour changes) so the
    content stream - and therefore text extraction/ATS parsing - follows the text."""
    area = pymupdf.Rect()
    tw, color = None, None
    for pl in placements:
        if not pl.text.strip():
            continue
        bw = fonts.embolden.get(pl.style)
        if bw:  # fill + stroke text (TextWriter cannot stroke)
            if tw is not None:
                tw.write_text(page, color=_rgb(color)); tw = None
            rf = fonts.cache[pl.style]
            name = fonts.page_fonts.setdefault(id(rf), f"FB{len(fonts.page_fonts)}{page.number}")
            page.insert_font(fontname=name, fontbuffer=rf.buffer)
            rgb = _rgb(pl.style.color)
            size = pl.style.size * scale
            page.insert_text((pl.x, pl.y), pl.text, fontsize=size, fontname=name, color=rgb, fill=rgb,
                             render_mode=2, border_width=bw)
            w = rf.font.text_length(pl.text, size)
            area |= pymupdf.Rect(pl.x, pl.y - rf.font.ascender * size, pl.x + w, pl.y - rf.font.descender * size)
            continue
        if tw is None or pl.style.color != color:
            if tw is not None:
                tw.write_text(page, color=_rgb(color))
            tw, color = pymupdf.TextWriter(page.rect), pl.style.color
        rect, _ = tw.append((pl.x, pl.y), pl.text, font=fonts.cache[pl.style].font, fontsize=pl.style.size * scale)
        area |= rect
    if tw is not None:
        tw.write_text(page, color=_rgb(color))
    return area


def _style_of(span: dict) -> Style:
    st = span_style(span)
    return Style(st["font"], st["size"], st["color"], st["bold"], st["italic"], st["serif"], st["monospace"])


def _op_overrides(op: _Style) -> dict:
    return {k: v for k, v in (("size", op.size), ("color", op.color), ("bold", op.bold), ("italic", op.italic)) if v is not None}


def _pieces(op, old: list[Piece], base: Style, res: OpResult) -> list[Piece]:
    pieces = build_pieces(styled_input(op), old, base, _op_overrides(op))
    if len({(st.bold, st.italic, st.color, st.size) for t, st in pieces if t.strip()}) > 1:
        res.styled = describe(pieces)
    return pieces


def _replace_inline(ctx: _Ctx, page, op: ReplaceText, res: OpResult) -> None:
    hits = _select(find_line_hits(page, op.search), op.occurrence, f"search text {op.search!r}")
    # Process right-to-left / bottom-up so earlier hits keep valid geometry.
    hits.sort(key=lambda h: (h.line["bbox"][1], h.chars[h.start][0]["bbox"][0]), reverse=True)
    geo = _PageGeometry(page)
    page_lines = logical_lines(page)
    fonts = _Fonts(ctx, page, op, res)
    plans = []
    for h in hits:
        if any(p[0].line is h.line for p in plans):
            raise PdfToolError("several matches in one line; use occurrence or a rect-based replace")
        old: list[Piece] = []
        for c, span in h.chars[h.start : h.end]:
            st = _style_of(span)
            old = old[:-1] + [(old[-1][0] + c["c"], st)] if old and old[-1][1] == st else old + [(c["c"], st)]
        pieces = _pieces(op, old, old[0][1], res)
        fonts.prepare(pieces)
        c0 = h.chars[h.start][0]
        line_rect = pymupdf.Rect(h.line["bbox"])
        left_lim, right_lim = geo.limits(line_rect, h.block, h.line["rects"])
        x_start = c0["origin"][0]
        old_end = h.chars[h.end][0]["origin"][0] if h.end < len(h.chars) else h.chars[h.end - 1][0]["bbox"][2]
        old_w = old_end - x_start
        line_x0 = h.chars[0][0]["bbox"][0]
        line_x1 = h.chars[-1][0]["bbox"][2]
        align = op.align if op.align in ("right", "center") else "left"
        # a justified line keeps its right edge: the width change is spread over the following word gaps
        def ink_right(chars) -> float:  # trailing spaces do not count
            return max((c["bbox"][2] for c in chars if not c["c"].isspace()), default=0.0)

        tail = [c for c, _ in h.chars[h.end :]]
        while tail and tail[-1]["c"].isspace():
            tail = tail[:-1]
        trailing_spaces = sum(1 for c in tail if c["c"].isspace())
        # column neighbours: lines starting at the same x within a few line heights (blocks may be split)
        x0, size = h.chars[0][0]["bbox"][0], h.chars[0][1]["size"]
        column = [ln for ln in page_lines if abs(ln["bbox"].x0 - x0) < 2 and abs(ln["y"] - h.line["y"]) < 6 * size]
        col_right = max(ink_right([c for c, _ in ln["chars"]]) for ln in column)
        has_next = any(0 < ln["y"] - h.line["y"] < 1.8 * size for ln in column)
        justified = (
            align == "left" and trailing_spaces and len(column) >= 3 and has_next
            and abs(ink_right([c for c, _ in h.chars]) - col_right) < 1.5
        )  # fmt: skip
        # subset fonts often lack a space glyph (justified PDFs position words instead)
        space_w = max(fonts.width(" ", pieces[0][1]), 0.25 * pieces[0][1].size)
        scale = 1.0
        while True:
            new_w = sum(fonts.width(t, st, scale) for t, st in pieces)
            delta = new_w - old_w
            shift_pre = -delta if align == "right" else -delta / 2 if align == "center" else 0.0
            shift_post = 0.0 if align == "right" else delta / 2 if align == "center" else delta
            new_x0 = line_x0 + shift_pre if h.start > 0 else x_start + shift_pre
            new_x1 = line_x1 + shift_post if h.end < len(h.chars) else x_start + shift_pre + new_w
            distribute = bool(justified) and abs(delta) / trailing_spaces <= 0.6 * space_w
            if distribute:
                new_x1 = line_x1
            fits = new_x0 >= left_lim - 0.5 and new_x1 <= right_lim + 0.5
            if fits or op.overflow == "allow":
                if not fits:
                    res.warnings.append(f"line extends to x={new_x1:.1f} (limit {right_lim:.1f}); may overlap neighbours")
                break
            if op.overflow == "shrink" and scale - 0.02 >= op.min_scale - 1e-9:
                scale = round(scale - 0.02, 2)
                continue
            raise PdfToolError(
                f"replacement does not fit: line would span x={new_x0:.1f}..{new_x1:.1f}, available "
                f"{left_lim:.1f}..{right_lim:.1f} (needs {new_w:.1f}pt, original {old_w:.1f}pt). "
                "Shorten the text, use overflow 'shrink', or replace the whole paragraph with a rect."
            )
        plans.append((h, pieces, (delta, shift_pre, shift_post, distribute, trailing_spaces), scale))

    for h, pieces, (delta, shift_pre, shift_post, distribute, n_sp), scale in plans:
        moved: list[tuple[int, int, list[float]]] = []  # char ranges to rewrite, with a shift per char
        if abs(delta) > 0.05:
            if shift_pre and h.start > 0:
                moved.append((0, h.start, [shift_pre] * h.start))
            if h.end < len(h.chars) and distribute:
                shifts, passed = [], 0
                for c, _ in h.chars[h.end :]:
                    passed += c["c"].isspace()
                    shifts.append(delta * max(0.0, 1 - passed / n_sp))
                moved.append((h.end, len(h.chars), shifts))
            elif shift_post and h.end < len(h.chars):
                moved.append((h.end, len(h.chars), [shift_post] * (len(h.chars) - h.end)))
        redact_idx = [(h.start, h.end)] + [(a, b) for a, b, _ in moved]
        rects = [_union(c["bbox"] for c, _ in h.chars[a:b]) for a, b in redact_idx]
        y = h.chars[h.start][0]["origin"][1]
        _redact(page, rects, op.fill)
        x = h.chars[h.start][0]["origin"][0] + shift_pre
        placements = []
        for text, st in pieces:
            placements.append(Placement(x, y, text, st))
            x += fonts.width(text, st, scale)
        written = _write_placements(page, placements, fonts, scale)
        for a, b, shifts in moved:
            # rewrite untouched neighbours span by span (and shift by shift), moved horizontally
            k = a
            while k < b:
                span, shift = h.chars[k][1], shifts[k - a]
                j = k
                while j < b and h.chars[j][1] is span and abs(shifts[j - a] - shift) < 1e-6:
                    j += 1
                seg = "".join(c["c"] for c, _ in h.chars[k:j])
                if seg.strip():
                    st = span_style(span)
                    seg_font = resolve_font(
                        ctx.doc, page, seg, original=st["font"], bold=st["bold"], italic=st["italic"],
                        serif=st["serif"], mono=st["monospace"], font_dirs=ctx.font_dirs,
                    )  # fmt: skip
                    if seg_font.source != "original-embedded":
                        msg = f"neighbouring text {seg.strip()!r} re-set in {seg_font.label} (original font not reusable)"
                        res.warnings.append(msg)
                    # letter by letter at the original positions: keeps the producer's tracking/kerning
                    tw = pymupdf.TextWriter(page.rect)
                    for c, _ in h.chars[k:j]:
                        if not c["c"].isspace():
                            rect, _ = tw.append((c["origin"][0] + shift, c["origin"][1]), c["c"], font=seg_font.font, fontsize=st["size"])
                            written |= rect
                    tw.write_text(page, color=_rgb(st["color"]))
                k = j
        res.rects.append(rnd(_union(rects) | written))
        if scale < 1:
            res.scale = scale
            res.warnings.append(f"replacement scaled to {scale:.0%} to fit")
    fonts.summary(res)


def _replace_rect(ctx: _Ctx, page, op: ReplaceText | InsertText, res: OpResult) -> None:
    box = pymupdf.Rect(op.rect) & page.rect
    if box.is_empty:
        raise PdfToolError("rect is empty or outside the page")
    old: list[Piece] = []
    spans, line_rects, baselines = [], [], []
    for b in page.get_text("dict", flags=RAW_FLAGS)["blocks"]:
        if b["type"] != 0:
            continue
        for ln in b["lines"]:
            inside = [s for s in ln["spans"] if s["text"].strip() and pymupdf.Rect(s["bbox"]) in box + (-1, -1, 1, 1)]
            if not inside:
                continue
            if old:
                old.append((" ", old[-1][1]))  # line break inside a paragraph = space
            old += [(s["text"], _style_of(s)) for s in inside]
            spans += inside
            line_rects.append(pymupdf.Rect(ln["bbox"]))
            baselines.append(inside[0]["origin"][1])
    is_replace = isinstance(op, ReplaceText)
    if is_replace and not spans:
        res.warnings.append("no existing text found inside rect; text inserted with default style")
    if not is_replace and spans:
        res.warnings.append("rect overlaps existing text; use replace_text to replace it")
    if spans:
        dom = dominant_style(spans) or {}
        first = span_style(spans[0])
        base = Style(dom["font"], dom["size"], dom["color"], dom["bold"], dom["italic"], first["serif"], first["monospace"])
    else:
        base = Style(op.font or "Helvetica", op.size or 11.0, op.color or "#000000", bool(op.bold), bool(op.italic))
    pieces = _pieces(op, old, base, res)
    size = op.size or base.size
    gaps = sorted(b - a for a, b in zip(baselines, baselines[1:]) if b - a > 0.5)
    pitch = op.line_height or (gaps[len(gaps) // 2] if gaps else size * 1.2)
    align = op.align or (infer_align(line_rects, box) if is_replace else "left")
    if is_replace and not op.align and len(line_rects) > 1:
        res.warnings.append(f"alignment inferred as '{align}'")
    fonts = _Fonts(ctx, page, op, res)
    fonts.prepare(pieces)
    if is_replace and spans:
        _redact(page, [box], op.fill)
    # first-line and following-line starts (keeps hanging indents such as "- " bullets)
    x_first = line_rects[0].x0 if line_rects else box.x0
    x_rest = min(r.x0 for r in line_rects[1:]) if len(line_rects) > 1 else x_first
    top = (baselines[0] - BASELINE_RATIO * size - (pitch / size - 1) * size / 2) if baselines else box.y0
    first = baselines[0] if baselines else box.y0 + (pitch - size) / 2 + BASELINE_RATIO * size
    available = box.y1 - top

    def run(scale: float):
        try:
            return layout(pieces, lambda t, st: fonts.width(t, st, scale), x_first, x_rest, box.x1,
                          top + (first - top) * scale, pitch * scale, align)  # fmt: skip
        except ValueError as exc:
            raise PdfToolError(str(exc)) from exc

    scale = 1.0
    placements, n = run(scale)
    needed = n * pitch
    while n * pitch * scale > available + 0.5:
        if op.overflow == "shrink" and scale - 0.02 >= op.min_scale - 1e-9:
            scale = round(scale - 0.02, 2)
            placements, n = run(scale)
            continue
        if op.overflow == "allow":
            if top + n * pitch > page.rect.y1:
                raise PdfToolError(f"text needs {needed:.1f}pt height and would run off the page")
            res.warnings.append(f"text extends {needed - available:.1f}pt below the box; check for overlap")
            break
        if op.overflow == "shrink":
            raise PdfToolError(f"text does not fit even at {op.min_scale:.0%} scale (needs {needed:.1f}pt, box {available:.1f}pt)")
        raise PdfToolError(
            f"text needs {needed:.1f}pt height but the box has {available:.1f}pt (y {top:.1f}-{box.y1:.1f}). "
            "Shorten the text, enlarge the rect, use overflow 'shrink', or reconstruct the document."
        )
    _write_placements(page, placements, fonts, scale)
    used_h = n * pitch * scale
    if scale < 1:
        res.scale = scale
        res.warnings.append(f"text scaled to {scale:.0%} to fit")
    res.rects.append(rnd(pymupdf.Rect(min(x_first, x_rest), top, box.x1, top + used_h)))
    res.reference_rects.append(rnd(box))
    if is_replace and baselines:
        res.height_change = round(used_h - (baselines[-1] - baselines[0] + pitch), 2)
    fonts.summary(res)


def _insert_point(ctx: _Ctx, page, op: InsertText, res: OpResult) -> None:
    base = Style(op.font or "Helvetica", op.size or 11.0, op.color or "#000000", bool(op.bold), bool(op.italic))
    pieces = _pieces(op, [], base, res)
    fonts = _Fonts(ctx, page, op, res)
    fonts.prepare(pieces)
    x, y = op.point
    scale = 1.0
    while True:
        w = sum(fonts.width(t, st, scale) for t, st in pieces)
        if x + w <= page.rect.x1 or op.overflow == "allow":
            break
        if op.overflow == "shrink" and scale - 0.02 >= op.min_scale - 1e-9:
            scale = round(scale - 0.02, 2)
            continue
        raise PdfToolError(f"text is {w:.1f}pt wide and runs past the page edge at x={page.rect.x1:.1f}")
    big = max((st for _, st in pieces), key=lambda st: st.size)
    f = fonts.cache[big].font
    area = pymupdf.Rect(x, y - f.ascender * big.size * scale, x + w, y - f.descender * big.size * scale)
    if page.get_text("words", clip=area):
        res.warnings.append("inserted line overlaps existing text")
    placements = []
    for text, st in pieces:
        placements.append(Placement(x, y, text, st))
        x += fonts.width(text, st, scale)
    rect = _write_placements(page, placements, fonts, scale)
    if scale < 1:
        res.scale = scale
    res.rects.append(rnd(rect))
    fonts.summary(res)


def _redact_op(ctx: _Ctx, page, op: Redact, res: OpResult) -> None:
    if op.rect is not None:
        rects = [pymupdf.Rect(op.rect) & page.rect]
    else:
        hits = find_line_hits(page, op.search)
        rects = [h.rect for h in hits] or page.search_for(op.search)  # search_for also spans line breaks
        rects = _select(rects, op.occurrence, f"search text {op.search!r}")
    _redact(page, rects, op.fill, op.remove_images, op.remove_graphics)
    res.rects += [rnd(r) for r in rects]


def _replace_image(ctx: _Ctx, page, op: ReplaceImage, res: OpResult) -> None:
    from PIL import Image

    img_path = Path(ctx.path(op.image))
    if not img_path.is_file():
        raise PdfToolError(f"image not found: {img_path}")
    try:
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            new_ratio = im.width / im.height
    except Exception as exc:
        raise PdfToolError(f"cannot read image {img_path}: {exc}") from exc

    infos = page.get_image_info(xrefs=True)
    if op.rect is not None:
        target = pymupdf.Rect(op.rect)
        scored = [(abs(pymupdf.Rect(i["bbox"]) & target), i) for i in infos]
        scored = [s for s in scored if s[0] > 0]
        if not scored:
            raise PdfToolError("no image overlaps rect")
        info = max(scored, key=lambda s: s[0])[1]
        xref = info["xref"]
        placements = [(page, pymupdf.Rect(info["bbox"]))]
    else:
        xref = op.xref if op.xref is not None else _xref_from_asset(op.asset_id)
        placements = []
        for p in ctx.doc:
            placements += [(p, pymupdf.Rect(i["bbox"])) for i in p.get_image_info(xrefs=True) if i["xref"] == xref]
        if not any(p.number == page.number for p, _ in placements):
            raise PdfToolError(f"image xref {xref} is not placed on page {op.page}")

    if xref == 0:  # inline image: no shared object to swap
        box = placements[0][1]
        _redact(page, [box], images=True)
        page.insert_image(box, filename=str(img_path), keep_proportion=op.keep_proportion)
        res.warnings.append("inline image: removed by redaction and re-inserted")
        res.rects.append(rnd(box))
        return

    old_box = placements[0][1]
    old_ratio = old_box.width / old_box.height if old_box.height else new_ratio
    if not op.keep_proportion or abs(new_ratio / old_ratio - 1) < 0.02:
        page.replace_image(xref, filename=str(img_path))
    else:
        page.delete_image(xref)
        for p, box in placements:
            p.insert_image(box, filename=str(img_path), keep_proportion=True)
    pages = sorted({p.number + 1 for p, _ in placements})
    if len(pages) > 1 or len(placements) > 1:
        res.warnings.append(f"image object is shared; replaced at {len(placements)} placements on pages {pages}")
    res.rects += [rnd(b) for p, b in placements if p.number == page.number]
    res.other_pages += [{"page": p.number + 1, "rect": rnd(b)} for p, b in placements if p.number != page.number]
    res.message = f"replaced xref {xref}"


def _insert_image(ctx: _Ctx, page, op: InsertImage, res: OpResult) -> None:
    from PIL import Image

    path = Path(ctx.path(op.image))
    if not path.is_file():
        raise PdfToolError(f"image not found: {path}")
    with Image.open(path) as im:
        w, h = im.size
    box = pymupdf.Rect(op.rect) & page.rect
    if box.is_empty:
        raise PdfToolError("rect is empty or outside the page")
    page.insert_image(box, filename=str(path), keep_proportion=op.keep_proportion)
    if op.keep_proportion:  # the placed area actually used
        s = min(box.width / w, box.height / h)
        cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        box = pymupdf.Rect(cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2)
    res.rects.append(rnd(box))


def _remove_image(ctx: _Ctx, page, op: RemoveImage, res: OpResult) -> None:
    """Delete image placements on this page (content-stream removal, not a cover-up)."""
    infos = page.get_image_info(xrefs=True)
    if op.rect is not None:
        target = pymupdf.Rect(op.rect)
        scored = [(abs(pymupdf.Rect(i["bbox"]) & target), k) for k, i in enumerate(infos)]
        scored = [s for s in scored if s[0] > 0]
        if not scored:
            raise PdfToolError("no image overlaps rect")
        chosen = [max(scored)[1]]
    else:
        xref = op.xref if op.xref is not None else _xref_from_asset(op.asset_id)
        chosen = [k for k, i in enumerate(infos) if i["xref"] == xref]
        if not chosen:
            raise PdfToolError(f"image xref {xref} is not placed on page {op.page}")
    for k in chosen:
        box = pymupdf.Rect(infos[k]["bbox"]) & page.rect
        if any(j not in chosen and pymupdf.Rect(i["bbox"]).intersects(box) for j, i in enumerate(infos)):
            raise PdfToolError("another image overlaps this one; removing it would remove both")
        w, h = box.width, box.height
        page.add_redact_annot(box + (w / 4, h / 4, -w / 4, -h / 4), fill=False, cross_out=False)
        res.rects.append(rnd(box))
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
        text=pymupdf.PDF_REDACT_TEXT_NONE,
    )


def _shift_content(ctx: _Ctx, page, op: ShiftContent, res: OpResult) -> None:
    """Re-place the page as two clipped copies: above ``y`` unchanged, below moved by ``dy``.

    Text stays text, vectors stay vectors; links below ``y`` are moved too.
    """
    W, H = page.rect.width, page.rect.height
    y, dy = op.y, op.dy
    if y >= H:
        raise PdfToolError(f"y={y} is outside the page")
    if page.first_annot or page.first_widget:
        raise PdfToolError("page has annotations or form fields, which would not move; shift_content is not supported")
    drawings = page.get_drawings()
    backgrounds = [d for d in drawings if d.get("fill") and not d.get("color") and d["rect"].width >= 0.9 * W]
    content = [
        pymupdf.Rect(ln["bbox"])
        for b in page.get_text("dict", flags=RAW_FLAGS)["blocks"] if b["type"] == 0 for ln in b["lines"]
    ]  # fmt: skip
    content += [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    content += [d["rect"] for d in drawings if not any(d is b for b in backgrounds)]
    crossing = [r for r in content if r.y0 < y - 0.5 and r.y1 > y + 0.5]
    if crossing:
        raise PdfToolError(f"content crosses y={y} (e.g. {rnd(crossing[0], 1)}); choose a y in an empty gap")
    if dy < 0:
        blocked = [r for r in content if r.y1 > y + dy + 0.5 and r.y0 < y - 0.5]
        if blocked:
            raise PdfToolError(f"the {-dy:.1f}pt above y={y} are not empty (e.g. {rnd(blocked[0], 1)}); content would overlap")
    else:
        below = [r.y1 for r in content if r.y0 >= y - 0.5]
        if below and max(below) + dy > H:
            raise PdfToolError(f"moving down {dy:.1f}pt pushes content off the page (lowest content at y={max(below):.1f})")

    words_before = len(page.get_text("words"))
    links = page.get_links()
    src = pymupdf.open()
    src.insert_pdf(ctx.doc, from_page=page.number, to_page=page.number)
    empty = ctx.doc.get_new_xref()
    ctx.doc.update_object(empty, "<<>>")
    ctx.doc.update_stream(empty, b" ")
    page.set_contents(empty)
    for link in page.get_links():
        page.delete_link(link)
    gap = pymupdf.Rect(0, H + dy, W, H) if dy < 0 else pymupdf.Rect(0, y, W, y + dy)
    fills = [d["fill"] for d in backgrounds if d["rect"].intersects(gap)]
    if fills:  # the uncovered strip gets the page background colour
        page.draw_rect(gap, color=None, fill=fills[-1], width=0)
    top = pymupdf.Rect(0, 0, W, y + dy if dy < 0 else y)
    page.show_pdf_page(top, src, 0, clip=top)
    bottom_clip = pymupdf.Rect(0, y, W, H if dy < 0 else H - dy)
    page.show_pdf_page(bottom_clip + (0, dy, 0, dy), src, 0, clip=bottom_clip)
    src.close()
    for link in links:
        r = link["from"]
        if r.y0 >= y - 0.5:
            link = {**link, "from": r + (0, dy, 0, dy)}
        page.insert_link(link)
    try:
        page.clean_contents(sanitize=True)  # drop resources no longer referenced by the page
    except Exception:
        pass
    words_after = len(page.get_text("words"))
    if words_after != words_before:
        raise PdfToolError(f"internal check failed: {words_before} words before the shift, {words_after} after")
    res.verified = True  # word count unchanged (checked above)
    res.message = f"moved content below y={y:g} by {dy:+g}pt ({len(links)} links kept)"


def _xref_from_asset(asset_id: str) -> int:
    m = re.fullmatch(r"img-(\d+)", asset_id or "")
    if not m:
        raise PdfToolError(f"asset_id {asset_id!r} is not an xref image id like 'img-12' (inline images: use rect)")
    return int(m.group(1))


def _verify(doc: pymupdf.Document, op, res: OpResult, base: Path) -> None:
    page = doc[op.page - 1]
    if isinstance(op, (ReplaceText, InsertText)) and res.rects:
        expected = "".join(t for t, _ in styled_input(op))
        area = _union(res.rects) + (-2, -2, 2, 2)
        words = page.get_text("words", clip=pymupdf.Rect(0, area.y0, page.rect.x1, area.y1), sort=True)
        res.verified = _squash(expected) in "".join(w[4] for w in words)
        if not res.verified:
            res.warnings.append("could not read the new text back from the output")
    elif isinstance(op, Redact):
        if op.search is not None:
            res.verified = _squash(op.search) not in _squash(page.get_text("text"))
        else:
            res.verified = not page.get_text("text", clip=pymupdf.Rect(op.rect) + (1, 1, -1, -1)).strip()
    elif isinstance(op, RemoveImage):
        left = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
        res.verified = not any(abs(r & pymupdf.Rect(b)) > 0.5 * abs(pymupdf.Rect(b)) for r in left for b in res.rects)
    elif isinstance(op, (ReplaceImage, InsertImage)) and res.rects:
        res.verified = _placed_matches_file(doc, page, pymupdf.Rect(res.rects[0]), _resolve(base, op.image))
        if not res.verified:
            res.warnings.append("the image placed in the output does not match the new file pixel for pixel")


def _resolve(base: Path, p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else base / q


def _placed_matches_file(doc: pymupdf.Document, page, box: pymupdf.Rect, path: Path) -> bool:
    """Decoded pixels of the image placed at ``box`` equal the file's (JPEG decoders may differ by a level)."""
    import numpy as np
    from PIL import Image

    from .compare import _image_pixels

    def iou(i) -> float:
        r = pymupdf.Rect(i["bbox"])
        return abs(r & box) / (abs(r | box) or 1)

    best = max(page.get_image_info(xrefs=True), key=iou, default=None)  # not e.g. a page background under it
    if best is None or iou(best) < 0.9:
        return False
    if best["xref"] and doc.xref_stream_raw(best["xref"]) == path.read_bytes():
        return True  # stored as-is (e.g. JPEG pass-through)
    got = _image_pixels(doc, page, best["xref"], box).astype(np.int16)
    with Image.open(path) as im:
        want = np.asarray(im.convert("RGBA" if got.shape[2] == 4 else "RGB")).astype(np.int16)
    if got.shape != want.shape:
        return False
    if got.shape[2] == 4:  # colour under fully transparent pixels is irrelevant: compare premultiplied
        got = np.dstack([got[..., :3] * got[..., 3:] // 255, got[..., 3]])
        want = np.dstack([want[..., :3] * want[..., 3:] // 255, want[..., 3]])
    return float(np.abs(got - want).mean()) < 1.0


def _dispatch(ctx: _Ctx, page, op, res: OpResult) -> None:
    if isinstance(op, ReplaceText):
        (_replace_inline if op.search is not None else _replace_rect)(ctx, page, op, res)
    elif isinstance(op, InsertText):
        (_insert_point if op.point is not None else _replace_rect)(ctx, page, op, res)
    elif isinstance(op, Redact):
        _redact_op(ctx, page, op, res)
    elif isinstance(op, RemoveImage):
        _remove_image(ctx, page, op, res)
    elif isinstance(op, InsertImage):
        _insert_image(ctx, page, op, res)
    elif isinstance(op, ShiftContent):
        _shift_content(ctx, page, op, res)
    else:
        _replace_image(ctx, page, op, res)


def apply_edits(
    pdf: str | Path,
    changes: ChangeSet | str | Path,
    output: str | Path,
    overwrite: bool = False,
) -> EditReport:
    pdf, output = Path(pdf), Path(output)
    if isinstance(changes, ChangeSet):
        cs, base = changes, Path.cwd()
    else:
        cs, base = load_changes(changes)
    ensure_writable_output(output, [pdf], overwrite)
    doc = open_pdf(pdf)
    try:
        ctx = _Ctx(doc, base, tuple(str(Path(d).expanduser()) for d in cs.font_dirs))
        results: list[OpResult] = []
        # shifts run last so every other operation can use original-page coordinates
        order = sorted(range(len(cs.operations)), key=lambda k: isinstance(cs.operations[k], ShiftContent))
        for k in order:
            op, i = cs.operations[k], k + 1
            res = OpResult(index=i, op=op.op, page=op.page, status="applied")
            try:
                if op.page > doc.page_count:
                    raise PdfToolError(f"page {op.page} does not exist (document has {doc.page_count})")
                page = doc[op.page - 1]
                if page.rotation:
                    raise PdfToolError("editing rotated pages is not supported; use reconstruction")
                _dispatch(ctx, page, op, res)
            except PdfToolError as exc:
                res.status, res.message = "failed", str(exc)
            results.append(res)
        results.sort(key=lambda r: r.index)
        report = EditReport(source=str(pdf.resolve()), output=None, ok=all(r.status == "applied" for r in results), results=results)
        if not report.ok:
            raise EditError(report)
        if ctx.inserted_fonts:
            try:
                doc.subset_fonts()
            except Exception as exc:
                report.warnings.append(f"font subsetting failed, output is larger: {exc}")
        with atomic_output(output) as tmp:
            # deflate also losslessly compresses images stored uncompressed (pixels unchanged, bytes differ)
            doc.save(tmp, garbage=3, deflate=True)
    finally:
        doc.close()

    out_doc = open_pdf(output)
    try:
        for op, res in zip(cs.operations, report.results):
            _verify(out_doc, op, res, base)
    finally:
        out_doc.close()
    report.output = str(output.resolve())
    for r in report.results:
        refs = r.reference_rects or r.rects
        for ref, out in zip(refs, r.rects):
            mask = {"page": r.page, "rect": ref, "label": f"op {r.index} {r.op}"}
            if out != ref:
                mask["candidate_rect"] = out
            report.masks.append(mask)
        report.masks += [{**m, "label": f"op {r.index} {r.op}"} for m in r.other_pages]
    for op in cs.operations:  # areas below a shift line sit elsewhere in the output
        if isinstance(op, ShiftContent):
            report.shifts.append({"page": op.page, "y": op.y, "dy": op.dy})
            for m in report.masks:
                out = m.get("candidate_rect", m["rect"])
                if m["page"] == op.page and out[1] >= op.y - 0.5:
                    m["candidate_rect"] = rnd(pymupdf.Rect(out) + (0, op.dy, 0, op.dy))
    return report
