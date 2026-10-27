"""`pdf-tool edit`: structured in-place edits of an existing PDF (Strategy A).

Text is removed with real PDF redaction (content-stream removal, not a white
box) and rewritten with the original font when it has the needed glyphs,
otherwise with an explicitly reported substitute. All operations are applied
to an in-memory copy; nothing is written unless every operation succeeds.
"""

from __future__ import annotations

import html as htmllib
import json
import re
import tempfile
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
from .inspect import dominant_style, span_style
from .models import BBox

RAW_FLAGS = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES & ~pymupdf.TEXT_PRESERVE_LIGATURES
BASELINE_RATIO = 0.8  # MuPDF's HTML layout puts the first baseline 0.8em below the line box top
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


class ReplaceText(_Op, _Style):
    op: Literal["replace_text"]
    search: str | None = Field(None, description="Exact text to replace (within one line)")
    rect: BBox | None = Field(None, description="Replace all text in this area (paragraph mode)")
    occurrence: int | Literal["all"] = Field("all", description="1-based occurrence for search mode")
    text: str | None = None
    html: str | None = Field(None, description="Simple HTML instead of text (rect mode only)")
    fill: str | None = Field(None, description="Paint the cleared area with this color (default: transparent)")

    @model_validator(mode="after")
    def _check(self):
        if (self.search is None) == (self.rect is None):
            raise ValueError("replace_text needs exactly one of 'search' or 'rect'")
        if (self.text is None) == (self.html is None):
            raise ValueError("replace_text needs exactly one of 'text' or 'html'")
        if self.html is not None and self.search is not None:
            raise ValueError("'html' is only supported with 'rect'")
        if self.search is not None and not self.search.strip():
            raise ValueError("'search' must not be empty")
        return self


class InsertText(_Op, _Style):
    op: Literal["insert_text"]
    rect: BBox | None = Field(None, description="Text box; text wraps inside it")
    point: tuple[float, float] | None = Field(None, description="Baseline start point for a single line")
    text: str | None = None
    html: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if (self.rect is None) == (self.point is None):
            raise ValueError("insert_text needs exactly one of 'rect' or 'point'")
        if (self.text is None) == (self.html is None):
            raise ValueError("insert_text needs exactly one of 'text' or 'html'")
        if self.point is not None and (self.html is not None or "\n" in (self.text or "")):
            raise ValueError("'point' inserts a single line of plain text; use 'rect' for multi-line or html")
        return self


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
    ReplaceText | InsertText | Redact | ReplaceImage | RemoveImage | ShiftContent, Field(discriminator="op")
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
    height_change: float | None = Field(
        None, description="rect replace: new minus old text height in pt (pair with shift_content)"
    )
    rects: list[BBox] = Field([], description="Changed areas in the output")
    reference_rects: list[BBox] = Field([], description="Matching areas in the input when they differ (rect replace)")
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
            if -0.5 <= gap < 1.5 * cur["size"]:
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


def _write(page: pymupdf.Page, x: float, y: float, text: str, rf: ResolvedFont, size: float, color) -> pymupdf.Rect:
    tw = pymupdf.TextWriter(page.rect)
    rect, _ = tw.append((x, y), text, font=rf.font, fontsize=size)
    tw.write_text(page, color=color)
    return rect


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


def _font_for(ctx: _Ctx, page, text: str, op: _Style, style: dict | None) -> ResolvedFont:
    style = style or {}
    rf = resolve_font(
        ctx.doc,
        page,
        text,
        original=style.get("font"),
        family=op.font,
        font_file=ctx.path(op.font_file),
        bold=op.bold if op.bold is not None else style.get("bold"),
        italic=op.italic if op.italic is not None else style.get("italic"),
        serif=style.get("serif", False),
        mono=style.get("monospace", False),
        font_dirs=ctx.font_dirs,
    )
    ctx.inserted_fonts = True  # every rewrite embeds a full font program; subset on save
    return rf


def _replace_inline(ctx: _Ctx, page, op: ReplaceText, res: OpResult) -> None:
    hits = _select(find_line_hits(page, op.search), op.occurrence, f"search text {op.search!r}")
    # Process right-to-left / bottom-up so earlier hits keep valid geometry.
    hits.sort(key=lambda h: (h.line["bbox"][1], h.chars[h.start][0]["bbox"][0]), reverse=True)
    geo = _PageGeometry(page)
    plans = []
    for h in hits:
        if any(p[0].line is h.line for p in plans):
            raise PdfToolError("several matches in one line; use occurrence or a rect-based replace")
        c0, span0 = h.chars[h.start]
        style = span_style(span0)
        size = op.size or style["size"]
        color = _rgb(op.color or style["color"])
        rf = _font_for(ctx, page, op.text, op, style)
        res.warnings += [w for w in rf.warnings if w not in res.warnings]
        line_rect = pymupdf.Rect(h.line["bbox"])
        left_lim, right_lim = geo.limits(line_rect, h.block, h.line["rects"])
        x_start = c0["origin"][0]
        old_end = h.chars[h.end][0]["origin"][0] if h.end < len(h.chars) else h.chars[h.end - 1][0]["bbox"][2]
        old_w = old_end - x_start
        line_x0 = h.chars[0][0]["bbox"][0]
        line_x1 = h.chars[-1][0]["bbox"][2]
        align = op.align if op.align in ("right", "center") else "left"
        scale, layout = 1.0, None
        while True:
            new_w = rf.font.text_length(op.text, size * scale)
            delta = new_w - old_w
            shift_pre = -delta if align == "right" else -delta / 2 if align == "center" else 0.0
            shift_post = 0.0 if align == "right" else delta / 2 if align == "center" else delta
            new_x0 = line_x0 + shift_pre if h.start > 0 else x_start + shift_pre
            new_x1 = line_x1 + shift_post if h.end < len(h.chars) else x_start + shift_pre + new_w
            fits = new_x0 >= left_lim - 0.5 and new_x1 <= right_lim + 0.5
            layout = (delta, shift_pre, shift_post, new_x0, new_x1)
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
        plans.append((h, style, size * scale, color, rf, layout, scale))

    for h, _style, size, color, rf, (delta, shift_pre, shift_post, _x0, _x1), scale in plans:
        moved: list[tuple[int, int, float]] = []  # (from, to, shift) char ranges that must be rewritten
        if abs(delta) > 0.05:
            if shift_pre and h.start > 0:
                moved.append((0, h.start, shift_pre))
            if shift_post and h.end < len(h.chars):
                moved.append((h.end, len(h.chars), shift_post))
        redact_idx = [(h.start, h.end)] + [(a, b) for a, b, _ in moved]
        rects = [_union(c["bbox"] for c, _ in h.chars[a:b]) for a, b in redact_idx]
        y = h.chars[h.start][0]["origin"][1]
        _redact(page, rects, op.fill)
        written = _write(page, h.chars[h.start][0]["origin"][0] + shift_pre, y, op.text, rf, size, color)
        for a, b, shift in moved:
            # rewrite untouched neighbours span by span, shifted horizontally
            k = a
            while k < b:
                span = h.chars[k][1]
                j = k
                while j < b and h.chars[j][1] is span:
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
                    c0 = h.chars[k][0]
                    written |= _write(page, c0["origin"][0] + shift, c0["origin"][1], seg, seg_font, st["size"], _rgb(st["color"]))
                k = j
        res.rects.append(rnd(_union(rects) | written))
        res.font, res.font_source = rf.label, rf.source
        if scale < 1:
            res.scale = scale
            res.warnings.append(f"replacement scaled to {scale:.0%} to fit")


TEXT_ALIGN = {
    "left": pymupdf.TEXT_ALIGN_LEFT,
    "center": pymupdf.TEXT_ALIGN_CENTER,
    "right": pymupdf.TEXT_ALIGN_RIGHT,
    "justify": pymupdf.TEXT_ALIGN_JUSTIFY,
}


def _place_text(
    page, box: pymupdf.Rect, first_baseline: float, text: str, rf: ResolvedFont, size: float, color: str,
    pitch: float, align: str, op: _Style, res: OpResult,
) -> tuple[pymupdf.Rect, float]:  # fmt: skip
    """Wrap plain text into ``box`` starting exactly at ``first_baseline``.

    Uses TextWriter (one glyph per character): MuPDF's HTML layout forms ligatures
    that subset fonts cannot map back to Unicode, which breaks text extraction.
    """
    ratio = pitch / size

    def layout(scale: float, target_page) -> tuple[pymupdf.TextWriter, float]:
        fs = size * scale
        base = box.y0 + (first_baseline - box.y0) * scale  # keep the line-box top fixed when shrinking
        tw = pymupdf.TextWriter(target_page.rect)
        # fill_textbox indents every line after the first by 0.2em; start the rect that much earlier
        rect = pymupdf.Rect(box.x0 - 0.2 * fs, box.y0, box.x1, target_page.rect.y1)
        over = tw.fill_textbox(rect, text, pos=(box.x0, base), font=rf.font, fontsize=fs, lineheight=ratio, align=TEXT_ALIGN[align])
        if over:
            raise PdfToolError("text runs past the bottom of the page")
        lines = round((tw.last_point.y - base) / (pitch * scale)) + 1
        return tw, lines * pitch * scale

    scratch = pymupdf.open()
    probe = scratch.new_page(width=page.rect.width, height=box.y0 + 10000)
    scale = 1.0
    _, needed = layout(1.0, probe)
    used = needed
    while used > box.height + 0.5:
        if op.overflow == "shrink" and scale - 0.02 >= op.min_scale - 1e-9:
            scale = round(scale - 0.02, 2)
            _, used = layout(scale, probe)
            continue
        if op.overflow == "allow":
            if box.y0 + used > page.rect.y1:
                raise PdfToolError(f"text needs {used:.1f}pt height and would run off the page")
            res.warnings.append(f"text extends {used - box.height:.1f}pt below the box; check for overlap")
            break
        hint = "Shorten the text, enlarge the rect, use overflow 'shrink', or reconstruct the document."
        if op.overflow == "shrink":
            raise PdfToolError(f"text does not fit even at {op.min_scale:.0%} scale (needs {needed:.1f}pt, box {box.height:.1f}pt)")
        raise PdfToolError(f"text needs {needed:.1f}pt height but the box has {box.height:.1f}pt. {hint}")
    scratch.close()
    tw, used = layout(scale, page)
    tw.write_text(page, color=_rgb(color))
    if scale < 1:
        res.scale = scale
        res.warnings.append(f"text scaled to {scale:.0%} to fit")
    return pymupdf.Rect(box.x0, box.y0, box.x1, box.y0 + used), used


def _css(rf: ResolvedFont, size: float, color: str, line_ratio: float, align: str, arc_dir: Path) -> str:
    weight = "bold" if rf.bold else "normal"
    style = "italic" if rf.italic else "normal"
    if rf.generic:
        family, face = rf.generic, ""
    else:
        fname = f"face.{rf.ext}"
        (arc_dir / fname).write_bytes(rf.buffer or b"")
        family = '"EditFace"'
        # the file already carries its weight/style; declare it for every combination
        face = "".join(
            f'@font-face {{font-family: {family}; src: url("{fname}"); font-weight: {w}; font-style: {s};}}\n'
            for w in ("normal", "bold")
            for s in ("normal", "italic")
        )
        weight = style = "normal"
    return (
        # note: a '*' margin reset shifts MuPDF's first baseline by 1pt; reset body/p only
        f"{face}body {{margin: 0; padding: 0; font-family: {family}; font-size: {size:.2f}pt; color: {color}; line-height: {line_ratio:.4f}; "
        f"text-align: {align}; font-weight: {weight}; font-style: {style};}}\n"
        "p { margin: 0; }\n"
    )


def _place_html(
    ctx: _Ctx, page, box: pymupdf.Rect, html: str, rf: ResolvedFont, size: float, color: str,
    line_ratio: float, align: str, op: _Style, res: OpResult,
) -> pymupdf.Rect:  # fmt: skip
    """Measure, apply the overflow policy, then insert HTML into ``box``."""
    with tempfile.TemporaryDirectory(prefix="pdf-tool-font-") as tmp:
        arc_dir = Path(tmp)
        css = _css(rf, size, color, line_ratio, align, arc_dir)
        arc = pymupdf.Archive(str(arc_dir))
        scratch = pymupdf.open()
        probe_h = 10000
        sp = scratch.new_page(width=page.rect.width, height=box.y0 + probe_h + 10)
        spare, _ = sp.insert_htmlbox(pymupdf.Rect(box.x0, box.y0, box.x1, box.y0 + probe_h), html, css=css, archive=arc, scale_low=1)
        scratch.close()
        needed = probe_h - spare if spare >= 0 else float("inf")
        target, scale_low = box, 1
        if needed > box.height + 0.5:
            if op.overflow == "shrink":
                scale_low = op.min_scale
            elif op.overflow == "allow":
                target = pymupdf.Rect(box.x0, box.y0, box.x1, box.y0 + needed + 1)
                if target.y1 > page.rect.y1:
                    raise PdfToolError(f"text needs {needed:.1f}pt height and would run off the page")
                res.warnings.append(f"text extends {needed - box.height:.1f}pt below the box; check for overlap")
            else:
                raise PdfToolError(
                    f"text needs {needed:.1f}pt height but the box has {box.height:.1f}pt. "
                    "Shorten the text, enlarge the rect, use overflow 'shrink', or reconstruct the document."
                )
        spare, scale = page.insert_htmlbox(target, html, css=css, archive=arc, scale_low=scale_low)
        if spare < 0:
            raise PdfToolError(f"text does not fit even at {op.min_scale:.0%} scale (needs {needed:.1f}pt, box {box.height:.1f}pt)")
        if scale < 0.999:
            res.scale = round(scale, 3)
            res.warnings.append(f"text scaled to {scale:.0%} to fit")
        return target, min(needed, target.height) * scale


def _infer_align(lines: list[pymupdf.Rect], box: pymupdf.Rect) -> str:
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
    if same_left:
        return "left"
    if max(rights) - min(rights) < tol:
        return "right"
    centers = [(r.x0 + r.x1) / 2 for r in lines]
    if max(centers) - min(centers) < tol * 2:
        return "center"
    return "left"


def _replace_rect(ctx: _Ctx, page, op: ReplaceText | InsertText, res: OpResult) -> None:
    box = pymupdf.Rect(op.rect) & page.rect
    if box.is_empty:
        raise PdfToolError("rect is empty or outside the page")
    spans, line_rects, baselines = [], [], []
    for b in page.get_text("dict", flags=RAW_FLAGS)["blocks"]:
        if b["type"] != 0:
            continue
        for ln in b["lines"]:
            inside = [s for s in ln["spans"] if s["text"].strip() and pymupdf.Rect(s["bbox"]) in box + (-1, -1, 1, 1)]
            if inside:
                spans += inside
                line_rects.append(pymupdf.Rect(ln["bbox"]))
                baselines.append(inside[0]["origin"][1])
    is_replace = isinstance(op, ReplaceText)
    if is_replace and not spans:
        res.warnings.append("no existing text found inside rect; text inserted with default style")
    if not is_replace and spans:
        res.warnings.append("rect overlaps existing text; use replace_text to replace it")
    style = {}
    if spans:
        style = dominant_style(spans) or {}
        first = span_style(spans[0])
        style |= {"serif": first["serif"], "monospace": first["monospace"]}
    size = op.size or style.get("size") or 11.0
    color = op.color or style.get("color") or "#000000"
    gaps = sorted(b - a for a, b in zip(baselines, baselines[1:]) if b - a > 0.5)
    pitch = op.line_height or (gaps[len(gaps) // 2] if gaps else size * 1.2)
    line_ratio = pitch / size
    align = op.align or (_infer_align(line_rects, box) if is_replace else "left")
    if is_replace and not op.align and len(line_rects) > 1:
        res.warnings.append(f"alignment inferred as '{align}'")
    plain = op.text if op.text is not None else re.sub(r"<[^>]+>", "", op.html or "")
    rf = _font_for(ctx, page, htmllib.unescape(plain), op, style)
    res.warnings += [w for w in rf.warnings if w not in res.warnings]
    if is_replace and spans:
        _redact(page, [box], op.fill)
    top = (baselines[0] - BASELINE_RATIO * size - (line_ratio - 1) * size / 2) if baselines else box.y0
    left = min(r.x0 for r in line_rects) if line_rects else box.x0  # the original text edge, not the rough rect
    target = pymupdf.Rect(max(box.x0, left), top, box.x1, box.y1)
    if op.text is not None:
        first = baselines[0] if baselines else top + (line_ratio - 1) * size / 2 + BASELINE_RATIO * size
        placed, used_h = _place_text(page, target, first, op.text, rf, size, color, pitch, align, op, res)
    else:
        placed, used_h = _place_html(ctx, page, target, op.html, rf, size, color, line_ratio, align, op, res)
    res.rects.append(rnd(placed))
    res.reference_rects.append(rnd(box))
    if is_replace and baselines:
        res.height_change = round(used_h - (baselines[-1] - baselines[0] + pitch), 2)
    res.font, res.font_source = rf.label, rf.source


def _insert_point(ctx: _Ctx, page, op: InsertText, res: OpResult) -> None:
    rf = _font_for(ctx, page, op.text or "", op, None)
    res.warnings += rf.warnings
    size = op.size or 11.0
    x, y = op.point
    scale = 1.0
    while True:
        w = rf.font.text_length(op.text, size * scale)
        if x + w <= page.rect.x1 or op.overflow == "allow":
            break
        if op.overflow == "shrink" and scale - 0.02 >= op.min_scale - 1e-9:
            scale = round(scale - 0.02, 2)
            continue
        raise PdfToolError(f"text is {w:.1f}pt wide and runs past the page edge at x={page.rect.x1:.1f}")
    s = size * scale
    area = pymupdf.Rect(x, y - rf.font.ascender * s, x + w, y - rf.font.descender * s)
    if page.get_text("words", clip=area):
        res.warnings.append("inserted line overlaps existing text")
    rect = _write(page, x, y, op.text, rf, s, _rgb(op.color))
    if scale < 1:
        res.scale = scale
    res.rects.append(rnd(rect))
    res.font, res.font_source = rf.label, rf.source


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
    res.message = f"replaced xref {xref}"


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


def _verify(doc: pymupdf.Document, op, res: OpResult) -> None:
    page = doc[op.page - 1]
    if isinstance(op, (ReplaceText, InsertText)) and res.rects:
        expected = op.text if op.text is not None else htmllib.unescape(re.sub(r"<[^>]+>", "", op.html or ""))
        area = _union(res.rects) + (-2, -2, 2, 2)
        found = page.get_text("text", clip=pymupdf.Rect(0, area.y0, page.rect.x1, area.y1))
        res.verified = _squash(expected) in _squash(found)
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


def _dispatch(ctx: _Ctx, page, op, res: OpResult) -> None:
    if isinstance(op, ReplaceText):
        (_replace_inline if op.search is not None else _replace_rect)(ctx, page, op, res)
    elif isinstance(op, InsertText):
        (_insert_point if op.point is not None else _replace_rect)(ctx, page, op, res)
    elif isinstance(op, Redact):
        _redact_op(ctx, page, op, res)
    elif isinstance(op, RemoveImage):
        _remove_image(ctx, page, op, res)
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
            doc.save(tmp, garbage=3, deflate=True)
    finally:
        doc.close()

    out_doc = open_pdf(output)
    try:
        for op, res in zip(cs.operations, report.results):
            _verify(out_doc, op, res)
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
    for op in cs.operations:  # areas below a shift line sit elsewhere in the output
        if isinstance(op, ShiftContent):
            report.shifts.append({"page": op.page, "y": op.y, "dy": op.dy})
            for m in report.masks:
                out = m.get("candidate_rect", m["rect"])
                if m["page"] == op.page and out[1] >= op.y - 0.5:
                    m["candidate_rect"] = rnd(pymupdf.Rect(out) + (0, op.dy, 0, op.dy))
    return report
