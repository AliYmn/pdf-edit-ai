"""`pdf-tool scaffold`: an editable HTML replica of a PDF, the starting point for
reconstruction (Strategy B).

Every text segment is absolutely positioned on its original baseline, shapes and
images keep their paint order, fonts come from the PDF itself, and links are kept.
The HTML is then rendered with `generate`, every segment is measured against the
original, and baseline offsets are corrected per style (calibration).
"""

from __future__ import annotations

import html
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .assets import export_fonts, export_images
from .common import floats_to_hex, open_pdf, parse_pages, require_dir_output, write_json
from .fonts import BUILTIN, CSS_ALIASES, embedded_fonts, font_class, norm_name, split_font_name, system_font
from .inspect import infer_align, page_background, page_links, span_style
from .vector import padded_rect, path_d, path_element

FLAGS = pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES & ~pymupdf.TEXT_PRESERVE_LIGATURES
GENERIC = {"sans": "sans-serif", "serif": "serif", "mono": "monospace"}


def _n(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


@dataclass
class FontFace:
    stack: str  # CSS font-family value
    weight: int
    italic: bool
    metrics: pymupdf.Font


@dataclass
class Segment:
    """A run of text on one baseline without large gaps, written as one absolutely positioned div."""

    page: int
    x: float
    baseline: float
    width: float
    spans: list[tuple[str, str, float, str | None, bool]]  # (text, class, baseline offset, href, zero advance)
    key: str  # calibration group (dominant class + line height)
    line_height: float
    offset: float  # baseline distance from the div top, from font metrics
    chars: list[tuple[str, float]] = field(default_factory=list)  # (char, origin x) incl. spaces
    justify: bool = False
    block: str | None = None
    correction: float = 0.0  # baseline fix from calibration
    letter_spacing: float = 0.0  # fitted from calibration
    word_spacing: float = 0.0

    @property
    def text(self) -> str:
        return "".join(t for t, *_ in self.spans)


@dataclass
class Block:
    id: str
    flow: dict  # CSS for rewriting the block as one flowing paragraph
    key: str
    offset: float
    first_baseline: float
    segments: list[Segment] = field(default_factory=list)
    flow_correction: float | None = None  # measured with the flow recipe itself
    breaks: str | None = None  # "same" if flowing the original text reproduces its line breaks

    def flow_css(self) -> str:
        f = self.flow
        corr = self.flow_correction if self.flow_correction is not None else self.segments[0].correction
        return (
            f"left:{_n(f['left'])}pt;top:{_n(self.first_baseline - self.offset + corr)}pt;width:{_n(f['width'])}pt;"
            f"line-height:{_n(f['line-height'])}pt;text-align:{f['text-align']}"
            + (f";padding-left:{_n(-f['indent'])}pt;text-indent:{_n(f['indent'])}pt" if f["indent"] < -0.5 else "")
            + (f";text-indent:{_n(f['indent'])}pt" if f["indent"] > 0.5 else "")
        )


class _Fonts:
    """Map PDF fonts to CSS: embedded web fonts via @font-face, otherwise system families."""

    def __init__(self, doc: pymupdf.Document, font_assets, out: Path):
        self.doc, self.out = doc, out
        self.by_xref = {a.xref: a for a in font_assets}
        self.faces: dict[str, str] = {}  # @font-face rules
        self.cache: dict[tuple[int, str], FontFace] = {}
        self.report: dict[str, dict] = {}

    def face(self, page: pymupdf.Page, span: dict) -> FontFace:
        key = (page.number, span["font"])
        if key in self.cache:
            return self.cache[key]
        st = span_style(span)
        fam, bold, italic = split_font_name(span["font"])
        bold, italic = bold or st["bold"], italic or st["italic"]
        cls = font_class(fam, st["serif"], st["monospace"])
        display = span["font"].split("-")[0].split(",")[0]
        emb = embedded_fonts(self.doc, page).get(norm_name(span["font"]))
        asset = self.by_xref.get(emb[0]) if emb else None
        metrics = None
        if asset is not None and asset.web_usable and asset.file:
            css_fam = f"pdf-{fam}"
            rule = (
                f'@font-face {{ font-family: "{css_fam}"; src: url("{asset.file}"); '
                f"font-weight: {700 if bold else 400}; font-style: {'italic' if italic else 'normal'}; }}"
            )
            self.faces.setdefault(rule, rule)
            stack = f'"{css_fam}", "{display}", {GENERIC[cls]}'
            source = "embedded subset" if asset.subset else "embedded"
            metrics = pymupdf.Font(fontbuffer=(self.out / asset.file).read_bytes())
        else:
            names = [display] + [n for n in CSS_ALIASES.get(fam, []) if n != display]
            stack = ", ".join(f'"{n}"' for n in names) + f", {GENERIC[cls]}"
            path = system_font(fam, bold, italic)
            source = f"system ({path.name})" if path else "browser fallback"
            if emb:
                try:
                    metrics = pymupdf.Font(fontbuffer=emb[2])
                except Exception:
                    metrics = None
            if metrics is None and path:
                metrics = pymupdf.Font(fontfile=str(path))
        if metrics is None:
            metrics = pymupdf.Font(BUILTIN[cls][(2 if italic else 0) + (1 if bold else 0)])
        ff = FontFace(stack, 700 if bold else 400, italic, metrics)
        self.cache[key] = ff
        self.report.setdefault(span["font"], {"css": stack, "source": source})
        return ff


class _Classes:
    def __init__(self):
        self.rules: dict[tuple, str] = {}

    def get(self, face: FontFace, size: float, color: str) -> str:
        k = (face.stack, face.weight, face.italic, round(size, 2), color)
        if k not in self.rules:
            self.rules[k] = f"s{len(self.rules) + 1}"
        return self.rules[k]

    def css(self) -> str:
        return "\n".join(
            f".{name} {{ font-family: {stack}; font-weight: {w}; font-style: {'italic' if it else 'normal'}; "
            f"font-size: {_n(size)}pt; color: {color}; }}"
            for (stack, w, it, size, color), name in self.rules.items()
        )


def _offset(face: FontFace, size: float, line_height: float) -> float:
    """Baseline position inside a CSS line box (half-leading model)."""
    asc, desc = face.metrics.ascender, face.metrics.descender
    return (line_height - (asc - desc) * size) / 2 + asc * size


def _split_segments(line: dict) -> list[list[tuple[dict, dict]]]:
    """Split a text line at large gaps (column/tab-like spacing)."""
    chars = [(c, s) for s in line["spans"] for c in s["chars"]]
    segs, cur = [], []
    for c, s in chars:
        wide_space = c["c"].isspace() and (c["bbox"][2] - c["bbox"][0]) > 0.6 * s["size"]
        jump = cur and c["bbox"][0] - cur[-1][0]["bbox"][2] > 0.6 * s["size"]
        if wide_space or jump:
            if any(not x[0]["c"].isspace() for x in cur):
                segs.append(cur)
            cur = [] if wide_space else [(c, s)]
            continue
        cur.append((c, s))
    if any(not x[0]["c"].isspace() for x in cur):
        segs.append(cur)
    return [_strip(seg) for seg in segs]


def _strip(seg):
    while seg and seg[0][0]["c"].isspace():
        seg = seg[1:]
    while seg and seg[-1][0]["c"].isspace():
        seg = seg[:-1]
    return seg


def _href(link) -> str:
    return link.uri if link.uri else f"#page-{link.target_page}"


def _page_text(doc, page, fonts: _Fonts, classes: _Classes, links, warnings) -> tuple[list[Segment], list[Block], set]:
    segments: list[Segment] = []
    blocks: list[Block] = []
    used_links: set[int] = set()
    for bi, block in enumerate(page.get_text("rawdict", flags=FLAGS)["blocks"]):
        if block["type"] != 0:
            continue
        lines = []
        for line in block["lines"]:
            if abs(line["dir"][0] - 1) > 1e-3:
                text = "".join(c["c"] for s in line["spans"] for c in s["chars"])
                warnings.append(f"page {page.number + 1}: rotated text not reproduced: {text[:40]!r}")
                continue
            lines.append(line)
        block_segments = []
        for line in lines:
            for seg in _split_segments(line):
                if not seg:
                    continue
                # group characters back into style runs
                runs: list[tuple[str, dict]] = []
                for c, s in seg:
                    if runs and runs[-1][1] is s:
                        runs[-1] = (runs[-1][0] + c["c"], s)
                    else:
                        runs.append((c["c"], s))
                main = max((s for _, s in runs), key=lambda s: (s["size"], len(s["chars"])))
                baseline = main["origin"][1]
                face = fonts.face(page, main)
                lh = round(main["size"], 2)
                spans = []
                starts = [next(c for c, sp in seg if sp is s)["origin"][0] for _, s in runs]
                for k, (text, s) in enumerate(runs):
                    # hidden marks (e.g. a 0.75pt bullet for screen readers) that take no space in the PDF
                    zero = k + 1 < len(runs) and abs(starts[k + 1] - starts[k]) < 0.05
                    f = fonts.face(page, s)
                    cls = classes.get(f, s["size"], span_style(s)["color"])
                    center = pymupdf.Rect(s["bbox"]).tl + (pymupdf.Rect(s["bbox"]).br - pymupdf.Rect(s["bbox"]).tl) * 0.5
                    href = None
                    for k, ln in enumerate(links):
                        if center in pymupdf.Rect(ln.bbox) + (-0.5, -0.5, 0.5, 0.5):
                            href = _href(ln)
                            used_links.add(k)
                            break
                    spans.append((text, cls, s["origin"][1] - baseline, href, zero))
                x0 = seg[0][0]["origin"][0]
                width = seg[-1][0]["bbox"][2] - x0
                key = f"{classes.get(face, main['size'], span_style(main)['color'])}@{lh}"
                block_segments.append(
                    Segment(page.number + 1, x0, baseline, width, spans, key, lh, _offset(face, main["size"], lh),
                            chars=[(c["c"], c["origin"][0]) for c, _ in seg])  # fmt: skip
                )
        segments += block_segments
        # multi-line paragraph blocks: justified lines + a ready "flow" recipe
        rows = sorted({round(s.baseline, 1) for s in block_segments})
        if len(rows) >= 2 and len(block_segments) == len(rows):
            pitches = [b - a for a, b in zip(rows, rows[1:])]
            if max(pitches) - min(pitches) < 0.6:
                segs = sorted(block_segments, key=lambda s: s.baseline)
                rects = [pymupdf.Rect(s.x, s.baseline - 1, s.x + s.width, s.baseline) for s in segs]
                align = infer_align(rects)
                if align == "justify":
                    for s in segs[:-1]:
                        s.justify = " " in s.text
                pitch = statistics.median(pitches)
                first_face = fonts.face(page, max((sp for ln in lines for sp in ln["spans"]), key=lambda sp: len(sp["chars"])))
                size = segs[0].line_height
                x_rest = min(s.x for s in segs[1:])
                bid = f"b{page.number + 1}-{bi + 1}"
                left = min(segs[0].x, x_rest)
                flow = {
                    "left": left,
                    "width": _wrap_right(segs, align) - left,
                    "line-height": pitch,
                    "text-align": align,
                    "indent": segs[0].x - x_rest,
                }
                blk = Block(bid, flow, segs[0].key.split("@")[0], _offset(first_face, size, pitch), segs[0].baseline, segs)
                for s in segs:
                    s.block = bid
                blocks.append(blk)
    return segments, blocks, used_links


def _image_clip(clips: list[dict], box: pymupdf.Rect) -> str:
    """CSS clip-path for an image clipped by a non-rectangular path (e.g. a round photo)."""
    for d in reversed(clips):
        sc = pymupdf.Rect(d["scissor"])
        if abs(sc & box) > 0.8 * max(abs(sc), abs(box)):
            rule = "evenodd, " if d.get("even_odd") else ""
            return f";clip-path:path({rule}'{path_d(d, -box.x0, -box.y0, 4 / 3)}')"  # CSS px
    return ""


def _wrap_right(segs: list[Segment], align: str) -> float:
    """Estimate the right edge the original text wrapped at: at least the longest line,
    and less than any line plus a space plus the next line's first word (it did not fit).
    Justified lines are stretched to that edge, so there it is the longest line."""
    longest = max(s.x + s.width for s in segs)
    if align == "justify":
        return longest + 0.25
    gaps = [b[1] - a[1] for s in segs for a, b in zip(s.chars, s.chars[1:]) if a[0] == " "]
    space = statistics.median(gaps) if gaps else 0.25 * segs[0].line_height
    limits = []
    for cur, nxt in zip(segs, segs[1:]):
        first_space = next((x for ch, x in nxt.chars if ch.isspace()), None)
        word = (first_space if first_space is not None else nxt.x + nxt.width) - nxt.x
        limits.append(cur.x + cur.width + space + word)
    upper = min(limits) if limits else longest + 1
    return upper - 0.5 if upper - 0.5 > longest + 0.5 else longest + 1


def _graphics(page, images_by_page) -> tuple[list[str], dict]:
    """Shapes and images in paint order. Simple boxes/rules become divs; other paths inline SVG."""
    clips = [
        d for d in page.get_drawings(extended=True)
        if d["type"] == "clip" and d.get("scissor") and not (len(d["items"]) == 1 and d["items"][0][0] == "re")
    ]  # fmt: skip
    log = page.get_bboxlog()
    items: list[tuple[int, str, object]] = []
    for d in page.get_drawings():
        items.append((d.get("seqno", 0), "draw", d))
    image_seqs = [i for i, (kind, _) in enumerate(log) if kind == "fill-image"]
    for pl, asset in images_by_page.get(page.number + 1, []):
        seq = next((i for i in image_seqs if pymupdf.Rect(log[i][1]) & pymupdf.Rect(pl.bbox)), len(log))
        items.append((seq, "img", (pl, asset)))
    items.sort(key=lambda t: t[0])
    out: list[str] = []
    counts = {"boxes": 0, "svg": 0, "images": 0}
    pending: list[dict] = []

    def flush():
        if pending:
            r = padded_rect(pending)
            body = "".join(path_element(d, -r.x0, -r.y0) for d in pending)
            out.append(
                f'<svg class="v" style="left:{_n(r.x0)}pt;top:{_n(r.y0)}pt;width:{_n(r.width)}pt;height:{_n(r.height)}pt" '
                f'viewBox="0 0 {_n(r.width)} {_n(r.height)}">{body}</svg>'
            )
            counts["svg"] += 1
            pending.clear()

    for _, kind, obj in items:
        if kind == "img":
            flush()
            pl, asset = obj
            x0, y0, x1, y1 = pl.bbox
            clip = _image_clip(clips, pymupdf.Rect(pl.bbox))
            out.append(
                f'<img class="i" src="{asset.file}" alt="" '
                f'style="left:{_n(x0)}pt;top:{_n(y0)}pt;width:{_n(x1 - x0)}pt;height:{_n(y1 - y0)}pt{clip}">'
            )
            counts["images"] += 1
            continue
        d = obj
        items_ = d["items"]
        fill, stroke = floats_to_hex(d.get("fill")), floats_to_hex(d.get("color"))
        op = ""
        if (d.get("fill_opacity") or 1) < 1:
            op = f";opacity:{_n(d['fill_opacity'])}"
        if len(items_) == 1 and items_[0][0] == "re" and fill and not stroke:
            r = items_[0][1]
            flush()
            out.append(
                f'<div class="r" style="left:{_n(r.x0)}pt;top:{_n(r.y0)}pt;width:{_n(r.width)}pt;height:{_n(r.height)}pt;background:{fill}{op}"></div>'
            )
            counts["boxes"] += 1
        elif len(items_) == 1 and items_[0][0] == "l" and stroke and not fill and _axis(items_[0]) and _solid(d):
            p1, p2, w = items_[0][1], items_[0][2], d.get("width") or 1
            flush()
            if abs(p1.y - p2.y) < 0.01:
                x0, x1 = sorted((p1.x, p2.x))
                style = f"left:{_n(x0)}pt;top:{_n(p1.y - w / 2)}pt;width:{_n(x1 - x0)}pt;height:{_n(w)}pt"
            else:
                y0, y1 = sorted((p1.y, p2.y))
                style = f"left:{_n(p1.x - w / 2)}pt;top:{_n(y0)}pt;width:{_n(w)}pt;height:{_n(y1 - y0)}pt"
            out.append(f'<div class="r" style="{style};background:{stroke}"></div>')
            counts["boxes"] += 1
        else:
            pending.append(d)
    flush()
    return out, counts


def _axis(item) -> bool:
    p1, p2 = item[1], item[2]
    return abs(p1.x - p2.x) < 0.01 or abs(p1.y - p2.y) < 0.01


def _solid(d) -> bool:
    dashes = (d.get("dashes") or "").strip()
    return dashes in ("", "[] 0") or dashes.startswith("[] ")


def _render_page(pno: int, width: float, height: float, size_idx: int, segments, blocks, graphics, overlays, background,
                 flow_test: bool = False) -> str:  # fmt: skip
    seg_by_block: dict[str, list[Segment]] = defaultdict(list)
    loose = []
    for s in segments:
        (seg_by_block[s.block] if s.block else loose).append(s)
    parts = [
        f'<section class="page" id="page-{pno}" data-size="{size_idx}" '
        f'style="width:{_n(width)}pt;height:{_n(height)}pt;background:{background}">'
    ]
    parts.append("  <!-- graphics (paint order) -->")
    parts += ["  " + g for g in graphics]
    parts.append("  <!-- text: each .l sits on its original baseline; .block = paragraph (see data-flow) -->")
    emitted = set()
    for s in sorted(segments, key=lambda s: (s.baseline, s.x)):
        if s.block:
            if s.block in emitted:
                continue
            emitted.add(s.block)
            blk = next(b for b in blocks if b.id == s.block)
            members = sorted(seg_by_block[s.block], key=lambda x: x.baseline)
            if flow_test:  # calibration: the block written with its own flow recipe
                inner = " ".join(_spans_html(x) for x in members)
                parts.append(f'  <p class="flow {blk.key}" style="{blk.flow_css()}">{inner}</p>')
                continue
            breaks = f' data-flow-breaks="{blk.breaks}"' if blk.breaks else ""
            parts.append(f'  <div class="block" id="{blk.id}" data-class="{blk.key}" data-flow="{blk.flow_css()}"{breaks}>')
            parts += ["    " + _segment_html(x) for x in members]
            parts.append("  </div>")
        else:
            parts.append("  " + _segment_html(s))
    parts += ["  " + o for o in overlays]
    parts.append("</section>")
    return "\n".join(parts)


def _segment_html(s: Segment) -> str:
    style = f"left:{_n(s.x)}pt;top:{_n(s.baseline - s.offset + s.correction)}pt;line-height:{_n(s.line_height)}pt"
    if s.justify:
        style += f";width:{_n(s.width)}pt;text-align:justify;text-align-last:justify"
    if abs(s.letter_spacing) >= 0.005:
        style += f";letter-spacing:{s.letter_spacing:.3f}pt"
    if abs(s.word_spacing) >= 0.01:
        style += f";word-spacing:{s.word_spacing:.3f}pt"
    return f'<div class="l" style="{style}">{_spans_html(s)}</div>'


def _spans_html(s: Segment) -> str:
    inner = []
    for text, cls, dy, href, zero in s.spans:
        css = (f"position:relative;top:{_n(dy)}pt;" if abs(dy) > 0.3 else "") + ("display:inline-block;width:0;line-height:0;" if zero else "")
        attr = f' style="{css.rstrip(";")}"' if css else ""
        span = f'<span class="{cls}"{attr}>{html.escape(text, quote=False)}</span>'
        inner.append(f'<a href="{html.escape(href)}">{span}</a>' if href else span)
    return "".join(inner)


BASE_CSS = """* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
.page { position: relative; overflow: hidden; break-after: page; }
.page:last-child { break-after: auto; }
.l, .r, .i, .v, .la { position: absolute; margin: 0; }
.l { white-space: pre; }
.r, .i, .v { display: block; }
.v { overflow: visible; }
a { color: inherit; text-decoration: none; }
a.la { display: block; }
/* rewrite a .block as flowing text: <p class="flow sN" style="{data-flow}">new text</p> */
.flow { position: absolute; margin: 0; white-space: normal; }"""


def _document(title: str, sizes: list[tuple[float, float]], fonts: _Fonts, classes: _Classes, pages_html: list[str]) -> str:
    page_rules = "\n".join(
        f"@page s{k} {{ size: {_n(w)}pt {_n(h)}pt; margin: 0; }}\n.page[data-size=\"{k}\"] {{ page: s{k}; }}"
        for k, (w, h) in enumerate(sizes)
    )
    return (
        f'<!doctype html>\n<html>\n<head>\n<meta charset="utf-8">\n<title>{html.escape(title)}</title>\n'
        "<!-- generated by pdf-tool scaffold: absolute positions in pt; edit text in place, or rewrite a .block\n"
        "     as flowing text using its data-flow style; render with: pdf-tool generate index.html -o out.pdf -->\n"
        f"<style>\n{page_rules}\n{BASE_CSS}\n{chr(10).join(fonts.faces)}\n{classes.css()}\n</style>\n</head>\n<body>\n"
        + "\n".join(pages_html)
        + "\n</body>\n</html>\n"
    )


def _measure(pdf: Path) -> dict[int, list[tuple[str, float, float, list[float]]]]:
    """(text without spaces, x, baseline, x of every non-space char) per text segment."""
    out: dict[int, list] = defaultdict(list)
    with pymupdf.open(pdf) as doc:
        for page in doc:
            for block in page.get_text("rawdict", flags=FLAGS)["blocks"]:
                for line in block.get("lines", []):
                    for seg in _split_segments(line):
                        if seg:
                            ink = [c for c, _ in seg if not c["c"].isspace()]
                            text = "".join(c["c"] for c in ink)
                            xs = [c["origin"][0] for c in ink]
                            out[page.number + 1].append((text, seg[0][0]["origin"][0], seg[0][0]["origin"][1], xs))
    return out


def _match(segments: list[Segment], measured) -> list[tuple[Segment, float, float, list[float]]]:
    """Pair each source segment with the closest rendered segment of the same text."""
    pairs = []
    for s in segments:
        want = "".join(s.text.split())
        cands = [(abs(x - s.x) + abs(y - s.baseline), x, y, xs) for t, x, y, xs in measured.get(s.page, []) if t == want]
        cands = [c for c in cands if c[0] < 25]
        if cands:
            _, x, y, xs = min(cands, key=lambda c: c[0])
            pairs.append((s, s.x - x, s.baseline - y, xs))
    return pairs


def _offsets(s: Segment, rendered_xs: list[float]) -> list[tuple[int, int, float]]:
    """(chars before, spaces before, original minus rendered x) per visible glyph.
    Isolated one-glyph spikes are dropped: letters inside a ligature get positions
    split from the ligature glyph, differently in each PDF."""
    rows = []
    spaces = ink = 0
    for i, (ch, x) in enumerate(s.chars):
        if ch.isspace():
            spaces += 1
            continue
        if ink < len(rendered_xs):
            rows.append((i, spaces, (x - s.chars[0][1]) - (rendered_xs[ink] - rendered_xs[0])))
        ink += 1
    return [
        r for k, r in enumerate(rows)
        if not (0 < k < len(rows) - 1 and abs(r[2] - rows[k - 1][2]) > 0.8 and abs(r[2] - rows[k + 1][2]) > 0.8)
    ]  # fmt: skip


def _fit_spacing(s: Segment, rendered_xs: list[float]) -> tuple[float, float]:
    """Least-squares letter-spacing a and word-spacing b so that every glyph lands on
    its original x: offset_k = a * (chars before k) + b * (spaces before k)."""
    rows = _offsets(s, rendered_xs)
    sii = sum(i * i for i, _, _ in rows)
    sis = sum(i * sp for i, sp, _ in rows)
    sss = sum(sp * sp for _, sp, _ in rows)
    sdi = sum(d * i for i, _, d in rows)
    sds = sum(d * sp for _, sp, d in rows)
    det = sii * sss - sis * sis
    if sss and abs(det) > 1e-9 * max(1.0, sii * sss):
        a, b = (sdi * sss - sds * sis) / det, (sii * sds - sis * sdi) / det
    else:
        a, b = (sdi / sii if sii else 0.0), 0.0
    if abs(a) < 0.05 and sss:  # browsers drop ligatures when letter-spacing != 0; use word spacing alone
        a, b = 0.0, sds / sss
    return max(-1.0, min(1.0, a)), max(-5.0, min(5.0, b))


def scaffold(pdf: str | Path, out_dir: str | Path, pages: str | None = None, calibrate: bool = True) -> dict:
    from .generate import generate_pdf

    pdf = Path(pdf)
    out = require_dir_output(Path(out_dir))
    doc = open_pdf(pdf)
    warnings: list[str] = []
    try:
        selected = parse_pages(pages, doc.page_count)
        images = export_images(doc, out, 1, warnings)
        images_by_page: dict[int, list] = defaultdict(list)
        for a in images:
            for pl in a.placements:
                images_by_page[pl.page].append((pl, a))
        fonts = _Fonts(doc, export_fonts(doc, out, warnings), out)
        classes = _Classes()
        sizes: list[tuple[float, float]] = []
        page_data = []
        all_segments: list[Segment] = []
        counts = defaultdict(int)
        for i in selected:
            page = doc[i]
            size = (round(page.rect.width, 2), round(page.rect.height, 2))
            if size not in sizes:
                sizes.append(size)
            links = page_links(page)
            segments, blocks, used = _page_text(doc, page, fonts, classes, links, warnings)
            graphics, gcounts = _graphics(page, images_by_page)
            overlays = [
                f'<a class="la" href="{html.escape(_href(ln))}" style="left:{_n(ln.bbox[0])}pt;top:{_n(ln.bbox[1])}pt;'
                f'width:{_n(ln.bbox[2] - ln.bbox[0])}pt;height:{_n(ln.bbox[3] - ln.bbox[1])}pt"></a>'
                for k, ln in enumerate(links) if k not in used
            ]  # fmt: skip
            for k, v in gcounts.items():
                counts[k] += v
            counts["text_segments"] += len(segments)
            counts["paragraph_blocks"] += len(blocks)
            counts["links"] += len(links)
            page_data.append((page.number + 1, size[0], size[1], sizes.index(size), segments, blocks, graphics, overlays,
                              page_background(page) or "#ffffff"))  # fmt: skip
            all_segments += segments
        title = (doc.metadata or {}).get("title") or pdf.stem
    finally:
        doc.close()

    html_path = out / "index.html"
    preview = out / "scaffold.pdf"

    def write(path: Path, flow_test: bool = False) -> None:
        pages_html = [_render_page(*pd, flow_test=flow_test) for pd in page_data]
        path.write_text(_document(title, sizes, fonts, classes, pages_html), encoding="utf-8")

    def build():
        write(html_path)
        return generate_pdf(html_path, preview, overwrite=True)

    def calibrate_flows() -> dict:
        """Render every block with its flow recipe; fix its top and check its line breaks."""
        blocks = [b for pd in page_data for b in pd[5]]
        if not blocks:
            return {}
        test_html, test_pdf = out / ".flow-check.html", out / ".flow-check.pdf"
        try:
            write(test_html, flow_test=True)
            generate_pdf(test_html, test_pdf, overwrite=True)
            measured = _measure(test_pdf)
        finally:
            test_html.unlink(missing_ok=True)
            test_pdf.unlink(missing_ok=True)
        same = 0
        for blk in blocks:
            got = _match(blk.segments, measured)
            first = next((dy for seg, _dx, dy, _xs in got if seg is blk.segments[0]), None)
            base = blk.flow_correction if blk.flow_correction is not None else blk.segments[0].correction
            if first is not None:
                blk.flow_correction = base + first
            ok = len(got) == len(blk.segments) and first is not None and all(abs(dy - first) < 1 for _, _, dy, _ in got)
            blk.breaks = "same" if ok else "different"
            same += blk.breaks == "same"
        write(html_path)  # data-flow attributes only: the preview render is unchanged
        return {"paragraph_blocks": len(blocks), "flow_reproduces_line_breaks": same}

    def drift(pairs) -> list[float]:
        """Largest glyph x error per segment (start and spacing errors combined)."""
        return [max((abs(d + dx) for _, _, d in _offsets(s, xs)), default=abs(dx)) for s, dx, _dy, xs in pairs]

    def stats(values: list[float]) -> dict | None:
        return {"median": round(statistics.median(values), 2), "max": round(max(values), 2)} if values else None

    gen = build()
    calibration = None
    if calibrate:
        pairs = _match(all_segments, _measure(preview))
        before = drift(pairs)
        by_key: dict[str, list[float]] = defaultdict(list)
        for (s, _dx, dy, xs), err in zip(pairs, before):
            by_key[s.key].append(dy)
            if not s.justify and err > 0.3:  # only where glyphs visibly drift
                s.letter_spacing, s.word_spacing = _fit_spacing(s, xs)
        corr = {k: statistics.median(v) for k, v in by_key.items()}
        own = {id(seg): dy for seg, _dx, dy, _xs in pairs}
        for s in all_segments:  # each matched segment gets its own fix (Chromium snaps to sub-pixels)
            s.correction = own.get(id(s), corr.get(s.key, 0.0))
        gen = build()
        pairs = _match(all_segments, _measure(preview))
        residual = {id(seg): dy for seg, _dx, dy, _xs in pairs}
        if any(abs(dy) > 0.375 for dy in residual.values()):
            # Chromium snaps baselines to whole CSS px, sometimes the far way: one more round, keep the better
            previous = {id(seg): seg.correction for seg in all_segments}
            for seg in all_segments:
                seg.correction += residual.get(id(seg), 0.0)
            build()
            trial = {id(seg): dy for seg, _dx, dy, _xs in _match(all_segments, _measure(preview))}
            for seg in all_segments:
                if abs(trial.get(id(seg), 0.0)) >= abs(residual.get(id(seg), 0.0)):
                    seg.correction = previous[id(seg)]
            gen = build()
            pairs = _match(all_segments, _measure(preview))
        calibration = {
            "matched_segments": len(pairs),
            "unmatched_segments": len(all_segments) - len(pairs),
            "baseline_error_pt": stats([abs(dy) for _, _, dy, _ in pairs]),
            "glyph_x_error_pt": stats(drift(pairs)),
            "glyph_x_error_before_spacing_fit_pt": stats(before),
            "segments_with_spacing_fix": sum(1 for s in all_segments if s.letter_spacing or s.word_spacing),
            "note": "Chromium places text baselines on a 0.75pt (1 CSS px) grid; up to 0.375pt is the floor",
        }
        calibration["flow_recipes"] = calibrate_flows()
    report = {
        "source": str(pdf.resolve()),
        "html": str(html_path),
        "preview_pdf": str(preview),
        "pages": len(page_data),
        "elements": dict(counts),
        "fonts": fonts.report,
        "calibration": calibration,
        "warnings": warnings + gen["warnings"],
    }
    write_json(report, out / "scaffold.json")
    return report
