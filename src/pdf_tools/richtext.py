"""Styled text for edits: simple HTML parsing, style transfer from the text being
replaced, and line layout where every run can use its own font."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from html.parser import HTMLParser


@dataclass(frozen=True)
class Style:
    font: str | None  # original PDF font name; used to find the matching font program
    size: float
    color: str
    bold: bool = False
    italic: bool = False
    serif: bool = False
    monospace: bool = False


Piece = tuple[str, Style]
Overrides = dict  # any of: bold, italic, color, size, font


class _HtmlRuns(HTMLParser):
    BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[tuple[str, Overrides]] = []
        self.stack: list[tuple[str, Overrides]] = []

    def _current(self) -> Overrides:
        merged: Overrides = {}
        for _, o in self.stack:
            merged |= o
        return merged

    def _newline(self) -> None:
        if self.out and not self.out[-1][0].endswith("\n"):
            self.out.append(("\n", {}))

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.out.append(("\n", {}))
            return
        if tag in self.BLOCK:
            self._newline()
        o: Overrides = {}
        if tag in ("b", "strong"):
            o["bold"] = True
        elif tag in ("i", "em"):
            o["italic"] = True
        style = dict(attrs).get("style") or ""
        if m := re.search(r"color\s*:\s*(#[0-9a-fA-F]{6})", style):
            o["color"] = m.group(1)
        if m := re.search(r"font-weight\s*:\s*(\w+)", style):
            o["bold"] = m.group(1) in ("bold", "bolder", "600", "700", "800", "900")
        if m := re.search(r"font-style\s*:\s*(\w+)", style):
            o["italic"] = m.group(1) in ("italic", "oblique")
        self.stack.append((tag, o))

    def handle_endtag(self, tag):
        for k in range(len(self.stack) - 1, -1, -1):
            if self.stack[k][0] == tag:
                del self.stack[k:]
                break
        if tag in self.BLOCK:
            self._newline()

    def handle_data(self, data):
        text = re.sub(r"\s+", " ", data)
        if text:
            self.out.append((text, self._current()))


def parse_html(html: str) -> list[tuple[str, Overrides]]:
    """'<b>Note:</b> text<br>more' -> [('Note:', {'bold': True}), (' text', {}), ('\\n', {}), ...]"""
    p = _HtmlRuns()
    p.feed(html)
    p.close()
    out = p.out
    while out and out[-1][0] == "\n":
        out.pop()
    if out and out[0][0].startswith(" "):
        out[0] = (out[0][0].lstrip(), out[0][1])
    return out


def _tokens(text: str) -> list[str]:
    return re.findall(r"\n|[^\S\n]+|[^\s]+", text)


def transfer_styles(old: list[Piece], new_text: str, base: Style) -> list[Style]:
    """Per-character styles for ``new_text``, borrowed word by word from the text it
    replaces: unchanged words keep their style, replaced words take the style of the
    words they replace (proportionally), inserted words continue the previous style."""
    if not new_text:
        return []
    old_tokens = [(t, st) for text, st in old for t in _tokens(text)]
    new_tokens = _tokens(new_text)
    if not old_tokens or len({st for _, st in old_tokens}) == 1:
        style = old_tokens[0][1] if old_tokens else base
        return [style] * len(new_text)
    styles: list[Style] = [base] * len(new_tokens)
    sm = SequenceMatcher(None, [t for t, _ in old_tokens], new_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for k in range(j2 - j1):
            if tag == "equal":
                styles[j1 + k] = old_tokens[i1 + k][1]
            elif tag == "replace":
                styles[j1 + k] = old_tokens[i1 + min(i2 - i1 - 1, k * (i2 - i1) // (j2 - j1))][1]
            elif tag == "insert":
                styles[j1 + k] = old_tokens[i1 - 1][1] if i1 > 0 else old_tokens[0][1]
    for k, tok in enumerate(new_tokens):  # a space between differently styled words follows the previous word
        if tok.isspace() and k > 0:
            styles[k] = styles[k - 1]
    return [st for tok, st in zip(new_tokens, styles) for _ in tok]


def build_pieces(
    inputs: list[tuple[str, Overrides]], old: list[Piece], base: Style, op_overrides: Overrides
) -> list[Piece]:
    """Combine transferred styles, op-level overrides and per-run overrides into pieces."""
    text = "".join(t for t, _ in inputs)
    char_styles = transfer_styles(old, text, base)
    out: list[Piece] = []
    pos = 0
    for chunk, overrides in inputs:
        for k, ch in enumerate(chunk):
            st = replace(char_styles[pos + k], **{**op_overrides, **overrides})
            if out and out[-1][1] == st:
                out[-1] = (out[-1][0] + ch, st)
            else:
                out.append((ch, st))
        pos += len(chunk)
    return out


def describe(pieces: list[Piece]) -> str:
    """Markdown-ish rendering (**bold**, _italic_) to show the styles that were applied."""
    out = []
    for text, st in pieces:
        if not text.strip():
            out.append(text)
            continue
        lead, core, trail = re.match(r"^(\s*)(.*?)(\s*)$", text, re.S).groups()
        if st.italic:
            core = f"_{core}_"
        if st.bold:
            core = f"**{core}**"
        out.append(lead + core + trail)
    return "".join(out).replace("\n", "⏎")


@dataclass
class Placement:
    x: float
    y: float
    text: str
    style: Style


def layout(
    pieces: list[Piece],
    width_of,  # (text, style) -> width in pt at style.size
    x_first: float,
    x_rest: float,
    right: float,
    first_baseline: float,
    pitch: float,
    align: str,
) -> tuple[list[Placement], int]:
    """Greedy line breaking of styled pieces. Returns placements and the line count.
    Raises ValueError when a single word is wider than the available width."""
    tokens = [(tok, st) for text, st in pieces for tok in _tokens(text)]
    lines: list[tuple[list[tuple[str, Style]], bool]] = []  # (tokens, ends paragraph)
    cur: list[tuple[str, Style]] = []
    cur_w = 0.0
    for tok, st in tokens:
        if tok == "\n":
            lines.append((cur, True))
            cur, cur_w = [], 0.0
            continue
        avail = right - (x_first if not lines else x_rest)
        w = width_of(tok, st)
        if tok.isspace():
            if cur:
                cur.append((tok, st))
                cur_w += w
            continue
        if w > avail + 0.01:
            raise ValueError(f"the word {tok!r} ({w:.1f}pt) is wider than the box ({avail:.1f}pt)")
        trailing = sum(width_of(t, s) for t, s in _trailing_spaces(cur))
        if cur and cur_w - trailing + w > avail + 0.01:
            lines.append((_strip_trailing(cur), False))
            cur, cur_w = [], 0.0
        cur.append((tok, st))
        cur_w += w
    if cur or not lines:
        lines.append((cur, True))

    out: list[Placement] = []
    for i, (toks, para_end) in enumerate(lines):
        toks = _strip_trailing(toks)
        x0 = x_first if i == 0 else x_rest
        avail = right - x0
        y = first_baseline + i * pitch
        widths = [width_of(t, s) for t, s in toks]
        line_w = sum(widths)
        gaps = sum(1 for t, _ in toks if t.isspace())
        if align == "justify" and not para_end and gaps:
            extra = (avail - line_w) / gaps
            x = x0
            for (t, s), w in zip(toks, widths):
                if t.isspace():
                    x += w + extra
                else:
                    out.append(Placement(x, y, t, s))
                    x += w
            continue
        x = x0 + {"right": avail - line_w, "center": (avail - line_w) / 2}.get(align, 0.0)
        for (t, s), w in zip(toks, widths):  # merge same-style neighbours, spaces included
            if out and out[-1].y == y and out[-1].style == s and abs(out[-1].x + width_of(out[-1].text, s) - x) < 0.01:
                out[-1].text += t
            else:
                out.append(Placement(x, y, t, s))
            x += w
    return out, len(lines)


def _trailing_spaces(toks):
    k = len(toks)
    while k and toks[k - 1][0].isspace():
        k -= 1
    return toks[k:]


def _strip_trailing(toks):
    k = len(toks)
    while k and toks[k - 1][0].isspace():
        k -= 1
    return toks[:k]
