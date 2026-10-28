"""Font resolution for edits: original embedded font -> system font -> built-in.

Every candidate is checked for glyph coverage of the text to be written, so a
missing glyph never silently turns into a substituted font.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pymupdf

from .common import PdfToolError, strip_subset

SYSTEM_FONT_DIRS = [
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
    "~/Library/Fonts",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "~/.fonts",
    "~/.local/share/fonts",
    "C:/Windows/Fonts",
]
# PDF base names -> common metric-compatible system family
ALIASES = {
    "helvetica": "arial",
    "arialmt": "arial",
    "times": "timesnewroman",
    "timesroman": "timesnewroman",
    "courier": "couriernew",
    "liberationsans": "arial",
    "liberationserif": "timesnewroman",
    "nimbussans": "arial",
    "nimbusroman": "timesnewroman",
    "nimbusmono": "couriernew",
}
# CSS family names of metric-compatible system fonts, for browser rendering
CSS_ALIASES = {
    "nimbussans": ["Helvetica", "Arial"],
    "helvetica": ["Helvetica", "Arial"],
    "arial": ["Arial", "Helvetica"],
    "nimbusroman": ["Times New Roman", "Times"],
    "times": ["Times New Roman", "Times"],
    "timesnewroman": ["Times New Roman", "Times"],
    "nimbusmono": ["Courier New", "Courier"],
    "courier": ["Courier New", "Courier"],
}
STYLE_WORDS = (
    "bolditalic", "boldoblique", "semibold", "demibold", "extrabold", "bold", "italic", "oblique",
    "regular", "roman", "book", "medium", "normal",
)  # fmt: skip
BUILTIN = {
    "sans": ("helv", "hebo", "heit", "hebi"),
    "serif": ("tiro", "tibo", "tiit", "tibi"),
    "mono": ("cour", "cobo", "coit", "cobi"),
}
GENERIC_CSS = {"sans": "sans-serif", "serif": "serif", "mono": "monospace"}


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def split_font_name(name: str) -> tuple[str, bool, bool]:
    """'ABCDEF+TimesNewRomanPS-BoldItalicMT' -> ('timesnewroman', True, True)."""
    n = strip_subset(name)
    family, _, style = re.sub(r",", "-", n).partition("-")
    family = re.sub(r"(PSMT|PS|MT)$", "", family)
    style = re.sub(r"(PSMT|PS|MT)$", "", style)
    fam, st = norm_name(family), norm_name(style)
    changed = True
    while changed:
        changed = False
        for w in STYLE_WORDS:
            if fam.endswith(w) and len(fam) > len(w) + 2:
                fam, st, changed = fam[: -len(w)], w + st, True
    bold = any(w in st for w in ("bold", "black", "heavy", "semibold", "demi"))
    italic = "italic" in st or "oblique" in st
    return fam, bold, italic


SANS_FAMILIES = ("sans", "helvetica", "arial", "swiss", "univers", "frutiger", "futura", "gill", "verdana",
                 "tahoma", "segoe", "roboto", "lato", "inter", "myriad", "franklin", "avenir", "din", "calibri")


def font_class(family: str, serif: bool = False, mono: bool = False) -> str:
    """Known family names win over the PDF's flags (which are often wrong); 'Roman'
    and 'Book' are weights, not a serif hint."""
    if mono or any(w in family for w in ("courier", "mono", "consol", "menlo", "code")):
        return "mono"
    if any(w in family for w in SANS_FAMILIES):
        return "sans"
    if serif or any(w in family for w in ("times", "serif", "georgia", "garamond", "minion", "cambria", "baskerville")):
        return "serif"
    return "sans"


def is_condensed(name: str) -> bool:
    n = norm_name(name)
    return any(w in n for w in ("cond", "narrow", "compressed"))


@cache
def _font_index(extra_dirs: tuple[str, ...]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for d in (*extra_dirs, *SYSTEM_FONT_DIRS):
        root = Path(os.path.expanduser(d))
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in (".ttf", ".otf") and p.is_file():
                index.setdefault(norm_name(p.stem), p)
    return index


def system_font(family: str, bold: bool, italic: bool, extra_dirs: tuple[str, ...] = ()) -> Path | None:
    fam = ALIASES.get(family, family)
    if bold and italic:
        styles = ["bolditalic", "boldoblique", "bi", "z"]
    elif bold:
        styles = ["bold", "bd", "b"]
    elif italic:
        styles = ["italic", "oblique", "i"]
    else:
        styles = ["", "regular", "roman", "book", "normal"]
    index = _font_index(extra_dirs)
    for st in styles:
        if (p := index.get(fam + st)) is not None:
            return p
    return None


@dataclass
class ResolvedFont:
    font: pymupdf.Font
    label: str
    source: str  # font_file | original-embedded | system | builtin
    bold: bool
    italic: bool
    buffer: bytes | None = None
    ext: str = "ttf"
    generic: str | None = None  # CSS generic family for built-in fonts
    warnings: list[str] = field(default_factory=list)


def _missing(font: pymupdf.Font, text: str) -> list[str]:
    return sorted({c for c in text if not c.isspace() and not font.has_glyph(ord(c))})


def embedded_fonts(doc: pymupdf.Document, page: pymupdf.Page) -> dict[str, tuple[int, str, bytes]]:
    """Embedded font programs on a page, keyed by normalized BaseFont (no subset
    prefix) and by the font program's own name (closer to what spans report)."""
    out: dict[str, tuple[int, str, bytes]] = {}
    for xref, ext, _t, basefont, *_ in page.get_fonts(full=True):
        if ext == "n/a":
            continue
        try:
            _, fext, _, buf = doc.extract_font(xref)
        except Exception:
            continue
        if not buf:
            continue
        out.setdefault(norm_name(strip_subset(basefont)), (xref, fext, buf))
        try:
            out.setdefault(norm_name(pymupdf.Font(fontbuffer=buf).name), (xref, fext, buf))
        except Exception:
            pass
    return out


def embedded_variant(doc: pymupdf.Document, page: pymupdf.Page, family: str, bold: bool, italic: bool) -> str | None:
    """Normalized name of an embedded font on the page with this family and style
    (e.g. the Bold sibling of 'FiraSans-Regular'); matches BaseFont or program name."""
    for key in embedded_fonts(doc, page):
        if split_font_name(key) == (family, bold, italic):
            return key
    return None


def _subset_with_glyphs(doc: pymupdf.Document, key: str, text: str):
    seen: set[int] = set()
    for page in doc:
        for k, (xref, ext, buf) in embedded_fonts(doc, page).items():
            if xref in seen or not (k.startswith(key) or key.startswith(k)) or min(len(k), len(key)) < 12:
                continue
            seen.add(xref)
            try:
                f = pymupdf.Font(fontbuffer=buf)
            except Exception:
                continue
            if not _missing(f, text):
                return f, buf, ext
    return None


def resolve_font(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    text: str,
    original: str | None = None,
    family: str | None = None,
    font_file: str | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
    serif: bool = False,
    mono: bool = False,
    font_dirs: tuple[str, ...] = (),
) -> ResolvedFont:
    notes: list[str] = []
    name = family or original or "Helvetica"
    fam, name_bold, name_italic = split_font_name(name)
    want_bold = name_bold if bold is None else bold
    want_italic = name_italic if italic is None else italic

    if font_file:
        p = Path(font_file)
        if not p.is_file():
            raise PdfToolError(f"font_file not found: {p}")
        f = pymupdf.Font(fontfile=str(p))
        if miss := _missing(f, text):
            raise PdfToolError(f"font_file {p.name} has no glyphs for {''.join(miss)!r}")
        return ResolvedFont(f, f.name, "font_file", want_bold, want_italic, p.read_bytes(), p.suffix.lstrip(".").lower())

    if original and not family and (want_bold, want_italic) != (name_bold, name_italic):
        sibling = embedded_variant(doc, page, fam, want_bold, want_italic)
        if sibling:
            original = sibling
        else:
            notes.append(f"no embedded {'bold ' if want_bold else ''}{'italic ' if want_italic else ''}variant of '{original}'")
            original = None
    if original and not family:
        fonts_here = embedded_fonts(doc, page)
        key = norm_name(original)
        emb = fonts_here.get(key)
        if emb is None and len(key) >= 12:  # producers truncate names ('...BlackCondense' vs '...BlackCondensed')
            emb = next((v for k, v in fonts_here.items() if k.startswith(key) or key.startswith(k) and len(k) >= 12), None)
        if emb:
            _, ext, buf = emb
            f = pymupdf.Font(fontbuffer=buf)
            miss = _missing(f, text)
            if not miss:
                return ResolvedFont(f, original, "original-embedded", want_bold, want_italic, buf, ext)
            # the same font is often embedded as several subsets: look for one with every glyph
            other = _subset_with_glyphs(doc, key, text)
            if other:
                f, buf, ext = other
                return ResolvedFont(f, original, "original-embedded", want_bold, want_italic, buf, ext,
                                    warnings=[f"glyphs taken from another embedded subset of '{original}'"])
            notes.append(f"embedded '{original}' lacks glyphs for {''.join(miss)!r} (subset font or no cmap)")
        else:
            notes.append(f"'{original}' is not embedded or not extractable")

    path = system_font(fam, want_bold, want_italic, font_dirs)
    if path is None and is_condensed(name) and font_class(fam, serif, mono) == "sans":
        path = system_font("arialnarrow", want_bold, want_italic, font_dirs)  # keeps condensed widths
    if path:
        try:
            f = pymupdf.Font(fontfile=str(path))
            miss = _missing(f, text)
            if not miss:
                if original and norm_name(original) != norm_name(f.name):
                    notes.append(f"using system font {path.name} for '{original}'")
                return ResolvedFont(f, f.name, "system", want_bold, want_italic, path.read_bytes(), path.suffix.lstrip(".").lower(), warnings=notes)
            notes.append(f"system font {path.name} lacks glyphs for {''.join(miss)!r}")
        except Exception as exc:
            notes.append(f"cannot load {path}: {exc}")

    cls = font_class(fam, serif, mono)
    code = BUILTIN[cls][(2 if want_italic else 0) + (1 if want_bold else 0)]
    f = pymupdf.Font(code)
    if miss := _missing(f, text):
        raise PdfToolError(
            f"no available font covers {''.join(miss)!r}; provide font_file (or font_dirs) with a suitable font. "
            + "; ".join(notes)
        )
    notes.append(f"substituted built-in {f.name} ({cls}) for '{name}'")
    return ResolvedFont(f, f.name, "builtin", want_bold, want_italic, generic=GENERIC_CSS[cls], warnings=notes)
