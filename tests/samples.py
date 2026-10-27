"""Programmatic sample PDFs for the test-suite (no binary fixtures in the repo)."""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

LOREM = (
    "This quarterly report summarises operational performance, customer growth and the main risks "
    "identified during the period. All figures are unaudited and subject to revision."
)
TURKISH = "Türkçe karakterler: ğüşıöç İĞÜŞÖÇ — Çağrı Merkezi İstanbul"


def _write(page: pymupdf.Page, xy, text: str, size: float = 11, font: str = "helv", color=(0, 0, 0)) -> None:
    """Write with a Unicode-capable embedded font (built-in Nimbus clones cover Turkish)."""
    tw = pymupdf.TextWriter(page.rect)
    tw.append(xy, text, font=pymupdf.Font(font), fontsize=size)
    tw.write_text(page, color=color)


def _logo_png(size=(240, 80)) -> bytes:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([2, 2, size[0] - 3, size[1] - 3], radius=16, fill=(20, 70, 160, 255))
    d.ellipse([16, 16, 64, 64], fill=(250, 190, 20, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _photo_jpeg(size=(320, 200)) -> bytes:
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)
    for y in range(size[1]):
        d.line([(0, y), (size[0], y)], fill=(40 + y // 2, 120, 200 - y // 3))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def multipage(path: Path, pages: int = 3) -> Path:
    doc = pymupdf.open()
    for n in range(1, pages + 1):
        page = doc.new_page(width=595.28, height=841.89)
        _write(page, (72, 90), f"Section {n}: Quarterly Report", size=18, font="hebo")
        _write(page, (72, 115), "Report date: March 2024", size=11)
        y = 150
        for k in range(4):
            _write(page, (72, y), f"Paragraph {k + 1} on page {n}. Revenue grew steadily across all regions.")
            y += 14
        _write(page, (72, 800), f"Page {n} of {pages}", size=9, color=(0.4, 0.4, 0.4))
    doc.set_metadata({"title": "Sample Report", "author": "pdf-tool tests"})
    doc.save(path)
    return path


def table(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=595.28, height=841.89)
    _write(page, (72, 80), "Sales by region", size=16, font="hebo")
    rows = [("Region", "Q1", "Q2"), ("North", "120", "135"), ("South", "98", "110"), ("West", "143", "150")]
    x0, y0, cw, rh = 72, 100, 140, 24
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            page.draw_rect(rect, color=(0, 0, 0), fill=(0.9, 0.93, 1) if r == 0 else None, width=0.8)
            _write(page, (rect.x0 + 6, rect.y1 - 8), cell, font="hebo" if r == 0 else "helv")
    doc.save(path)
    return path


def images(path: Path) -> Path:
    doc = pymupdf.open()
    for n in (1, 2):
        page = doc.new_page(width=595.28, height=841.89)
        page.insert_image(pymupdf.Rect(72, 36, 192, 76), stream=_logo_png())  # same logo on both pages
        _write(page, (72, 110), f"Product sheet page {n}", size=16, font="hebo")
        if n == 1:
            page.insert_image(pymupdf.Rect(72, 140, 392, 340), stream=_photo_jpeg())
            _write(page, (72, 360), "Figure 1: Gradient photo placeholder.")
        # vector "logo": a filled circle drawn with curves
        page.draw_circle((500, 60), 22, color=(0.8, 0.1, 0.1), fill=(0.9, 0.2, 0.2))
    doc.save(path)
    return path


def turkish(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=595.28, height=841.89)
    _write(page, (72, 100), "Başlık: Şirket Raporu", size=16, font="hebo")
    _write(page, (72, 130), TURKISH)
    _write(page, (72, 150), "Tarih: 15 Mart 2024")
    doc.save(path)
    return path


def mixed_sizes(path: Path) -> Path:
    doc = pymupdf.open()
    for w, h, label in ((595.28, 841.89, "A4 portrait"), (792, 612, "Letter landscape"), (400, 400, "Square")):
        page = doc.new_page(width=w, height=h)
        _write(page, (40, 60), f"{label} page", size=14)
    doc.save(path)
    return path


def paragraph(path: Path) -> Path:
    """A justified-looking multi-line paragraph block with a heading and text below."""
    doc = pymupdf.open()
    page = doc.new_page(width=595.28, height=841.89)
    _write(page, (72, 80), "Summary", size=16, font="hebo")
    page.insert_htmlbox(
        pymupdf.Rect(72, 95, 523, 200),
        f"<p>{LOREM} {LOREM}</p>",
        css="* {font-family: sans-serif; font-size: 11pt; line-height: 1.35; margin: 0;}",
    )
    _write(page, (72, 260), "Next section starts here.")
    doc.save(path)
    return path


def scanned(path: Path) -> Path:
    """Image-only page: text rendered to pixels, no text layer."""
    src = pymupdf.open()
    sp = src.new_page(width=595.28, height=841.89)
    _write(sp, (72, 100), "Scanned Invoice 2024", size=24, font="hebo")
    _write(sp, (72, 140), "Total amount due: 1,250.00 EUR", size=14)
    pix = sp.get_pixmap(dpi=200, alpha=False)
    doc = pymupdf.open()
    page = doc.new_page(width=595.28, height=841.89)
    page.insert_image(page.rect, stream=pix.tobytes("png"))
    doc.save(path)
    return path


def new_image(path: Path, color=(10, 160, 90), size=(240, 80)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def mixed_styles(path: Path) -> Path:
    """Bold labels followed by regular text, and a bullet with a hanging indent."""
    doc = pymupdf.open()
    page = doc.new_page(width=595.28, height=841.89)
    for y, label, rest in ((100, "Backend:", " Python, FastAPI, Django"), (118, "Infra:", " Docker, Kubernetes")):
        tw = pymupdf.TextWriter(page.rect)
        _, end = tw.append((72, y), label, font=pymupdf.Font("hebo"), fontsize=10)
        tw.append(end, rest, font=pymupdf.Font("helv"), fontsize=10)
        tw.write_text(page)
    _write(page, (72, 160), "- Led backend development of real-time trading and financial transaction systems,", size=10)
    _write(page, (80, 172), "covering order execution and balance management.", size=10)
    _write(page, (72, 220), "Next block.", size=10)
    doc.save(path)
    return path
