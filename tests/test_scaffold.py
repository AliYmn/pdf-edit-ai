import json
from pathlib import Path

import pymupdf
import pytest

from pdf_tools.assets import extract_assets
from pdf_tools.compare import compare_pdfs
from pdf_tools.inspect import inspect_pdf
from pdf_tools.scaffold import scaffold
from samples import TURKISH


@pytest.fixture(scope="module")
def linked(tmp_path_factory, pdfs):
    """The image sample plus links: one on a text span, one on a picture, one internal."""
    d = tmp_path_factory.mktemp("linked")
    src = d / "linked.pdf"
    with pymupdf.open(pdfs["images"]) as doc:
        page = doc[0]
        page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(72, 96, 250, 114), "uri": "https://example.com/a"})
        for piece in (pymupdf.Rect(72, 96, 150, 114), pymupdf.Rect(150, 96, 250, 114)):  # word-piece duplicates
            page.insert_link({"kind": pymupdf.LINK_URI, "from": piece, "uri": "https://example.com/a"})
        page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(72, 140, 392, 340), "uri": "https://example.com/photo"})
        page.insert_link({"kind": pymupdf.LINK_GOTO, "from": pymupdf.Rect(72, 350, 300, 365), "page": 1})
        doc[1].draw_circle((540, 60), 10, color=(0.8, 0.1, 0.1), fill=(0.9, 0.2, 0.2))  # same icon twice on page 2
        doc[1].draw_circle((540, 200), 10, color=(0.8, 0.1, 0.1), fill=(0.9, 0.2, 0.2))
        doc.save(src)
    return src


def test_inspect_lists_links_without_duplicates(linked):
    links = inspect_pdf(linked).pages[0].links
    assert [(ln.uri, ln.target_page) for ln in links] == [
        ("https://example.com/a", None), ("https://example.com/photo", None), (None, 2),
    ]  # fmt: skip


def test_assets_export_small_vector_icons_deduplicated(linked, tmp_path):
    m = extract_assets(linked, tmp_path / "a")
    vectors = [a for a in m.assets if a.kind == "vector"]
    assert vectors
    circle2 = next(v for v in vectors if v.pages == [2] and len(v.placements) == 2)  # identical icons merged
    svg = (tmp_path / "a" / circle2.file).read_text()
    assert svg.startswith("<svg") and "<path" in svg and "<text" not in svg
    assert (tmp_path / "a" / circle2.file).with_suffix(".png").is_file()


def test_scaffold_replicates_document(linked, tmp_path):
    rep = scaffold(linked, tmp_path / "s")
    cal = rep["calibration"]
    assert cal["unmatched_segments"] == 0
    assert cal["baseline_error_pt"]["max"] <= 0.38  # Chromium puts baselines on a 0.75pt (1 CSS px) grid
    assert cal["glyph_x_error_pt"]["max"] <= 2
    html = Path(rep["html"]).read_text(encoding="utf-8")
    assert 'href="https://example.com/a"' in html and 'href="#page-2"' in html
    assert all(f["source"] for f in rep["fonts"].values()) and '"Helvetica"' in html  # CFF fonts -> metric-compatible system font
    assert html.count("<img ") == 3  # logo x2, photo x1
    cmp = compare_pdfs(linked, rep["preview_pdf"], tmp_path / "c")
    assert cmp.document_missing_text == [] and cmp.document_added_text == []
    assert all(p.text_similarity == 1.0 and not p.moved_blocks and not p.warnings for p in cmp.pages)
    with pymupdf.open(rep["preview_pdf"]) as out:
        uris = {ln.get("uri") for ln in out[0].get_links()}
        assert {"https://example.com/a", "https://example.com/photo"} <= uris
    assert json.loads((tmp_path / "s" / "scaffold.json").read_text())["pages"] == 2


def test_scaffold_page_sizes_turkish_and_paragraph_blocks(pdfs, tmp_path):
    rep = scaffold(pdfs["mixed_sizes"], tmp_path / "m")
    with pymupdf.open(rep["preview_pdf"]) as out:
        sizes = [(round(p.rect.width), round(p.rect.height)) for p in out]
    assert sizes == [(595, 842), (792, 612), (400, 400)]
    rep = scaffold(pdfs["turkish"], tmp_path / "t")
    with pymupdf.open(rep["preview_pdf"]) as out:
        assert TURKISH in out[0].get_text()
    rep = scaffold(pdfs["paragraph"], tmp_path / "p")
    assert rep["elements"]["paragraph_blocks"] == 1
    assert 'data-flow="left:' in Path(rep["html"]).read_text(encoding="utf-8")


def test_flow_recipe_rewrites_paragraph_on_original_baseline(pdfs, tmp_path):
    import re

    from pdf_tools.generate import generate_pdf

    rep = scaffold(pdfs["paragraph"], tmp_path / "p")
    html = Path(rep["html"]).read_text(encoding="utf-8")
    m = re.search(r'<div class="block" id="[^"]+" data-class="([^"]+)" data-flow="([^"]+)"[^>]*>.*?\n  </div>', html, re.S)
    assert m and rep["calibration"]["flow_recipes"]["flow_reproduces_line_breaks"] == 1
    new_html = html.replace(m.group(0), f'<p class="flow {m.group(1)}" style="{m.group(2)}">A much shorter summary.</p>')
    (tmp_path / "p" / "edited.html").write_text(new_html, encoding="utf-8")
    generate_pdf(tmp_path / "p" / "edited.html", tmp_path / "p" / "edited.pdf")
    def first_line(path, needle):
        with pymupdf.open(path) as doc:
            return next(ln["spans"][0]["origin"] for b in doc[0].get_text("dict")["blocks"] for ln in b.get("lines", [])
                        if needle in "".join(s["text"] for s in ln["spans"]))  # fmt: skip
    x0, y0 = first_line(pdfs["paragraph"], "This quarterly report")
    x1, y1 = first_line(tmp_path / "p" / "edited.pdf", "A much shorter")
    assert abs(x1 - x0) < 0.05 and abs(y1 - y0) <= 0.38
