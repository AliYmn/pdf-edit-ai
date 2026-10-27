import json

import pytest
from PIL import Image

from pdf_tools.assets import extract_assets
from pdf_tools.common import parse_pages
from pdf_tools.inspect import inspect_pdf
from pdf_tools.models import InputError
from pdf_tools.render import render_pdf
from samples import TURKISH


def test_inspect_multipage_structure(pdfs):
    doc = inspect_pdf(pdfs["multipage"])
    assert doc.meta.page_count == 3
    assert doc.meta.text_layer == "full"
    assert doc.meta.metadata["title"] == "Sample Report"
    page = doc.pages[0]
    assert (page.width, page.height) == (595.28, 841.89)
    assert page.background == "#ffffff"
    texts = [e for e in page.elements if e.type == "text"]
    heading = next(e for e in texts if "Section 1" in e.text)
    assert heading.provenance == "extracted"
    assert heading.typography.size == 18 and heading.typography.bold
    assert heading.lines[0].spans[0].origin[1] == pytest.approx(90, abs=0.1)
    footer = next(e for e in texts if e.text.startswith("Page 1 of 3"))
    assert footer.typography.color == "#666666"
    orders = [e.reading_order for e in page.elements if e.reading_order]
    assert orders == sorted(orders) and heading.reading_order < footer.reading_order


def test_inspect_fonts_are_linked_to_spans(pdfs):
    doc = inspect_pdf(pdfs["multipage"])
    by_name = {f.name: f for f in doc.fonts}
    assert by_name["Helvetica"].embedded and 11.0 in by_name["Helvetica"].sizes
    assert by_name["Helvetica-Bold"].chars > 0


def test_inspect_table(pdfs):
    doc = inspect_pdf(pdfs["table"])
    tables = [e for e in doc.pages[0].elements if e.type == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert t.provenance == "inferred"
    assert (t.table.rows, t.table.cols) == (4, 3)
    assert t.table.cells[1] == ["North", "120", "135"]
    shapes = [e for e in doc.pages[0].elements if e.type == "shape"]
    fills = [bytes.fromhex(s.fill[1:]) for s in shapes if s.fill]
    assert any(max(abs(a - b) for a, b in zip(f, (230, 237, 255))) <= 2 for f in fills)  # header fill (0.9, 0.93, 1)


def test_inspect_images_and_page_selection(pdfs):
    doc = inspect_pdf(pdfs["images"], pages="2")
    assert [p.number for p in doc.pages] == [2]
    imgs = [e for e in doc.pages[0].elements if e.type == "image"]
    assert len(imgs) == 1 and imgs[0].asset_id.startswith("img-")
    assert imgs[0].image_size == (240, 80)
    assert not doc.pages[0].needs_ocr  # short title + logo is not a scan


def test_inspect_turkish_text(pdfs):
    doc = inspect_pdf(pdfs["turkish"])
    assert any(e.text == TURKISH for e in doc.pages[0].elements if e.type == "text")


def test_inspect_mixed_page_sizes(pdfs):
    doc = inspect_pdf(pdfs["mixed_sizes"])
    assert [(p.width, p.height) for p in doc.pages] == [(595.28, 841.89), (792, 612), (400, 400)]


def test_inspect_scanned_needs_ocr(pdfs):
    doc = inspect_pdf(pdfs["scanned"])
    assert doc.meta.text_layer == "none"
    assert doc.pages[0].needs_ocr and doc.pages[0].image_coverage > 0.95
    assert any("OCR" in w for w in doc.pages[0].warnings)


def test_inspect_json_roundtrip(pdfs, tmp_path):
    out = tmp_path / "a.json"
    out.write_text(inspect_pdf(pdfs["table"]).model_dump_json(), encoding="utf-8")
    data = json.loads(out.read_text())
    assert data["schema_version"] == "1.0" and data["pages"][0]["elements"]


def test_inspect_errors(tmp_path):
    with pytest.raises(InputError, match="not found"):
        inspect_pdf(tmp_path / "missing.pdf")
    bad = tmp_path / "bad.pdf"
    bad.write_text("not a pdf")
    with pytest.raises(InputError):
        inspect_pdf(bad)


def test_parse_pages():
    assert parse_pages("1,3-4", 5) == [0, 2, 3]
    assert parse_pages(None, 2) == [0, 1]
    for bad in ("0", "6", "2-1", "x"):
        with pytest.raises(InputError):
            parse_pages(bad, 5)


def test_render_dpi_and_selection(pdfs, tmp_path):
    res = render_pdf(pdfs["mixed_sizes"], tmp_path / "out dir", dpi=72, pages="2-3")
    assert [r["page"] for r in res] == [2, 3]
    with Image.open(res[0]["file"]) as im:
        assert im.size == (792, 612)
    res = render_pdf(pdfs["multipage"], tmp_path / "hi", dpi=144, pages="1")
    with Image.open(res[0]["file"]) as im:
        assert im.size == (1191, 1684)
    with pytest.raises(InputError):
        render_pdf(pdfs["multipage"], tmp_path / "x", dpi=5)


def test_assets(pdfs, tmp_path):
    m = extract_assets(pdfs["images"], tmp_path / "assets")
    images = {a.id: a for a in m.assets if a.kind == "image"}
    logo = next(a for a in images.values() if a.width == 240)
    assert logo.role == "logo-candidate" and logo.pages == [1, 2] and len(logo.placements) == 2
    assert (tmp_path / "assets" / logo.file).is_file() and logo.format == "png"
    photo = next(a for a in images.values() if a.width == 320)
    assert photo.format == "jpeg" and photo.role == "image"
    vectors = [a for a in m.assets if a.kind == "vector"]
    assert vectors and all((tmp_path / "assets" / v.file).read_text().startswith("<svg") for v in vectors)
    fonts = [a for a in m.assets if a.kind == "font"]
    assert {f.font_name for f in fonts} >= {"Helvetica", "Helvetica-Bold"}
    assert json.loads((tmp_path / "assets" / "manifest.json").read_text())["assets"]
