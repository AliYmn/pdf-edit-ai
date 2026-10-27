import json

import pymupdf
import pytest

from conftest import squash, text_of
from pdf_tools.edit import ChangeSet, EditError, apply_edits, load_changes
from pdf_tools.models import InputError
from samples import new_image


def edit(src, tmp_path, ops, name="out.pdf", **kw):
    return apply_edits(src, ChangeSet.model_validate({"operations": ops}), tmp_path / name, **kw)


def line_with(path, page, needle):
    with pymupdf.open(path) as doc:
        for b in doc[page - 1].get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                text = "".join(s["text"] for s in ln["spans"])
                if needle in text:
                    return ln
    return None


def test_replace_same_line_keeps_baseline_and_font(pdfs, tmp_path):
    rep = edit(pdfs["multipage"], tmp_path, [{"op": "replace_text", "page": 1, "search": "March 2024", "text": "April 2025"}])
    r = rep.results[0]
    assert rep.ok and r.verified and r.font_source == "original-embedded"
    out = tmp_path / "out.pdf"
    assert "Report date: April 2025" in text_of(out, 1)
    assert "March 2024" in text_of(out, 2)  # other pages untouched
    before = line_with(pdfs["multipage"], 1, "Report date")
    after_spans = [s for b in pymupdf.open(out)[0].get_text("dict")["blocks"] for ln in b.get("lines", [])
                   for s in ln["spans"] if "April" in s["text"]]  # fmt: skip
    assert after_spans[0]["origin"][1] == pytest.approx(before["spans"][0]["origin"][1], abs=0.01)
    assert after_spans[0]["size"] == pytest.approx(11)


def test_replace_shifts_rest_of_line(pdfs, tmp_path):
    rep = edit(pdfs["multipage"], tmp_path, [
        {"op": "replace_text", "page": 1, "search": "Report", "occurrence": 2, "text": "Final report"},
        {"op": "replace_text", "page": 2, "search": "grew steadily", "occurrence": 2, "text": "grew"},
    ])  # fmt: skip
    assert rep.ok and all(r.verified for r in rep.results)
    out = tmp_path / "out.pdf"
    assert "Final report date: March 2024" in text_of(out, 1)
    t2 = text_of(out, 2)
    assert "Paragraph 2 on page 2. Revenue grew across all regions." in t2
    assert t2.count("grew steadily") == 3


def test_replace_overflow_error_and_shrink(pdfs, tmp_path):
    # paragraph lines are limited to their block width: any growth fails
    ops = [{"op": "replace_text", "page": 1, "search": "steadily", "occurrence": 1, "text": "steadily and strongly"}]
    with pytest.raises(EditError, match="does not fit") as exc:
        edit(pdfs["multipage"], tmp_path, ops)
    assert not (tmp_path / "out.pdf").exists()
    assert exc.value.report.results[0].status == "failed"
    # a single-line block may grow up to the page's content margin (72pt here)
    long = "March 2024 and the following eleven months of the fiscal year ending in late February"
    ops = [{"op": "replace_text", "page": 1, "search": "March 2024", "text": long}]
    with pytest.raises(EditError, match="does not fit"):
        edit(pdfs["multipage"], tmp_path, ops)
    ops[0] |= {"overflow": "shrink", "min_scale": 0.7}
    rep = edit(pdfs["multipage"], tmp_path, ops, name="shrunk.pdf")
    assert rep.ok and rep.results[0].verified and 0.7 <= rep.results[0].scale < 1
    ops[0] |= {"overflow": "allow"}
    rep = edit(pdfs["multipage"], tmp_path, ops, name="allowed.pdf")
    assert rep.ok and any("overlap" in w for w in rep.results[0].warnings)


def test_replace_turkish_text(pdfs, tmp_path):
    rep = edit(pdfs["turkish"], tmp_path, [{"op": "replace_text", "page": 1, "search": "15 Mart 2024", "text": "22 Ağustos 2025 Çarşamba"}])
    assert rep.ok and rep.results[0].verified
    assert "Tarih: 22 Ağustos 2025 Çarşamba" in text_of(tmp_path / "out.pdf")


def test_replace_paragraph_rect_reflows_on_original_baseline(pdfs, tmp_path):
    new = "Yeni paragraf: şirketin büyüme stratejisi ikinci yarıda güncellenmiştir. " * 3
    rep = edit(pdfs["paragraph"], tmp_path, [{"op": "replace_text", "page": 1, "rect": [72, 95, 523, 200], "text": new}])
    assert rep.ok and rep.results[0].verified
    out = tmp_path / "out.pdf"
    assert squash(new) in squash(text_of(out))
    assert "quarterly report" not in text_of(out)
    assert "Next section starts here." in text_of(out)
    first_before = line_with(pdfs["paragraph"], 1, "This quarterly")["spans"][0]["origin"][1]
    first_after = line_with(out, 1, "Yeni paragraf")["spans"][0]["origin"][1]
    assert first_after == pytest.approx(first_before, abs=0.6)


def test_replace_paragraph_overflow_fails_without_output(pdfs, tmp_path):
    ops = [{"op": "replace_text", "page": 1, "rect": [72, 95, 523, 140], "text": "word " * 400}]
    with pytest.raises(EditError, match="height"):
        edit(pdfs["paragraph"], tmp_path, ops)
    assert not (tmp_path / "out.pdf").exists()


def test_redaction_removes_text_from_content(pdfs, tmp_path):
    rep = edit(pdfs["multipage"], tmp_path, [
        {"op": "redact", "page": 3, "search": "Page 3 of 3"},
        {"op": "redact", "page": 1, "rect": [70, 145, 530, 200], "fill": "#000000"},
    ])  # fmt: skip
    assert rep.ok and all(r.verified for r in rep.results)
    out = tmp_path / "out.pdf"
    assert "Page 3 of 3" not in text_of(out, 3)
    assert "Paragraph 1 on page 1" not in text_of(out, 1)
    assert "Section 1" in text_of(out, 1)
    pix = pymupdf.open(out)[0].get_pixmap(clip=pymupdf.Rect(100, 160, 110, 170))
    assert pix.pixel(2, 2) == (0, 0, 0)


def test_insert_text_point_and_box(pdfs, tmp_path):
    rep = edit(pdfs["multipage"], tmp_path, [
        {"op": "insert_text", "page": 1, "point": [72, 300], "text": "Onaylandı: Çağrı", "color": "#aa0000", "size": 12},
        {"op": "insert_text", "page": 1, "rect": [72, 320, 300, 380], "text": "Line one\nLine two", "align": "center"},
    ])  # fmt: skip
    assert rep.ok and all(r.verified for r in rep.results)
    t = text_of(tmp_path / "out.pdf", 1)
    assert "Onaylandı: Çağrı" in t and "Line one" in t and "Line two" in t


def test_replace_images(pdfs, tmp_path):
    img = new_image(tmp_path / "new.png", size=(240, 80))
    tall = new_image(tmp_path / "tall.png", color=(200, 0, 0), size=(100, 300))
    with pymupdf.open(pdfs["images"]) as doc:
        logo_xref = next(i["xref"] for i in doc[0].get_image_info(xrefs=True) if i["width"] == 240)
    rep = edit(pdfs["images"], tmp_path, [
        {"op": "replace_image", "page": 1, "asset_id": f"img-{logo_xref}", "image": str(img)},
        {"op": "replace_image", "page": 1, "rect": [100, 200, 300, 300], "image": str(tall)},
    ])  # fmt: skip
    assert rep.ok
    assert "shared" in rep.results[0].warnings[0]
    with pymupdf.open(tmp_path / "out.pdf") as doc:
        page = doc[0]
        assert page.rect.width == pytest.approx(595.28) and page.rect.height == pytest.approx(841.89)
        red = [i for i in page.get_image_info() if i["width"] == 100]
        assert red and red[0]["bbox"][3] - red[0]["bbox"][1] == pytest.approx(200, abs=1)  # fit inside, proportional
        logo_px = page.get_pixmap(clip=pymupdf.Rect(100, 50, 101, 51)).pixel(0, 0)
        assert logo_px == (10, 160, 90)


def test_invalid_changes(tmp_path, pdfs):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(InputError, match="valid JSON"):
        load_changes(bad)
    bad.write_text(json.dumps([{"op": "replace_text", "page": 1, "text": "x"}]))
    with pytest.raises(InputError, match="exactly one of 'search' or 'rect'"):
        load_changes(bad)
    bad.write_text(json.dumps([{"op": "rotate", "page": 1}]))
    with pytest.raises(InputError):
        load_changes(bad)
    with pytest.raises(EditError, match="does not exist"):
        edit(pdfs["table"], tmp_path, [{"op": "redact", "page": 9, "search": "x"}])
    with pytest.raises(EditError, match="not found"):
        edit(pdfs["table"], tmp_path, [{"op": "replace_text", "page": 1, "search": "nope", "text": "x"}])


def test_never_overwrites_input(pdfs, copy_of, tmp_path):
    src = copy_of(pdfs["table"])
    before = src.read_bytes()
    ops = ChangeSet.model_validate({"operations": [{"op": "replace_text", "page": 1, "search": "North", "text": "Kuzey"}]})
    with pytest.raises(InputError, match="overwrite"):
        apply_edits(src, ops, src)
    assert src.read_bytes() == before
    rep = apply_edits(src, ops, src, overwrite=True)
    assert rep.ok and "Kuzey" in text_of(src)


def test_table_cell_edit_respects_cell_border(pdfs, tmp_path):
    ops = [{"op": "replace_text", "page": 1, "search": "North", "text": "North-East Region Long Name"}]
    with pytest.raises(EditError, match="does not fit"):
        edit(pdfs["table"], tmp_path, ops)
    rep = edit(pdfs["table"], tmp_path, [{"op": "replace_text", "page": 1, "search": "North", "text": "Kuzey"}])
    assert rep.ok and "Kuzey" in text_of(tmp_path / "out.pdf")
