"""Regression tests for layout, font and image fixes found on real documents
(manuals, CVs): justified lines, columns, substitute fonts, image verification,
print swaps on photos and document-wide replacements."""

import io
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw

from conftest import text_of
from pdf_tools.compare import Mask, compare_pdfs
from pdf_tools.edit import ChangeSet, EditError, apply_edits, logical_lines
from pdf_tools.fonts import font_class, is_condensed, resolve_font, system_font
from pdf_tools.printswap import DEFAULT_FONTS
from samples import _write, new_image

SUPP = Path("/System/Library/Fonts/Supplemental")


def edit(src, tmp_path, ops, name="out.pdf"):
    return apply_edits(src, ChangeSet.model_validate({"operations": ops}), tmp_path / name)


def line_right(path, needle):
    with pymupdf.open(path) as doc:
        for ln in logical_lines(doc[0]):
            text = "".join(c["c"] for c, _ in ln["chars"])
            if needle in text:
                return max(c["bbox"][2] for c, _ in ln["chars"] if not c["c"].isspace())
    raise AssertionError(f"{needle!r} not found")


# ---------------------------------------------------------------- text layout


def test_justified_line_keeps_right_edge(tmp_path):
    """Words positioned one by one across a fixed measure."""
    src = tmp_path / "justified.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    font, size, x0, x1 = pymupdf.Font("helv"), 10, 72.0, 320.0
    rows = ["Battery maintenance requires regular checks of",
            "each terminal and the electrolyte level steadily",
            "so that capacity and cranking performance stay",
            "Short last line."]  # fmt: skip
    for k, row in enumerate(rows):
        words, y = row.split(), 100 + k * 13
        gap = (x1 - x0 - sum(font.text_length(w, size) for w in words)) / (len(words) - 1)
        assert gap > font.text_length(" ", size)
        if k == len(rows) - 1:
            gap = font.text_length(" ", size)
        tw, x = pymupdf.TextWriter(page.rect), x0
        for w in words:  # word, a real space glyph, then the next word positioned (as Word writes it)
            tw.append((x, y), w + " ", font=font, fontsize=size)
            x += font.text_length(w, size) + gap
        tw.write_text(page)
    doc.save(src)
    before = line_right(src, "steadily")
    rep = edit(src, tmp_path, [{"op": "replace_text", "page": 1, "search": "steadily", "text": "daily"}])
    assert rep.ok and rep.results[0].verified
    out = tmp_path / "out.pdf"
    assert line_right(out, "electrolyte level daily") == pytest.approx(before, abs=0.6)
    assert line_right(out, "Short last line") < 200  # untouched


def test_columns_on_one_baseline_are_not_merged(tmp_path):
    src = tmp_path / "columns.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    _write(page, (72, 100), "Left column text", size=10)
    _write(page, (300, 100), "Right column text", size=10)
    tw = pymupdf.TextWriter(page.rect)  # a label and its value 2pt apart do form one line
    _, end = tw.append((72, 130), "Name:", font=pymupdf.Font("hebo"), fontsize=10)
    tw.append((end.x + 2, end.y), "Jane", font=pymupdf.Font("helv"), fontsize=10)
    tw.write_text(page)
    doc.save(src)
    with pymupdf.open(src) as d:
        lines = ["".join(c["c"] for c, _ in ln["chars"]).strip() for ln in logical_lines(d[0])]
    assert "Left column text" in lines and "Right column text" in lines
    assert any(t.replace(" ", "") == "Name:Jane" for t in lines)
    rep = edit(src, tmp_path, [{"op": "replace_text", "page": 1, "search": "Left column text", "text": "Left"}])
    assert rep.ok and "Right column text" in text_of(tmp_path / "out.pdf")


# ---------------------------------------------------------------- fonts


def test_font_class_and_condensed_names():
    assert font_class("swiss721", serif=True) == "sans"  # known family wins over a wrong serif flag
    assert font_class("helveticaroman") == "sans"
    assert font_class("timesnewroman") == "serif"
    assert font_class("couriernew") == "mono"
    assert is_condensed("Swiss721BT-BlackCondensed") and is_condensed("ArialNarrow")
    assert not is_condensed("Arial-BoldMT")


@pytest.mark.skipif(system_font("arialnarrow", True, False) is None, reason="Arial Narrow not installed")
def test_condensed_sans_falls_back_to_arial_narrow():
    doc = pymupdf.open()
    rf = resolve_font(doc, doc.new_page(), "TEST", original="FooSansCondensed-Bold")
    assert rf.source == "system" and "narrow" in rf.font.name.lower()


def _embedded_doc(fontfile: Path) -> pymupdf.Document:
    """The same font embedded twice, as separate objects: page 1 holds a subset whose glyphs
    cannot be reused (no cmap), page 2 the full program."""
    out = pymupdf.open()
    for subset in (True, False):
        d = pymupdf.open()
        p = d.new_page()
        p.insert_font(fontname="F1", fontfile=str(fontfile))
        p.insert_text((72, 100), "ABC", fontname="F1", fontsize=12)
        if subset:
            d.subset_fonts()
        out.insert_pdf(pymupdf.open("pdf", d.tobytes()))
    return pymupdf.open("pdf", out.tobytes())


@pytest.mark.skipif(not (SUPP / "Arial Narrow Bold.ttf").is_file(), reason="Arial Narrow Bold not installed")
def test_truncated_name_and_glyphs_from_another_subset():
    doc = _embedded_doc(SUPP / "Arial Narrow Bold.ttf")
    span_font = doc[1].get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["font"]
    rf = resolve_font(doc, doc[1], "CAB", original=span_font[:-1])  # producers cut long names short
    assert rf.source == "original-embedded"
    rf = resolve_font(doc, doc[0], "ZAX", original=span_font)  # page 1's own subset is unusable
    assert rf.source == "original-embedded" and any("another embedded subset" in w for w in rf.warnings)


@pytest.mark.skipif(system_font("arial", True, False) is None, reason="Arial Bold not installed")
def test_substituted_black_face_is_emboldened(tmp_path):
    src = tmp_path / "black.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Heading text", fontname="helv", fontsize=20)
    doc.xref_set_key(page.get_fonts()[0][0], "BaseFont", "/Arial-Black")  # not embedded
    doc.save(src)
    rep = edit(src, tmp_path, [{"op": "replace_text", "page": 1, "search": "Heading", "text": "Title"}])
    assert rep.ok and any("emboldened" in w for w in rep.results[0].warnings)


# ---------------------------------------------------------------- compare and images


def _inline_image_pdf(path: Path, extra_overlap: bool = False) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    _write(page, (72, 300), "Overlapping words", size=11)
    _write(page, (74, 303), "Overlapping words", size=11)  # present in the reference too
    if extra_overlap:
        _write(page, (72, 500), "Spilled line here", size=11)
        _write(page, (73, 502), "Spilled line here", size=11)
    page.clean_contents()
    xref = page.get_contents()[0]
    inline = b"\nq 100 0 0 50 72 92 cm BI /W 2 /H 2 /CS /RGB /BPC 8 /F /AHx ID ff000000ff000000ffffff00> EI Q\n"
    doc.update_stream(xref, doc.xref_stream(xref) + inline)
    doc.save(path)
    return path


def test_compare_flags_inline_images_and_reports_only_new_warnings(tmp_path):
    ref = _inline_image_pdf(tmp_path / "ref.pdf")
    same = compare_pdfs(ref, ref, tmp_path / "c1")
    page = same.pages[0]
    assert same.status == "ok" and not page.warnings
    assert len(page.images) == 1 and page.images[0].inline and page.images[0].status in ("identical", "same_pixels")
    worse = compare_pdfs(ref, _inline_image_pdf(tmp_path / "cand.pdf", extra_overlap=True), tmp_path / "c2")
    assert worse.status == "review" and any("overlapping" in w for w in worse.pages[0].warnings)


def test_insert_image_is_verified_and_masked(pdfs, tmp_path):
    logo = new_image(tmp_path / "logo.png")
    rep = edit(pdfs["multipage"], tmp_path, [{"op": "insert_image", "page": 1, "rect": [400, 60, 520, 100], "image": str(logo)}])
    assert rep.ok and rep.results[0].verified and rep.masks
    cmp = compare_pdfs(pdfs["multipage"], tmp_path / "out.pdf", tmp_path / "cmp", masks=None)
    assert cmp.status == "review"  # unmasked: the new image is reported
    masked = compare_pdfs(pdfs["multipage"], tmp_path / "out.pdf", tmp_path / "cmp2",
                          masks=[Mask.model_validate(m) for m in rep.masks])  # fmt: skip
    assert masked.status == "ok" and any(i.status == "added" and i.masked for i in masked.pages[0].images)


# ---------------------------------------------------------------- print swap


def _bold_font():
    return next((p for p in DEFAULT_FONTS if Path(p).is_file()), None)


ORANGE, NEIGHBOUR = (240, 130, 20), (235, 235, 235)


def _product_photo(path: Path) -> tuple[Path, list]:
    """Dark battery casing: sheared orange brand print with a white spec line below it."""
    from PIL import ImageFont

    w, h = 420, 220
    img = Image.new("RGB", (w, h), (28, 30, 34))
    tile = Image.new("L", (300, 90), 0)
    ImageDraw.Draw(tile).text((10, 5), "YUASA", font=ImageFont.truetype(_bold_font(), 64), fill=255)
    tile = tile.transform((300, 110), Image.AFFINE, (1, 0, 0, -0.12, 1, 0), resample=Image.BICUBIC)  # baseline slopes
    img.paste(Image.new("RGB", tile.size, ORANGE), (60, 30), tile)
    ImageDraw.Draw(img).text((70, 160), "12V 7Ah", font=ImageFont.truetype(_bold_font(), 28), fill=NEIGHBOUR)
    img.save(path, quality=95, subsampling=0)
    return path, [(55, 25), (370, 25), (370, 150), (55, 150)]


@pytest.mark.skipif(_bold_font() is None, reason="no bold sans font for print text")
def test_swap_print_on_product_photo(tmp_path):
    photo, poly = _product_photo(tmp_path / "photo.jpg")
    src = tmp_path / "manual.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(72, 72, 492, 292), filename=str(photo))
    doc.save(src)
    rep = edit(src, tmp_path, [{"op": "swap_print", "page": 1, "rect": [100, 100, 400, 250],
                                "prints": [{"polygon": poly, "text": "UPLUS", "old": "YUASA"}]}])  # fmt: skip
    r = rep.results[0]
    assert rep.ok and r.verified and rep.masks, r.warnings
    ink = r.message.split("ink #")[1][:6]
    assert all(abs(int(ink[2 * k : 2 * k + 2], 16) - ORANGE[k]) < 30 for k in range(3))  # same ink colour
    with pymupdf.open(tmp_path / "out.pdf") as out:
        xref = out[0].get_images()[0][0]
        after = np.asarray(Image.open(io.BytesIO(out.extract_image(xref)["image"])).convert("RGB")).astype(int)
    before = np.asarray(Image.open(photo).convert("RGB")).astype(int)
    assert np.abs(after[155:200] - before[155:200]).mean() < 1.5  # spec line below is untouched
    assert np.abs(after[30:140] - before[30:140]).mean() > 3  # the brand print changed
    orange = (after[30:140, :, 0] > 180) & (after[30:140, :, 2] < 90)
    assert orange.sum() > 500  # new print in the brand colour


def test_swap_print_validation(pdfs, tmp_path):
    with pytest.raises(Exception, match="exactly one of 'text' or 'logo'"):
        ChangeSet.model_validate({"operations": [{"op": "swap_print", "page": 1, "xref": 5,
                                                  "prints": [{"polygon": [[0, 0], [1, 0], [1, 1]]}]}]})  # fmt: skip
    with pytest.raises(EditError, match="no image overlaps"):
        edit(pdfs["multipage"], tmp_path, [{"op": "swap_print", "page": 1, "rect": [0, 0, 50, 50],
                                            "prints": [{"polygon": [[0, 0], [9, 0], [9, 9]], "text": "A", "old": "B"}]}])  # fmt: skip
    assert not (tmp_path / "out.pdf").exists()


# ---------------------------------------------------------------- replace_all


def test_replace_all_across_pages_preserves_case(tmp_path):
    src = tmp_path / "brand.pdf"
    doc = pymupdf.open()
    for n in range(3):
        page = doc.new_page()
        _write(page, (72, 100), "YUASA battery manual", size=14, font="hebo")
        _write(page, (72, 130), f"Yuasa and yuasa are on line two of page {n + 1}.", size=10)
        _write(page, (72, 160), "Contact YUASA service.", size=10)
    doc.save(src)
    rep = edit(src, tmp_path, [{"op": "replace_all", "pages": "1-2", "search": "yuasa", "text": "Uplus"}])
    r = rep.results[0]
    assert rep.ok and r.verified and "replaced 8" in r.message, r.message
    out = tmp_path / "out.pdf"
    for p in (1, 2):
        t = text_of(out, p)
        assert "UPLUS battery manual" in t and "Uplus and uplus are on line two" in t and "Contact UPLUS service." in t
        assert "yuasa" not in t.lower()
    assert "YUASA battery manual" in text_of(out, 3)  # outside the range
    assert {m["page"] for m in rep.masks} == {1, 2}
    with pytest.raises(Exception, match="contains the search text"):
        ChangeSet.model_validate({"operations": [{"op": "replace_all", "search": "Uplus", "text": "UPLUS Pro"}]})
