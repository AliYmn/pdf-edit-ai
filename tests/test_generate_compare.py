import shutil
from pathlib import Path

import pymupdf
import pytest

from conftest import text_of
from pdf_tools.compare import compare_pdfs, load_masks, parse_mask, ssim
from pdf_tools.edit import ChangeSet, apply_edits
from pdf_tools.generate import generate_pdf, parse_page_size
from pdf_tools.models import InputError
from samples import TURKISH, new_image

SYSTEM_TTF = next(
    (Path(p) for p in (
        "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "C:/Windows/Fonts/georgia.ttf",
    ) if Path(p).is_file()),
    None,
)  # fmt: skip

HTML = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 20mm; }}
body {{ font-family: Helvetica, Arial, sans-serif; font-size: 11pt; }}
.band {{ background: #1446a0; color: #fff; padding: 6px; }}
table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #555; padding: 4px; }}
.next {{ break-before: page; }}
.clip {{ width: 40px; height: 14px; overflow: hidden; white-space: nowrap; }}
</style></head><body>
<div class="band">Header band</div>
<img src="img/logo.png" style="height: 30px">
<img src="https://example.com/tracker.png">
<img src="../outside.png">
<h1>Rapor</h1><p>{TURKISH}</p>
<table><tr><th>Bölge</th><th>Q1</th></tr><tr><td>Kuzey</td><td>120</td></tr></table>
<div class="clip">This sentence is clipped by its box</div>
<section class="next"><h1>Second page</h1></section>
<script>document.body.append('JS-RAN')</script>
</body></html>"""


@pytest.fixture
def site(tmp_path):
    root = tmp_path / "site dir"
    (root / "img").mkdir(parents=True)
    new_image(root / "img" / "logo.png")
    new_image(tmp_path / "outside.png", color=(255, 0, 0))
    (root / "doc.html").write_text(HTML, encoding="utf-8")
    return root


def test_generate_basic_and_security(site, tmp_path):
    out = tmp_path / "out file.pdf"
    res = generate_pdf(site / "doc.html", out)
    assert res["pages"] == 2
    w, h = res["page_sizes_pt"][0]
    assert w == pytest.approx(595.28, abs=0.5) and h == pytest.approx(841.89, abs=0.5)
    text = text_of(out)
    assert TURKISH in text and "Kuzey" in text and "Second page" in text
    assert "JS-RAN" not in text  # scripts disabled
    assert any("example.com" in w for w in res["warnings"])
    assert any("outside.png" in w for w in res["warnings"])  # not served from outside the root
    assert any("clipped" in w for w in res["warnings"])
    with pymupdf.open(out) as doc:
        assert doc[0].get_image_info()  # local logo embedded
        band = doc[0].get_pixmap(clip=pymupdf.Rect(100, 70, 101, 71)).pixel(0, 0)
        assert max(abs(a - b) for a, b in zip(band, (20, 70, 160))) <= 2  # background colors printed


def test_generate_sizes_header_footer(site, tmp_path, pdfs):
    footer = tmp_path / "footer.html"
    footer.write_text('<div style="font-size:8pt;width:100%;text-align:center">Sayfa <span class="pageNumber"></span>'
                      ' / <span class="totalPages"></span></div>', encoding="utf-8")  # fmt: skip
    res = generate_pdf(site / "doc.html", tmp_path / "a.pdf", size_from=pdfs["mixed_sizes"], size_from_page=2,
                       margin="15mm", footer_template=footer)  # fmt: skip
    w, h = res["page_sizes_pt"][0]
    assert w == pytest.approx(792, abs=0.5) and h == pytest.approx(612, abs=0.5)
    assert "Sayfa 1 / " in text_of(tmp_path / "a.pdf", 1)
    res = generate_pdf(site / "doc.html", tmp_path / "b.pdf", page_size="A5-landscape")
    w, h = res["page_sizes_pt"][0]
    assert w == pytest.approx(595.28, abs=0.5) and h == pytest.approx(419.53, abs=0.5)


@pytest.mark.skipif(SYSTEM_TTF is None, reason="no system TTF available")
def test_generate_custom_font(tmp_path):
    shutil.copy(SYSTEM_TTF, tmp_path / "brand.ttf")
    (tmp_path / "f.html").write_text(
        '<style>@font-face{font-family:Brand;src:url("brand.ttf")} body{font-family:Brand}</style><p>Özel yazı tipi</p>',
        encoding="utf-8",
    )
    res = generate_pdf(tmp_path / "f.html", tmp_path / "f.pdf")
    family = SYSTEM_TTF.stem.split("Serif")[0].lower()
    assert any(family in f.lower() for f in res["fonts"]), res["fonts"]
    assert "Özel yazı tipi" in text_of(tmp_path / "f.pdf")


def test_generate_validation(site, tmp_path):
    out = tmp_path / "x.pdf"
    out.write_bytes(b"existing")
    with pytest.raises(InputError, match="already exists"):
        generate_pdf(site / "doc.html", out)
    assert out.read_bytes() == b"existing"
    with pytest.raises(InputError):
        generate_pdf(site / "missing.html", tmp_path / "y.pdf")
    with pytest.raises(InputError):
        parse_page_size("B9-huge")
    assert parse_page_size("210x297mm") == pytest.approx((595.28, 841.89), abs=0.01)


# ------------------------------------------------------------------ compare


def test_compare_identical(pdfs, tmp_path):
    rep = compare_pdfs(pdfs["multipage"], pdfs["multipage"], tmp_path / "cmp")
    assert rep.status == "ok"
    assert all(p.ssim == pytest.approx(1.0) and p.changed_pixels == 0 for p in rep.pages)
    assert Path(rep.pages[0].side_by_side).is_file() and Path(rep.pages[0].diff_image).is_file()
    assert (tmp_path / "cmp" / "report.json").is_file()


def test_compare_detects_edits_and_masks_them(pdfs, tmp_path):
    cs = ChangeSet.model_validate({"operations": [{"op": "replace_text", "page": 1, "search": "March 2024", "text": "June 2026"}]})
    edit_rep = apply_edits(pdfs["multipage"], cs, tmp_path / "e.pdf")
    rep = compare_pdfs(pdfs["multipage"], tmp_path / "e.pdf", tmp_path / "c1")
    p1 = rep.pages[0]
    assert rep.status == "review" and p1.changed_pixels > 0
    assert "March 2024" in rep.document_missing_text and "June 2026" in rep.document_added_text
    region = pymupdf.Rect(p1.changed_regions[0])
    assert region.intersects(pymupdf.Rect(edit_rep.masks[0]["rect"]))
    assert rep.pages[1].changed_pixels == 0

    masks_file = tmp_path / "report.json"
    masks_file.write_text(edit_rep.model_dump_json(), encoding="utf-8")
    rep = compare_pdfs(pdfs["multipage"], tmp_path / "e.pdf", tmp_path / "c2", masks=load_masks(masks_file),
                       expect=["June 2026"], forbid=["March 2024 (page 1)"])  # fmt: skip
    assert rep.pages[0].changed_pixels == 0 and rep.pages[0].masked_regions == 1
    assert rep.status == "ok" and all(e.ok for e in rep.expectations)


def test_compare_page_count_size_and_missing_content(pdfs, tmp_path):
    rep = compare_pdfs(pdfs["multipage"], pdfs["mixed_sizes"], tmp_path / "c")
    assert rep.status == "review"
    assert any("page size differs" in w for w in rep.pages[1].warnings)
    assert rep.document_missing_text
    two = tmp_path / "two.pdf"
    with pymupdf.open(pdfs["multipage"]) as doc:
        doc.select([0, 1])
        doc.save(two)
    rep = compare_pdfs(pdfs["multipage"], two, tmp_path / "d")
    assert any("missing 1 page" in w for w in rep.warnings)
    assert "page missing in candidate" in rep.pages[2].warnings
    assert any("Section 3" in t for t in rep.document_missing_text)


def test_compare_overflow_warnings(pdfs, tmp_path):
    bad = tmp_path / "bad.pdf"
    with pymupdf.open(pdfs["multipage"]) as doc:
        page = doc[0]
        page.insert_text((540, 300), "runs past the right edge of the page", fontsize=11)
        page.insert_text((74, 152), "OVERLAPPING", fontsize=11)
        doc.save(bad)
    rep = compare_pdfs(pdfs["multipage"], bad, tmp_path / "c")
    w = " ".join(rep.pages[0].warnings)
    assert "beyond the page edge" in w and "overlapping text" in w and "reference content area" in w
    assert any(e.ok is False for e in compare_pdfs(pdfs["multipage"], bad, tmp_path / "c2", expect=["nonexistent"]).expectations)


def test_masks_and_ssim_helpers():
    m = parse_mask("2:10,20,30.5,40")
    assert m.page == 2 and m.rect == (10, 20, 30.5, 40)
    with pytest.raises(InputError):
        parse_mask("2:10,20")
    import numpy as np

    a = np.full((50, 50), 200, np.uint8)
    assert ssim(a, a) == pytest.approx(1.0)
    b = a.copy()
    b[10:40, 10:40] = 0
    assert ssim(a, b) < 0.8
