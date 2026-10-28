"""Detect images, update them, put them back, and prove nothing else changed."""

import json

import numpy as np
import pymupdf
from PIL import Image

from pdf_tools.assets import extract_assets
from pdf_tools.compare import compare_pdfs, load_masks
from pdf_tools.edit import ChangeSet, apply_edits


def statuses(rep):
    return [sorted((im.status, im.masked, im.moved) for im in p.images) for p in rep.pages]


def test_manifest_describes_images(pdfs, tmp_path):
    m = extract_assets(pdfs["images"], tmp_path / "a")
    logo = next(a for a in m.assets if a.kind == "image" and a.width == 240)
    photo = next(a for a in m.assets if a.kind == "image" and a.width == 320)
    assert logo.has_alpha and logo.format == "png" and logo.effective_dpi == 144.0  # 240 px over 120 pt
    assert not photo.has_alpha and photo.format == "jpeg" and photo.colorspace


def test_update_image_and_put_it_back(pdfs, tmp_path):
    src = pdfs["images"]
    m = extract_assets(src, tmp_path / "a")
    logo = next(a for a in m.assets if a.kind == "image" and a.width == 240)
    # "make the logo red": recolor the blue body, keep size and transparency
    arr = np.array(Image.open(tmp_path / "a" / logo.file).convert("RGBA"))
    blue = (arr[..., 2] > 120) & (arr[..., 0] < 60) & (arr[..., 3] > 0)
    arr[blue, :3] = (200, 30, 40)
    Image.fromarray(arr).save(tmp_path / "logo-red.png")
    rep = apply_edits(src, ChangeSet.model_validate({"operations": [
        {"op": "replace_image", "page": 1, "asset_id": logo.id, "image": str(tmp_path / "logo-red.png")}]}), tmp_path / "out.pdf")
    r = rep.results[0]
    assert rep.ok and r.verified  # placed pixels == the edited file
    assert sorted(m["page"] for m in rep.masks) == [1, 2]  # shared logo: masked on both pages
    (tmp_path / "rep.json").write_text(rep.model_dump_json(), encoding="utf-8")
    masks, shifts = load_masks(tmp_path / "rep.json")
    cmp = compare_pdfs(src, tmp_path / "out.pdf", tmp_path / "c", masks=masks, shifts=shifts)
    assert statuses(cmp) == [[("changed", True, False), ("identical", False, False)], [("changed", True, False)]]
    assert cmp.status == "ok"  # the photo is byte-identical, only the intended image changed
    with pymupdf.open(tmp_path / "out.pdf") as doc:  # still transparent around the logo body
        assert doc[0].get_pixmap(clip=pymupdf.Rect(73, 37, 74, 38)).pixel(0, 0) == (255, 255, 255)


def test_compare_reports_image_statuses(pdfs, tmp_path):
    src = pdfs["images"]
    with pymupdf.open(src) as doc:
        photo = next(i for i in doc[0].get_image_info(xrefs=True) if i["width"] == 320)
        pix = pymupdf.Pixmap(doc, photo["xref"])
        pix.save(tmp_path / "same.png")  # same pixels, different encoding (JPEG -> PNG)
    rep = apply_edits(src, ChangeSet.model_validate({"operations": [
        {"op": "replace_image", "page": 1, "xref": photo["xref"], "image": str(tmp_path / "same.png")},
        {"op": "remove_image", "page": 2, "rect": [72, 36, 192, 76]}]}), tmp_path / "out.pdf")
    assert rep.ok
    cmp = compare_pdfs(src, tmp_path / "out.pdf", tmp_path / "c")
    page1 = {im.reference_size: im.status for im in cmp.pages[0].images}
    assert page1[(320, 200)] == "same_pixels"
    # untouched logo: MuPDF re-encodes a shared image object when it is redacted on another page
    assert page1[(240, 80)] in ("identical", "same_pixels")
    assert [im.status for im in cmp.pages[1].images] == ["missing"]
    assert cmp.status == "review"  # an unmasked image went missing
    summary = json.loads(cmp.model_dump_json())["pages"][1]["images"][0]
    assert summary["reference_bbox"] == [72.0, 36.0, 192.0, 76.0]
