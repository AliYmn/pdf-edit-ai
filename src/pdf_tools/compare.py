"""`pdf-tool compare`: visual + structural comparison of a reference and a candidate PDF.

Pixel metrics alone cannot judge intentional edits, so the report combines:
- per-page SSIM / changed-pixel ratio / changed regions (with masks for areas
  that were edited on purpose),
- word-level text diffs per page and across the whole document (catches text
  that went missing even when it reflowed to another page),
- text-block position/size changes,
- layout warnings: text off the page, overlapping text, content outside the
  reference margins, extra/missing pages.
"""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pymupdf
from pydantic import BaseModel, Field, ValidationError

from .common import InputError, open_pdf, require_dir_output, rnd, write_json
from .models import BBox

MAX_ITEMS = 20
WORD_FLAGS = pymupdf.TEXTFLAGS_WORDS & ~pymupdf.TEXT_PRESERVE_LIGATURES
DICT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES & ~pymupdf.TEXT_PRESERVE_LIGATURES


class Mask(BaseModel):
    page: int = Field(ge=1)
    rect: BBox = Field(description="Area in the reference")
    candidate_rect: BBox | None = Field(None, description="Area in the candidate, if it differs")
    label: str = ""


class Shift(BaseModel):
    """Intentional move of everything at or below ``y`` (from `edit` shift_content)."""

    page: int = Field(ge=1)
    y: float
    dy: float


class BlockChange(BaseModel):
    """Geometry change of a matched text block. Measured on glyph origins and
    baselines, not bbox tops, so fonts with different ascender metrics compare fairly."""

    text: str
    reference_bbox: BBox | None = None
    candidate_bbox: BBox | None = None
    dx: float | None = Field(None, description="Left edge shift")
    dy: float | None = Field(None, description="First baseline shift")
    dw: float | None = Field(None, description="Width change")
    dh: float | None = Field(None, description="Change of first-to-last baseline distance (line count/leading)")


class ImageChange(BaseModel):
    """One image on the page, matched between reference and candidate."""

    status: Literal["identical", "same_pixels", "changed", "missing", "added"] = Field(
        description="identical: same bytes; same_pixels: re-encoded, pixels equal; changed: pixels differ"
    )
    reference_bbox: BBox | None = None
    candidate_bbox: BBox | None = None
    reference_size: tuple[int, int] | None = Field(None, description="Pixels")
    candidate_size: tuple[int, int] | None = None
    moved: bool = Field(False, description="Placement differs by more than the tolerance")
    pixel_diff: float | None = Field(None, description="Mean absolute difference 0..1 (same pixel size only)")
    masked: bool = Field(False, description="Inside a mask: an intended change")
    inline: bool = Field(False, description="Inline image (inside the content stream): compared by pixels only")


class PageComparison(BaseModel):
    page: int
    reference_size: tuple[float, float] | None = None
    candidate_size: tuple[float, float] | None = None
    ssim: float | None = None
    changed_pixels: float | None = Field(None, description="Fraction of unmasked pixels that differ noticeably")
    changed_regions: list[BBox] = Field([], description="Largest changed areas, in reference page points")
    masked_regions: int = 0
    text_similarity: float | None = Field(None, description="Word-sequence similarity 0..1 (masked words excluded)")
    missing_text: list[str] = []
    added_text: list[str] = []
    moved_blocks: list[BlockChange] = []
    missing_blocks: list[BlockChange] = []
    added_blocks: list[BlockChange] = []
    images: list[ImageChange] = []
    warnings: list[str] = []
    side_by_side: str | None = None
    diff_image: str | None = None


class Expectation(BaseModel):
    text: str
    kind: str  # "present" | "absent"
    ok: bool
    pages: list[int] = []


class ComparisonReport(BaseModel):
    reference: str
    candidate: str
    dpi: int
    reference_pages: int
    candidate_pages: int
    status: str = Field(description="'ok' or 'review' (warnings, differences or failed expectations)")
    document_missing_text: list[str] = Field([], description="Reference text absent from the whole candidate")
    document_added_text: list[str] = []
    expectations: list[Expectation] = []
    warnings: list[str] = []
    pages: list[PageComparison]


def load_masks(path: str | Path) -> tuple[list[Mask], list[Shift]]:
    """Masks from a JSON list or an object with "masks" (an edit report also carries "shifts")."""
    p = Path(path)
    if not p.is_file():
        raise InputError(f"mask file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        shifts = [Shift.model_validate(x) for x in data.get("shifts", [])] if isinstance(data, dict) else []
        if isinstance(data, dict):
            data = data.get("masks", [])
        return [Mask.model_validate(m) for m in data], shifts
    except (json.JSONDecodeError, ValidationError) as exc:
        raise InputError(f"invalid mask file {p}: {exc}") from exc


def parse_mask(spec: str) -> Mask:
    m = re.fullmatch(r"\s*(\d+)\s*:\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*", spec)
    if not m:
        raise InputError(f"invalid mask {spec!r}; use PAGE:x0,y0,x1,y1 (points)")
    return Mask(page=int(m.group(1)), rect=tuple(float(m.group(i)) for i in range(2, 6)), label="cli")


# ------------------------------------------------------------------ pixels


def _render(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity of two grayscale images (Wang et al. 2004)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return float(m.mean())


def _pixel_compare(ref: np.ndarray, cand: np.ndarray, masks_px: list[tuple[int, int, int, int]], scale: float):
    """Return (ssim, changed fraction, regions in pt, diff overlay image)."""
    ga = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY)
    valid = np.ones_like(ga, dtype=bool)
    for x0, y0, x1, y1 in masks_px:
        ga[y0:y1, x0:x1] = 128
        gb[y0:y1, x0:x1] = 128
        valid[y0:y1, x0:x1] = False
    score = ssim(ga, gb)
    diff = cv2.GaussianBlur(cv2.absdiff(ga, gb), (3, 3), 0)
    changed = (diff > 40) & valid
    frac = float(changed.sum() / max(1, valid.sum()))
    kx, ky = max(3, int(round(8 / scale))), max(3, int(round(4 / scale)))  # join glyphs/words into regions
    grown = cv2.dilate(changed.astype(np.uint8), np.ones((ky, kx), np.uint8))
    contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] * scale * scale >= 4]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    regions = [rnd((x * scale, y * scale, (x + w) * scale, (y + h) * scale), 1) for x, y, w, h in boxes[:MAX_ITEMS]]

    overlay = (ref.astype(np.float32) * 0.35 + 255 * 0.65).astype(np.uint8)
    overlay[changed] = (230, 30, 30)
    for x, y, w, h in boxes[:MAX_ITEMS]:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (230, 30, 30), 1)
    for x0, y0, x1, y1 in masks_px:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (40, 90, 230), 2)
    return score, frac, regions, overlay


def _labelled(img: np.ndarray, label: str) -> np.ndarray:
    bar = np.full((28, img.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _side_by_side(ref: np.ndarray | None, cand: np.ndarray | None, page: int) -> np.ndarray:
    imgs = []
    for img, name in ((ref, "reference"), (cand, "candidate")):
        if img is None:
            other = cand if ref is None else ref
            img = np.full_like(other, 255)
            cv2.putText(img, "no page", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 0, 0), 2)
        imgs.append(_labelled(img, f"{name} p{page}"))
    h = max(i.shape[0] for i in imgs)
    imgs = [np.vstack([i, np.full((h - i.shape[0], i.shape[1], 3), 255, np.uint8)]) for i in imgs]
    gap = np.full((h, 12, 3), 200, np.uint8)
    return np.hstack([imgs[0], gap, imgs[1]])


def _save(path: Path, rgb: np.ndarray) -> str:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return str(path)


# -------------------------------------------------------------------- text


def _in_masks(rect, masks: list[pymupdf.Rect]) -> bool:
    r = pymupdf.Rect(rect)
    area = abs(r) or 1
    return any(abs(r & m) / area > 0.5 for m in masks)


def _words(page: pymupdf.Page, masks: list[pymupdf.Rect]) -> list[tuple]:
    return [w for w in page.get_text("words", sort=True, flags=WORD_FLAGS) if not _in_masks(w[:4], masks)]


def _phrases(a: list[str], b: list[str]) -> tuple[list[str], list[str], float]:
    sm = SequenceMatcher(None, a, b, autojunk=False)
    missing, added = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            missing.append(" ".join(a[i1:i2])[:200])
        if tag in ("insert", "replace"):
            added.append(" ".join(b[j1:j2])[:200])
    return missing, added, round(sm.ratio(), 4) if (a or b) else 1.0


def _blocks(page: pymupdf.Page, masks: list[pymupdf.Rect]) -> list[tuple[pymupdf.Rect, str, tuple]]:
    """(bbox, text, (x0, first baseline, x1, last baseline)) of text blocks that do
    not touch a mask (a block touching an edit is expected to change)."""
    out = []
    for b in page.get_text("rawdict", sort=True, flags=DICT_FLAGS)["blocks"]:
        lines = []
        for ln in b.get("lines", []):
            ink = [c for s in ln["spans"] for c in s["chars"] if not c["c"].isspace()]
            if ink:
                lines.append((ln, ink))
        if not lines:
            continue
        rect = pymupdf.Rect(b["bbox"])
        text = " ".join("".join(c["c"] for s in ln["spans"] for c in s["chars"]).strip() for ln, _ in lines)
        text = " ".join(text.split())
        if any(rect.intersects(m) for m in masks):
            continue
        # extents of visible glyphs only: trailing spaces must not count as a width change
        x0 = min(ink[0]["origin"][0] for _, ink in lines)
        x1 = max(c["bbox"][2] for _, ink in lines for c in ink)
        geom = (x0, lines[0][1][0]["origin"][1], x1, lines[-1][1][0]["origin"][1])
        out.append((rect, text, geom))
    return out


def _expected_dy(geom: tuple, shifts: list[Shift]) -> float:
    return sum(sh.dy for sh in shifts if geom[1] >= sh.y)


def _match_blocks(ra, rb, tol: float, shifts: list[Shift] = ()):
    moved, used = [], set()
    unmatched_a = []
    for rect_a, text_a, ga in ra:
        best, best_j = 0.0, None
        for j, (_, text_b, _) in enumerate(rb):
            if j in used:
                continue
            if text_a == text_b:
                best, best_j = 1.0, j
                break
            sm = SequenceMatcher(None, text_a, text_b, autojunk=False)
            if sm.real_quick_ratio() > best and sm.quick_ratio() > best:
                r = sm.ratio()
                if r > best:
                    best, best_j = r, j
        if best_j is None or best < 0.85:
            # block segmentation differs (e.g. two columns merged into one block): not missing
            if not any(text_a in text_b or text_b in text_a for _, text_b, _ in rb):
                unmatched_a.append(BlockChange(text=text_a[:120], reference_bbox=rnd(rect_a)))
            continue
        used.add(best_j)
        rect_b, _, gb = rb[best_j]
        dy = gb[1] - ga[1] - _expected_dy(ga, shifts)  # intentional shifts are not regressions
        d = (gb[0] - ga[0], dy, (gb[2] - gb[0]) - (ga[2] - ga[0]), (gb[3] - gb[1]) - (ga[3] - ga[1]))
        if max(abs(v) for v in d) > tol:
            moved.append(
                BlockChange(text=text_a[:120], reference_bbox=rnd(rect_a), candidate_bbox=rnd(rect_b), dx=round(d[0], 2),
                            dy=round(d[1], 2), dw=round(d[2], 2), dh=round(d[3], 2))
            )  # fmt: skip
    added = [BlockChange(text=t[:120], candidate_bbox=rnd(r)) for j, (r, t, _) in enumerate(rb) if j not in used]
    moved.sort(key=lambda m: -max(abs(m.dx), abs(m.dy), abs(m.dw), abs(m.dh)))
    return moved[:MAX_ITEMS], unmatched_a[:MAX_ITEMS], added[:MAX_ITEMS]


def _image_pixels(doc: pymupdf.Document, page: pymupdf.Page, xref: int, bbox: pymupdf.Rect) -> np.ndarray:
    """Decoded RGB(A) pixels of an image object (soft mask applied); inline images are rendered."""
    if not xref:
        pix = page.get_pixmap(clip=bbox, dpi=150, alpha=False)
    else:
        pix = pymupdf.Pixmap(doc, xref)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        if pix.colorspace is None or pix.colorspace.n != 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        kind, value = doc.xref_get_key(xref, "SMask")
        if kind == "xref":
            mask = pymupdf.Pixmap(doc, int(value.split()[0]))
            if (mask.width, mask.height) == (pix.width, pix.height):
                pix = pymupdf.Pixmap(pix, mask)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)


def _page_images(doc: pymupdf.Document, page: pymupdf.Page) -> list[dict]:
    out = []
    for info in page.get_image_info(xrefs=True):
        bbox = pymupdf.Rect(info["bbox"]) & page.rect
        if bbox.is_empty:
            continue
        xref = info.get("xref", 0)
        arr = _image_pixels(doc, page, xref, bbox)
        raw = doc.xref_stream_raw(xref) if xref else None
        out.append({
            "bbox": bbox, "arr": arr, "size": (arr.shape[1], arr.shape[0]),
            "pix": hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest(),
            "raw": hashlib.sha256(raw).hexdigest() if raw else None, "inline": not xref,
        })  # fmt: skip
    return out


def _match_images(ra: list[dict], rb: list[dict], ref_masks, shifts: list[Shift], tol: float, cand_masks=()) -> list[ImageChange]:
    """Same pixels first (wherever they moved), then by placement overlap."""
    out: list[ImageChange] = []
    used: set[int] = set()

    def expected(r: pymupdf.Rect) -> pymupdf.Rect:
        dy = sum(sh.dy for sh in shifts if r.y0 >= sh.y - 0.5)
        return r + (0, dy, 0, dy)

    def moved(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
        e = expected(a)
        return max(abs(e.x0 - b.x0), abs(e.y0 - b.y0), abs(e.x1 - b.x1), abs(e.y1 - b.y1)) > tol

    pending = []
    for a in ra:
        same = [(j, b) for j, b in enumerate(rb) if j not in used and b["pix"] == a["pix"]]
        if same:
            j, b = min(same, key=lambda t: abs(expected(a["bbox"]).tl - t[1]["bbox"].tl))
            used.add(j)
            status = "identical" if a["raw"] and a["raw"] == b["raw"] else "same_pixels"
            out.append(ImageChange(status=status, reference_bbox=rnd(a["bbox"]), candidate_bbox=rnd(b["bbox"]),
                                   reference_size=a["size"], candidate_size=b["size"], inline=a["inline"],
                                   moved=moved(a["bbox"], b["bbox"])))  # fmt: skip
        else:
            pending.append(a)
    for a in pending:
        e = expected(a["bbox"])
        overlap = [(abs(e & b["bbox"]) / (abs(e | b["bbox"]) or 1), j, b) for j, b in enumerate(rb) if j not in used]
        overlap = [o for o in overlap if o[0] > 0.3]
        masked = _in_masks(a["bbox"], ref_masks)
        if not overlap:
            out.append(ImageChange(status="missing", reference_bbox=rnd(a["bbox"]), reference_size=a["size"], masked=masked))
            continue
        _, j, b = max(overlap, key=lambda o: o[0])
        used.add(j)
        diff = None
        if a["arr"].shape == b["arr"].shape:
            diff = round(float(np.abs(a["arr"].astype(np.int16) - b["arr"].astype(np.int16)).mean() / 255), 4)
        out.append(ImageChange(status="changed", reference_bbox=rnd(a["bbox"]), candidate_bbox=rnd(b["bbox"]),
                               reference_size=a["size"], candidate_size=b["size"], moved=moved(a["bbox"], b["bbox"]),
                               pixel_diff=diff, masked=masked))  # fmt: skip
    for j, b in enumerate(rb):
        if j not in used:
            out.append(ImageChange(status="added", candidate_bbox=rnd(b["bbox"]), candidate_size=b["size"],
                                   masked=_in_masks(b["bbox"], cand_masks)))  # fmt: skip
    return out


def _layout_warnings(page: pymupdf.Page, ref_page: pymupdf.Page | None) -> list[str]:
    warnings = []
    W, H = page.rect.width, page.rect.height
    words = page.get_text("words", flags=WORD_FLAGS)
    off = [w[4] for w in words if w[0] < -0.5 or w[1] < -0.5 or w[2] > W + 0.5 or w[3] > H + 0.5]
    if off:
        warnings.append(f"{len(off)} words extend beyond the page edge, e.g. {' '.join(off[:5])!r}")
    # overlapping words from different lines (sweep over y)
    ws = sorted(words, key=lambda w: w[1])
    overlaps = []
    for i, a in enumerate(ws):
        ra = pymupdf.Rect(a[:4])
        for b in ws[i + 1 :]:
            if b[1] >= a[3]:
                break
            if (a[5], a[6]) == (b[5], b[6]):
                continue
            rb = pymupdf.Rect(b[:4])
            inter = abs(ra & rb)
            if inter > 0.3 * min(abs(ra), abs(rb)):
                overlaps.append(f"{a[4]!r}/{b[4]!r} at ({ra.x0:.0f},{ra.y0:.0f})")
        if len(overlaps) >= 5:
            break
    if overlaps:
        warnings.append("overlapping text (possible overflow): " + "; ".join(overlaps))
    if ref_page is not None and words and abs(ref_page.rect - page.rect) < 1:
        ref_words = ref_page.get_text("words", flags=WORD_FLAGS)
        if ref_words:
            rc = pymupdf.Rect(min(w[0] for w in ref_words), min(w[1] for w in ref_words),
                              max(w[2] for w in ref_words), max(w[3] for w in ref_words))  # fmt: skip
            cc = pymupdf.Rect(min(w[0] for w in words), min(w[1] for w in words),
                              max(w[2] for w in words), max(w[3] for w in words))  # fmt: skip
            out = {"left": rc.x0 - cc.x0, "top": rc.y0 - cc.y0, "right": cc.x1 - rc.x1, "bottom": cc.y1 - rc.y1}
            beyond = {k: round(v, 1) for k, v in out.items() if v > 3}
            if beyond:
                warnings.append(f"text exceeds the reference content area by {beyond} pt")
    return warnings


# -------------------------------------------------------------------- main


def compare_pdfs(
    reference: str | Path,
    candidate: str | Path,
    out_dir: str | Path,
    dpi: int = 100,
    masks: list[Mask] | None = None,
    shifts: list[Shift] | None = None,
    expect: list[str] | None = None,
    forbid: list[str] | None = None,
    tolerance: float = 2.0,
) -> ComparisonReport:
    if not 36 <= dpi <= 300:
        raise InputError("dpi must be between 36 and 300")
    out = require_dir_output(Path(out_dir))
    da, db = open_pdf(reference), open_pdf(candidate)
    masks = masks or []
    try:
        warnings: list[str] = []
        if da.page_count != db.page_count:
            diff = db.page_count - da.page_count
            warnings.append(
                f"candidate has {diff} extra page(s); content may have spilled over" if diff > 0
                else f"candidate is missing {-diff} page(s)"
            )  # fmt: skip
        pages: list[PageComparison] = []
        all_a: list[str] = []
        all_b: list[str] = []
        scale = 72 / dpi
        for i in range(max(da.page_count, db.page_count)):
            pa = da[i] if i < da.page_count else None
            pb = db[i] if i < db.page_count else None
            pm = [pymupdf.Rect(m.rect) for m in masks if m.page == i + 1]  # reference side
            cm = [pymupdf.Rect(m.candidate_rect or m.rect) for m in masks if m.page == i + 1]  # candidate side
            psh = [sh for sh in shifts or [] if sh.page == i + 1]
            pc = PageComparison(page=i + 1, masked_regions=len(pm))
            img_a = _render(pa, dpi) if pa else None
            img_b = _render(pb, dpi) if pb else None
            if pa:
                pc.reference_size = rnd((pa.rect.width, pa.rect.height))
            if pb:
                pc.candidate_size = rnd((pb.rect.width, pb.rect.height))
                # only problems the candidate introduced; the reference's own quirks are not regressions
                known = set(_layout_warnings(pa, None)) if pa else set()
                pc.warnings += [w for w in _layout_warnings(pb, pa) if w not in known]
            if pa and pb:
                if abs(pa.rect.width - pb.rect.width) > 0.5 or abs(pa.rect.height - pb.rect.height) > 0.5:
                    pc.warnings.append(f"page size differs: reference {pc.reference_size} vs candidate {pc.candidate_size} pt")
                    img_b_cmp = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_AREA)
                else:
                    img_b_cmp = img_b[: img_a.shape[0], : img_a.shape[1]]
                    if img_b_cmp.shape != img_a.shape:  # off-by-one pixel rounding
                        img_b_cmp = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_AREA)
                masks_px = [
                    tuple(int(round(v / scale)) for v in (m.x0, m.y0, m.x1, m.y1)) for m in pm + cm
                ]
                masks_px = [(max(0, x0), max(0, y0), x1, y1) for x0, y0, x1, y1 in masks_px]
                score, frac, regions, overlay = _pixel_compare(img_a, img_b_cmp, masks_px, scale)
                pc.ssim, pc.changed_pixels, pc.changed_regions = round(score, 4), round(frac, 5), regions
                pc.diff_image = _save(out / f"diff-{i + 1:03d}.png", overlay)
                wa, wb = _words(pa, pm), _words(pb, cm)
                pc.missing_text, pc.added_text, pc.text_similarity = _phrases([w[4] for w in wa], [w[4] for w in wb])
                pc.missing_text, pc.added_text = pc.missing_text[:MAX_ITEMS], pc.added_text[:MAX_ITEMS]
                pc.moved_blocks, pc.missing_blocks, pc.added_blocks = _match_blocks(_blocks(pa, pm), _blocks(pb, cm), tolerance, psh)
                pc.images = _match_images(_page_images(da, pa), _page_images(db, pb), pm, psh, tolerance, cm)
            elif pa:
                pc.warnings.append("page missing in candidate")
            else:
                pc.warnings.append("extra page in candidate")
            if pa:
                all_a += [w[4] for w in _words(pa, pm)]
            if pb:
                all_b += [w[4] for w in _words(pb, cm)]
            pc.side_by_side = _save(out / f"side-by-side-{i + 1:03d}.png", _side_by_side(img_a, img_b, i + 1))
            pages.append(pc)
        if pages and db.page_count > 1 and len(db[-1].get_text("words")) < 15 and len(da[-1].get_text("words")) >= 15:
            warnings.append("candidate's last page is nearly empty; check for a few lines spilling over")
        doc_missing, doc_added, _ = _phrases(all_a, all_b)

        expectations = []
        texts_b = ["".join(p.get_text("text").split()) for p in db]
        for kind, items in (("present", expect or []), ("absent", forbid or [])):
            for t in items:
                needle = "".join(t.split())
                found = [k + 1 for k, txt in enumerate(texts_b) if needle in txt]
                ok = bool(found) if kind == "present" else not found
                expectations.append(Expectation(text=t, kind=kind, ok=ok, pages=found))
    finally:
        da.close()
        db.close()

    review = (
        warnings or doc_missing or any(not e.ok for e in expectations)
        or any(p.warnings or p.missing_blocks or p.moved_blocks or (p.changed_pixels or 0) > 0.002 for p in pages)
        or any(not im.masked and (im.moved or im.status in ("changed", "missing", "added")) for p in pages for im in p.images)
    )  # fmt: skip
    report = ComparisonReport(
        reference=str(Path(reference).resolve()),
        candidate=str(Path(candidate).resolve()),
        dpi=dpi,
        reference_pages=len([p for p in pages if p.reference_size]),
        candidate_pages=len([p for p in pages if p.candidate_size]),
        status="review" if review else "ok",
        document_missing_text=doc_missing[:MAX_ITEMS],
        document_added_text=doc_added[:MAX_ITEMS],
        expectations=expectations,
        warnings=warnings,
        pages=pages,
    )
    write_json(report, out / "report.json")
    return report
