"""`pdf-tool assets`: export embedded images, vector graphics and fonts."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pymupdf

from .common import open_pdf, require_dir_output, rnd, strip_subset, write_json
from .models import Asset, AssetManifest, Placement
from .vector import is_background, is_graphic, padded_rect, svg_document

WEB_FONT_EXTS = {"ttf", "otf", "woff", "woff2"}
MAX_VECTORS_PER_PAGE = 40


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _image_bytes(doc: pymupdf.Document, xref: int) -> tuple[bytes, str, int, int]:
    """Return (data, ext, width, height), converting to RGB(A) PNG when the raw
    stream is not directly usable in a browser (CMYK, soft masks, JBIG2...)."""
    info = doc.extract_image(xref)
    if info and not info.get("smask") and info.get("colorspace") in (1, 3) and info["ext"] in ("png", "jpeg", "jpg"):
        return info["image"], info["ext"], info["width"], info["height"]
    pix = pymupdf.Pixmap(doc, xref)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)  # drop alpha before colorspace conversion
    if pix.colorspace is None or pix.colorspace.n not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    smask = info.get("smask") if info else 0
    if smask:
        mask = pymupdf.Pixmap(doc, smask)
        if (mask.width, mask.height) == (pix.width, pix.height):
            pix = pymupdf.Pixmap(pix, mask)
    return pix.tobytes("png"), "png", pix.width, pix.height


def _role(placements: list[Placement], doc: pymupdf.Document) -> str:
    pages = {p.page for p in placements}
    for p in placements:
        page_rect = doc[p.page - 1].rect
        r = pymupdf.Rect(p.bbox)
        frac = abs(r) / abs(page_rect)
        if frac > 0.85:
            return "page-background-or-scan"
        near_edge = r.y1 <= page_rect.height * 0.2 or r.y0 >= page_rect.height * 0.9
        if frac < 0.2 and (near_edge or len(pages) > 1):
            return "logo-candidate"
    return "image"


def export_images(doc: pymupdf.Document, out: Path, min_size: int, warnings: list[str]) -> list[Asset]:
    placements: dict[str, list[Placement]] = defaultdict(list)
    inline: dict[str, tuple[int, pymupdf.Rect]] = {}
    for page in doc:
        for info in page.get_image_info(xrefs=True):
            bbox = pymupdf.Rect(info["bbox"]) & page.rect
            if bbox.is_empty:
                continue
            xref = info.get("xref", 0)
            aid = f"img-{xref}" if xref else f"img-p{page.number + 1}-inline{info['number']}"
            placements[aid].append(Placement(page=page.number + 1, bbox=rnd(bbox)))
            if not xref:
                inline[aid] = (page.number, bbox)

    assets = []
    for aid, places in placements.items():
        try:
            if aid in inline:
                pno, bbox = inline[aid]
                pix = doc[pno].get_pixmap(clip=bbox, dpi=300, alpha=False)
                data, ext, w, h = pix.tobytes("png"), "png", pix.width, pix.height
                notes = ["inline image: exported as a 300 dpi render of its placement"]
                xref = None
            else:
                xref = int(aid.split("-")[1])
                data, ext, w, h = _image_bytes(doc, xref)
                raw = doc.extract_image(xref) or {}
                notes = []
        except Exception as exc:
            warnings.append(f"{aid}: could not export image ({exc})")
            continue
        if w < min_size or h < min_size:
            continue
        dpi = min(w / (pymupdf.Rect(p.bbox).width / 72) for p in places if pymupdf.Rect(p.bbox).width > 0)
        file = out / "images" / f"{aid}.{ext}"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
        assets.append(
            Asset(
                id=aid,
                kind="image",
                file=str(file.relative_to(out)),
                pages=sorted({p.page for p in places}),
                placements=places,
                width=w,
                height=h,
                format=ext,
                role=_role(places, doc),
                sha256=_sha(data),
                xref=xref,
                has_alpha=bool(raw.get("smask")) if xref else False,
                colorspace=(raw.get("cs-name") or None) if xref else "DeviceRGB",
                effective_dpi=round(dpi, 1),
                notes=notes,
            )
        )
    return assets


def export_vectors(doc: pymupdf.Document, out: Path, warnings: list[str]) -> list[Asset]:
    """Export icon/logo-like vector groups as SVG (from the path data) + 300 dpi PNG.

    Large background fills are ignored so they do not merge everything into one
    cluster; plain rules, boxes and table grids are skipped. Identical graphics
    (e.g. the same icon used several times) become one asset with several placements.
    """
    by_svg: dict[str, Asset] = {}
    for page in doc:
        drawings = [d for d in page.get_drawings() if not is_background(d, page.rect)]
        if not drawings:
            continue
        page_area = abs(page.rect)
        groups = []
        for rect in page.cluster_drawings(drawings=drawings):
            members = [d for d in drawings if (rect + (-0.5, -0.5, 0.5, 0.5)).contains(d["rect"])]
            if members and rect.width >= 4 and rect.height >= 4 and abs(rect) < 0.5 * page_area and is_graphic(members):
                groups.append((rect, members))
        if len(groups) > MAX_VECTORS_PER_PAGE:
            warnings.append(f"page {page.number + 1}: {len(groups)} vector graphics, exporting the largest {MAX_VECTORS_PER_PAGE}")
            groups = sorted(groups, key=lambda g: abs(g[0]), reverse=True)[:MAX_VECTORS_PER_PAGE]
        for _rect, members in groups:
            box = padded_rect(members)
            svg = svg_document(members, box)
            place = Placement(page=page.number + 1, bbox=rnd(box))
            if svg in by_svg:
                asset = by_svg[svg]
                asset.placements.append(place)
                if place.page not in asset.pages:
                    asset.pages.append(place.page)
                continue
            aid = f"vec-p{page.number + 1}-{sum(a.pages[0] == page.number + 1 for a in by_svg.values()) + 1}"
            base = out / "vectors" / aid
            base.parent.mkdir(parents=True, exist_ok=True)
            base.with_suffix(".svg").write_text(svg, encoding="utf-8")
            with pymupdf.open(stream=svg.encode(), filetype="svg") as rendered:
                pix = rendered[0].get_pixmap(dpi=300, alpha=True)
            pix.save(base.with_suffix(".png"))
            small = max(box.width, box.height) < 24
            role = "icon" if small else "logo-candidate" if box.y1 <= page.rect.height * 0.2 else "vector-graphic"
            by_svg[svg] = Asset(
                id=aid,
                kind="vector",
                file=str(base.with_suffix(".svg").relative_to(out)),
                pages=[page.number + 1],
                placements=[place],
                width=pix.width,
                height=pix.height,
                format="svg",
                role=role,
                sha256=_sha(svg.encode()),
                notes=[f"PNG (300 dpi, transparent): {base.with_suffix('.png').relative_to(out)}"],
            )
    return list(by_svg.values())


def export_fonts(doc: pymupdf.Document, out: Path, warnings: list[str]) -> list[Asset]:
    seen: dict[int, Asset] = {}
    for page in doc:
        for xref, ext, ftype, basefont, *_ in page.get_fonts(full=True):
            if xref in seen:
                if page.number + 1 not in seen[xref].pages:
                    seen[xref].pages.append(page.number + 1)
                continue
            name = strip_subset(basefont)
            asset = Asset(
                id=f"font-{xref}",
                kind="font",
                pages=[page.number + 1],
                format=ext,
                xref=xref,
                font_name=name,
                subset=name != basefont,
                notes=[f"type {ftype}"],
            )
            seen[xref] = asset
            if ext == "n/a":
                asset.notes.append("not embedded; the viewer substitutes a system font")
                asset.web_usable = False
                continue
            try:
                _, fext, _, content = doc.extract_font(xref)
            except Exception as exc:
                warnings.append(f"font-{xref} ({name}): extraction failed ({exc})")
                continue
            if not content:
                asset.notes.append("font program not extractable")
                continue
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60]
            file = out / "fonts" / f"font-{xref}-{safe}.{fext}"
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(content)
            asset.file = str(file.relative_to(out))
            asset.format = fext
            asset.sha256 = _sha(content)
            asset.web_usable = fext in WEB_FONT_EXTS
            if not asset.web_usable:
                asset.notes.append(f"'{fext}' font programs are not loadable via CSS @font-face")
            if asset.subset:
                asset.notes.append("subset font: only glyphs used in the original are present")
                try:
                    cps = pymupdf.Font(fontbuffer=content).valid_codepoints()
                    if len(cps) <= 400:
                        asset.notes.append("available characters: " + "".join(chr(c) for c in cps if c >= 32))
                except Exception:
                    pass
    return list(seen.values())


def extract_assets(path: str | Path, out_dir: str | Path, min_size: int = 8, vectors: bool = True) -> AssetManifest:
    out = require_dir_output(Path(out_dir))
    doc = open_pdf(path)
    try:
        warnings: list[str] = []
        assets = export_images(doc, out, min_size, warnings)
        if vectors:
            assets += export_vectors(doc, out, warnings)
        assets += export_fonts(doc, out, warnings)
        manifest = AssetManifest(source=str(Path(path).resolve()), assets=assets, warnings=warnings)
        write_json(manifest, out / "manifest.json")
        return manifest
    finally:
        doc.close()
