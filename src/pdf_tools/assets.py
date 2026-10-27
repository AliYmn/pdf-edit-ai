"""`pdf-tool assets`: export embedded images, vector graphics and fonts."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pymupdf

from .common import open_pdf, require_dir_output, rnd, strip_subset, write_json
from .models import Asset, AssetManifest, Placement

WEB_FONT_EXTS = {"ttf", "otf", "woff", "woff2"}
MAX_VECTORS_PER_PAGE = 20


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


def _export_images(doc: pymupdf.Document, out: Path, min_size: int, warnings: list[str]) -> list[Asset]:
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
                notes = []
        except Exception as exc:
            warnings.append(f"{aid}: could not export image ({exc})")
            continue
        if w < min_size or h < min_size:
            continue
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
                notes=notes,
            )
        )
    return assets


def _has_curves(drawings: list[dict], rect: pymupdf.Rect) -> bool:
    for d in drawings:
        if d["rect"].intersects(rect) and any(item[0] in ("c", "qu") for item in d["items"]):
            return True
    return False


def _export_vectors(doc: pymupdf.Document, out: Path, warnings: list[str]) -> list[Asset]:
    """Export clusters of curved vector paths (typical for vector logos/icons)."""
    assets = []
    for page in doc:
        drawings = page.get_drawings()
        if not drawings:
            continue
        page_area = abs(page.rect)
        clusters = [
            r
            for r in page.cluster_drawings(drawings=drawings)
            if r.width >= 8 and r.height >= 8 and 0.0005 < abs(r) / page_area < 0.5 and _has_curves(drawings, r)
        ]
        if len(clusters) > MAX_VECTORS_PER_PAGE:
            warnings.append(f"page {page.number + 1}: {len(clusters)} vector clusters, exporting the largest {MAX_VECTORS_PER_PAGE}")
            clusters = sorted(clusters, key=abs, reverse=True)[:MAX_VECTORS_PER_PAGE]
        for k, rect in enumerate(clusters, 1):
            aid = f"vec-p{page.number + 1}-{k}"
            base = out / "vectors" / aid
            base.parent.mkdir(parents=True, exist_ok=True)
            # SVG: show the clipped region on a page of the same size, then export.
            tmp = pymupdf.open()
            tpage = tmp.new_page(width=rect.width, height=rect.height)
            tpage.show_pdf_page(tpage.rect, doc, page.number, clip=rect)
            svg = tpage.get_svg_image(text_as_path=True)
            tmp.close()
            base.with_suffix(".svg").write_text(svg, encoding="utf-8")
            pix = page.get_pixmap(clip=rect, dpi=300, alpha=True)
            png = pix.tobytes("png")
            base.with_suffix(".png").write_bytes(png)
            place = Placement(page=page.number + 1, bbox=rnd(rect))
            role = "logo-candidate" if rect.y1 <= page.rect.height * 0.2 else "vector-graphic"
            assets.append(
                Asset(
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
                    notes=[
                        f"PNG render (300 dpi, transparent): {base.with_suffix('.png').relative_to(out)}",
                        "region crop: any text or images inside the region are included",
                    ],
                )
            )
    return assets


def _export_fonts(doc: pymupdf.Document, out: Path, warnings: list[str]) -> list[Asset]:
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
        assets = _export_images(doc, out, min_size, warnings)
        if vectors:
            assets += _export_vectors(doc, out, warnings)
        assets += _export_fonts(doc, out, warnings)
        manifest = AssetManifest(source=str(Path(path).resolve()), assets=assets, warnings=warnings)
        write_json(manifest, out / "manifest.json")
        return manifest
    finally:
        doc.close()
