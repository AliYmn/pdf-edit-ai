"""`pdf-tool render`: rasterize PDF pages to PNG."""

from __future__ import annotations

from pathlib import Path

from .common import InputError, open_pdf, parse_pages, require_dir_output


def render_pdf(path: str | Path, out_dir: str | Path, dpi: int = 150, pages: str | None = None) -> list[dict]:
    if not 18 <= dpi <= 1200:
        raise InputError("dpi must be between 18 and 1200")
    out = require_dir_output(Path(out_dir))
    doc = open_pdf(path)
    try:
        results = []
        for i in parse_pages(pages, doc.page_count):
            page = doc[i]
            pix = page.get_pixmap(dpi=dpi, alpha=False, annots=True)
            file = out / f"page-{i + 1:03d}.png"
            pix.save(file)
            results.append(
                {
                    "page": i + 1,
                    "file": str(file),
                    "width_px": pix.width,
                    "height_px": pix.height,
                    "width_pt": round(page.rect.width, 2),
                    "height_pt": round(page.rect.height, 2),
                    "dpi": dpi,
                }
            )
        return results
    finally:
        doc.close()
