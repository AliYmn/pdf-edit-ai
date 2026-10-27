"""`pdf-tool generate`: render HTML/CSS to PDF with headless Chromium (Playwright).

The HTML is served from a virtual origin whose root is a local directory.
Every request is intercepted: files under the root are served, everything else
(remote URLs, paths outside the root) is blocked. Page JavaScript is disabled
unless explicitly allowed.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from .common import InputError, PdfToolError, atomic_output, ensure_writable_output, open_pdf, strip_subset

ORIGIN = "http://pdf-tool.localhost"
PAGE_SIZES = {
    "a3": (841.89, 1190.55),
    "a4": (595.28, 841.89),
    "a5": (419.53, 595.28),
    "letter": (612.0, 792.0),
    "legal": (612.0, 1008.0),
}
TEXT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml; charset=utf-8",
}
UNITS = {"pt": 1.0, "mm": 72 / 25.4, "cm": 72 / 2.54, "in": 72.0, "px": 0.75}

# Elements whose content is clipped or that stick out of the page (JS-free check via CDP evaluate).
OVERFLOW_JS = """() => {
  const out = [];
  const pageW = document.documentElement.clientWidth;
  for (const el of document.body.querySelectorAll('*')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none') continue;
    const desc = el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') +
      (el.classList.length ? '.' + [...el.classList].join('.') : '');
    const text = (el.innerText || '').trim().slice(0, 60);
    const clips = cs.overflow !== 'visible' || cs.overflowX !== 'visible' || cs.overflowY !== 'visible';
    if (clips && (el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1))
      out.push({element: desc, issue: 'content clipped', text});
    const r = el.getBoundingClientRect();
    if (r.width && r.right > pageW + 1 && el.children.length === 0)
      out.push({element: desc, issue: 'extends past page width', text});
    if (out.length >= 50) break;
  }
  return out;
}"""


def parse_length(value: str) -> float:
    """'12mm' -> points. Bare numbers are points."""
    m = re.fullmatch(r"\s*([\d.]+)\s*(pt|mm|cm|in|px)?\s*", value)
    if not m:
        raise InputError(f"invalid length {value!r}; use e.g. '12mm', '0.5in', '36pt'")
    return float(m.group(1)) * UNITS[m.group(2) or "pt"]


def parse_page_size(spec: str) -> tuple[float, float]:
    key = spec.strip().lower()
    landscape = key.endswith("-landscape")
    key = key.removesuffix("-landscape")
    if key in PAGE_SIZES:
        w, h = PAGE_SIZES[key]
    else:
        m = re.fullmatch(r"([\d.]+)\s*x\s*([\d.]+)\s*(pt|mm|cm|in|px)?", key)
        if not m:
            raise InputError(f"invalid page size {spec!r}; use A4, Letter, A4-landscape or e.g. 210x297mm")
        unit = UNITS[m.group(3) or "pt"]
        w, h = float(m.group(1)) * unit, float(m.group(2)) * unit
    return (h, w) if landscape else (w, h)


def parse_margin(spec: str) -> dict[str, str]:
    parts = [parse_length(p) for p in spec.split()]
    if not 1 <= len(parts) <= 4:
        raise InputError("margin takes 1-4 values like CSS: 'top right bottom left'")
    trbl = parts * 4 if len(parts) == 1 else parts * 2 if len(parts) == 2 else parts + parts[1:2] if len(parts) == 3 else parts
    return {k: f"{v / 72:.4f}in" for k, v in zip(("top", "right", "bottom", "left"), trbl)}


def size_from_pdf(pdf: str | Path, page: int = 1) -> tuple[float, float]:
    doc = open_pdf(pdf)
    try:
        if not 1 <= page <= doc.page_count:
            raise InputError(f"page {page} outside 1-{doc.page_count}")
        r = doc[page - 1].rect
        return r.width, r.height
    finally:
        doc.close()


def generate_pdf(
    html: str | Path,
    output: str | Path,
    page_size: str | None = None,
    size_from: str | Path | None = None,
    size_from_page: int = 1,
    margin: str | None = None,
    header_template: str | Path | None = None,
    footer_template: str | Path | None = None,
    root: str | Path | None = None,
    allow_js: bool = False,
    overwrite: bool = False,
    timeout_s: float = 60,
) -> dict:
    html, output = Path(html).resolve(), Path(output)
    if not html.is_file():
        raise InputError(f"HTML file not found: {html}")
    root_dir = Path(root).resolve() if root else html.parent
    if not html.is_relative_to(root_dir):
        raise InputError(f"{html} is not inside --root {root_dir}")
    if page_size and size_from:
        raise InputError("use either --page-size or --size-from, not both")
    ensure_writable_output(output, [html], overwrite)

    # CLI page size / margins are injected as a later @page rule so they beat the document's CSS.
    pdf_opts: dict = {"print_background": True, "prefer_css_page_size": True}
    page_css = []
    if page_size or size_from:
        w, h = parse_page_size(page_size) if page_size else size_from_pdf(size_from, size_from_page)
        page_css.append(f"size: {w:.3f}pt {h:.3f}pt;")
    if margin:
        page_css.append("margin: " + " ".join(f"{v}" for v in parse_margin(margin).values()) + ";")
    if header_template or footer_template:
        pdf_opts["display_header_footer"] = True
        pdf_opts["header_template"] = Path(header_template).read_text(encoding="utf-8") if header_template else "<span></span>"
        pdf_opts["footer_template"] = Path(footer_template).read_text(encoding="utf-8") if footer_template else "<span></span>"

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise PdfToolError("playwright is not installed; run scripts/setup.sh") from exc

    blocked: list[str] = []
    missing: list[str] = []

    def handle(route) -> None:
        url = route.request.url
        if url.startswith(ORIGIN + "/"):
            rel = unquote(urlparse(url).path).lstrip("/")
            target = (root_dir / rel).resolve()
            if target.is_relative_to(root_dir) and target.is_file():
                ctype = TEXT_TYPES.get(target.suffix.lower())
                # text files are read as UTF-8 even without <meta charset>
                route.fulfill(path=str(target), **({"content_type": ctype} if ctype else {}))
                return
            missing.append(rel)
            route.fulfill(status=404, body="")
            return
        if url.startswith(("data:", "blob:", "about:")):
            route.continue_()
            return
        blocked.append(url)
        route.abort()

    url = f"{ORIGIN}/{html.relative_to(root_dir).as_posix()}"
    with atomic_output(output) as tmp:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                try:
                    context = browser.new_context(java_script_enabled=allow_js, service_workers="block")
                    context.route("**/*", handle)
                    page = context.new_page()
                    page.emulate_media(media="print")
                    page.goto(url, wait_until="load", timeout=timeout_s * 1000)
                    if page_css:
                        # add_style_tag waits on a script load event, which never fires with JS disabled
                        page.evaluate(
                            "css => { const s = document.createElement('style'); s.textContent = css;"
                            " document.head.appendChild(s); }",
                            "@page { " + " ".join(page_css) + " }",
                        )
                    page.evaluate("() => document.fonts.ready.then(() => true)")
                    layout_issues = page.evaluate(OVERFLOW_JS)
                    page.pdf(path=str(tmp), **pdf_opts)
                    chromium = browser.version
                finally:
                    browser.close()
        except PlaywrightError as exc:
            msg = str(exc)
            if "Executable doesn't exist" in msg:
                msg = "Chromium is not installed for Playwright; run `python -m playwright install chromium`"
            raise PdfToolError(f"PDF generation failed: {msg}") from exc

    doc = open_pdf(output)
    try:
        sizes = sorted({(round(pg.rect.width, 2), round(pg.rect.height, 2)) for pg in doc})
        fonts = sorted({strip_subset(f[3]) for pg in doc for f in pg.get_fonts(full=True)})
        page_count = doc.page_count
    finally:
        doc.close()
    warnings = [f"blocked non-local request: {u}" for u in blocked]
    warnings += [f"missing local file: {m}" for m in missing]
    warnings += [f"{i['issue']}: <{i['element']}> {i['text']!r}" for i in layout_issues]
    return {
        "output": str(output.resolve()),
        "pages": page_count,
        "page_sizes_pt": sizes,
        "fonts": fonts,
        "chromium": chromium,
        "javascript": allow_js,
        "warnings": warnings,
    }
