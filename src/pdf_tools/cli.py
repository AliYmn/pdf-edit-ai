"""Command-line interface. Exit codes: 0 ok, 1 processing failure, 2 invalid input."""

from __future__ import annotations

import functools
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .common import write_json
from .models import InputError, PdfToolError

app = typer.Typer(
    name="pdf-tool",
    help="Local PDF toolkit for AI agents: inspect, extract, render, assets, edit, scaffold, generate, compare.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

PdfArg = Annotated[Path, typer.Argument(help="Input PDF", dir_okay=False)]
PagesOpt = Annotated[str | None, typer.Option("--pages", "-p", help="1-based pages, e.g. '1,3-5' (default: all)")]


def handle_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except PdfToolError as exc:
            report = getattr(exc, "report", None)
            if report is not None:
                typer.echo(write_json(report, None))
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(exc.exit_code) from None
        except typer.Exit:
            raise
        except Exception as exc:  # unexpected: keep the message short unless debugging
            if os.environ.get("PDF_TOOL_DEBUG"):
                traceback.print_exc()
            typer.echo(f"internal error: {type(exc).__name__}: {exc} (set PDF_TOOL_DEBUG=1 for a traceback)", err=True)
            raise typer.Exit(1) from None

    return wrapper


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version")] = False,
) -> None:
    pass


@app.command()
@handle_errors
def inspect(
    pdf: PdfArg,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Write JSON here; print a summary instead")] = None,
    pages: PagesOpt = None,
    shapes: Annotated[bool, typer.Option(help="Include vector shapes")] = True,
    tables: Annotated[bool, typer.Option(help="Run table detection")] = True,
) -> None:
    """Analyze layout: pages, text blocks with fonts/colors/bboxes, images, tables, shapes, metadata."""
    from .inspect import inspect_pdf

    doc = inspect_pdf(pdf, pages=pages, shapes=shapes, tables=tables)
    text = write_json(doc, output)
    if output is None:
        typer.echo(text)
        return
    summary = {
        "output": str(output),
        "page_count": doc.meta.page_count,
        "text_layer": doc.meta.text_layer,
        "pages": [
            {
                "page": p.number,
                "size_pt": [p.width, p.height],
                "background": p.background,
                "counts": {t: sum(e.type == t for e in p.elements) for t in ("text", "image", "table", "shape")},
                **({"warnings": p.warnings} if p.warnings else {}),
            }
            for p in doc.pages
        ],
        "fonts": [f"{f.name} ({'embedded' if f.embedded else 'not embedded'}{', subset' if f.subset else ''}) sizes {f.sizes}" for f in doc.fonts],
        "warnings": doc.warnings,
    }
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))


@app.command()
@handle_errors
def extract(
    pdfs: Annotated[list[Path], typer.Argument(help="One or more PDFs (models are loaded once)", dir_okay=False)],
    fmt: Annotated[str, typer.Option("--format", "-f", help="markdown | json | html")] = "markdown",
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output directory (default: <stem>_extract)")] = None,
    engine: Annotated[str, typer.Option(help="marker (structured, OCR) | pymupdf (fast, text layer only)")] = "marker",
    ocr: Annotated[str, typer.Option(help="auto | force | never")] = "auto",
    pages: PagesOpt = None,
    images: Annotated[bool, typer.Option(help="Extract embedded images")] = True,
    device: Annotated[str | None, typer.Option(help="Torch device for Marker: mps | cpu | cuda (default: auto)")] = None,
) -> None:
    """Extract structured content (Markdown/JSON/HTML + images) with Datalab Marker."""
    from .extract import extract as run

    if fmt not in ("markdown", "json", "html"):
        raise InputError("--format must be markdown, json or html")
    if engine not in ("marker", "pymupdf"):
        raise InputError("--engine must be marker or pymupdf")
    if ocr not in ("auto", "force", "never"):
        raise InputError("--ocr must be auto, force or never")
    if output and len(pdfs) > 1:
        results = [run(p, output / p.stem, fmt, engine, ocr, pages, images, device) for p in pdfs]
    else:
        results = [run(p, output or Path(f"{p.stem}_extract"), fmt, engine, ocr, pages, images, device) for p in pdfs]
    slim = [{k: v for k, v in r.items() if k != "marker_metadata"} for r in results]
    typer.echo(json.dumps(slim[0] if len(slim) == 1 else slim, indent=2, ensure_ascii=False, default=str))


@app.command()
@handle_errors
def render(
    pdf: PdfArg,
    dpi: Annotated[int, typer.Option(help="Resolution")] = 150,
    pages: PagesOpt = None,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output directory (default: <stem>_pages)")] = None,
) -> None:
    """Render pages to PNG images (page-001.png, ...)."""
    from .render import render_pdf

    typer.echo(json.dumps(render_pdf(pdf, output or Path(f"{pdf.stem}_pages"), dpi=dpi, pages=pages), indent=2))


@app.command()
@handle_errors
def assets(
    pdf: PdfArg,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output directory (default: <stem>_assets)")] = None,
    min_size: Annotated[int, typer.Option(help="Skip images smaller than this (px)")] = 8,
    vectors: Annotated[bool, typer.Option(help="Export curved vector clusters (logos/icons) as SVG+PNG")] = True,
) -> None:
    """Export images, vector graphics and fonts with ids, page references and manifest.json."""
    from .assets import extract_assets

    out = output or Path(f"{pdf.stem}_assets")
    m = extract_assets(pdf, out, min_size=min_size, vectors=vectors)
    counts = {k: sum(a.kind == k for a in m.assets) for k in ("image", "vector", "font")}
    typer.echo(json.dumps({"manifest": str(out / "manifest.json"), "counts": counts, "warnings": m.warnings}, indent=2))


@app.command()
@handle_errors
def edit(
    pdf: PdfArg,
    changes: Annotated[Path, typer.Option("--changes", "-c", help="JSON edit instructions", dir_okay=False)],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output PDF")],
    overwrite: Annotated[bool, typer.Option(help="Allow replacing an existing file (including the input)")] = False,
    report: Annotated[Path | None, typer.Option(help="Also write the JSON report here")] = None,
) -> None:
    """Apply structured edits (replace/insert/redact text, replace images) to an existing PDF."""
    from .edit import EditError, apply_edits

    try:
        result = apply_edits(pdf, changes, output, overwrite=overwrite)
    except EditError as exc:
        if report:
            write_json(exc.report, report)
        raise
    typer.echo(write_json(result, report))


@app.command()
@handle_errors
def generate(
    html: Annotated[Path, typer.Argument(help="Input HTML file", dir_okay=False)],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output PDF")],
    page_size: Annotated[str | None, typer.Option(help="A4, Letter, A4-landscape, 210x297mm ... (default: CSS @page)")] = None,
    size_from: Annotated[Path | None, typer.Option(help="Use the page size of this PDF", dir_okay=False)] = None,
    size_from_page: Annotated[int, typer.Option(help="Page of --size-from to measure")] = 1,
    margin: Annotated[str | None, typer.Option(help="CSS-style margins, e.g. '20mm' or '18mm 15mm'")] = None,
    header_template: Annotated[Path | None, typer.Option(help="Chromium header template HTML")] = None,
    footer_template: Annotated[Path | None, typer.Option(help="Chromium footer template HTML")] = None,
    root: Annotated[Path | None, typer.Option(help="Directory served to the page (default: the HTML's folder)")] = None,
    allow_js: Annotated[bool, typer.Option(help="Run page JavaScript (off by default)")] = False,
    overwrite: Annotated[bool, typer.Option(help="Allow replacing an existing output file")] = False,
    timeout: Annotated[float, typer.Option(help="Page load timeout in seconds")] = 60,
) -> None:
    """Render HTML/CSS to PDF with headless Chromium. Network and files outside --root are blocked."""
    from .generate import generate_pdf

    result = generate_pdf(
        html, output, page_size=page_size, size_from=size_from, size_from_page=size_from_page, margin=margin,
        header_template=header_template, footer_template=footer_template, root=root, allow_js=allow_js,
        overwrite=overwrite, timeout_s=timeout,
    )  # fmt: skip
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
@handle_errors
def scaffold(
    pdf: PdfArg,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Output directory (default: <stem>_scaffold)")] = None,
    pages: PagesOpt = None,
    calibrate: Annotated[bool, typer.Option(help="Render, measure and correct baselines")] = True,
) -> None:
    """Write an editable HTML replica (index.html + fonts/images) as the start for reconstruction."""
    from .scaffold import scaffold as run

    typer.echo(json.dumps(run(pdf, output or Path(f"{pdf.stem}_scaffold"), pages=pages, calibrate=calibrate), indent=2, ensure_ascii=False))


@app.command()
@handle_errors
def compare(
    reference: Annotated[Path, typer.Argument(help="Reference (original) PDF", dir_okay=False)],
    candidate: Annotated[Path, typer.Argument(help="Candidate (revised) PDF", dir_okay=False)],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output directory")] = Path("comparison"),
    dpi: Annotated[int, typer.Option(help="Render resolution for pixel comparison")] = 100,
    masks: Annotated[list[Path] | None, typer.Option("--masks", help="JSON masks (list, or an edit report incl. its shifts)")] = None,
    mask: Annotated[list[str] | None, typer.Option("--mask", help="PAGE:x0,y0,x1,y1 in points; repeatable")] = None,
    expect: Annotated[list[str] | None, typer.Option("--expect", help="Text that must appear; repeatable")] = None,
    forbid: Annotated[list[str] | None, typer.Option("--forbid", help="Text that must not appear; repeatable")] = None,
    tolerance: Annotated[float, typer.Option(help="Block movement tolerance in points")] = 2.0,
) -> None:
    """Compare PDFs: side-by-side + diff images, metrics, text/bbox diffs, overflow warnings."""
    from .compare import compare_pdfs, load_masks, parse_mask

    all_masks, shifts = [parse_mask(s) for s in mask or []], []
    for f in masks or []:
        file_masks, file_shifts = load_masks(f)
        all_masks += file_masks
        shifts += file_shifts
    rep = compare_pdfs(reference, candidate, output, dpi=dpi, masks=all_masks, shifts=shifts, expect=expect,
                       forbid=forbid, tolerance=tolerance)  # fmt: skip
    summary = {
        "status": rep.status,
        "report": str(output / "report.json"),
        "pages": f"{rep.reference_pages} reference / {rep.candidate_pages} candidate",
        "warnings": rep.warnings,
        "document_missing_text": rep.document_missing_text,
        "document_added_text": rep.document_added_text,
        "expectations": [e.model_dump() for e in rep.expectations],
        "per_page": [
            {
                "page": p.page,
                "ssim": p.ssim,
                "changed_pixels": p.changed_pixels,
                "text_similarity": p.text_similarity,
                "moved_blocks": len(p.moved_blocks),
                "missing_blocks": len(p.missing_blocks),
                "warnings": p.warnings,
                "side_by_side": p.side_by_side,
                "diff": p.diff_image,
            }
            for p in rep.pages
        ],
    }
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))
    if any(not e.ok for e in rep.expectations):
        typer.echo("error: expectation check failed", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    sys.exit(app())
