"""`pdf-tool extract`: structured content extraction with Datalab Marker.

PyMuPDF decides first whether OCR is needed. Digital PDFs with a usable text
layer run Marker with ``disable_ocr`` (layout + structure only, no VLM). OCR on
macOS/CPU uses Surya's llama.cpp backend, which needs the ``llama-server``
binary (installed project-locally by ``scripts/setup.sh --with-ocr``).
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import re
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from .common import InputError, PdfToolError, open_pdf, parse_pages, require_dir_output, write_json
from .inspect import inspect_pdf, page_needs_ocr

Format = Literal["markdown", "json", "html"]
OcrMode = Literal["auto", "force", "never"]
EXT = {"markdown": "md", "json": "json", "html": "html"}
RENDERER = {
    "markdown": "marker.renderers.markdown.MarkdownRenderer",
    "json": "marker.renderers.json.JSONRenderer",
    "html": "marker.renderers.html.HTMLRenderer",
}

_models: dict | None = None


def local_llama_server() -> str | None:
    """llama-server from LLAMA_CPP_BINARY, the venv (setup.sh) or PATH."""
    env = os.environ.get("LLAMA_CPP_BINARY")
    if env and os.path.isfile(env):
        return env
    for cand in sorted((Path(sys.prefix) / "opt" / "llama.cpp").glob("**/llama-server")):
        if os.access(cand, os.X_OK):
            return str(cand)
    return shutil.which("llama-server")


def _configure_env(device: str | None) -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if device:
        os.environ["TORCH_DEVICE"] = device
    binary = local_llama_server()
    if binary:
        os.environ.setdefault("LLAMA_CPP_BINARY", binary)


def marker_models(device: str | None = None) -> dict:
    """Create Marker's predictor set once per process and reuse it."""
    global _models
    if _models is None:
        _configure_env(device)
        try:
            from marker.models import create_model_dict, shutdown_models
        except ImportError as exc:
            raise PdfToolError(f"Marker is not installed: {exc}. Run scripts/setup.sh") from exc
        _models = create_model_dict()
        atexit.register(shutdown_models, _models)
    return _models


def ocr_plan(pdf: Path, pages: list[int], mode: OcrMode) -> dict:
    doc = open_pdf(pdf)
    try:
        need = []
        for i in pages:
            needs, *_ = page_needs_ocr(doc[i])
            if needs:
                need.append(i + 1)
    finally:
        doc.close()
    if mode == "never":
        decision = "disabled by --ocr never" + (f"; pages {need} have no usable text" if need else "")
        return {"mode": mode, "pages_needing_ocr": need, "disable_ocr": True, "force_ocr": False, "decision": decision}
    if mode == "force":
        return {"mode": mode, "pages_needing_ocr": need, "disable_ocr": False, "force_ocr": True, "decision": "forced"}
    if not need:
        decision = "text layer usable on all pages; OCR skipped"
        return {"mode": mode, "pages_needing_ocr": [], "disable_ocr": True, "force_ocr": False, "decision": decision}
    full = len(need) == len(pages)
    decision = "no usable text layer; full OCR" if full else f"OCR for empty/garbled content on pages {need}"
    return {"mode": mode, "pages_needing_ocr": need, "disable_ocr": False, "force_ocr": full, "decision": decision}


def _save_json_images(node: dict, out: Path, files: list[str]) -> None:
    """Move base64 images out of Marker JSON into files; keep file references."""
    images = node.get("images")
    if images:
        for name, b64 in list(images.items()):
            data = base64.b64decode(b64)
            ext = ".png" if data[:4] == b"\x89PNG" else ".jpeg"
            target = out / (name.strip("/").replace("/", "_") + ext)  # '/page/0/Picture/1' -> page_0_Picture_1
            target.write_bytes(data)
            images[name] = target.name
            files.append(target.name)
    for child in node.get("children") or []:
        _save_json_images(child, out, files)


def _run_marker(pdf: Path, out: Path, fmt: Format, pages: list[int], plan: dict, images: bool, device: str | None) -> dict:
    if not plan["disable_ocr"] and not local_llama_server():
        raise PdfToolError(
            "OCR is required ("
            + plan["decision"]
            + ") but Marker's OCR backend needs the llama.cpp `llama-server` binary. "
            "Install it with `scripts/setup.sh --with-ocr` (project-local) or `brew install llama.cpp`, "
            "or rerun with --ocr never for text-layer-only output."
        )
    models = marker_models(device)
    from marker.converters.pdf import PdfConverter
    from marker.settings import settings

    config = {
        "output_format": fmt,
        "page_range": pages,
        "disable_ocr": plan["disable_ocr"],
        "force_ocr": plan["force_ocr"],
        "paginate_output": True,
        "extract_images": images,
    }
    converter = PdfConverter(artifact_dict=models, renderer=RENDERER[fmt], config=config)
    rendered = converter(str(pdf))
    stem = pdf.stem
    files: list[str] = []
    if fmt == "json":
        data = json.loads(rendered.model_dump_json(exclude={"metadata"}))
        if images:
            _save_json_images(data, out, files)
        (out / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        text = rendered.markdown if fmt == "markdown" else rendered.html
        (out / f"{stem}.{EXT[fmt]}").write_text(text, encoding="utf-8")
        for name, img in (rendered.images or {}).items():
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(out / name)
            files.append(name)
    return {
        "page_numbering": "Marker page ids and markdown separators {N} are 0-based (page N+1)",
        "marker_version": version("marker-pdf"),
        "mode": converter.mode,
        "torch_device": settings.TORCH_DEVICE_MODEL,
        "image_files": files,
        "marker_metadata": rendered.metadata,
    }


def _run_pymupdf(pdf: Path, out: Path, fmt: Format, pages: list[int], images: bool) -> dict:
    stem = pdf.stem
    page_spec = ",".join(str(p + 1) for p in pages)
    if fmt == "json":
        write_json(inspect_pdf(pdf, pages=page_spec), out / f"{stem}.json")
    elif fmt == "html":
        doc = open_pdf(pdf)
        try:
            body = "\n".join(
                f'<section class="page" data-page="{i + 1}">\n{doc[i].get_text("xhtml")}\n</section>' for i in pages
            )
            # MuPDF writes non-ASCII as numeric entities; keep markup entities (&lt; etc.) intact
            body = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)) if int(m.group(1), 16) > 127 else m.group(0), body)
        finally:
            doc.close()
        html = f'<!doctype html>\n<html><head><meta charset="utf-8"><title>{stem}</title></head>\n<body>\n{body}\n</body></html>\n'
        (out / f"{stem}.html").write_text(html, encoding="utf-8")
    else:
        import pymupdf4llm

        md = pymupdf4llm.to_markdown(
            str(pdf), pages=pages, page_separators=True, write_images=images, image_path=str(out), show_progress=False
        )
        (out / f"{stem}.md").write_text(md, encoding="utf-8")
    return {}


def extract(
    pdf: str | Path,
    out_dir: str | Path,
    fmt: Format = "markdown",
    engine: Literal["marker", "pymupdf"] = "marker",
    ocr: OcrMode = "auto",
    pages: str | None = None,
    images: bool = True,
    device: str | None = None,
) -> dict:
    pdf = Path(pdf)
    if fmt not in EXT:
        raise InputError(f"unsupported format {fmt!r}")
    doc = open_pdf(pdf)
    selected = parse_pages(pages, doc.page_count)
    doc.close()
    out = require_dir_output(Path(out_dir))
    plan = ocr_plan(pdf, selected, ocr)
    started = time.time()
    if engine == "marker":
        details = _run_marker(pdf, out, fmt, selected, plan, images, device)
    elif engine == "pymupdf":
        if plan["pages_needing_ocr"] and ocr != "never":
            plan["decision"] += "; pymupdf engine cannot OCR, these pages will be empty"
        details = _run_pymupdf(pdf, out, fmt, selected, images)
    else:
        raise InputError(f"unknown engine {engine!r}")
    meta = {
        "source": str(pdf.resolve()),
        "engine": engine,
        "format": fmt,
        "output": str(out / f"{pdf.stem}.{EXT[fmt]}"),
        "pages": [p + 1 for p in selected],
        "ocr": plan,
        "elapsed_s": round(time.time() - started, 2),
        **details,
    }
    write_json(meta, out / f"{pdf.stem}_meta.json")
    return meta
