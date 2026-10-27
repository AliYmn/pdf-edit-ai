"""Small helpers shared by the commands."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pymupdf

from .models import InputError, PdfToolError  # noqa: F401 - re-exported for the command modules

SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")


def open_pdf(path: str | Path) -> pymupdf.Document:
    """Open a PDF for reading. Embedded JavaScript is never executed by PyMuPDF."""
    p = Path(path)
    if not p.is_file():
        raise InputError(f"file not found: {p}")
    try:
        doc = pymupdf.open(p)
    except Exception as exc:  # PyMuPDF raises various types for corrupt input
        raise InputError(f"cannot open {p} as PDF: {exc}") from exc
    if not doc.is_pdf:
        doc.close()
        raise InputError(f"{p} is not a PDF")
    if doc.needs_pass:
        doc.close()
        raise InputError(f"{p} is password-protected; decrypt it first")
    if doc.page_count == 0:
        doc.close()
        raise InputError(f"{p} has no pages")
    return doc


def parse_pages(spec: str | None, page_count: int) -> list[int]:
    """Parse a 1-based page spec like "1,3-5" into sorted 0-based indices."""
    if not spec or spec.strip().lower() == "all":
        return list(range(page_count))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise InputError(f"invalid page spec {part!r}; use e.g. '1,3-5'")
        a = int(m.group(1))
        b = int(m.group(2) or a)
        if a < 1 or b < a or b > page_count:
            raise InputError(f"page range {part!r} outside 1-{page_count}")
        pages.update(range(a - 1, b))
    return sorted(pages)


def int_to_hex(color: int) -> str:
    return f"#{color & 0xFFFFFF:06x}"


def floats_to_hex(color: tuple[float, ...] | list[float] | None) -> str | None:
    if not color:
        return None
    if len(color) == 1:  # gray
        color = (color[0],) * 3
    elif len(color) == 4:  # CMYK -> RGB, naive
        c, m, y, k = color
        color = ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return "#" + "".join(f"{round(max(0.0, min(1.0, v)) * 255):02x}" for v in color[:3])


def hex_to_floats(color: str) -> tuple[float, float, float]:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", color.strip())
    if not m:
        raise InputError(f"invalid color {color!r}; use '#rrggbb'")
    h = m.group(1)
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def strip_subset(name: str) -> str:
    return SUBSET_PREFIX.sub("", name)


def rnd(values, nd: int = 2):
    return tuple(round(float(v), nd) for v in values)


def ensure_writable_output(output: Path, inputs: list[Path], overwrite: bool) -> None:
    """Refuse to clobber an input or an existing file unless overwrite is set."""
    out = output.resolve()
    if out.exists() and out.is_dir():
        raise InputError(f"output {output} is a directory")
    if not overwrite:
        if any(out == i.resolve() for i in inputs):
            raise InputError(f"refusing to overwrite input file {output}; pass --overwrite to allow")
        if out.exists():
            raise InputError(f"{output} already exists; pass --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def atomic_output(output: Path) -> Iterator[Path]:
    """Yield a temp path next to ``output``; move it into place only on success."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=output.suffix, dir=output.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        yield tmp_path
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)  # mkstemp creates 0600; give the result normal permissions
        os.replace(tmp_path, output)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_json(data, output: Path | None) -> str:
    """Serialize pydantic models / plain data; write to ``output`` if given."""
    if hasattr(data, "model_dump_json"):
        text = data.model_dump_json(indent=2, exclude_none=True)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return text


def require_dir_output(path: Path) -> Path:
    if path.exists() and not path.is_dir():
        raise InputError(f"{path} exists and is not a directory")
    path.mkdir(parents=True, exist_ok=True)
    return path

