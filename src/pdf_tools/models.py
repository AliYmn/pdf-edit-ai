"""Normalized document model shared by all commands.

Coordinates are PDF points (1/72 inch) with the origin at the top-left of the
page, x to the right and y downwards (PyMuPDF convention). Bounding boxes are
``(x0, y0, x1, y1)``.

Every element carries ``provenance``:
- ``extracted``: read directly from the PDF content (text spans, image placements,
  vector paths, embedded fonts).
- ``inferred``: produced by a heuristic (table detection, page background,
  reading order, OCR need). Treat these as hints, not facts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]
Provenance = Literal["extracted", "inferred"]

SCHEMA_VERSION = "1.0"


class PdfToolError(Exception):
    """Error with a CLI exit code. 1 = processing failure, 2 = invalid input."""

    exit_code = 1


class InputError(PdfToolError):
    exit_code = 2


class Span(BaseModel):
    text: str
    bbox: BBox
    origin: tuple[float, float] = Field(description="Baseline start point")
    font: str
    size: float
    color: str
    bold: bool
    italic: bool
    serif: bool
    monospace: bool


class Line(BaseModel):
    bbox: BBox
    spans: list[Span]


class Typography(BaseModel):
    """Dominant style of a text block, weighted by character count."""

    font: str
    size: float
    color: str
    bold: bool
    italic: bool
    line_height: float | None = Field(
        None, description="Median baseline-to-baseline distance in points (multi-line blocks only)"
    )


class Table(BaseModel):
    rows: int
    cols: int
    header: list[str | None] = []
    cells: list[list[str | None]]
    cell_bboxes: list[list[BBox | None]] = []


class Element(BaseModel):
    id: str
    type: Literal["text", "image", "table", "shape"]
    page: int
    bbox: BBox
    provenance: Provenance
    source: str
    reading_order: int | None = Field(None, description="Inferred top-to-bottom, left-to-right order")
    text: str | None = None
    typography: Typography | None = None
    lines: list[Line] | None = None
    asset_id: str | None = None
    image_size: tuple[int, int] | None = Field(None, description="Intrinsic image size in pixels")
    table: Table | None = None
    fill: str | None = None
    stroke: str | None = None
    stroke_width: float | None = None


class Link(BaseModel):
    bbox: BBox
    uri: str | None = Field(None, description="External target (https:, mailto:, tel:, ...)")
    target_page: int | None = Field(None, description="1-based page for internal links")


class Page(BaseModel):
    number: int = Field(description="1-based page number")
    width: float
    height: float
    rotation: int
    background: str | None = Field(None, description="Inferred dominant page-edge color")
    text_chars: int
    image_coverage: float = Field(description="Fraction of page area covered by images")
    needs_ocr: bool = Field(description="Inferred: little or no usable text layer")
    elements: list[Element]
    links: list[Link] = []
    warnings: list[str] = []


class FontInfo(BaseModel):
    name: str = Field(description="Font name without subset prefix")
    basefont: str
    type: str
    embedded: bool
    subset: bool
    xrefs: list[int]
    pages: list[int]
    sizes: list[float] = Field([], description="Sizes seen in text spans")
    chars: int = Field(0, description="Characters drawn with this font")
    span_names: list[str] = Field([], description="Names text spans report for this font (font program name)")


class DocumentMeta(BaseModel):
    source: str
    page_count: int
    metadata: dict[str, str]
    pdf_format: str | None = None
    text_layer: Literal["full", "partial", "none"]


class Document(BaseModel):
    schema_version: str = SCHEMA_VERSION
    coordinate_system: str = "PDF points, origin top-left, y down; bbox = (x0, y0, x1, y1)"
    meta: DocumentMeta
    fonts: list[FontInfo]
    pages: list[Page]
    warnings: list[str] = []


class Placement(BaseModel):
    page: int
    bbox: BBox


class Asset(BaseModel):
    id: str
    kind: Literal["image", "vector", "font"]
    file: str | None = Field(None, description="Path relative to the manifest")
    pages: list[int] = []
    placements: list[Placement] = []
    width: int | None = None
    height: int | None = None
    format: str | None = None
    role: str | None = Field(None, description="Inferred role such as 'logo-candidate'")
    sha256: str | None = None
    xref: int | None = None
    font_name: str | None = None
    subset: bool | None = None
    web_usable: bool | None = None
    has_alpha: bool | None = Field(None, description="Image has transparency (soft mask)")
    colorspace: str | None = Field(None, description="Image colour space as stored, e.g. DeviceRGB, DeviceCMYK")
    effective_dpi: float | None = Field(None, description="Lowest pixels-per-inch over its placements")
    notes: list[str] = []


class AssetManifest(BaseModel):
    source: str
    assets: list[Asset]
    warnings: list[str] = []
