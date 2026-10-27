"""Convert PyMuPDF vector drawings (`page.get_drawings()`) to SVG."""

from __future__ import annotations

import re

import pymupdf

from .common import floats_to_hex

LINECAP = {0: "butt", 1: "round", 2: "square"}
LINEJOIN = {0: "miter", 1: "round", 2: "bevel"}


def _n(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def path_d(drawing: dict, dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> str:
    """SVG path data for one drawing, translated by (dx, dy) then scaled."""
    parts: list[str] = []
    cur = None

    def pt(p) -> str:
        return f"{_n((p.x + dx) * scale)} {_n((p.y + dy) * scale)}"

    def xs(v: float) -> str:
        return _n((v + dx) * scale)

    def ys(v: float) -> str:
        return _n((v + dy) * scale)

    for item in drawing["items"]:
        op = item[0]
        if op == "re":
            r = item[1]
            parts.append(f"M{xs(r.x0)} {ys(r.y0)}H{xs(r.x1)}V{ys(r.y1)}H{xs(r.x0)}Z")
            cur = None
        elif op == "qu":
            q = item[1]
            parts.append(f"M{pt(q.ul)}L{pt(q.ur)}L{pt(q.lr)}L{pt(q.ll)}Z")
            cur = None
        else:
            start = item[1]
            if cur is None or abs(start.x - cur.x) > 1e-3 or abs(start.y - cur.y) > 1e-3:
                parts.append(f"M{pt(start)}")
            if op == "l":
                parts.append(f"L{pt(item[2])}")
                cur = item[2]
            elif op == "c":
                parts.append(f"C{pt(item[2])} {pt(item[3])} {pt(item[4])}")
                cur = item[4]
    if drawing.get("closePath"):
        parts.append("Z")
    return "".join(parts)


def _first(v, default=0):
    if isinstance(v, (tuple, list)):
        return v[0] if v else default
    return default if v is None else v


def path_element(drawing: dict, dx: float = 0.0, dy: float = 0.0) -> str:
    attrs = [f'd="{path_d(drawing, dx, dy)}"']
    fill = floats_to_hex(drawing.get("fill"))
    attrs.append(f'fill="{fill or "none"}"')
    if fill and drawing.get("even_odd"):
        attrs.append('fill-rule="evenodd"')
    if fill and (drawing.get("fill_opacity") or 1) < 1:
        attrs.append(f'fill-opacity="{_n(drawing["fill_opacity"])}"')
    stroke = floats_to_hex(drawing.get("color"))
    if stroke:
        attrs.append(f'stroke="{stroke}" stroke-width="{_n(drawing.get("width") or 1)}"')
        attrs.append(f'stroke-linecap="{LINECAP.get(int(_first(drawing.get("lineCap"))), "butt")}"')
        attrs.append(f'stroke-linejoin="{LINEJOIN.get(int(_first(drawing.get("lineJoin"))), "miter")}"')
        dashes = re.findall(r"[\d.]+", (drawing.get("dashes") or "").split("]")[0])
        if dashes and any(float(x) for x in dashes):
            attrs.append(f'stroke-dasharray="{" ".join(dashes)}"')
        if (drawing.get("stroke_opacity") or 1) < 1:
            attrs.append(f'stroke-opacity="{_n(drawing["stroke_opacity"])}"')
    return "<path " + " ".join(attrs) + "/>"


def padded_rect(drawings: list[dict]) -> pymupdf.Rect:
    """Union of drawing rects, grown by half the widest stroke so nothing is clipped."""
    r = pymupdf.Rect()
    pad = 0.0
    for d in drawings:
        r |= d["rect"]
        if d.get("color"):
            pad = max(pad, (d.get("width") or 1) / 2)
    return r + (-pad, -pad, pad, pad)


def svg_document(drawings: list[dict], rect: pymupdf.Rect | None = None) -> str:
    """Standalone SVG of the drawings, origin moved to the rect's top-left (pt units)."""
    rect = rect or padded_rect(drawings)
    body = "".join(path_element(d, -rect.x0, -rect.y0) for d in drawings)
    w, h = _n(rect.width), _n(rect.height)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}pt" height="{h}pt" viewBox="0 0 {w} {h}">'
        f"{body}</svg>\n"
    )


def is_graphic(drawings: list[dict]) -> bool:
    """True for icon/logo-like groups: curves, diagonal lines or nested fills.
    Plain rules, boxes and table grids are not graphics."""
    for d in drawings:
        for item in d["items"]:
            if item[0] in ("c", "qu"):
                return True
            if item[0] == "l" and abs(item[1].x - item[2].x) > 0.5 and abs(item[1].y - item[2].y) > 0.5:
                return True
    fills = [d["rect"] for d in drawings if d.get("fill")]
    return any(a != b and a.contains(b) for a in fills for b in fills)


def is_background(drawing: dict, page_rect: pymupdf.Rect) -> bool:
    """Large fill-only rectangles (page/body/header backgrounds)."""
    r = drawing["rect"]
    return bool(drawing.get("fill")) and not drawing.get("color") and (
        r.width >= 0.9 * page_rect.width or abs(r) >= 0.25 * abs(page_rect)
    )
