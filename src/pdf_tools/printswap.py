"""Swap a mark printed on a photo (a logo or a word on a product) for new content,
in the same ink colour, size, angle and position.

1. Ink inside a polygon is separated from its background (Otsu); only letter-shaped
   components mostly inside the polygon count, so neighbouring lines are left alone.
2. Its rotated bounding box and ink colour (strongest ink pixels) are measured.
3. The ink is removed by inpainting from the surroundings.
4. The new content is drawn into exactly that box: a logo image (its alpha, tinted
   with the ink colour) or text mapped with the metrics of the old text, so letter
   proportions and the baseline match. ``shear`` keeps vertical strokes vertical
   (print seen in perspective); ``rotate`` is for designs printed at an angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import PdfToolError

DEFAULT_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
SUPERSAMPLE = 4


@dataclass
class PrintSpec:
    polygon: list[tuple[float, float]]
    text: str | None = None  # new text, or None when ``logo`` is given
    old: str | None = None  # the printed text being replaced (for its metrics)
    logo: str | None = None  # image file whose alpha is the new mark
    ink: str = "light"  # light ink on a dark surface, "dark", or "auto" (anything unlike the surface)
    mode: str = "shear"  # "shear" (perspective) or "rotate" (design printed at an angle)
    clip: list[tuple[float, float]] | None = None  # visible area (e.g. not behind a cutaway)
    grow: float = 1.0  # widen the box for small blurry print whose ink core is thinner
    stroke: int = 0  # extra weight for text (stroke width at 200 px font size)
    min_aspect: float | None = None  # extend a partly hidden mark to at least this width/height
    font_file: str | None = None
    color: str | None = None  # ink of the new mark (default: the measured ink colour)
    refill: str = "inpaint"  # "surface": refill from surface-coloured pixels only (flat labels next to other print)
    threshold: float | None = None  # auto ink: colour distance from the surface that counts as ink
    fit: str = "stretch"  # logo: "stretch" to the old box, or "contain" (keep the logo's proportions)


def _surface_distance(img: np.ndarray, pm: np.ndarray) -> np.ndarray:
    """Colour distance of every pixel from the surface around the polygon (median of its rim)."""
    rim = cv2.dilate(pm, np.ones((7, 7), np.uint8)) - pm
    surface = np.median(img[rim > 0].reshape(-1, 3), axis=0) if rim.any() else np.median(img[pm > 0], axis=0)
    return np.linalg.norm(img.astype(np.float32) - surface, axis=2)


def _surface_fill(img: np.ndarray, out: np.ndarray, remove: np.ndarray, poly, H: float) -> np.ndarray:
    """Refill the removed area from nearby surface pixels only (normalised convolution).

    Inpainting pulls in whatever borders the hole - including neighbouring print, which leaves
    dark smudges on a flat label. Pixels that look like the surface around the polygon (and
    follow its gradients) are averaged instead; where too few exist, the inpainting stays."""
    h, w = img.shape[:2]
    pm = cv2.fillPoly(np.zeros((h, w), np.uint8), [np.int32(poly)], 1)
    dist = _surface_distance(img, pm)
    valid = (~remove & (dist < 45)).astype(np.float32)
    sigma = max(3.0, H * 0.6)
    num = cv2.GaussianBlur(img.astype(np.float32) * valid[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(valid, (0, 0), sigma)
    ok = remove & (den > 0.05)
    bg = num / np.maximum(den, 1e-6)[..., None]
    res = out.copy()
    res[ok] = bg[ok]
    # soften the seam between the refilled area and the untouched photo
    soft = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), 1.0)[..., None]
    return out * (1 - soft) + np.where(ok[..., None], res, out) * soft


def ink_mask(img: np.ndarray, poly, ink: str = "light", threshold: float | None = None) -> np.ndarray:
    """Letter-shaped ink inside the polygon: components with at least half their pixels inside."""
    h, w = img.shape[:2]
    pm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(pm, [np.int32(poly)], 1)
    if not pm.any():
        raise PdfToolError("print polygon is empty or outside the image")
    if ink == "auto":  # multi-coloured marks (e.g. a red symbol next to white letters)
        dist = np.clip(_surface_distance(img, pm), 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(dist[pm > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cand = dist > (threshold if threshold else min(max(t, 25), 60))  # a dim red symbol next to white letters still counts
    else:
        lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        t, _ = cv2.threshold(lum[pm > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cand = (lum > t) if ink == "light" else (lum < t)
    region = cv2.dilate(pm, np.ones((9, 9), np.uint8))
    cand = (cand & (region > 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
    inside = np.bincount(lab[pm > 0].ravel(), minlength=n)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        keep[k] = st[k, 4] >= 2 and inside[k] >= 0.5 * st[k, 4]
    mask = keep[lab]
    if not mask.any():
        raise PdfToolError("no print found inside the polygon")
    return mask


def rotated_box(mask: np.ndarray) -> tuple[tuple[float, float], float, float, float]:
    """((cx, cy), width along the text, height, angle in degrees)."""
    ys, xs = np.nonzero(mask)
    rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    box = cv2.boxPoints(rect)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    long_e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    W, H = max(np.linalg.norm(e1), np.linalg.norm(e2)), min(np.linalg.norm(e1), np.linalg.norm(e2))
    angle = math.degrees(math.atan2(long_e[1], long_e[0]))
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180
    if abs(angle) > 60:  # a near-vertical long side means a tall box: keep the text direction horizontal
        angle = angle - 90 if angle > 0 else angle + 90
        W, H = H, W
    return rect[0], float(W), float(H), angle


def _font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    for p in [path] if path else DEFAULT_FONTS:
        if p and Path(p).is_file():
            return ImageFont.truetype(p, size)
    raise PdfToolError("no bold sans font found for print text; pass font_file")


def _text_tile(old: str, new: str, W: float, H: float, font_file: str | None, stroke: int):
    """``new`` rendered with the transform that maps ``old`` (same font and size) onto W x H."""
    f = _font(font_file, 200)

    def ink(text):
        l, t, r, b = f.getbbox(text, stroke_width=stroke)
        im = Image.new("L", (r + 40, b + 40), 0)
        ImageDraw.Draw(im).text((20, 20), text, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
        arr = np.array(im)
        ys, xs = np.nonzero(arr > 20)
        return arr, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    _, (ox0, oy0, ox1, oy1) = ink(old)
    arr, _ = ink(new)
    S = SUPERSAMPLE
    sx, sy = W * S / (ox1 - ox0), H * S / (oy1 - oy0)
    Wp, Hp = max(1, round(W * S)), max(1, round(H * S))
    pad = int(Hp * 0.6)  # room for ascenders/descenders outside the old text's box
    M = np.float32([[sx, 0, -ox0 * sx], [0, sy, -oy0 * sy + pad]])
    tile = cv2.warpAffine(arr.astype(np.float32), M, (int(Wp * 1.4) + 4, Hp + 2 * pad), flags=cv2.INTER_AREA)
    return np.clip(tile, 0, 255) / 255, float(pad)


def _logo_tile(path: str, W: float, H: float, fit: str = "stretch"):
    p = Path(path)
    if not p.is_file():
        raise PdfToolError(f"logo file not found: {p}")
    with Image.open(p) as im:
        im = im.convert("RGBA")
        alpha = np.array(im)[..., 3]
    if alpha.min() == 255:  # no transparency: use darkness as coverage
        alpha = 255 - np.array(im.convert("L"))
    ys, xs = np.nonzero(alpha > 10)
    alpha = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    S = SUPERSAMPLE
    Wp, Hp = max(1, round(W * S)), max(1, round(H * S))
    if fit == "contain":  # keep the logo's proportions, centred in the old box
        k = min(Wp / alpha.shape[1], Hp / alpha.shape[0])
        w, h = max(1, round(alpha.shape[1] * k)), max(1, round(alpha.shape[0] * k))
        small = np.array(Image.fromarray(alpha).resize((w, h), Image.LANCZOS))
        tile = np.zeros((Hp, Wp), np.uint8)
        tile[(Hp - h) // 2:(Hp - h) // 2 + h, (Wp - w) // 2:(Wp - w) // 2 + w] = small
    else:
        tile = np.array(Image.fromarray(alpha).resize((Wp, Hp), Image.LANCZOS))
    return tile.astype(np.float32) / 255, 0.0


def swap_print(img: np.ndarray, spec: PrintSpec) -> tuple[np.ndarray, dict]:
    """Apply one print swap to an RGB image; returns the new image and what was measured."""
    if (spec.text is None) == (spec.logo is None):
        raise PdfToolError("a print swap needs exactly one of 'text' or 'logo'")
    if spec.text is not None and not spec.old:
        raise PdfToolError("a text print swap needs 'old' (the printed text being replaced)")
    h, w = img.shape[:2]
    mask = ink_mask(img, spec.polygon, spec.ink, spec.threshold)
    clip_m = None
    if spec.clip is not None:
        clip_m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(clip_m, [np.int32(spec.clip)], 1)
        mask &= cv2.erode(clip_m, np.ones((2, 2), np.uint8)).astype(bool)
        if not mask.any():
            raise PdfToolError("no print found inside the clip area")
    (cx, cy), W, H, angle = rotated_box(mask)
    if spec.min_aspect and W < spec.min_aspect * H:  # the rest of the mark is hidden: extend it
        d = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        ys, xs = np.nonzero(mask)
        start = (np.column_stack([xs, ys]) @ d).min()
        Wn = spec.min_aspect * H
        mid = np.array([cx, cy]) + d * ((start + Wn / 2) - (np.array([cx, cy]) @ d))
        (cx, cy), W = (float(mid[0]), float(mid[1])), Wn
    W, H = W + (spec.grow - 1) * H, H * spec.grow
    if spec.ink == "auto":
        dist = _surface_distance(img, cv2.fillPoly(np.zeros((h, w), np.uint8), [np.int32(spec.polygon)], 1))[mask]
        strong = dist >= np.percentile(dist, 75)
    else:
        lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[mask]
        strong = lum >= np.percentile(lum, 75) if spec.ink == "light" else lum <= np.percentile(lum, 25)
    color = np.median(img[mask][strong], axis=0)
    measured = color
    if spec.color:
        hx = spec.color.lstrip("#")
        color = np.array([int(hx[k:k + 2], 16) for k in (0, 2, 4)], np.float32)
    k = max(5, int(H * 0.12) | 1)  # large print has wider soft edges that must go too
    remove = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((k, k), np.uint8))
    if clip_m is not None:
        remove &= clip_m * 255
    out = cv2.inpaint(img, remove, 4, cv2.INPAINT_TELEA).astype(np.float32)
    if spec.refill == "surface":
        out = _surface_fill(img, out, remove > 0, spec.polygon, H)
    if spec.logo:
        tile, v0 = _logo_tile(spec.logo, W, H, spec.fit)
    else:
        tile, v0 = _text_tile(spec.old, spec.text, W, H, spec.font_file, spec.stroke)
    S = SUPERSAMPLE
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    bx, by = W * S / 2, v0 + H * S / 2  # centre of the old box inside the tile
    if spec.mode == "rotate":
        M = np.float32([[c / S, -s / S, cx - c * bx / S + s * by / S], [s / S, c / S, cy - s * bx / S - c * by / S]])
    else:  # shear: the baseline slopes, vertical strokes stay vertical; vertical height = H / cos
        M = np.float32([[c / S, 0, cx - c * bx / S], [s / S, 1 / (S * c), cy - s * bx / S - by / (S * c)]])
    # warpAffine samples bilinearly (INTER_AREA is not applied to affine warps): low-pass the
    # supersampled tile first, or thin strokes of small print fall between the samples
    tile = cv2.GaussianBlur(tile, (0, 0), S * 0.45)
    alpha = cv2.warpAffine(tile, M, (w, h), flags=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.35, H * 0.012))  # match the photo's softness
    if clip_m is not None:
        alpha *= clip_m
    out = out * (1 - alpha[..., None]) + color * alpha[..., None]
    info = {
        "center": [round(float(cx), 1), round(float(cy), 1)],
        "size": [round(float(W), 1), round(float(H), 1)],
        "angle": round(float(angle), 2),
        "ink_color": "#" + "".join(f"{int(v):02x}" for v in measured),
        "new_color": "#" + "".join(f"{int(v):02x}" for v in color),
        "ink_pixels": int(mask.sum()),
    }
    return np.clip(out, 0, 255).astype(np.uint8), info
