"""Build examples/original.pdf: a two-page business report drawn with PyMuPDF.

It is deliberately *not* made with the HTML generator, so reconstructing it with
HTML/CSS is a real fidelity test. Run: .venv/bin/python examples/make_sample.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

OUT = Path(__file__).with_name("original.pdf")
W, H = 595.28, 841.89
NAVY = (15 / 255, 61 / 255, 110 / 255)
GRAY = (0.42, 0.42, 0.42)
M = 56  # side margin
CSS = "body {font-family: sans-serif; font-size: 10.5pt; line-height: 1.4; color: #222; text-align: justify;}"


def logo() -> bytes:
    img = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, 159, 159], radius=34, fill=(255, 255, 255, 255))
    d.polygon([(38, 122), (38, 38), (62, 38), (98, 92), (98, 38), (122, 38), (122, 122), (98, 122), (62, 68), (62, 122)],
              fill=(15, 61, 110, 255))  # fmt: skip
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def text(page, xy, s, size, bold=False, color=(0.13, 0.13, 0.13)):
    tw = pymupdf.TextWriter(page.rect)
    tw.append(xy, s, font=pymupdf.Font("hebo" if bold else "helv"), fontsize=size)
    tw.write_text(page, color=color)


def right_text(page, x1, y, s, size, bold=False, color=(0.13, 0.13, 0.13)):
    w = pymupdf.Font("hebo" if bold else "helv").text_length(s, size)
    text(page, (x1 - w, y), s, size, bold, color)


def chrome(page, n, total, logo_png):
    page.draw_rect(pymupdf.Rect(0, 0, W, 70), color=None, fill=NAVY)
    page.insert_image(pymupdf.Rect(M, 17, M + 36, 53), stream=logo_png)
    text(page, (M + 48, 41), "Northwind Analytics", 16, bold=True, color=(1, 1, 1))
    right_text(page, W - M, 41, "Quarterly Business Review", 10, color=(0.85, 0.9, 0.97))
    page.draw_line((M, H - 48), (W - M, H - 48), color=(0.75, 0.78, 0.82), width=0.6)
    text(page, (M, H - 34), "Northwind Analytics · Confidential", 8, color=GRAY)
    right_text(page, W - M, H - 34, f"Page {n} of {total}", 8, color=GRAY)


def heading(page, y, s):
    text(page, (M, y), s, 13, bold=True, color=NAVY)


def para(page, y0, y1, html):
    page.insert_htmlbox(pymupdf.Rect(M, y0, W - M, y1), f"<p>{html}</p>", css=CSS)


def table(page, y0, rows):
    widths = [200, 95, 95, 93.28]
    x = [M]
    for w in widths:
        x.append(x[-1] + w)
    rh = 22
    for r, row in enumerate(rows):
        top = y0 + r * rh
        if r == 0:
            page.draw_rect(pymupdf.Rect(M, top, W - M, top + rh), color=None, fill=(0.91, 0.933, 0.965))
        for c, cell in enumerate(row):
            base = top + 14.5
            if c == 0:
                text(page, (x[c] + 6, base), cell, 9.5, bold=r == 0)
            else:
                right_text(page, x[c + 1] - 6, base, cell, 9.5, bold=r == 0)
        page.draw_line((M, top + rh), (W - M, top + rh), color=(0.69, 0.74, 0.8), width=0.75)
    page.draw_line((M, y0), (W - M, y0), color=(0.69, 0.74, 0.8), width=0.75)
    return y0 + rh * len(rows)


def main() -> None:
    lg = logo()
    doc = pymupdf.open()
    p1 = doc.new_page(width=W, height=H)
    chrome(p1, 1, 2, lg)
    text(p1, (M, 112), "Q1 2024 Performance Summary", 20, bold=True, color=NAVY)
    text(p1, (M, 132), "Prepared for: Yıldız Lojistik A.Ş.  ·  Date: 12 April 2024", 10, color=GRAY)
    heading(p1, 168, "1. Overview")
    para(p1, 178, 270,
         "Northwind Analytics closed the first quarter of 2024 ahead of plan. Revenue grew across all three "
         "regions, driven by new logistics customers in Istanbul and Ankara and by higher utilisation of the "
         "route-optimisation platform. Operating costs stayed within budget despite the expansion of the "
         "support team, and customer churn fell to its lowest level since the service launched.")  # fmt: skip
    heading(p1, 300, "2. Key Metrics")
    end = table(p1, 312, [
        ("Metric", "Q4 2023", "Q1 2024", "Change"),
        ("Revenue (EUR)", "3.8M", "4.2M", "+10.5%"),
        ("Active customers", "212", "247", "+16.5%"),
        ("Platform utilisation", "71%", "78%", "+7 pts"),
        ("Monthly churn", "2.1%", "1.4%", "-0.7 pts"),
    ])  # fmt: skip
    heading(p1, end + 34, "3. Outlook")
    para(p1, end + 44, end + 140,
         "For the second quarter we expect continued growth as two enterprise contracts move from pilot to "
         "production. Hiring will focus on data engineering, and the next platform release will add carbon "
         "reporting for every delivery route.")  # fmt: skip

    p2 = doc.new_page(width=W, height=H)
    chrome(p2, 2, 2, lg)
    heading(p2, 112, "4. Risks")
    y = 134
    for item in (
        "Fuel price volatility may compress customer margins and delay renewals.",
        "Integration backlog with two ERP vendors could slow onboarding in Q2.",
        "Currency movements (EUR/TRY) affect reported revenue from Turkish customers.",
    ):
        text(p2, (M + 4, y), "•", 10.5, color=NAVY)
        text(p2, (M + 18, y), item, 10.5)
        y += 18
    heading(p2, y + 22, "5. Next Steps")
    para(p2, y + 32, y + 120,
         "The leadership team will review the hiring plan in May and present the carbon-reporting roadmap to "
         "key customers in June. A detailed budget update will follow with the half-year report.")  # fmt: skip
    doc.set_metadata({"title": "Q1 2024 Performance Summary", "author": "Northwind Analytics"})
    doc.save(OUT, garbage=3, deflate=True)
    print(OUT)


if __name__ == "__main__":
    main()
