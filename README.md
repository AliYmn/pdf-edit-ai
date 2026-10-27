# pdf-agent-tools

A local PDF toolkit for AI coding agents (Claude Code, Codex CLI). It reads a PDF's
structure and style, edits small things in place, renders HTML/CSS reconstructions with
Chromium, and compares the result against the original, both visually and structurally.

The agent does the reasoning. `pdf-tool` does the measuring, editing, rendering and checking.
Everything runs locally. The only downloads are dependencies, Chromium and Marker model weights
at setup or first use.

```
inspect ─► render ─► extract / assets ─► (edit | HTML + generate) ─► render ─► compare ─► fix
```

## Install

Primary target: macOS on Apple Silicon. Linux works the same way. The OCR backend binary is
pinned for macOS only (see Limitations).

```bash
scripts/setup.sh                 # .venv (Python 3.12 via uv, or a local python3.11-3.13) + Playwright Chromium
scripts/setup.sh --with-ocr      # + llama.cpp `llama-server` in .venv/opt (needed to OCR scanned PDFs)
scripts/setup.sh --warm-models   # + download Marker models now instead of on first use
```

What the script does:
- creates `.venv` and installs the package with `pip install -e .[dev]`; no global Python packages are touched,
- installs Playwright's Chromium into Playwright's own cache,
- reports the torch backend. Marker uses **fast** mode on MPS/CPU and **balanced** mode on CUDA.
  CUDA is never forced on macOS,
- with `--with-ocr`, downloads a pinned llama.cpp build (`b11480`), checks its SHA-256 and unpacks it into
  `.venv/opt/llama.cpp`. Marker 2.0 runs OCR through Surya's VLM, and on macOS/CPU that means
  a `llama-server` process with Metal. `pdf-tool` finds this binary automatically. An existing
  `llama-server` on `PATH` or `LLAMA_CPP_BINARY` also works.

Model weights go to `~/.cache/huggingface`: about 140 MB of layout models for digital PDFs, plus
about 1.5 GB of Surya OCR weights the first time a scanned PDF is processed.

Then use `.venv/bin/pdf-tool` or `source .venv/bin/activate`.

Installed and tested here: Python 3.12.12, PyMuPDF 2.0, marker-pdf 2.0.0 (surya-ocr 0.22.1, torch
2.14.1 with MPS), Playwright 1.64.1 (Chromium 156), opencv-python-headless 4.11, Pillow 10.4
(pinned below 11 by Marker), pydantic 2.14, typer 0.27, pytest 9.1.

## Commands

All coordinates are PDF points (1/72 in), with the origin at the top-left: `bbox = [x0, y0, x1, y1]`.
Page numbers are 1-based, and `--pages` takes `1,3-5`.
Exit codes: `0` success, `1` processing failure, `2` invalid input or refused overwrite.
Paths with spaces work. Existing files, including the input PDF, are never overwritten without `--overwrite`.

| Command | Purpose |
|---|---|
| `pdf-tool inspect in.pdf -o analysis.json` | Layout JSON: pages, text blocks → lines → spans (font, size, color, bold/italic, baseline origin), line pitch, images (asset ids), tables (cells and cell boxes), vector shapes (fill/stroke), page background, fonts (embedded/subset, sizes used), metadata, OCR need. With `-o`, a short summary goes to stdout. Without it, the full JSON does. |
| `pdf-tool extract in.pdf --format markdown\|json\|html [-o dir]` | Marker extraction (headings, lists, tables, images, page-aware). PyMuPDF decides first whether OCR is needed (`--ocr auto\|force\|never`). Several PDFs per call reuse the loaded models. `--engine pymupdf` is a fast fallback that never OCRs. |
| `pdf-tool render in.pdf --dpi 150 [-p 1-3] [-o dir]` | PNG per page (`page-001.png`, …). |
| `pdf-tool assets in.pdf -o assets` | Images (original JPEG/PNG bytes when browser-usable, otherwise RGB(A) PNG), curved vector clusters (SVG + 300 dpi PNG), embedded font programs, all listed in `manifest.json` with ids, placements and an inferred role (`logo-candidate`, …). |
| `pdf-tool edit in.pdf --changes changes.json -o out.pdf [--report r.json]` | In-place edits: `replace_text` (search within a line, or a paragraph `rect`), `insert_text` (point or box), `redact`, `replace_image`. Either every operation is applied or nothing is written. |
| `pdf-tool generate doc.html -o out.pdf [--size-from in.pdf \| --page-size A4] [--margin 20mm] [--header-template h.html] [--footer-template f.html]` | HTML/CSS to PDF with headless Chromium: print CSS, page breaks, backgrounds, local images and fonts, tables. Page JS is off, the network is blocked, and only files under the HTML's folder (or `--root`) are served. |
| `pdf-tool compare ref.pdf cand.pdf -o dir [--masks r.json] [--mask 1:x0,y0,x1,y1] [--expect T] [--forbid T]` | Side-by-side and diff PNGs, per-page SSIM, changed pixels and changed regions, word-level text diffs (per page and across the whole document), text-block moves measured on baselines, page size/count checks, and overflow warnings. Exits `1` if an `--expect`/`--forbid` check fails. |

### Edit instructions

```json
{
  "font_dirs": ["./brand-fonts"],
  "operations": [
    {"op": "replace_text", "page": 1, "search": "12 April 2024", "text": "19 April 2024"},
    {"op": "replace_text", "page": 1, "search": "4.2M", "text": "4.35M", "align": "right"},
    {"op": "replace_text", "page": 2, "rect": [56, 140, 539, 230], "text": "Rewritten paragraph…"},
    {"op": "insert_text", "page": 1, "point": [56, 780], "text": "Onaylandı", "size": 9, "color": "#666666"},
    {"op": "insert_text", "page": 1, "rect": [56, 700, 300, 760], "html": "<b>Note:</b> draft"},
    {"op": "redact", "page": 3, "search": "Internal only", "fill": "#000000"},
    {"op": "replace_image", "page": 1, "asset_id": "img-12", "image": "new-logo.png"}
  ]
}
```

How the edit engine behaves:
- **Removal is real redaction.** Characters are deleted from the content stream. By default
  the area is left transparent, so backgrounds survive. `fill` paints a box, and
  `remove_images`/`remove_graphics` widen the redaction.
- **Search replacement works on visual lines** and ignores whitespace differences. Text objects
  that share a baseline and nearly touch are treated as one line. The rest of the line is re-set
  so spacing stays natural. `align: right|center` keeps the line's end or centre fixed. Use
  `occurrence: n` to target one match.
- **Paragraph replacement (`rect`)** puts the new text on the original first baseline, with the
  original line pitch, font, size and color. Alignment is inferred unless given.
- **Fonts.** The original embedded font is reused if it has every glyph needed. Otherwise the tool
  tries a matching system font, then a built-in metric-compatible font (Helvetica/Times/Courier
  clones with Latin Extended, including Turkish). Each step is checked for glyph coverage, and the
  report states `font_source` and any substitution.
- **Overflow.** `"overflow": "error"` (default) fails with measured numbers. `"shrink"` scales down
  to `min_scale`. `"allow"` writes anyway and warns. Lines are limited by paragraph width, table
  cell borders, neighbouring text and images, and the page's content margin.
- **Image replacement** swaps every placement of a shared image object and keeps proportions
  inside the original box.
- **Verification.** After saving, every text operation is read back from the output
  (`verified: true`), and the report's `masks` can be passed straight to `compare --masks`.

## Agent skill

`skills/pdf-revision/SKILL.md` is the full workflow: inspect, render, extract and assets,
understand the style, choose direct edit or reconstruction, edit or generate, render, compare,
fix (at most 3 iterations), verify, and report. It tells the agent to look at the page images
itself, never to drop content to make it fit, and never to overwrite the original.

**Claude Code**: install for all projects, or per project:

```bash
mkdir -p ~/.claude/skills && ln -s "$PWD/skills/pdf-revision" ~/.claude/skills/pdf-revision
```

```bash
mkdir -p .claude/skills && ln -s /path/to/pdf-agent-tools/skills/pdf-revision .claude/skills/pdf-revision
```

Claude Code picks the skill up from its description when you ask for a PDF revision
("update the date and add a row to the table in report.pdf, keep the design"). You can also
invoke it explicitly with `/pdf-revision`.

**Codex CLI**: Codex loads skills from `~/.codex/skills`:

```bash
mkdir -p ~/.codex/skills && ln -s "$PWD/skills/pdf-revision" ~/.codex/skills/pdf-revision
```

Then ask Codex to "use the pdf-revision skill to …", or mention `$pdf-revision`. With an older
Codex without skills support, add a line to the project's `AGENTS.md`: *"For PDF revisions,
follow /path/to/pdf-agent-tools/skills/pdf-revision/SKILL.md"*.

The skill looks for `pdf-tool` on `PATH`, and otherwise at `<repo>/.venv/bin/pdf-tool`
(it resolves the symlinked skill directory). To put the CLI on your PATH:

```bash
ln -s "$PWD/.venv/bin/pdf-tool" ~/.local/bin/pdf-tool
```

## End-to-end example

`examples/run_demo.sh` reproduces a full revision:

1. `make_sample.py` draws `examples/original.pdf`, a two-page report with a header band, logo,
   justified paragraphs, a metrics table, bullets, footers and Turkish text. It is drawn with
   PyMuPDF on purpose, so rebuilding it with Chromium is a real fidelity test.
2. **Strategy A** (`direct-edit/changes.json`): change the date, extend the client name, and
   correct two right-aligned table figures. Result: every operation reused the original
   embedded font and was verified. The masked compare shows SSIM 1.0 and 0 changed pixels
   outside the edits, with no missing text and all expectations met
   (`direct-edit/side-by-side-001.png`).
3. **Strategy B** (`reconstruction/revised.html`): a more concise Overview, a new table row,
   and a new Outlook paragraph, built only from `inspect` measurements. Two compare/fix iterations
   (Chromium rounds table rows to whole pixels; per-cell header backgrounds left hairline seams).
   Final state: every unchanged text block is within ±1.25 pt of its original baseline and x
   position, and the intended changes moved only the Outlook section (+21.75 pt for the added
   22 pt row). All expectations pass (`reconstruction/side-by-side-001.png`). SSIM is 0.85 on
   page 1, because the content changed, and 0.96 on page 2, from sub-point rasterization differences.

## Tests

```bash
.venv/bin/python -m pytest -q                 # everything (Marker tests take ~20 s once models are cached)
.venv/bin/python -m pytest -q -m "not marker" # fast suite, no Marker models needed (~5 s)
```

All sample PDFs are generated in code (`tests/samples.py`): multi-page, ruled tables, images
(shared logo, JPEG photo, vector circle), Turkish text, mixed page sizes, a paragraph block, and
a scanned (image-only) page. The suite covers inspection, rendering, assets, every edit operation
(baseline preservation, line shifting, overflow error/shrink/allow, table-cell limits, Turkish,
redaction removal, image swaps, atomic failure, overwrite protection), generation (security,
sizes, header/footer, custom fonts), comparison (identical, edited, masked, missing pages, size
differences, overflow and overlap detection, expectations), the CLI's exit codes, and Marker
extraction on both digital and scanned input.

## Architecture

```
src/pdf_tools/
  models.py    Pydantic document/asset models (extracted vs inferred provenance), error types
  common.py    PDF opening/validation, page specs, colors, safe atomic output
  inspect.py   PyMuPDF layout analysis
  extract.py   Marker integration (lazy, cached models, OCR decision) + PyMuPDF fallback
  render.py    page rasterization
  assets.py    image / vector / font export with manifest
  fonts.py     glyph-checked font resolution for edits
  edit.py      edit schema + engine (Strategy A)
  generate.py  Playwright/Chromium HTML→PDF with request sandbox (Strategy B)
  compare.py   visual + structural comparison
  cli.py       Typer CLI (thin wrapper; core functions return Pydantic models)
```

The CLI only parses arguments and prints JSON. Each command is a plain function that returns a
Pydantic model or dict (`inspect_pdf`, `extract`, `render_pdf`, `extract_assets`, `apply_edits`,
`generate_pdf`, `compare_pdfs`), so an MCP adapter can wrap these functions directly later.

## Known limitations

- **Not pixel-perfect.** In-place edits preserve everything they don't touch byte-for-byte.
  Rewritten text matches the original baseline, font and size, but a substitute font is used
  whenever the embedded font lacks glyphs. This is common with subset fonts, which often have no
  usable cmap. Reconstructions in Chromium land on a 0.75 pt pixel grid and use slightly
  different font metrics and anti-aliasing than the source, so expect sub-point offsets.
- **Subset fonts** (`ABCDEF+Name`) usually can't render new characters. `assets` lists which
  characters a subset font contains, where it can tell.
- **Search replacement** works inside one visual line. Text that spans lines needs a `rect`
  replacement. Editing rotated pages is refused. Inline images can only be replaced through
  `rect` (redact + insert).
- **Tables, reading order, page background, vector-logo detection and OCR need are heuristics**
  (marked `provenance: inferred`). Reading order assumes a single column.
- **Marker 2.0 OCR on macOS** needs `llama-server` (installed by `--with-ocr`). Without it, scanned
  pages fail with a clear message. `--ocr never` and `--engine pymupdf` still work. The first OCR
  run downloads about 1.5 GB of weights, and a warm OCR page took about 15 s on an M3 Pro.
  The pinned llama.cpp binary covers macOS (arm64/x64). On Linux, install `llama-server` yourself
  and set `LLAMA_CPP_BINARY`.
- **Marker's own optional LLM features are not used.** marker-pdf still installs its
  LLM-provider client libraries as dependencies.
- **`generate`** blocks page JavaScript (`--allow-js` to enable), so JS-driven layouts don't render.
  Header/footer templates can't load external resources (use inline styles and data URIs).
- **Password-protected PDFs** are rejected. Form field values are not part of the text analysis.
