#!/usr/bin/env bash
# End-to-end revision demo. Outputs go to examples/work/ (inspection data, renders,
# comparisons) and to examples/direct-edit/ and examples/reconstruction/ (PDFs).
set -euo pipefail
cd "$(dirname "$0")"
P="../.venv/bin/pdf-tool"
W=work
mkdir -p "$W"

../.venv/bin/python make_sample.py                                 # original.pdf (PyMuPDF-drawn)
"$P" inspect original.pdf -o "$W/analysis.json" > /dev/null
"$P" render original.pdf --dpi 110 -o "$W/ref-pages" > /dev/null
"$P" assets original.pdf -o "$W/assets" > /dev/null
"$P" extract original.pdf --format markdown -o "$W/content" > /dev/null

CHECKS=(--expect "19 April 2024" --expect "Yıldız Lojistik ve Ticaret A.Ş." --expect "4.35M" --expect "+14.5%"
        --forbid "12 April 2024" --forbid "4.2M")

echo "== Strategy A: direct edit"
"$P" edit original.pdf --changes direct-edit/changes.json -o direct-edit/revised-direct.pdf \
     --report "$W/edit-report.json" --overwrite > /dev/null
"$P" compare original.pdf direct-edit/revised-direct.pdf -o "$W/cmp-direct" --masks "$W/edit-report.json" "${CHECKS[@]}" \
  | ../.venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); print("status:", d["status"], "| expectations ok:", all(e["ok"] for e in d["expectations"]), "| missing text:", d["document_missing_text"])'

echo "== Strategy B: reconstruction"
"$P" generate reconstruction/revised.html -o reconstruction/revised-reconstructed.pdf --size-from original.pdf --overwrite > /dev/null
"$P" compare original.pdf reconstruction/revised-reconstructed.pdf -o "$W/cmp-recon" --dpi 110 "${CHECKS[@]}" \
     --expect "Net Promoter Score" --expect "İzmir" \
  | ../.venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); print("status:", d["status"], "| expectations ok:", all(e["ok"] for e in d["expectations"])); [print(" page", p["page"], "ssim", p["ssim"], "moved blocks", p["moved_blocks"], "warnings", p["warnings"]) for p in d["per_page"]]'
"$P" render reconstruction/revised-reconstructed.pdf --dpi 110 -o "$W/rev-pages" > /dev/null
echo "Look at $W/cmp-*/side-by-side-*.png and diff-*.png"
