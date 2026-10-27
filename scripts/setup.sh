#!/usr/bin/env bash
# Set up pdf-agent-tools in a project-local virtual environment (.venv).
#
#   scripts/setup.sh                 # Python deps + Playwright Chromium
#   scripts/setup.sh --with-ocr      # + project-local llama.cpp server for Marker OCR
#   scripts/setup.sh --warm-models   # + download Marker models now (otherwise on first use)
#
# Nothing is installed globally: packages go to .venv, Chromium to Playwright's
# cache, llama.cpp to .venv/opt/llama.cpp, model weights to ~/.cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PY_VERSION="${PDF_TOOL_PYTHON:-3.12}"  # torch/Marker wheels are most reliable on 3.11-3.12
WITH_OCR=0
WARM=0

# Pinned llama.cpp build (Surya's OCR backend on macOS/CPU spawns `llama-server`).
LLAMA_BUILD="b11480"
LLAMA_SHA256_macos_arm64="a015c26a9bd94c409c95c7abd79583da9333eeaa3998685295f36da5f36e7aed"
LLAMA_SHA256_macos_x64="587044f3ea4f0b83980af0af14590e59cabebeee908f322914b6dc86902d0af4"

for arg in "$@"; do
  case "$arg" in
    --with-ocr) WITH_OCR=1 ;;
    --warm-models) WARM=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "== Environment"
OS="$(uname -s)"; ARCH="$(uname -m)"
echo "OS: $OS $ARCH"
[ "$OS" = "Darwin" ] && sysctl -n machdep.cpu.brand_string 2>/dev/null || true

echo "== Python virtual environment ($VENV)"
if command -v uv >/dev/null 2>&1; then
  [ -x "$VENV/bin/python" ] || uv venv "$VENV" --python "$PY_VERSION"
  uv pip install --python "$VENV/bin/python" -e "$ROOT[dev]"
else
  PY=""
  for cand in "python$PY_VERSION" python3.12 python3.11 python3.13 python3; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(not (3,11) <= sys.version_info[:2] <= (3,13))'; then
      PY="$cand"; break
    fi
  done
  [ -n "$PY" ] || { echo "Python 3.11-3.13 not found (install one, or install uv)" >&2; exit 1; }
  [ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -e "$ROOT[dev]"
fi

echo "== Playwright Chromium"
"$VENV/bin/python" -m playwright install chromium

echo "== Hardware acceleration"
"$VENV/bin/python" - <<'PY'
import torch
mps = torch.backends.mps.is_available()
cuda = torch.cuda.is_available()
print(f"torch {torch.__version__}: cuda={cuda} mps={mps}")
if cuda:
    print("Marker: 'balanced' mode on CUDA (VLM layout + OCR via vLLM if installed).")
elif mps:
    print("Marker: 'fast' mode; layout models run on MPS/CPU, OCR via llama.cpp (Metal).")
else:
    print("Marker: 'fast' mode on CPU; OCR via llama.cpp.")
PY

if [ "$WITH_OCR" = 1 ]; then
  echo "== llama.cpp (OCR backend)"
  if command -v llama-server >/dev/null 2>&1; then
    echo "using llama-server on PATH: $(command -v llama-server)"
  else
    case "$OS-$ARCH" in
      Darwin-arm64) PLAT="macos-arm64"; SHA="$LLAMA_SHA256_macos_arm64" ;;
      Darwin-x86_64) PLAT="macos-x64"; SHA="$LLAMA_SHA256_macos_x64" ;;
      *) echo "no pinned llama.cpp build for $OS-$ARCH; install llama-server and set LLAMA_CPP_BINARY" >&2; exit 1 ;;
    esac
    DEST="$VENV/opt/llama.cpp"
    if [ ! -x "$(find "$DEST" -name llama-server -type f 2>/dev/null | head -1)" ]; then
      TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
      URL="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/llama-$LLAMA_BUILD-bin-$PLAT.tar.gz"
      echo "downloading $URL"
      curl -fsSL "$URL" -o "$TMP/llama.tar.gz"
      echo "$SHA  $TMP/llama.tar.gz" | shasum -a 256 -c -
      mkdir -p "$DEST"
      tar -xzf "$TMP/llama.tar.gz" -C "$DEST"
    fi
    BIN="$(find "$DEST" -name llama-server -type f | head -1)"
    "$BIN" --version 2>&1 | head -2
    echo "installed: $BIN (found automatically by pdf-tool)"
  fi
fi

if [ "$WARM" = 1 ]; then
  echo "== Warming Marker models"
  TMPD="$(mktemp -d)"
  "$VENV/bin/python" - "$TMPD" <<'PY'
import sys, pymupdf
d = pymupdf.open(); p = d.new_page()
p.insert_text((72, 72), "Warm-up document for Marker models.", fontsize=12)
d.save(sys.argv[1] + "/warm.pdf")
PY
  "$VENV/bin/pdf-tool" extract "$TMPD/warm.pdf" -o "$TMPD/out" >/dev/null
  if [ "$WITH_OCR" = 1 ]; then
    "$VENV/bin/pdf-tool" extract "$TMPD/warm.pdf" --ocr force -o "$TMPD/out-ocr" >/dev/null
  fi
  rm -rf "$TMPD"
fi

echo "== Done"
echo "CLI: $VENV/bin/pdf-tool --help   (or: source .venv/bin/activate)"
