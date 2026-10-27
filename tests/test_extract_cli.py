import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import squash, text_of
from pdf_tools.cli import app
from pdf_tools.extract import extract, local_llama_server, ocr_plan

runner = CliRunner()


def test_ocr_plan_decisions(pdfs):
    digital = ocr_plan(pdfs["multipage"], [0, 1, 2], "auto")
    assert digital["disable_ocr"] and not digital["force_ocr"] and digital["pages_needing_ocr"] == []
    scan = ocr_plan(pdfs["scanned"], [0], "auto")
    assert scan["force_ocr"] and not scan["disable_ocr"] and scan["pages_needing_ocr"] == [1]
    assert ocr_plan(pdfs["scanned"], [0], "never")["disable_ocr"]
    assert ocr_plan(pdfs["multipage"], [0], "force")["force_ocr"]


@pytest.mark.parametrize("fmt,ext", [("markdown", "md"), ("json", "json"), ("html", "html")])
def test_extract_pymupdf_engine(pdfs, tmp_path, fmt, ext):
    meta = extract(pdfs["turkish"], tmp_path, fmt=fmt, engine="pymupdf")
    out = Path(meta["output"])
    assert out.name == f"turkish.{ext}" and out.is_file()
    assert "İstanbul" in out.read_text(encoding="utf-8")
    assert json.loads((tmp_path / "turkish_meta.json").read_text())["ocr"]["disable_ocr"]


@pytest.mark.marker
def test_extract_marker_digital(pdfs, tmp_path):
    meta = extract(pdfs["table"], tmp_path / "md", fmt="markdown")
    md = Path(meta["output"]).read_text(encoding="utf-8")
    assert meta["ocr"]["disable_ocr"] and "| North" in md and "{0}" in md  # page separator
    meta = extract(pdfs["images"], tmp_path / "js", fmt="json")
    data = json.loads(Path(meta["output"]).read_text(encoding="utf-8"))
    assert [c["block_type"] for c in data["children"]] == ["Page", "Page"]
    assert meta["image_files"] and all(f.startswith("page_") for f in meta["image_files"])
    assert all((tmp_path / "js" / f).is_file() for f in meta["image_files"])


@pytest.mark.marker
@pytest.mark.skipif(local_llama_server() is None, reason="llama-server not installed (scripts/setup.sh --with-ocr)")
def test_extract_marker_ocr_scanned(pdfs, tmp_path):
    meta = extract(pdfs["scanned"], tmp_path, fmt="markdown")
    md = Path(meta["output"]).read_text(encoding="utf-8")
    assert meta["ocr"]["force_ocr"]
    assert "Scanned Invoice 2024" in md and "1,250.00" in md


# ---------------------------------------------------------------------- CLI


def test_cli_inspect_summary_and_json(pdfs, tmp_path):
    out = tmp_path / "with space" / "analysis.json"
    r = runner.invoke(app, ["inspect", str(pdfs["table"]), "-o", str(out)])
    assert r.exit_code == 0, r.output
    summary = json.loads(r.stdout)
    assert summary["pages"][0]["counts"]["table"] == 1
    assert json.loads(out.read_text())["meta"]["page_count"] == 1


def test_cli_missing_file_exit_code(tmp_path):
    r = runner.invoke(app, ["inspect", str(tmp_path / "nope.pdf")])
    assert r.exit_code == 2 and "not found" in r.stderr


def test_cli_edit_success_failure_and_spaces(pdfs, tmp_path):
    src = tmp_path / "my input.pdf"
    src.write_bytes(pdfs["multipage"].read_bytes())
    changes = tmp_path / "my changes.json"
    changes.write_text(json.dumps({"operations": [{"op": "replace_text", "page": 1, "search": "March 2024", "text": "Mayıs 2025"}]}), encoding="utf-8")
    out = tmp_path / "revised file.pdf"
    r = runner.invoke(app, ["edit", str(src), "--changes", str(changes), "-o", str(out), "--report", str(tmp_path / "rep.json")])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["ok"] and "Mayıs 2025" in text_of(out, 1)

    r = runner.invoke(app, ["edit", str(src), "--changes", str(changes), "-o", str(out)])
    assert r.exit_code == 2 and "already exists" in r.stderr

    changes.write_text(json.dumps([{"op": "replace_text", "page": 1, "search": "absent", "text": "x"}]), encoding="utf-8")
    r = runner.invoke(app, ["edit", str(src), "--changes", str(changes), "-o", str(tmp_path / "f.pdf")])
    assert r.exit_code == 1 and json.loads(r.stdout)["results"][0]["status"] == "failed"
    assert not (tmp_path / "f.pdf").exists()


def test_cli_render_compare_expectations(pdfs, tmp_path):
    os.chdir(tmp_path)
    r = runner.invoke(app, ["render", str(pdfs["mixed_sizes"]), "--dpi", "50", "-p", "1"])
    assert r.exit_code == 0 and Path("mixed_sizes_pages/page-001.png").is_file()
    r = runner.invoke(app, ["compare", str(pdfs["multipage"]), str(pdfs["multipage"]), "-o", "cmp", "--expect", "Section 2"])
    assert r.exit_code == 0 and json.loads(r.stdout)["status"] == "ok"
    r = runner.invoke(app, ["compare", str(pdfs["multipage"]), str(pdfs["multipage"]), "-o", "cmp", "--forbid", "Section 2"])
    assert r.exit_code == 1


def test_cli_generate_and_extract_pymupdf(pdfs, tmp_path):
    html = tmp_path / "d.html"
    html.write_text("<!doctype html><meta charset='utf-8'><p>Merhaba dünya</p>", encoding="utf-8")
    r = runner.invoke(app, ["generate", str(html), "-o", str(tmp_path / "d.pdf"), "--page-size", "Letter"])
    assert r.exit_code == 0, r.output
    assert squash("Merhaba dünya") in squash(text_of(tmp_path / "d.pdf"))
    r = runner.invoke(app, ["extract", str(tmp_path / "d.pdf"), "--engine", "pymupdf", "-f", "html", "-o", str(tmp_path / "x")])
    assert r.exit_code == 0 and json.loads(r.stdout)["engine"] == "pymupdf"
    r = runner.invoke(app, ["extract", str(tmp_path / "d.pdf"), "-f", "docx"])
    assert r.exit_code == 2
