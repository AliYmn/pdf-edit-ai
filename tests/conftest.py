import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import samples  # noqa: E402


@pytest.fixture(scope="session")
def pdfs(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("samples")
    return {
        name: getattr(samples, name)(d / f"{name}.pdf")
        for name in ("multipage", "table", "images", "turkish", "mixed_sizes", "paragraph", "scanned")
    }


@pytest.fixture
def copy_of(tmp_path):
    def _copy(src: Path) -> Path:
        dst = tmp_path / src.name
        shutil.copy(src, dst)
        return dst

    return _copy


def text_of(path: Path, page: int | None = None) -> str:
    import pymupdf

    with pymupdf.open(path) as doc:
        pages = [doc[page - 1]] if page else list(doc)
        return "\n".join(p.get_text("text", sort=True) for p in pages)


def squash(s: str) -> str:
    return "".join(s.split())
