import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_partition_mod():
    path = PROJECT_ROOT / "scripts" / "partition_raw_from_delivery.py"
    name = "script_partition_raw_from_delivery"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def part():
    return _load_partition_mod()


def test_infer_year_month_strict(part, tmp_path):
    f = tmp_path / "consumo_2024_03.txt"
    f.write_text("a", encoding="utf-8")
    assert part.infer_year_month("consumo", str(f)) == (2024, 3)


def test_infer_year_month_loose_in_folder(part, tmp_path):
    f = tmp_path / "export_marzo_2024_03_extra.txt"
    f.write_text("a", encoding="utf-8")
    assert part.infer_year_month("consumo", str(f)) == (2024, 3)


def test_infer_year_month_iso(part, tmp_path):
    f = tmp_path / "data_2024-03-v1.txt"
    f.write_text("a", encoding="utf-8")
    assert part.infer_year_month("consumo", str(f)) == (2024, 3)


def test_partition_nested_copies(part, tmp_path):
    src = tmp_path / "in"
    (src / "consumo").mkdir(parents=True)
    raw_file = src / "consumo" / "export_2025_01.txt"
    raw_file.write_text("h|i\n", encoding="utf-8")
    dest = tmp_path / "raw"
    c, sk, warn = part.partition(
        str(src),
        str(dest),
        layout="nested",
        sources=["consumo"],
        dry_run=False,
        overwrite=False,
    )
    assert c == 1 and sk == 0
    out = dest / "consumo" / "consumo_2025_01.txt"
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == "h|i\n"


def test_partition_skip_without_overwrite(part, tmp_path):
    src = tmp_path / "in"
    (src / "consumo").mkdir(parents=True)
    (src / "consumo" / "c_2025_01.txt").write_text("a", encoding="utf-8")
    dest = tmp_path / "raw"
    (dest / "consumo").mkdir(parents=True)
    (dest / "consumo" / "consumo_2025_01.txt").write_text("old", encoding="utf-8")
    c, sk, warn = part.partition(
        str(src),
        str(dest),
        layout="nested",
        sources=["consumo"],
        dry_run=False,
        overwrite=False,
    )
    assert c == 0 and sk == 1
    assert "Ya existe" in warn[0]


def test_partition_flat_detect_source(part, tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "dump_inspecciones_2024_12.txt").write_text("x", encoding="utf-8")
    dest = tmp_path / "raw"
    c, sk, warn = part.partition(
        str(src),
        str(dest),
        layout="flat",
        sources=["inspecciones", "consumo", "maestro"],
        dry_run=False,
        overwrite=False,
    )
    assert c == 1 and sk == 0
    assert (dest / "inspecciones" / "inspecciones_2024_12.txt").is_file()
