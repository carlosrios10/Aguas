import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_mod():
    path = PROJECT_ROOT / "scripts" / "partition_bogota_v5_raw.py"
    name = "script_partition_bogota_v5_raw"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bogota():
    return _load_mod()


def test_vig_to_year_month(bogota):
    assert bogota.vig_to_year_month("202101") == (2021, 1)
    assert bogota.vig_to_year_month(202406.0) == (2024, 6)
    assert bogota.vig_to_year_month(202406) == (2024, 6)
    assert bogota.vig_to_year_month("n/a") is None
    assert bogota.vig_to_year_month(float("nan")) is None


def test_find_vigencia_column(bogota):
    assert bogota._find_vigencia_inspeccion_column(["a", "Vigencia_Inspección"]) == "Vigencia_Inspección"
    assert bogota._find_vigencia_inspeccion_column(["x", "Vigencia_Inspeccion"]) == "Vigencia_Inspeccion"


def test_partition_consumo_chunked(bogota, tmp_path):
    src = tmp_path / "c.txt"
    src.write_text(
        "CtaContrato,VIG,Consumo\n1,202401,10\n2,202401,20\n3,202402,5\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    max_v, ok, drop = bogota.partition_consumo(str(src), str(out), chunksize=2, dry_run=False)
    assert max_v == 202402
    assert ok == 3 and drop == 0
    p1 = out / "consumo_2024_01.csv"
    p2 = out / "consumo_2024_02.csv"
    assert p1.is_file() and p2.is_file()
    assert len(pd.read_csv(p1)) == 2
    assert len(pd.read_csv(p2)) == 1


def test_partition_consumo_drops_vig_after_june(bogota, tmp_path):
    src = tmp_path / "c.txt"
    src.write_text(
        "CtaContrato,VIG,Consumo\n1,202403,1\n2,202409,9\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    max_v, ok, drop = bogota.partition_consumo(str(src), str(out), chunksize=10, dry_run=False)
    assert max_v == 202403
    assert ok == 1 and drop == 1
    assert (out / "consumo_2024_03.csv").is_file()
    assert not (out / "consumo_2024_09.csv").exists()


def test_partition_inspecciones(bogota, tmp_path):
    src = tmp_path / "i.txt"
    src.write_text(
        "CtaContrato,Vigencia_Inspección,x\n10,202401.0,a\n11,202402,b\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    ok, drop = bogota.partition_inspecciones(str(src), str(out), dry_run=False)
    assert ok == 2 and drop == 0
    assert len(pd.read_csv(out / "inspecciones_2024_01.csv")) == 1


def test_copy_maestro(bogota, tmp_path):
    m = tmp_path / "m.txt"
    m.write_text("CtaContrato\n1\n", encoding="utf-8")
    out = tmp_path / "maestro"
    path = bogota.copy_maestro_max_vig(str(m), str(out), 202412, dry_run=False)
    assert Path(path).name == "maestro_2024_12.csv"
