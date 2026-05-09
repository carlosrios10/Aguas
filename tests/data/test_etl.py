"""Tests unitarios para src.data.etl (sin datos reales ni red)."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import etl


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_get_pending_months_all_when_overwrite(tmp_path):
    raw = tmp_path / "raw_consumo"
    raw.mkdir()
    (raw / "consumo_2024_01.txt").write_text("x")
    (raw / "consumo_2024_02.txt").write_text("x")

    interim = tmp_path / "interim"
    interim.mkdir()

    pending = etl.get_pending_months(str(raw), str(interim), "consumo", overwrite=True)
    assert pending == [(2024, 1), (2024, 2)]


def test_get_pending_months_detects_csv(tmp_path):
    raw = tmp_path / "raw_consumo"
    raw.mkdir()
    (raw / "consumo_2024_01.csv").write_text("x")

    interim = tmp_path / "interim"
    interim.mkdir()

    pending = etl.get_pending_months(str(raw), str(interim), "consumo", overwrite=True)
    assert pending == [(2024, 1)]


def test_get_pending_months_excludes_processed_partition(tmp_path):
    raw = tmp_path / "raw_consumo"
    raw.mkdir()
    (raw / "consumo_2024_01.txt").write_text("x")
    (raw / "consumo_2024_02.txt").write_text("x")

    interim = tmp_path / "interim"
    parquet = interim / "consumo" / "year=2024" / "month=01" / "consumo.parquet"
    _touch(parquet)

    pending = etl.get_pending_months(str(raw), str(interim), "consumo", overwrite=False)
    assert pending == [(2024, 2)]


def test_clean_inspecciones_raises_on_missing_columns():
    df = pd.DataFrame({"CtaContrato": ["1"]})
    with pytest.raises(ValueError, match="faltan columnas"):
        etl.clean_inspecciones(df)


def test_clean_inspecciones_dedup_by_contrato_date_target_1_positive():
    df = pd.DataFrame(
        {
            "CtaContrato": ["100", "100"],
            "Vigencia_Inspeccion": [202401, 202401],
            "AnomaliaCausaInefectividad": ["bypass", "bypass"],
            "Efectiva": [1, 1],
        }
    )
    out = etl.clean_inspecciones(df)
    assert len(out) == 1
    assert out["contrato"].iloc[0] == "100"
    assert out["is_fraud"].iloc[0] == 1
    assert pd.Timestamp("2024-01-01") == out["date"].iloc[0]


def test_clean_inspecciones_is_fraud_zero_when_not_in_target_1():
    df = pd.DataFrame(
        {
            "CtaContrato": ["200"],
            "Vigencia_Inspeccion": [202402],
            "AnomaliaCausaInefectividad": ["causa no catalogada"],
        }
    )
    out = etl.clean_inspecciones(df)
    assert out["is_fraud"].iloc[0] == 0


def test_clean_consumo_raises_on_missing_columns():
    df = pd.DataFrame({"CtaContrato": ["1"], "VIG": [202401], "Periodicidad": [2]})
    with pytest.raises(ValueError, match="faltan columnas"):
        etl.clean_consumo(df)


def test_clean_consumo_dedup_sums_consumo():
    df = pd.DataFrame(
        {
            "CtaContrato": ["1", "1"],
            "VIG": [202401, 202401],
            "Consumo": [10.0, 99.0],
            "Periodicidad": [2, 2],
        }
    )
    out = etl.clean_consumo(df)
    assert len(out) == 1
    assert out["consumo"].iloc[0] == 109.0


def test_clean_consumo_keeps_only_periodicidad_2():
    df = pd.DataFrame(
        {
            "CtaContrato": ["1", "2"],
            "VIG": [202401, 202401],
            "Consumo": [1.0, 2.0],
            "Periodicidad": [2, 1],
        }
    )
    out = etl.clean_consumo(df)
    assert len(out) == 1
    assert out["contrato"].iloc[0] == "1"


def test_clean_consumo_excludes_placeholder_vig():
    df = pd.DataFrame(
        {
            "CtaContrato": ["1", "2", "3"],
            "VIG": [202401, 202313, 202413],
            "Consumo": [1.0, 2.0, 3.0],
            "Periodicidad": [2, 2, 2],
        }
    )
    out = etl.clean_consumo(df)
    assert set(out["contrato"]) == {"1"}
    assert len(out) == 1


def test_clean_consumo_year_month_from_vig():
    df = pd.DataFrame(
        {
            "CtaContrato": ["9"],
            "VIG": [202403],
            "Consumo": [5.5],
            "Periodicidad": [2],
        }
    )
    out = etl.clean_consumo(df)
    assert out["year"].iloc[0] == 2024
    assert out["month"].iloc[0] == 3
    assert out["vig"].iloc[0] == "202403"


def test_clean_consumo_drops_vig_second_semester():
    df = pd.DataFrame(
        {
            "CtaContrato": ["1", "2"],
            "VIG": [202403, 202409],
            "Consumo": [1.0, 9.0],
            "Periodicidad": [2, 2],
        }
    )
    out = etl.clean_consumo(df)
    assert len(out) == 1
    assert out["contrato"].iloc[0] == "1"
    assert out["month"].iloc[0] == 3


def test_clean_maestro_rename_and_keep_last_duplicate():
    df = pd.DataFrame(
        {
            "CtaContrato": ["1", "1"],
            "oc_nme_localidad": ["A", "B"],
        }
    )
    out = etl.clean_maestro(df)
    assert len(out) == 1
    assert out["contrato"].iloc[0] == "1"
    assert out["oc_nme_localidad"].iloc[0] == "B"


def test_clean_maestro_raises_without_ctacontrato():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="ctacontrato"):
        etl.clean_maestro(df)


def test_process_month_writes_parquet(tmp_path):
    raw_base = tmp_path / "raw"
    cons_dir = raw_base / "consumo"
    cons_dir.mkdir(parents=True)
    raw_file = cons_dir / "consumo_2024_01.csv"
    raw_file.write_text(
        "CtaContrato,VIG,Consumo,Periodicidad\n1,202401,5,2\n",
        encoding="utf-8",
    )

    interim = tmp_path / "interim"
    ok = etl.process_month(
        str(raw_base),
        str(interim),
        "consumo",
        2024,
        1,
        etl.clean_consumo,
        overwrite=True,
    )
    assert ok is True
    out_parquet = interim / "consumo" / "year=2024" / "month=01" / "consumo.parquet"
    assert out_parquet.is_file()
    back = pd.read_parquet(out_parquet)
    assert len(back) == 1
    assert back["contrato"].iloc[0] == "1"
    assert back["consumo"].iloc[0] == 5.0


def test_process_month_no_parquet_when_clean_empty(tmp_path):
    """Si la limpieza deja 0 filas, no se guarda parquet y process_month devuelve False."""
    raw_base = tmp_path / "raw"
    cons_dir = raw_base / "consumo"
    cons_dir.mkdir(parents=True)
    (cons_dir / "consumo_2024_01.csv").write_text(
        "CtaContrato,VIG,Consumo,Periodicidad\n1,202401,5,1\n",
        encoding="utf-8",
    )
    interim = tmp_path / "interim"
    ok = etl.process_month(
        str(raw_base),
        str(interim),
        "consumo",
        2024,
        1,
        etl.clean_consumo,
        overwrite=True,
    )
    assert ok is False
    out_parquet = interim / "consumo" / "year=2024" / "month=01" / "consumo.parquet"
    assert not out_parquet.is_file()


def test_process_month_skips_when_parquet_exists_without_overwrite(tmp_path):
    raw_base = tmp_path / "raw"
    cons_dir = raw_base / "consumo"
    cons_dir.mkdir(parents=True)
    (cons_dir / "consumo_2024_01.csv").write_text(
        "CtaContrato,VIG,Consumo,Periodicidad\n1,202401,5,2\n",
        encoding="utf-8",
    )
    interim = tmp_path / "interim"
    existing = interim / "consumo" / "year=2024" / "month=01" / "consumo.parquet"
    _touch(existing)

    ok = etl.process_month(
        str(raw_base),
        str(interim),
        "consumo",
        2024,
        1,
        etl.clean_consumo,
        overwrite=False,
    )
    assert ok is False
