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
    df = pd.DataFrame(
        {"id_servicio": ["1"], "fecha": ["01/01/2024"], "resultado": ["OK"]}
    )
    with pytest.raises(ValueError, match="faltan columnas"):
        etl.clean_inspecciones(df)


def test_clean_inspecciones_fraud_flag_and_dedup():
    df = pd.DataFrame(
        {
            "id_servicio": ["100", "100"],
            "fecha": ["15/01/2024", "20/01/2024"],
            "resultado": ["NORMAL", "ANOMALO"],
            "id_inspeccion": [1, 2],
            "tiene_sancion": [0, 0],
        }
    )
    out = etl.clean_inspecciones(df)
    assert len(out) == 1
    assert out["contrato"].iloc[0] == "100"
    assert out["is_fraud"].iloc[0] == 1


def test_clean_consumo_dedup_keeps_first():
    df = pd.DataFrame(
        {
            "id_servicio": ["1", "1"],
            "fcm_anio": [2024, 2024],
            "fcm_mes": [1, 1],
            "fcm_m3_fact": [10, 99],
            "m3_fact_tipo": ["A", "A"],
            "cod_problema": [0, 0],
            "estatus": ["x", "x"],
            "tuvo_cm": ["N", "N"],
        }
    )
    out = etl.clean_consumo(df)
    assert len(out) == 1
    assert out["consumo"].iloc[0] == 10


def test_clean_consumo_string_consumo_before_comma():
    df = pd.DataFrame(
        {
            "id_servicio": ["1"],
            "fcm_anio": [2024],
            "fcm_mes": [1],
            "fcm_m3_fact": ["12,34"],
            "m3_fact_tipo": ["A"],
            "cod_problema": [0],
            "estatus": ["x"],
            "tuvo_cm": ["N"],
        }
    )
    out = etl.clean_consumo(df)
    assert out["consumo"].iloc[0] == 12


def test_clean_maestro_maps_category_and_keeps_last_duplicate():
    df = pd.DataFrame(
        {
            "id_servicio": ["1", "1"],
            "fecha_medidor": ["2024-01-01", "2024-06-01"],
            "desc_categoria": ["Residencia", "Residencia"],
            "municipio": ["A", "A"],
            "colonia": ["B", "B"],
            "zona": ["1", "1"],
            "tipo": ["T", "T"],
            "es_digital": [0, 1],
        }
    )
    out = etl.clean_maestro(df)
    assert len(out) == 1
    assert out["categoria"].iloc[0] == "Residencial"
    assert out["es_digital"].iloc[0] == 1


def test_process_month_writes_parquet(tmp_path):
    raw_base = tmp_path / "raw"
    cons_dir = raw_base / "consumo"
    cons_dir.mkdir(parents=True)
    raw_file = cons_dir / "consumo_2024_01.txt"
    raw_file.write_text(
        "id_servicio|fcm_anio|fcm_mes|fcm_m3_fact|m3_fact_tipo|cod_problema|estatus|tuvo_cm\n"
        "1|2024|1|5|A|0|x|N\n",
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
    assert back["consumo"].iloc[0] == 5


def test_process_month_skips_when_parquet_exists_without_overwrite(tmp_path):
    raw_base = tmp_path / "raw"
    cons_dir = raw_base / "consumo"
    cons_dir.mkdir(parents=True)
    (cons_dir / "consumo_2024_01.txt").write_text(
        "id_servicio|fcm_anio|fcm_mes|fcm_m3_fact|m3_fact_tipo|cod_problema|estatus|tuvo_cm\n"
        "1|2024|1|5|A|0|x|N\n",
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
