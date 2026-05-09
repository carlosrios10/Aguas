"""Tests unitarios de prioridad alta para src.data.make_dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import make_dataset


def _touch_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _write_consumo_partition(base: Path, year: int, month: int, df: pd.DataFrame) -> None:
    d = base / "consumo" / f"year={year}" / f"month={month:02d}"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "consumo.parquet", index=False)


def _write_maestro_partition(base: Path, year: int, month: int, df: pd.DataFrame) -> None:
    d = base / "maestro" / f"year={year}" / f"month={month:02d}"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "maestro.parquet", index=False)


def test_get_date_range_for_cutoff_normalizes_day_and_offsets_months():
    start_d, end_d = make_dataset.get_date_range_for_cutoff("2024-06-15", 12)
    assert end_d == pd.Timestamp(year=2024, month=6, day=1)
    assert start_d == pd.Timestamp(year=2023, month=6, day=1)


def test_get_fecha_fraud_list_empty_ordenes():
    df_o = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
    assert make_dataset.get_fecha_fraud_list(df_o, cant_periodos=12) == []


def test_get_fecha_fraud_list_requires_min_date_or_consumo():
    df_o = pd.DataFrame({"date": pd.to_datetime(["2024-06-01"])})
    assert make_dataset.get_fecha_fraud_list(df_o, df_consumo=None, cant_periodos=12, min_date_consumo=None) == []


def test_get_fecha_fraud_list_filters_by_min_date_consumo():
    df_o = pd.DataFrame({"date": pd.to_datetime(["2024-03-01", "2024-05-01"])})
    min_c = pd.Timestamp("2024-01-01")
    out = make_dataset.get_fecha_fraud_list(
        df_o,
        df_consumo=None,
        cant_periodos=3,
        cutoff_max=None,
        min_date_consumo=min_c,
    )
    assert out == ["2024-05-01"]


def test_get_fecha_fraud_list_uses_df_consumo_min_when_no_min_date_consumo():
    df_o = pd.DataFrame({"date": pd.to_datetime(["2024-05-01", "2025-01-01"])})
    df_c = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
    out = make_dataset.get_fecha_fraud_list(
        df_o,
        df_consumo=df_c,
        cant_periodos=12,
        cutoff_max=None,
        min_date_consumo=None,
    )
    assert out == ["2025-01-01"]


def test_get_fecha_fraud_list_notebook_fecha_ancla():
    df_o = pd.DataFrame({"date": pd.to_datetime(["2022-01-01", "2023-06-01", "2022-06-01"])})
    # 2021-01-01 + 24 cant_periodos = 2023-01-01
    out = make_dataset.get_fecha_fraud_list(
        df_o,
        cant_periodos=24,
        cutoff_max=None,
        min_date_consumo=None,
        fecha_ancla_ordenes="2021-01-01",
    )
    assert out == ["2023-06-01"]


def test_get_fecha_fraud_list_cutoff_max_string_filter():
    df_o = pd.DataFrame({"date": pd.to_datetime(["2024-05-01", "2024-07-01"])})
    min_c = pd.Timestamp("2024-01-01")
    out = make_dataset.get_fecha_fraud_list(
        df_o,
        cant_periodos=1,
        cutoff_max="2024-06-01",
        min_date_consumo=min_c,
    )
    assert out == ["2024-05-01"]


def test_get_consumo_date_range_none_when_no_files(tmp_path):
    assert make_dataset.get_consumo_date_range(str(tmp_path)) == (None, None)


def test_get_consumo_date_range_min_max_from_paths_only(tmp_path):
    base = Path(tmp_path)
    _touch_parquet(base / "consumo" / "year=2023" / "month=06" / "consumo.parquet")
    _touch_parquet(base / "consumo" / "year=2025" / "month=01" / "consumo.parquet")
    mn, mx = make_dataset.get_consumo_date_range(str(base))
    assert mn == pd.Timestamp(year=2023, month=6, day=1)
    assert mx == pd.Timestamp(year=2025, month=1, day=1)


def test_load_interim_data_empty_when_no_partitions(tmp_path):
    out = make_dataset.load_interim_data(str(tmp_path), "consumo")
    assert out.empty


def test_load_interim_data_consumo_loads_any_calendar_month_partition(tmp_path):
    base = Path(tmp_path)
    _write_consumo_partition(
        base,
        2024,
        8,
        pd.DataFrame({"contrato": ["aug"], "date": pd.to_datetime(["2024-08-01"]), "consumo": [99.0]}),
    )
    _write_consumo_partition(
        base,
        2024,
        3,
        pd.DataFrame({"contrato": ["mar"], "date": pd.to_datetime(["2024-03-01"]), "consumo": [1.0]}),
    )
    out = make_dataset.load_interim_data(str(base), "consumo", start_date="2024-01-01", end_date="2024-12-01")
    assert len(out) == 2
    assert set(out["contrato"]) == {"aug", "mar"}


def test_get_consumo_date_range_min_max_over_all_partitions(tmp_path):
    base = Path(tmp_path)
    _write_consumo_partition(base, 2024, 9, pd.DataFrame({"contrato": ["z"], "date": pd.to_datetime(["2024-09-01"]), "consumo": [1.0]}))
    _write_consumo_partition(base, 2024, 2, pd.DataFrame({"contrato": ["a"], "date": pd.to_datetime(["2024-02-01"]), "consumo": [2.0]}))
    mn, mx = make_dataset.get_consumo_date_range(str(base))
    assert mn == pd.Timestamp("2024-02-01")
    assert mx == pd.Timestamp("2024-09-01")


def test_load_interim_data_filters_partition_months_by_range(tmp_path):
    base = Path(tmp_path)
    _write_consumo_partition(
        base,
        2024,
        2,
        pd.DataFrame({"contrato": ["a"], "date": pd.to_datetime(["2024-02-01"]), "consumo": [1]}),
    )
    _write_consumo_partition(
        base,
        2024,
        4,
        pd.DataFrame({"contrato": ["b"], "date": pd.to_datetime(["2024-04-01"]), "consumo": [2]}),
    )
    _write_consumo_partition(
        base,
        2024,
        6,
        pd.DataFrame({"contrato": ["c"], "date": pd.to_datetime(["2024-06-01"]), "consumo": [3]}),
    )
    out = make_dataset.load_interim_data(
        str(base),
        "consumo",
        start_date="2024-03-01",
        end_date="2024-05-01",
    )
    assert len(out) == 1
    assert out["contrato"].iloc[0] == "b"


def test_load_interim_data_builds_date_from_year_month_columns(tmp_path):
    base = Path(tmp_path)
    d = base / "consumo" / "year=2024" / "month=03"
    d.mkdir(parents=True)
    df = pd.DataFrame({"contrato": ["x"], "year": [2024], "month": [3], "consumo": [5]})
    df.to_parquet(d / "consumo.parquet", index=False)
    out = make_dataset.load_interim_data(str(base), "consumo", start_date=None, end_date=None)
    assert len(out) == 1
    assert pd.to_datetime(out["date"].iloc[0]).normalize() == pd.Timestamp("2024-03-01")


def test_load_maestro_latest_picks_newest_partition(tmp_path):
    base = Path(tmp_path)
    _write_maestro_partition(
        base,
        2023,
        1,
        pd.DataFrame({"contrato": ["old"], "categoria": ["Residencial"]}),
    )
    _write_maestro_partition(
        base,
        2024,
        6,
        pd.DataFrame({"contrato": ["new"], "categoria": ["Comercial"]}),
    )
    out = make_dataset.load_maestro_latest(str(base))
    assert len(out) == 1
    assert out["contrato"].iloc[0] == "new"


def test_load_maestro_latest_empty_when_missing(tmp_path):
    assert make_dataset.load_maestro_latest(str(tmp_path)).empty


def test_add_consumo_flags_noop_on_empty_df():
    df = pd.DataFrame()
    make_dataset.add_consumo_flags(df)
    assert df.empty


def test_add_consumo_flags_bogota_raises_without_required_columns():
    df = pd.DataFrame({"contrato": ["1"], "consumo": [10]})
    with pytest.raises(ValueError, match="lectura1|codconsumo"):
        make_dataset.add_consumo_flags(df)


def test_add_consumo_flags_na_lecturas_and_codconsumo_flags():
    df = pd.DataFrame(
        {
            "contrato": ["1"],
            "consumo": [10],
            "lectura1": [np.nan],
            "lectura2": [np.nan],
            "codconsumo": ["Bajo consumo"],
        }
    )
    make_dataset.add_consumo_flags(df)
    assert "lectura1_flag_fraude" in df.columns
    assert int(df["lectura1_sin_obs"].iloc[0]) == 1
    assert int(df["codconsumo_flag_c_normal"].iloc[0]) == 0


def test_add_consumo_flags_bogota_lectura_groups_and_codconsumo():
    df = pd.DataFrame(
        {
            "contrato": ["a", "b", "c", "d"],
            "consumo": [1, 1, 1, 1],
            "lectura1": [7, 4, None, np.nan],
            "lectura2": [None, 16.0, "23", np.nan],
            "codconsumo": ["Consumo normal", "Alto consumo", "x", "Cmo prom hist"],
            "indicador": ["X", 0.0, None, pd.NA],
        }
    )
    make_dataset.add_consumo_flags(df)
    assert df["lectura1_flag_fraude"].tolist() == [1, 0, 0, 0]
    assert df["lectura1_flag_tecnico"].tolist() == [0, 1, 0, 0]
    assert df["lectura1_sin_obs"].tolist() == [0, 0, 1, 1]
    assert df["lectura2_flag_inaccesible"].tolist() == [0, 1, 0, 0]
    assert df["lectura2_flag_admin"].tolist() == [0, 0, 1, 0]
    assert df["lectura2_sin_obs"].tolist() == [1, 0, 0, 1]
    assert df["codconsumo_flag_c_normal"].tolist() == [1, 0, 0, 0]
    assert df["codconsumo_flag_c_alto"].tolist() == [0, 1, 0, 0]
    assert df["codconsumo_flag_c_avg"].tolist() == [0, 0, 0, 1]
    assert df["indicador"].tolist() == [1, 0, 0, 0]


def test_add_consumo_flags_ignores_extra_empagua_columns_if_bogota_present():
    """Columnas tipo EMPAGUA no activan otro ramal; sigue la lógica Bogotá."""
    df = pd.DataFrame(
        {
            "contrato": ["1", "2"],
            "consumo": [1, 1],
            "lectura1": ["7", "4"],
            "lectura2": ["7", "4"],
            "codconsumo": ["Consumo normal", "Consumo normal"],
            "cod_problema": [2, 0],
            "m3_fact_tipo": ["e", "x"],
        }
    )
    make_dataset.add_consumo_flags(df)
    assert "flag_fraude" not in df.columns
    assert int(df["lectura1_flag_fraude"].iloc[0]) == 1
