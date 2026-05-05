"""Tests unitarios de prioridad alta para src.data.make_dataset."""

from pathlib import Path

import pandas as pd

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


def test_add_consumo_flags_skips_when_no_cod_problema():
    df = pd.DataFrame({"contrato": ["1"], "consumo": [10]})
    make_dataset.add_consumo_flags(df)
    assert "flag_fraude" not in df.columns


def test_add_consumo_flags_cod_problema_and_aux_columns():
    df = pd.DataFrame(
        {
            "cod_problema": [2, 0, 3],
            "m3_fact_tipo": ["e", "x", "x"],
            "estatus": ["a", "s", "b"],
        }
    )
    make_dataset.add_consumo_flags(df)
    assert df["flag_fraude"].tolist() == [1, 0, 0]
    assert df["flag_inacceso"].tolist() == [0, 0, 1]
    assert df["flag_sin_problema"].tolist() == [0, 1, 0]
    assert df["fact_flag_estimado"].tolist() == [1, 0, 0]
    assert df["flag_status_a"].tolist() == [1, 0, 0]
    assert df["flag_status_s"].tolist() == [0, 1, 0]
    assert df["flag_status_b"].tolist() == [0, 0, 1]
