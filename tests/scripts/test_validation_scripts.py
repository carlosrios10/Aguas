import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(relative_name: str):
    path = PROJECT_ROOT / "scripts" / relative_name
    name = f"script_{relative_name.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def validate_raw_mod():
    return _load_script("validate_raw.py")


@pytest.fixture
def check_interim_mod():
    return _load_script("check_interim_completeness.py")


@pytest.fixture
def check_artifacts_mod():
    return _load_script("check_artifacts.py")


def _mk_paths(tmp_path):
    return {
        "raw": str(tmp_path / "raw"),
        "interim": str(tmp_path / "interim"),
        "processed": str(tmp_path / "processed"),
        "models": str(tmp_path / "models"),
        "predictions": str(tmp_path / "predictions"),
        "logs": str(tmp_path / "logs"),
    }


def test_validate_raw_ok_file(validate_raw_mod, tmp_path):
    raw = tmp_path / "raw" / "consumo"
    raw.mkdir(parents=True)
    (raw / "consumo_2025_01.txt").write_text(
        "id_servicio|fcm_anio|fcm_mes|fcm_m3_fact|m3_fact_tipo|cod_problema|estatus|tuvo_cm\n"
        "1|2025|1|10|A|0|A|N\n",
        encoding="utf-8",
    )
    results, has_errors = validate_raw_mod.validate_raw(str(tmp_path / "raw"), ["consumo"])
    assert has_errors is False
    assert results[0]["status"] in ("ok", "warn")


def test_validate_raw_quality_uses_normalized_column_names(validate_raw_mod, tmp_path):
    """CSV con distinto casing en cabeceras: no debe lanzar KeyError al medir nulos."""
    raw = tmp_path / "raw" / "consumo"
    raw.mkdir(parents=True)
    (raw / "consumo_2025_01.txt").write_text(
        "ID_SERVICIO|fcm_anio|FCM_MES|fcm_m3_fact|m3_fact_tipo|cod_problema|estatus|tuvo_cm\n"
        "1|2025|1|10|A|0|A|N\n",
        encoding="utf-8",
    )
    results, has_errors = validate_raw_mod.validate_raw(str(tmp_path / "raw"), ["consumo"])
    assert has_errors is False
    assert results[0]["status"] == "ok"


def test_validate_raw_detects_missing_columns(validate_raw_mod, tmp_path):
    raw = tmp_path / "raw" / "consumo"
    raw.mkdir(parents=True)
    (raw / "consumo_2025_01.txt").write_text(
        "id_servicio|fcm_anio|fcm_mes\n1|2025|1\n",
        encoding="utf-8",
    )
    results, has_errors = validate_raw_mod.validate_raw(str(tmp_path / "raw"), ["consumo"])
    assert has_errors is True
    assert "missing_columns" in results[0]["errors"][0]


def test_check_interim_reports_missing_months(check_interim_mod, tmp_path):
    interim = tmp_path / "interim" / "consumo" / "year=2025" / "month=07"
    interim.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(interim / "consumo.parquet", index=False)
    mdir = tmp_path / "interim" / "maestro" / "year=2025" / "month=07"
    mdir.mkdir(parents=True)
    pd.DataFrame({"contrato": ["1"], "categoria": ["A"]}).to_parquet(mdir / "maestro.parquet", index=False)
    report = check_interim_mod.check_interim(str(tmp_path / "interim"), "2025-09-01", 2)
    # Esperados: 2025-07 y 2025-08, falta 2025-08
    assert "2025-08" in report["missing_months"]
    assert report["ok"] is False


def test_check_interim_ok_with_full_window(check_interim_mod, tmp_path):
    for mm in (7, 8):
        interim = tmp_path / "interim" / "consumo" / "year=2025" / f"month={mm:02d}"
        interim.mkdir(parents=True)
        pd.DataFrame({"x": [1]}).to_parquet(interim / "consumo.parquet", index=False)
    mdir = tmp_path / "interim" / "maestro" / "year=2025" / "month=08"
    mdir.mkdir(parents=True)
    pd.DataFrame({"contrato": ["1"], "categoria": ["A"]}).to_parquet(mdir / "maestro.parquet", index=False)
    report = check_interim_mod.check_interim(str(tmp_path / "interim"), "2025-09-01", 2)
    assert report["missing_months"] == []
    assert report["ok"] is True


def test_check_artifacts_ok(check_artifacts_mod, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "lgbm_model.pkl").write_bytes(b"x")
    import joblib

    joblib.dump(["a", "b"], models / "features.pkl")
    report = check_artifacts_mod.check_artifacts(str(models))
    assert report["ok"] is True
    assert report["features_len"] == 2


def test_check_artifacts_fail_when_missing(check_artifacts_mod, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    report = check_artifacts_mod.check_artifacts(str(models))
    assert report["ok"] is False
