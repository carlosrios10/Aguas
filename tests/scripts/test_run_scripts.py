"""
Smoke tests para scripts/run_etl.py, run_train.py, run_inference.py.

- Con importlib + monkeypatch se verifica el código de salida de `main()` sin ETL/train real.
- Un test por script usa subprocess (config inexistente) para comprobar el entrypoint real.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(relative_name: str):
    """Carga un script de `scripts/` como módulo (misma lógica que al ejecutar con python)."""
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
def run_etl_mod():
    return _load_script("run_etl.py")


@pytest.fixture
def run_train_mod():
    return _load_script("run_train.py")


@pytest.fixture
def run_inference_mod():
    return _load_script("run_inference.py")


def _minimal_paths(tmp_path):
    return {
        "raw": str(tmp_path / "raw"),
        "interim": str(tmp_path / "interim"),
        "processed": str(tmp_path / "processed"),
        "models": str(tmp_path / "models"),
        "predictions": str(tmp_path / "predictions"),
        "logs": str(tmp_path / "logs"),
    }


def test_run_etl_main_exits_zero_when_etl_ok(monkeypatch, tmp_path, run_etl_mod):
    cfg = {
        "log_level": "ERROR",
        "paths": _minimal_paths(tmp_path),
        "etl": {"sources": ["consumo"], "overwrite": False},
    }
    summary = {"consumo": {"total": 0, "processed": 0, "skipped": 0}}

    def fake_run_monthly(**kwargs):
        fake_run_monthly.kwargs = kwargs
        return summary

    monkeypatch.setattr(run_etl_mod, "load_config", lambda c="config.yaml": cfg)
    monkeypatch.setattr(
        run_etl_mod,
        "get_paths",
        lambda c=None: {k: v for k, v in cfg["paths"].items()},
    )
    monkeypatch.setattr(run_etl_mod.etl, "run_monthly_etl", fake_run_monthly)

    monkeypatch.setattr(sys, "argv", ["run_etl.py"])
    assert run_etl_mod.main() == 0
    assert fake_run_monthly.kwargs["sources"] == ["consumo"]
    assert fake_run_monthly.kwargs["overwrite"] is False


def test_run_etl_main_exits_one_when_load_config_fails(monkeypatch, run_etl_mod):
    monkeypatch.setattr(run_etl_mod, "load_config", lambda *_args, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(sys, "argv", ["run_etl.py"])
    assert run_etl_mod.main() == 1


def test_run_etl_main_exits_one_when_etl_raises(monkeypatch, tmp_path, run_etl_mod):
    cfg = {"log_level": "ERROR", "paths": _minimal_paths(tmp_path), "etl": {}}

    monkeypatch.setattr(run_etl_mod, "load_config", lambda c="config.yaml": cfg)
    monkeypatch.setattr(
        run_etl_mod,
        "get_paths",
        lambda c=None: {k: v for k, v in cfg["paths"].items()},
    )
    monkeypatch.setattr(
        run_etl_mod.etl,
        "run_monthly_etl",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("fallo simulado")),
    )
    monkeypatch.setattr(sys, "argv", ["run_etl.py"])
    assert run_etl_mod.main() == 1


def test_run_train_main_exits_one_when_no_dataset(monkeypatch, tmp_path, run_train_mod):
    cfg = {
        "log_level": "ERROR",
        "paths": _minimal_paths(tmp_path),
        "train": {
            "cutoff": None,
            "cant_periodos": 12,
            "max_ctas_neg": 200,
            "sam_th": 0.3,
            "param_imb_method": "under",
            "preprocesor_num": 4,
        },
    }
    monkeypatch.setattr(run_train_mod, "load_config", lambda c="config.yaml": cfg)
    monkeypatch.setattr(
        run_train_mod,
        "get_paths",
        lambda c=None: {k: v for k, v in cfg["paths"].items()},
    )
    monkeypatch.setattr(run_train_mod, "create_train_dataset", lambda *a, **k: None)

    monkeypatch.setattr(sys, "argv", ["run_train.py"])
    assert run_train_mod.main() == 1


def test_run_train_main_exits_one_when_load_config_fails(monkeypatch, run_train_mod):
    monkeypatch.setattr(run_train_mod, "load_config", lambda *_a, **_k: (_ for _ in ()).throw(KeyError("bad yaml")))
    monkeypatch.setattr(sys, "argv", ["run_train.py"])
    assert run_train_mod.main() == 1


def test_run_inference_main_exits_one_when_no_dataset(monkeypatch, tmp_path, run_inference_mod):
    cfg = {
        "log_level": "ERROR",
        "paths": _minimal_paths(tmp_path),
        "inference": {
            "cutoff": "2025-06-01",
            "cant_periodos": 12,
            "contratos_list": None,
            "columns_filter": None,
            "output_columns": None,
        },
    }
    monkeypatch.setattr(run_inference_mod, "load_config", lambda c="config.yaml": cfg)
    monkeypatch.setattr(
        run_inference_mod,
        "get_paths",
        lambda c=None: {k: v for k, v in cfg["paths"].items()},
    )
    monkeypatch.setattr(run_inference_mod, "create_inference_dataset", lambda *a, **k: None)

    monkeypatch.setattr(sys, "argv", ["run_inference.py"])
    assert run_inference_mod.main() == 1


def test_run_inference_main_exits_one_when_load_config_fails(monkeypatch, run_inference_mod):
    monkeypatch.setattr(
        run_inference_mod, "load_config", lambda *_a, **_k: (_ for _ in ()).throw(OSError("no file"))
    )
    monkeypatch.setattr(sys, "argv", ["run_inference.py"])
    assert run_inference_mod.main() == 1


@pytest.mark.parametrize(
    "script,needle",
    [
        ("run_etl.py", "No se pudo cargar la config"),
        ("run_train.py", "No se pudo cargar la config"),
        ("run_inference.py", "No se pudo cargar la config"),
    ],
)
def test_subprocess_exits_one_when_config_file_missing(script, needle):
    import subprocess

    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), "--config", "__no_existe_12345__.yaml"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1
    assert needle in (r.stderr + r.stdout)
