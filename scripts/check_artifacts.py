#!/usr/bin/env python
"""
Valida artefactos mínimos para inferencia (modelo + features).
No modifica datos: solo reporta y retorna exit code.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config, get_paths


def check_artifacts(models_dir: str) -> dict:
    model_path = os.path.join(models_dir, "lgbm_model.pkl")
    features_path = os.path.join(models_dir, "features.pkl")
    out = {
        "models_dir": models_dir,
        "model_exists": os.path.isfile(model_path),
        "features_exists": os.path.isfile(features_path),
        "features_len": None,
        "ok": False,
    }
    if out["features_exists"]:
        try:
            features = joblib.load(features_path)
            out["features_len"] = len(features) if hasattr(features, "__len__") else None
        except Exception:
            out["features_exists"] = False
    out["ok"] = out["model_exists"] and out["features_exists"] and (out["features_len"] or 0) > 0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Chequeo de artefactos para inferencia.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        paths = get_paths(cfg)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar config: {e}", file=sys.stderr)
        return 1
    report = check_artifacts(paths["models"])
    logs_dir = paths.get("logs", os.path.join(PROJECT_ROOT, "data", "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    out_file = os.path.join(logs_dir, f"artifact_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Reporte guardado: {out_file}")
    print(
        f"[INFO] model_exists={report['model_exists']} features_exists={report['features_exists']} features_len={report['features_len']}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
