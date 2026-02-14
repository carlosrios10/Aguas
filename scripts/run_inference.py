#!/usr/bin/env python
"""
Script ejecutable de inferencia (equivalente a poc/inference.ipynb).

Uso (desde la raíz del proyecto):
  python scripts/run_inference.py

Lee config/config.yaml (inference.cutoff, paths), crea dataset wide, preprocesa,
carga modelo y features, obtiene scores P(fraud) y guarda data/predictions/scores_<CUTOFF>.csv.
"""
import argparse
import os
import sys
import traceback
import warnings

import joblib
import pandas as pd

# Raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config, get_paths
from src.data.make_dataset import create_inference_dataset
from src.preprocessing.preprocessing import preprocess_model_input

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Inferencia: dataset + scoring → CSV de scores.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Nombre del archivo de config en config/ (default: config.yaml).",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        paths = get_paths(cfg)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar la config: {e}", file=sys.stderr)
        return 1

    interim_dir = paths["interim"]
    processed_dir = paths["processed"]
    models_dir = paths["models"]
    predictions_dir = paths["predictions"]
    inf = cfg["inference"]
    cutoff = inf["cutoff"]
    cant_periodos = inf["cant_periodos"]
    contratos_list = inf["contratos_list"]

    print(f"CUTOFF     = {cutoff}")
    print(f"INTERIM    = {interim_dir}")
    print(f"PREDICTIONS= {predictions_dir}\n")

    try:
        # 1. Crear dataset de inferencia
        df = create_inference_dataset(
            interim_dir, processed_dir,
            cutoff=cutoff,
            cant_periodos=cant_periodos,
            contratos_list=contratos_list,
        )
        if df is None:
            print("[ERROR] No se pudo crear el dataset de inferencia.", file=sys.stderr)
            return 1
        print(f"Inferencia: {len(df)} filas")

        # 2. Preprocesamiento
        df = preprocess_model_input(df)

        # 3. Cargar modelo y features
        model = joblib.load(os.path.join(models_dir, "lgbm_model.pkl"))
        cols_for_model = joblib.load(os.path.join(models_dir, "features.pkl"))
        missing = [c for c in cols_for_model if c not in df.columns]
        if missing:
            print(f"[ERROR] Columnas faltantes en inference_wide: {missing}", file=sys.stderr)
            return 1

        # 4. Scores
        scores = model.predict_proba(df[cols_for_model])[:, 1]
        df_out = df[["contrato"]].copy()
        df_out["score"] = scores
        print(f"Scores calculados: {len(df_out)} registros")

        # 5. Guardar CSV
        os.makedirs(predictions_dir, exist_ok=True)
        out_path = os.path.join(predictions_dir, f"scores_{cutoff}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"[OK] Guardado: {out_path}")
        return 0

    except Exception:
        print("\n[ERROR] La inferencia falló.", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
