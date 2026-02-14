#!/usr/bin/env python
"""
Script ejecutable de entrenamiento (equivalente a poc/train.ipynb).

Uso (desde la raíz del proyecto):
  python scripts/run_train.py

Lee config/config.yaml (train.cutoff, paths, etc.), crea dataset, preprocesa, entrena LGBM
y guarda models/lgbm_model.pkl.
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
from src.data.make_dataset import create_train_dataset
from src.preprocessing.preprocessing import preprocess_model_input
from src.modeling.supervised_models import LGBMModel
from src.modeling.helpers import save_model

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Train LGBM: dataset + preprocesamiento + entrenamiento.")
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
    t = cfg["train"]
    cutoff = t["cutoff"]
    cant_periodos = t["cant_periodos"]
    max_ctas_neg = t["max_ctas_neg"]
    sam_th = t["sam_th"]
    param_imb_method = t["param_imb_method"]
    preprocesor_num = t["preprocesor_num"]

    print(f"CUTOFF     = {cutoff}")
    print(f"INTERIM    = {interim_dir}")
    print(f"MODELS     = {models_dir}\n")

    try:
        # 1. Crear dataset
        df = create_train_dataset(
            interim_dir, processed_dir,
            cant_periodos=cant_periodos,
            cutoff_max=cutoff,
            max_ctas=max_ctas_neg,
        )
        if df is None:
            print("[ERROR] No se pudo crear el dataset de train.", file=sys.stderr)
            return 1
        df = df.rename(columns={"is_fraud": "target"})
        print(f"Train: {len(df)} filas, target mean = {df['target'].mean():.4f}")

        # 2. Preprocesamiento
        df = preprocess_model_input(df)

        # 3. Cargar features e hiperparámetros
        cols_for_model = joblib.load(os.path.join(models_dir, "features.pkl"))
        hyperparams = joblib.load(os.path.join(models_dir, "hyperparams.pkl"))
        missing = [c for c in cols_for_model if c not in df.columns]
        if missing:
            print(f"[ERROR] Columnas faltantes en train_wide: {missing}", file=sys.stderr)
            return 1

        y_train = df["target"]

        # 4. Entrenar LGBM
        train_lgbm = LGBMModel(
            cols_for_model,
            hyperparams=hyperparams,
            search_hip=False,
            sampling_th=sam_th,
            preprocesor_num=preprocesor_num,
            sampling_method=param_imb_method,
        )
        lgbm_model = train_lgbm.train(
            df[cols_for_model], y_train,
            df_val=None, y_val=None,
        )
        print("LGBM entrenado.")

        # 5. Guardar modelo
        out_path = os.path.join(models_dir, "lgbm_model.pkl")
        save_model(lgbm_model, out_path)
        print(f"[OK] Modelo: {out_path}")
        return 0

    except Exception:
        print("\n[ERROR] El entrenamiento falló.", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
