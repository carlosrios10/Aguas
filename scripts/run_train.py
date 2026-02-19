#!/usr/bin/env python
"""Train LGBM. Uso: python scripts/run_train.py [--config config.yaml]"""
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config, get_paths
from src.data.make_dataset import create_train_dataset
from src.preprocessing.preprocessing import preprocess_model_input
from src.modeling.supervised_models import LGBMModel
from src.modeling.helpers import save_model
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")


def setup_logging(log_dir=None, log_level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(log_level)
    if root.handlers:
        return logging.getLogger(__name__)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info("Log guardado en: %s", log_file)
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train LGBM: dataset + preprocesamiento + entrenamiento.")
    parser.add_argument("--config", default="config.yaml", help="Archivo de config en config/.")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        paths = get_paths(cfg)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar la config: {e}", file=sys.stderr)
        return 1

    level_name = (cfg.get("log_level") or "INFO").strip().upper()
    log_level = getattr(logging, level_name, logging.INFO)
    log_dir = paths.get("logs")
    logger = setup_logging(log_dir=log_dir, log_level=log_level)

    interim_dir = paths["interim"]
    processed_dir = paths["processed"]
    models_dir = paths["models"]
    t = cfg["train"]
    cutoff = t["cutoff"]
    cant_periodos = t["cant_periodos"]
    sam_th = t["sam_th"]
    param_imb_method = t["param_imb_method"]
    preprocesor_num = t["preprocesor_num"]

    logger.info("=== ENTRENAMIENTO ===")
    logger.info("CUTOFF     = %s", cutoff)
    logger.info("INTERIM    = %s", interim_dir)
    logger.info("MODELS     = %s", models_dir)

    try:
        logger.info("Paso 1/6: Construyendo dataset de entrenamiento (ventana de %s meses)...", cant_periodos)
        df = create_train_dataset(interim_dir, processed_dir, cant_periodos=cant_periodos, cutoff_max=cutoff)
        if df is None:
            logger.error("No se pudo crear el dataset de train. Revise que existan inspecciones y consumo en interim.")
            return 1
        df = df.rename(columns={"is_fraud": "target"})
        logger.info("Dataset listo: %s filas, target mean = %.4f", len(df), df["target"].mean())

        logger.info("Paso 2/6: Aplicando preprocesamiento...")
        df = preprocess_model_input(df)

        logger.info("Paso 3/6: Cargando features e hiperparámetros...")
        cols_for_model = joblib.load(os.path.join(models_dir, "features.pkl"))
        hyperparams = joblib.load(os.path.join(models_dir, "hyperparams.pkl"))
        missing = [c for c in cols_for_model if c not in df.columns]
        if missing:
            logger.error("Columnas faltantes en train_wide: %s. Revise features.pkl y el dataset.", missing)
            return 1

        y_train = df["target"]
        logger.info("Paso 4/6: Entrenando LGBM...")
        train_lgbm = LGBMModel(cols_for_model, hyperparams=hyperparams, search_hip=False, sampling_th=sam_th,
            preprocesor_num=preprocesor_num, sampling_method=param_imb_method)
        lgbm_model = train_lgbm.train(df[cols_for_model], y_train, df_val=None, y_val=None)
        logger.info("LGBM entrenado.")

        logger.info("Paso 5/6: Calibrando modelo (isotonic, cv=5)...")
        calibrated_model = CalibratedClassifierCV(lgbm_model, method="isotonic", cv=5)
        calibrated_model.fit(df[cols_for_model], y_train)
        logger.info("Modelo calibrado (isotonic, cv=5).")

        logger.info("Paso 6/6: Guardando modelo calibrado...")
        out_path = os.path.join(models_dir, "lgbm_model_cal.pkl")
        save_model(calibrated_model, out_path)
        logger.info("Entrenamiento completado. Modelo calibrado guardado: %s", out_path)
        return 0
    except Exception:
        logger.exception("El entrenamiento falló. Compruebe config (train.*, paths) y datos en interim y models/.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
