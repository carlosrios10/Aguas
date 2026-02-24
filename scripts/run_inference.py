#!/usr/bin/env python
"""
Script ejecutable de inferencia (equivalente a poc/inference.ipynb).

Uso (desde la raíz del proyecto):
  python scripts/run_inference.py

Lee config/config.yaml (inference.cutoff, paths), crea dataset wide, preprocesa,
carga modelo y features, obtiene scores P(fraud) y guarda data/predictions/scores_<CUTOFF>.csv.
"""
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

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


def setup_logging(log_dir=None, log_level=logging.INFO):
    """
    Configura el logging: consola + archivo (si log_dir está definido).
    Devuelve el logger del módulo.
    """
    root = logging.getLogger()
    root.setLevel(log_level)
    if root.handlers:
        return logging.getLogger(__name__)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info("Log guardado en: %s", log_file)

    return logging.getLogger(__name__)


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

    level_name = (cfg.get("log_level") or "INFO").strip().upper()
    log_level = getattr(logging, level_name, logging.INFO)
    log_dir = paths.get("logs")
    logger = setup_logging(log_dir=log_dir, log_level=log_level)

    interim_dir = paths["interim"]
    processed_dir = paths["processed"]
    models_dir = paths["models"]
    predictions_dir = paths["predictions"]
    inf = cfg["inference"]
    cutoff = inf["cutoff"]
    cant_periodos = inf["cant_periodos"]
    contratos_list = inf["contratos_list"]
    columns_filter = inf.get("columns_filter")
    output_columns = inf.get("output_columns")

    logger.info("=== INFERENCIA ===")
    logger.info("CUTOFF     = %s", cutoff)
    logger.info("INTERIM    = %s", interim_dir)
    logger.info("PREDICTIONS= %s", predictions_dir)
    if columns_filter:
        logger.info("Filtro por columnas: %s", columns_filter)
    else:
        logger.info("Sin filtro por columnas (se procesan todos los contratos con consumo).")

    try:
        logger.info("Paso 1/5: Construyendo dataset de inferencia (ventana de %s meses)...", cant_periodos)
        df = create_inference_dataset(
            interim_dir, processed_dir,
            cutoff=cutoff,
            cant_periodos=cant_periodos,
            contratos_list=contratos_list,
            columns_filter=columns_filter,
        )
        if df is None:
            logger.error(
                "No se pudo crear el dataset de inferencia. "
                "Revise que existan datos en interim (consumo y maestro) y que inference.cutoff tenga formato YYYY-MM-DD."
            )
            return 1
        logger.info("Dataset listo: %s contratos.", len(df))

        logger.info("Paso 2/5: Aplicando preprocesamiento (categóricas, estrato, medidor)...")
        df = preprocess_model_input(df)

        logger.info("Paso 3/5: Cargando modelo y lista de features...")
        model = joblib.load(os.path.join(models_dir, "lgbm_model.pkl"))
        cols_for_model = joblib.load(os.path.join(models_dir, "features.pkl"))
        missing = [c for c in cols_for_model if c not in df.columns]
        if missing:
            logger.error(
                "Columnas faltantes en inference_wide: %s. "
                "Asegúrese de haber entrenado el modelo (run_train.py) y de que features.pkl coincida con el dataset.",
                missing,
            )
            return 1

        logger.info("Paso 4/5: Calculando scores P(riesgo) por contrato...")
        scores = model.predict_proba(df[cols_for_model])[:, 1]
        if output_columns and isinstance(output_columns, list):
            cols_out = [c for c in output_columns if c in df.columns]
            if "contrato" not in cols_out:
                cols_out = ["contrato"] + cols_out
        else:
            cols_out = ["contrato"]
        df_out = df[cols_out].copy()
        df_out["score"] = scores
        logger.info("Scores calculados: %s contratos. Columnas en CSV: %s + score", len(df_out), cols_out)

        logger.info("Paso 5/5: Guardando resultados...")
        os.makedirs(predictions_dir, exist_ok=True)
        exec_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(predictions_dir, f"scores_{cutoff}_{exec_ts}.csv")
        df_out.to_csv(out_path, index=False)
        logger.info("Inferencia completada. Archivo de scores: %s", out_path)
        return 0

    except Exception:
        logger.exception(
            "La inferencia falló. Compruebe config (inference.cutoff, paths) y que existan datos en interim y modelos en models/."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
