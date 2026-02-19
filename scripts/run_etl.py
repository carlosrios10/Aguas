#!/usr/bin/env python
"""
Script ejecutable del ETL mensual (equivalente a poc/1_etl.ipynb).

Uso (desde la raíz del proyecto):
  python scripts/run_etl.py
  python scripts/run_etl.py --overwrite
  python scripts/run_etl.py --config otro.yaml

Lee config/config.yaml (o --config) para paths y etl.sources/etl.overwrite.
Detecta meses pendientes en data/raw/ y escribe en data/interim/ (parquets por año/mes).
"""
import argparse
import logging
import os
import sys
from datetime import datetime

# Raíz del proyecto (carpeta que contiene src/, data/, scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import etl
from src.config import load_config, get_paths


def setup_logging(log_dir=None, log_level=logging.INFO):
    """
    Configura el logging para ETL: consola + archivo (si log_dir está definido).
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
            log_dir, f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info("Log guardado en: %s", log_file)

    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ETL mensual: raw → interim (inspecciones, consumo).")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Nombre del archivo de config en config/ (default: config.yaml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocesar todos los meses aunque ya existan en interim.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Directorio base raw (default: <proyecto>/data/raw).",
    )
    parser.add_argument(
        "--interim-dir",
        default=None,
        help="Directorio base interim (default: <proyecto>/data/interim).",
    )
    args = parser.parse_args()

    cfg = None
    try:
        cfg = load_config(args.config)
        paths = get_paths(cfg)
        default_raw = paths["raw"]
        default_interim = paths["interim"]
    except Exception as e:
        print(f"[ERROR] No se pudo cargar la config ({args.config}): {e}", file=sys.stderr)
        return 1

    level_name = (cfg.get("log_level") or "INFO").strip().upper()
    log_level = getattr(logging, level_name, logging.INFO)
    log_dir = paths.get("logs")
    logger = setup_logging(log_dir=log_dir, log_level=log_level)

    raw_dir = args.raw_dir or default_raw
    interim_dir = args.interim_dir or default_interim
    etl_cfg = (cfg or {}).get("etl", {})
    sources = etl_cfg.get("sources", ["inspecciones", "consumo"])
    overwrite = args.overwrite or etl_cfg.get("overwrite", False)

    logger.info("=== ETL MENSUAL ===")
    logger.info("RAW_DIR    = %s", raw_dir)
    logger.info("INTERIM_DIR= %s", interim_dir)
    logger.info("SOURCES    = %s", sources)
    logger.info("OVERWRITE  = %s", overwrite)

    try:
        summary = etl.run_monthly_etl(
            raw_dir=raw_dir,
            interim_dir=interim_dir,
            sources=sources,
            overwrite=overwrite,
        )
        logger.info("=" * 60)
        logger.info("RESUMEN")
        logger.info("=" * 60)
        for source, stats in summary.items():
            logger.info(
                "%s: Procesados=%s, saltados=%s, pendientes=%s",
                source.upper(),
                stats["processed"],
                stats["skipped"],
                stats["total_pending"],
            )
        logger.info("ETL completado.")
        return 0
    except Exception:
        logger.exception("El ETL falló. Revise paths (raw, interim) y que los archivos fuente existan.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
