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
import os
import sys
import traceback

# Raíz del proyecto (carpeta que contiene src/, data/, scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import etl
from src.config import load_config, get_paths


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

    raw_dir = args.raw_dir or default_raw
    interim_dir = args.interim_dir or default_interim
    etl_cfg = (cfg or {}).get("etl", {})
    sources = etl_cfg.get("sources", ["inspecciones", "consumo"])
    overwrite = args.overwrite or etl_cfg.get("overwrite", False)

    print(f"RAW_DIR    = {raw_dir}")
    print(f"INTERIM_DIR= {interim_dir}")
    print(f"SOURCES    = {sources}")
    print(f"OVERWRITE  = {overwrite}\n")

    try:
        summary = etl.run_monthly_etl(
            raw_dir=raw_dir,
            interim_dir=interim_dir,
            sources=sources,
            overwrite=overwrite,
        )
        print("\n" + "=" * 60)
        print("RESUMEN")
        print("=" * 60)
        for source, stats in summary.items():
            print(f"\n{source.upper()}:")
            print(f"  Procesados: {stats['processed']}, saltados: {stats['skipped']}, pendientes: {stats['total_pending']}")
        return 0
    except Exception:
        print("\n[ERROR] El ETL falló.", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
