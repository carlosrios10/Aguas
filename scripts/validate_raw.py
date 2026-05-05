#!/usr/bin/env python
"""
Valida calidad mínima de archivos en data/raw antes de ejecutar ETL.
No modifica datos: solo reporta y devuelve exit code (0=OK, 1=errores).
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config, get_paths


REQUIRED_COLS = {
    "inspecciones": {"id_servicio", "fecha", "resultado", "id_inspeccion", "tiene_sancion"},
    "consumo": {
        "id_servicio",
        "fcm_anio",
        "fcm_mes",
        "fcm_m3_fact",
        "m3_fact_tipo",
        "cod_problema",
        "estatus",
        "tuvo_cm",
    },
    "maestro": {
        "id_servicio",
        "fecha_medidor",
        "desc_categoria",
        "municipio",
        "colonia",
        "zona",
        "tipo",
        "es_digital",
    },
}


def _normalize_cols(df: pd.DataFrame) -> pd.Index:
    return df.columns.astype(str).str.strip().str.lower()


def _series_for_norm_col(df: pd.DataFrame, norm_name: str) -> pd.Series | None:
    """Devuelve la serie usando el nombre real de columna (mayúsculas/espacios) que normaliza a norm_name."""
    for c in df.columns:
        if str(c).strip().lower() == norm_name:
            return df[c]
    return None


def validate_one_file(source: str, path: str, min_rows: int = 1) -> dict:
    out = {
        "source": source,
        "file": os.path.basename(path),
        "path": path,
        "status": "ok",
        "rows": None,
        "errors": [],
        "warnings": [],
    }
    pattern = rf"^{source}_(\d{{4}})_(\d{{2}})\.txt$"
    m = re.match(pattern, os.path.basename(path))
    if not m:
        out["errors"].append("filename_pattern_invalid")
    else:
        out["year"] = int(m.group(1))
        out["month"] = int(m.group(2))
    try:
        df = pd.read_csv(path, sep="|", encoding="utf-8", encoding_errors="replace")
    except Exception as e:
        out["errors"].append(f"read_error:{type(e).__name__}")
        out["status"] = "error"
        return out

    out["rows"] = int(len(df))
    if len(df) < min_rows:
        out["errors"].append(f"min_rows_not_met:{len(df)}<{min_rows}")

    cols = set(_normalize_cols(df))
    missing = sorted(REQUIRED_COLS[source] - cols)
    if missing:
        out["errors"].append(f"missing_columns:{missing}")

    # quality checks minimalistas (acceso por nombre normalizado; el CSV puede traer otro casing)
    s_id = _series_for_norm_col(df, "id_servicio")
    if s_id is not None:
        null_pct = float(s_id.isna().mean() * 100.0)
        if null_pct > 5.0:
            out["warnings"].append(f"id_servicio_null_pct_high:{null_pct:.2f}")
    if source == "consumo":
        s_mes = _series_for_norm_col(df, "fcm_mes")
        if s_mes is not None:
            mnum = pd.to_numeric(s_mes, errors="coerce")
            bad = int(((~mnum.between(1, 12)) | mnum.isna()).sum())
            if bad > 0:
                out["warnings"].append(f"invalid_fcm_mes_rows:{bad}")
    if source == "inspecciones":
        s_fecha = _series_for_norm_col(df, "fecha")
        if s_fecha is not None:
            dt = pd.to_datetime(s_fecha, errors="coerce", dayfirst=True)
            bad = int(dt.isna().sum())
            if bad > 0:
                out["warnings"].append(f"invalid_fecha_rows:{bad}")

    if out["errors"]:
        out["status"] = "error"
    elif out["warnings"]:
        out["status"] = "warn"
    return out


def validate_raw(raw_dir: str, sources: list[str]) -> tuple[list[dict], bool]:
    results = []
    has_errors = False
    for source in sources:
        src_dir = os.path.join(raw_dir, source)
        if not os.path.isdir(src_dir):
            results.append(
                {
                    "source": source,
                    "file": None,
                    "path": src_dir,
                    "status": "error",
                    "rows": None,
                    "errors": ["source_dir_missing"],
                    "warnings": [],
                }
            )
            has_errors = True
            continue
        files = sorted(
            os.path.join(src_dir, x)
            for x in os.listdir(src_dir)
            if x.lower().endswith(".txt")
        )
        if not files:
            results.append(
                {
                    "source": source,
                    "file": None,
                    "path": src_dir,
                    "status": "error",
                    "rows": None,
                    "errors": ["no_txt_files_found"],
                    "warnings": [],
                }
            )
            has_errors = True
            continue
        for f in files:
            r = validate_one_file(source, f, min_rows=1)
            results.append(r)
            if r["status"] == "error":
                has_errors = True
    return results, has_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-check de calidad raw antes de ETL.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None, help="Fuente única: consumo|inspecciones|maestro")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        paths = get_paths(cfg)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar config: {e}", file=sys.stderr)
        return 1
    sources = [args.source] if args.source else (cfg.get("etl", {}).get("sources") or list(REQUIRED_COLS))
    sources = [s for s in sources if s in REQUIRED_COLS]
    results, has_errors = validate_raw(paths["raw"], sources)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_dir": paths["raw"],
        "sources": sources,
        "has_errors": has_errors,
        "results": results,
    }
    logs_dir = paths.get("logs", os.path.join(PROJECT_ROOT, "data", "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    out_file = os.path.join(logs_dir, f"raw_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Reporte guardado: {out_file}")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_warn = sum(1 for r in results if r["status"] == "warn")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"[INFO] Resultado raw validation -> ok={n_ok}, warn={n_warn}, error={n_err}")
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
