#!/usr/bin/env python
"""
Valida completitud temporal en interim para inferencia mensual.
No modifica datos: reporte + exit code (0=OK, 1=faltantes/errores).
"""
import argparse
import glob
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


def _extract_ym_from_path(path: str) -> tuple[int, int] | None:
    m = re.search(r"year=(\d{4})/month=(\d{2})", path.replace("\\", "/"))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def expected_months(cutoff: str, cant_periodos: int) -> list[tuple[int, int]]:
    """Meses cerrados previos al mes de inferencia: `cant_periodos` meses sin incluir el mes del cutoff."""
    end = pd.to_datetime(cutoff).replace(day=1)
    start = end - pd.DateOffset(months=cant_periodos)
    months = pd.period_range(start=start, end=end - pd.DateOffset(months=1), freq="M")
    return [(int(p.year), int(p.month)) for p in months]


def available_consumo_months(interim_dir: str) -> set[tuple[int, int]]:
    pattern = os.path.join(interim_dir, "consumo", "year=*", "month=*", "consumo.parquet")
    out = set()
    for f in glob.glob(pattern):
        ym = _extract_ym_from_path(f)
        if ym:
            out.add(ym)
    return out


def check_interim(interim_dir: str, cutoff: str, cant_periodos: int) -> dict:
    exp = expected_months(cutoff, cant_periodos)
    avail = available_consumo_months(interim_dir)
    missing = [x for x in exp if x not in avail]

    maestro_pattern = os.path.join(interim_dir, "maestro", "year=*", "month=*", "maestro.parquet")
    maestro_files = glob.glob(maestro_pattern)
    has_maestro = len(maestro_files) > 0

    return {
        "cutoff": str(pd.to_datetime(cutoff).date()),
        "cant_periodos": int(cant_periodos),
        "expected_months": [f"{y:04d}-{m:02d}" for y, m in exp],
        "available_months": sorted([f"{y:04d}-{m:02d}" for y, m in avail]),
        "missing_months": [f"{y:04d}-{m:02d}" for y, m in missing],
        "has_maestro": has_maestro,
        "ok": (len(missing) == 0 and has_maestro),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Chequeo de completitud temporal en interim.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cutoff", default=None, help="Opcional: override del cutoff")
    parser.add_argument("--cant-periodos", type=int, default=None, help="Opcional: override de cant_periodos")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        paths = get_paths(cfg)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar config: {e}", file=sys.stderr)
        return 1
    inf = cfg.get("inference", {})
    cutoff = args.cutoff or inf.get("cutoff")
    cant_periodos = args.cant_periodos or inf.get("cant_periodos", 12)
    try:
        report = check_interim(paths["interim"], cutoff=cutoff, cant_periodos=int(cant_periodos))
    except Exception as e:
        print(f"[ERROR] Falló check_interim: {e}", file=sys.stderr)
        return 1

    logs_dir = paths.get("logs", os.path.join(PROJECT_ROOT, "data", "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    out_file = os.path.join(logs_dir, f"interim_completeness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Reporte guardado: {out_file}")
    print(f"[INFO] cutoff={report['cutoff']}, cant_periodos={report['cant_periodos']}")
    print(f"[INFO] missing_months={report['missing_months']}, has_maestro={report['has_maestro']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
