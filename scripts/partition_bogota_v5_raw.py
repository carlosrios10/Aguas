#!/usr/bin/env python
"""
Particiona la entrega plana Bogotá v5 hacia data/raw en CSV (coma), sin pasar por el ETL EMPAGUA.

- Perfiles de consumo: una partición por VIG (YYYYMM) → consumo/consumo_AAAA_MM.csv
  (solo meses 1–6; vigencias de segundo semestre se omiten — ciclo bimestral Bogotá)
- Histórico inspecciones: por Vigencia_Inspección (YYYYMM) → inspecciones/inspecciones_AAAA_MM.csv
- Maestro: un solo archivo maestro_AAAA_MM.csv donde AAAA_MM es el VIG máximo encontrado en consumo

No procesa los Excel de anomalía. Encoding UTF-8; filas sin periodo válido se omiten (se reporta en log).

Uso (desde la raíz del proyecto):

  python scripts/partition_bogota_v5_raw.py \\
    --consumo "D:/.../Perfiles de Consumo 20250210.txt" \\
    --inspecciones "D:/.../Histórico_Inspecciones 20260210.txt" \\
    --maestro "D:/.../Datos maestro 20260210.txt"

  python scripts/partition_bogota_v5_raw.py --dry-run  # solo rutas por defecto v5
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Repo: .../Empresa-Bogota/proyecto/bogota_poc → datos en .../Empresa-Bogota/datos/v5
DEFAULT_V5 = os.path.join(
    os.path.dirname(os.path.dirname(PROJECT_ROOT)),
    "datos",
    "v5",
)

READ_KW = dict(encoding="utf-8", encoding_errors="replace", low_memory=False)


def vig_to_year_month(vig: Any) -> tuple[int, int] | None:
    """VIG o vigencia numérico/texto YYYYMM (p. ej. 202101, '202101', 202406.0)."""
    if pd.isna(vig):
        return None
    if isinstance(vig, float):
        if vig != vig:  # NaN
            return None
        vig = int(vig)
    s = str(vig).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) != 6 or not s.isdigit():
        return None
    y, m = int(s[:4]), int(s[4:6])
    if not (1 <= m <= 12):
        return None
    return y, m


def setup_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def partition_consumo(
    path: str,
    out_dir: str,
    chunksize: int,
    dry_run: bool,
) -> tuple[int | None, int, int]:
    """
    Escribe consumo_AAAA_MM.csv por chunk. Devuelve (max_yyyymm_int, filas_escritas, filas_omitidas).
    max_yyyymm como entero 202512 para nombrar maestro.
    """
    max_vig: int | None = None
    rows_ok = 0
    rows_drop = 0
    header_written: set[tuple[int, int]] = set()

    if dry_run:
        logging.info("[dry-run] consumo: escaneo completo (sin escribir) %s chunksize=%s", path, chunksize)
        for chunk in pd.read_csv(path, chunksize=chunksize, **READ_KW):
            if "VIG" not in chunk.columns:
                raise ValueError("Consumo: falta columna VIG")
            chunk = chunk.copy()
            chunk["_ym"] = chunk["VIG"].map(vig_to_year_month)
            valid = chunk["_ym"].notna() & (chunk["_ym"].map(lambda t: t[1]) <= 6)
            rows_ok += int(valid.sum())
            rows_drop += int((~valid).sum())
            if valid.any():
                ym = chunk.loc[valid, "_ym"]
                y = ym.map(lambda t: t[0])
                m = ym.map(lambda t: t[1])
                vig_int = y * 100 + m
                mv = int(vig_int.max())
                max_vig = mv if max_vig is None else max(max_vig, mv)
        return max_vig, rows_ok, rows_drop

    os.makedirs(out_dir, exist_ok=True)
    for chunk in pd.read_csv(path, chunksize=chunksize, **READ_KW):
        if "VIG" not in chunk.columns:
            raise ValueError("Consumo: falta columna VIG")
        chunk = chunk.copy()
        chunk["_ym"] = chunk["VIG"].map(vig_to_year_month)
        bad = chunk["_ym"].isna()
        rows_drop += int(bad.sum())
        chunk = chunk.loc[~bad].copy()
        if chunk.empty:
            continue

        chunk["_y"] = chunk["_ym"].map(lambda t: t[0])
        chunk["_m"] = chunk["_ym"].map(lambda t: t[1])
        bad_sem = chunk["_m"] > 6
        rows_drop += int(bad_sem.sum())
        chunk = chunk.loc[~bad_sem].copy()
        if chunk.empty:
            continue

        for (y, m), sub in chunk.groupby(["_y", "_m"], sort=False):
            key = (y, m)
            out_path = os.path.join(out_dir, f"consumo_{y}_{m:02d}.csv")
            out_df = sub.drop(columns=["_ym", "_y", "_m"])
            write_header = key not in header_written
            out_df.to_csv(out_path, mode="a", header=write_header, index=False)
            header_written.add(key)
            rows_ok += len(out_df)

            vig_int = y * 100 + m
            max_vig = vig_int if max_vig is None else max(max_vig, vig_int)

        logging.debug("Chunk consumo procesado, acumulado ok=%s", rows_ok)

    return max_vig, rows_ok, rows_drop


def _find_vigencia_inspeccion_column(cols: list[str]) -> str:
    if "Vigencia_Inspección" in cols:
        return "Vigencia_Inspección"
    for c in cols:
        norm = (
            c.lower()
            .replace("ó", "o")
            .replace("í", "i")
            .replace(" ", "_")
        )
        if "vigencia" in norm and "inspeccion" in norm:
            return c
    raise ValueError(
        "Inspecciones: no se encontró columna Vigencia_Inspección; columnas: " + str(cols)
    )


def partition_inspecciones(path: str, out_dir: str, dry_run: bool) -> tuple[int, int]:
    """Agrupa por Vigencia_Inspección. Devuelve (filas_ok, filas_omitidas)."""
    cols = list(pd.read_csv(path, nrows=0, **READ_KW).columns)
    vig_col = _find_vigencia_inspeccion_column(cols)

    if dry_run:
        df = pd.read_csv(path, **READ_KW)
        df["_ym"] = df[vig_col].map(vig_to_year_month)
        logging.info(
            "[dry-run] inspecciones: %s filas, válidas vig=%s",
            len(df),
            int(df["_ym"].notna().sum()),
        )
        return int(df["_ym"].notna().sum()), int(df["_ym"].isna().sum())

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(path, **READ_KW)
    df["_ym"] = df[vig_col].map(vig_to_year_month)
    rows_drop = int(df["_ym"].isna().sum())
    df = df.loc[df["_ym"].notna()].copy()
    df["_y"] = df["_ym"].map(lambda t: t[0])
    df["_m"] = df["_ym"].map(lambda t: t[1])

    rows_ok = 0
    for (y, m), sub in df.groupby(["_y", "_m"], sort=False):
        out_path = os.path.join(out_dir, f"inspecciones_{y}_{m:02d}.csv")
        sub.drop(columns=["_ym", "_y", "_m"]).to_csv(out_path, index=False)
        rows_ok += len(sub)
        logging.info("Inspecciones %s_%02d -> %s filas", y, m, len(sub))

    return rows_ok, rows_drop


def copy_maestro_max_vig(maestro_path: str, out_dir: str, max_vig: int, dry_run: bool) -> str:
    """Un solo maestro_Y_M.csv con Y,M derivados de max_vig (entero YYYYMM)."""
    y, m = max_vig // 100, max_vig % 100
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"maestro_{y}_{m:02d}.csv")
    if dry_run:
        logging.info("[dry-run] maestro -> %s (desde %s)", out_path, maestro_path)
        return out_path

    df = pd.read_csv(maestro_path, **READ_KW)
    df.to_csv(out_path, index=False)
    logging.info("Maestro %s (%s filas) max VIG consumo=%s", out_path, len(df), max_vig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Particiona entrega Bogotá v5 a data/raw en CSV por VIG / vigencia."
    )
    parser.add_argument(
        "--consumo",
        default=None,
        help="Ruta Perfiles de Consumo .txt/.csv (default: v5/Perfiles de Consumo 20250210.txt)",
    )
    parser.add_argument(
        "--inspecciones",
        default=None,
        help="Ruta Histórico Inspecciones (default: v5/Histórico_Inspecciones 20260210.txt)",
    )
    parser.add_argument(
        "--maestro",
        default=None,
        help="Ruta Datos maestro (default: v5/Datos maestro 20260210.txt)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("data", "raw"),
        help="Raíz raw del proyecto (default: data/raw)",
    )
    parser.add_argument("--chunksize", type=int, default=500_000, help="Filas por chunk en consumo")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)

    v5 = DEFAULT_V5 if os.path.isdir(DEFAULT_V5) else None

    def resolve(
        default_name: str,
        override: str | None,
        *,
        glob_fallback: str | None = None,
    ) -> str:
        if override:
            return os.path.abspath(override)
        if v5:
            p = os.path.join(v5, default_name)
            if os.path.isfile(p):
                return p
            if glob_fallback:
                matches = sorted(glob.glob(os.path.join(v5, glob_fallback)))
                if len(matches) == 1:
                    return matches[0]
                if len(matches) > 1:
                    raise SystemExit(
                        f"Varios archivos para {glob_fallback} en {v5}: {matches[:5]}..."
                    )
        raise SystemExit(
            f"No se encontró {default_name}. Pase la ruta explícita o coloque datos en {DEFAULT_V5}"
        )

    consumo_path = resolve(
        "Perfiles de Consumo 20250210.txt",
        args.consumo,
        glob_fallback="*Consumo*.txt",
    )
    insp_path = resolve(
        "Histórico_Inspecciones 20260210.txt",
        args.inspecciones,
        glob_fallback="*Inspecciones*.txt",
    )
    maestro_path = resolve(
        "Datos maestro 20260210.txt",
        args.maestro,
        glob_fallback="*maestro*.txt",
    )

    out_root = args.out
    if not os.path.isabs(out_root):
        out_root = os.path.join(PROJECT_ROOT, out_root)
    out_root = os.path.abspath(out_root)

    out_consumo = os.path.join(out_root, "consumo")
    out_insp = os.path.join(out_root, "inspecciones")
    out_maestro = os.path.join(out_root, "maestro")

    logging.info("Consumo       %s", consumo_path)
    logging.info("Inspecciones  %s", insp_path)
    logging.info("Maestro       %s", maestro_path)
    logging.info("Salida        %s", out_root)

    max_vig, c_ok, c_drop = partition_consumo(
        consumo_path, out_consumo, args.chunksize, args.dry_run
    )
    logging.info("Consumo filas escritas=%s omitidas=%s max_vig=%s", c_ok, c_drop, max_vig)

    i_ok, i_drop = partition_inspecciones(insp_path, out_insp, args.dry_run)
    logging.info("Inspecciones filas escritas=%s omitidas=%s", i_ok, i_drop)

    if max_vig is None:
        logging.error("No se pudo determinar max VIG en consumo; maestro no generado.")
        return 1

    copy_maestro_max_vig(maestro_path, out_maestro, max_vig, args.dry_run)
    logging.info("Listo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
