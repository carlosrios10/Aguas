#!/usr/bin/env python
"""
Validación de archivos raw (Excel) pendientes de ETL.

Solo valida meses (año/mes) que tienen archivo en data/raw/ y aún NO tienen
parquet en data/interim/ (misma lógica que get_pending_months con overwrite=False).

Por cada archivo pendiente:
  1) Lee el Excel.
  2) Aplica clean_inspecciones / clean_consumo (misma lógica que el ETL).
  3) Opcionalmente compara el resultado contra config/raw_manifest.generated.yaml
     (columnas observadas en interim, fracción de nulos, valores permitidos si
     el manifiesto lista allowed_values_observed).

Al final imprime un resumen agrupado por mes con lo que no pasó y qué esperaba el manifiesto.

Uso (desde la raíz del proyecto):
  python scripts/validate_raw.py
  python scripts/validate_raw.py --config config.yaml
  python scripts/validate_raw.py --skip-manifest

Códigos de salida: 0 = sin errores (puede haber advertencias); 1 = errores o
  advertencias con --warnings-as-errors.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config, get_paths
from src.data import etl

CLEANERS = {
    "inspecciones": etl.clean_inspecciones,
    "consumo": etl.clean_consumo,
}

# Columnas derivadas por el ETL: no usar allowed_values_observed frente al manifiesto
_IGNORE_MANIFEST_COLUMNS = frozenset(
    {"year", "month", "date", "is_fraud", "_manifest_year", "_manifest_month"}
)


@dataclass
class Finding:
    """Hallazgo de validación contra manifiesto o estado del archivo."""

    severity: str  # "error" | "warning"
    category: str
    title: str
    expected: str
    observed: str

    def lines(self) -> List[str]:
        out = [f"  [{self.severity.upper()}] {self.category}: {self.title}"]
        if self.expected:
            out.append(f"      esperado: {self.expected}")
        if self.observed:
            out.append(f"      observado: {self.observed}")
        return out


def setup_logging(log_dir: Optional[str] = None, log_level: int = logging.INFO) -> logging.Logger:
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
        log_file = os.path.join(log_dir, f"validate_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info("Log guardado en: %s", log_file)
    return logging.getLogger(__name__)


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_datetime_dtype(dtype_str: str) -> bool:
    return "datetime" in str(dtype_str).lower()


def _serialize_cell(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    return str(v).strip()


def _fmt_short_list(vals: Sequence[str], limit: int = 18) -> str:
    lst = sorted(set(vals), key=lambda x: str(x))[:limit]
    s = ", ".join(repr(x) for x in lst)
    if len(set(vals)) > limit:
        s += f" … (+{len(set(vals)) - limit} valores)"
    return s


def validate_against_manifest(
    cleaned: pd.DataFrame,
    manifest_sources: Dict[str, Any],
    source: str,
    *,
    null_slack: float,
    strict_enums: bool,
    findings_out: List[Finding],
    enum_errors_out: List[Finding],
) -> Tuple[int, int]:
    """
    Añade Findings a findings_out / enum_errors_out según strict_enums.
    Returns (errors, warnings).
    """
    errors, warnings = 0, 0
    spec = manifest_sources.get(source) or {}
    col_specs: Dict[str, Any] = spec.get("columns") or {}
    if not col_specs:
        findings_out.append(
            Finding(
                severity="warning",
                category="MANIFIESTO",
                title="El manifiesto no define columnas para esta fuente.",
                expected="Sección sources → %s → columns poblada." % source,
                observed="(vacío o ausente)",
            )
        )
        warnings += 1
        return errors, warnings

    manifest_cols = set(col_specs.keys())
    data_cols = set(cleaned.columns)

    missing_in_data = sorted(manifest_cols - data_cols)
    if missing_in_data:
        findings_out.append(
            Finding(
                severity="warning",
                category="COLUMNAS",
                title="Columnas definidas en el manifiesto y ausentes en el archivo tras clean().",
                expected="presentes: %s" % ", ".join(missing_in_data),
                observed="el DataFrame tras clean tiene: %s" % ", ".join(sorted(data_cols)),
            )
        )
        warnings += 1

    extra_in_data = sorted(data_cols - manifest_cols)
    if extra_in_data:
        findings_out.append(
            Finding(
                severity="warning",
                category="COLUMNAS",
                title="Columnas presentes tras clean() que no están en el manifiesto.",
                expected="solo columnas del manifiesto (o régénere el manifiesto con poc/profile_interim_manifest.ipynb).",
                observed="extras: %s" % ", ".join(extra_in_data),
            )
        )
        warnings += 1

    for col, rules in col_specs.items():
        if col not in cleaned.columns:
            continue
        if col in _IGNORE_MANIFEST_COLUMNS:
            continue
        s = cleaned[col]
        null_frac = float(s.isna().mean()) if len(s) else 0.0
        obs_null = float(rules.get("null_fraction_observed") or 0.0)
        if null_frac > obs_null + null_slack:
            findings_out.append(
                Finding(
                    severity="warning",
                    category="NULOS",
                    title=f"Columna {col!r}: demasiados nulos respecto al manifiesto.",
                    expected="fracción de nulos ≤ %.6f (manifiesto %.6f + slack %.6f)."
                    % (obs_null + null_slack, obs_null, null_slack),
                    observed="fracción observada en este archivo: %.6f" % null_frac,
                )
            )
            warnings += 1

        allowed = rules.get("allowed_values_observed")
        dtype_str = str(rules.get("pandas_dtype") or "")
        if not allowed or _is_datetime_dtype(dtype_str):
            continue

        allowed_set = {_serialize_cell(x) for x in allowed if x is not None and str(x).strip() != ""}
        present = {_serialize_cell(x) for x in s.dropna().unique()}
        present.discard("")
        unknown = sorted(present - allowed_set, key=str)
        if not unknown:
            continue

        f = Finding(
            severity="error" if strict_enums else "warning",
            category="VALORES",
            title=f"Columna {col!r}: aparecen valores no incluidos en allowed_values_observed del manifiesto.",
            expected="solo valores del conjunto del manifiesto (%d valores únicos observados históricamente); ej.: %s"
            % (
                len(allowed_set),
                _fmt_short_list(list(allowed_set), limit=12),
            ),
            observed="valores nuevos o distintos de normalización en este archivo: %s" % _fmt_short_list(unknown, limit=20),
        )
        if strict_enums:
            enum_errors_out.append(f)
            errors += 1
        else:
            findings_out.append(f)
            warnings += 1

    return errors, warnings


def validate_one_file(
    raw_path: str,
    source: str,
    year: int,
    month: int,
    manifest_sources: Optional[Dict[str, Any]],
    *,
    null_slack: float,
    strict_enums: bool,
    logger: logging.Logger,
    month_findings: List[Finding],
) -> Tuple[int, int]:
    """Completa month_findings y devuelve (errors, warnings)."""
    errors, warnings = 0, 0
    cleaner = CLEANERS.get(source)
    if not cleaner:
        logger.error("Fuente desconocida: %s", source)
        month_findings.append(
            Finding("error", "CONFIG", "Fuente no reconocida.", cleaner or "una de: inspecciones, consumo", repr(source))
        )
        return 1, 0

    if not os.path.isfile(raw_path):
        logger.error("No existe archivo: %s", raw_path)
        month_findings.append(Finding("error", "ARCHIVO", "No existe el archivo raw.", repr(raw_path), "(no encontrado)"))
        return 1, 0

    try:
        raw_df = pd.read_excel(raw_path)
    except Exception as e:
        logger.exception("No se pudo leer Excel %s: %s", raw_path, e)
        month_findings.append(
            Finding("error", "LECTURA", "No se pudo leer el Excel.", "archivo .xlsx legible por pandas", str(e))
        )
        return 1, 0

    n_raw = len(raw_df)
    try:
        cleaned = cleaner(raw_df)
    except Exception as e:
        logger.exception("Falló clean_* para %s %04d-%02d (%s)", source, year, month, raw_path)
        month_findings.append(
            Finding("error", "CLEAN", "Excepción en clean_* (misma función que usa el ETL).", "sin excepción", str(e))
        )
        return 1, warnings

    n_clean = len(cleaned)
    if n_raw > 0 and n_clean == 0:
        month_findings.append(
            Finding(
                severity="error",
                category="FILAS",
                title="Tras clean() no queda ninguna fila.",
                expected="al menos una fila si el Excel tenía registros válidos.",
                observed="filas raw=%s, tras clean=%s." % (n_raw, n_clean),
            )
        )
        logger.error("[%s] %04d-%02d: tras limpieza 0 filas (raw=%s).", source, year, month, n_raw)
        errors += 1

    logger.info("[%s] %04d-%02d: raw=%s filas, tras clean=%s filas", source, year, month, n_raw, n_clean)

    if manifest_sources is not None:
        enum_errors: List[Finding] = []
        e2, w2 = validate_against_manifest(
            cleaned,
            manifest_sources,
            source,
            null_slack=null_slack,
            strict_enums=strict_enums,
            findings_out=month_findings,
            enum_errors_out=enum_errors,
        )
        month_findings.extend(enum_errors)
        errors += e2
        warnings += w2

    return errors, warnings


def print_monthly_report(rows: List[Dict[str, Any]], logger: logging.Logger) -> None:
    """rows: cada dict tiene source, year, month, path, findings: List[Finding], errors_ct, warns_ct."""
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("RESUMEN POR MES — chequeos que no pasaron vs. lo que espera el manifiesto")
    lines.append("(Si no hay lista bajo un mes = ese archivo pasó estas comprobaciones; revise logs para filas OK.)")
    lines.append("=" * 78)

    for row in rows:
        y, m, src = row["year"], row["month"], row["source"]
        findings: List[Finding] = row["findings"]
        lines.append("")
        mes_lbl = f"{y}-{m:02d}"
        lines.append("[%s] %s  archivo: %s" % (src, mes_lbl, os.path.basename(row.get("path", ""))))
        if not findings:
            lines.append("  (sin advertencias ni errores de manifiesto en esta corrida)")
            continue
        for f in findings:
            lines.extend(f.lines())

    lines.append("")
    lines.append("=" * 78)
    text = "\n".join(lines)
    print(text)
    logger.info(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validar Excel raw pendientes (no aún en interim).")
    parser.add_argument("--config", default="config.yaml", help="YAML en config/ (default: config.yaml).")
    parser.add_argument(
        "--manifest",
        default="raw_manifest.generated.yaml",
        help="Nombre del manifiesto en config/ (default: raw_manifest.generated.yaml). Use '' para omitir.",
    )
    parser.add_argument("--skip-manifest", action="store_true", help="No cargar manifiesto; solo read+clean.")
    parser.add_argument(
        "--null-slack",
        type=float,
        default=0.08,
        help="Margen sobre null_fraction_observed del manifiesto antes de advertir (default: 0.08).",
    )
    parser.add_argument(
        "--strict-enums",
        action="store_true",
        help="Tratar valores fuera de allowed_values_observed como error (si hay manifiesto).",
    )
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Salir con código 1 si hubo advertencias.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="No imprimir el bloque RESUMEN POR MES al final.",
    )
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        paths = get_paths(cfg)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar config: {e}", file=sys.stderr)
        return 1

    level_name = (cfg.get("log_level") or "INFO").strip().upper()
    log_level = getattr(logging, level_name, logging.INFO)
    log_dir = paths.get("logs")
    logger = setup_logging(log_dir=log_dir, log_level=log_level)

    raw_base = paths["raw"]
    interim_base = paths["interim"]
    etl_cfg = cfg.get("etl", {}) or {}
    sources: List[str] = list(etl_cfg.get("sources", ["inspecciones", "consumo"]))

    manifest_sources: Optional[Dict[str, Any]] = None
    if not args.skip_manifest and args.manifest:
        manifest_path = os.path.join(PROJECT_ROOT, "config", args.manifest)
        if not os.path.isfile(manifest_path):
            logger.error(
                "No se encontró manifiesto: %s. Genérelo con poc/profile_interim_manifest.ipynb o use --skip-manifest.",
                manifest_path,
            )
            return 1
        data = load_manifest(manifest_path)
        manifest_sources = data.get("sources") or {}
        logger.info("Manifiesto cargado: %s", manifest_path)
    elif args.skip_manifest:
        logger.info("Validación sin manifiesto (solo read + clean).")

    total_errors = 0
    total_warnings = 0
    any_pending = False
    report_rows: List[Dict[str, Any]] = []

    for source in sources:
        raw_dir = os.path.join(raw_base, source)
        pending = etl.get_pending_months(raw_dir, interim_base, source, overwrite=False)
        if not pending:
            logger.info("[%s] Sin meses pendientes (raw ya reflejado en interim o sin xlsx en raw).", source)
            continue
        any_pending = True
        logger.info("[%s] Meses pendientes a validar: %s", source, pending)

        for year, month in pending:
            raw_file = os.path.join(raw_dir, f"{source}_{year}_{month:02d}.xlsx")
            month_findings: List[Finding] = []
            err, warn = validate_one_file(
                raw_file,
                source,
                year,
                month,
                manifest_sources,
                null_slack=args.null_slack,
                strict_enums=args.strict_enums,
                logger=logger,
                month_findings=month_findings,
            )
            total_errors += err
            total_warnings += warn
            report_rows.append(
                {
                    "source": source,
                    "year": year,
                    "month": month,
                    "path": raw_file,
                    "findings": month_findings,
                    "errors_ct": err,
                    "warns_ct": warn,
                }
            )

    if not any_pending:
        logger.info("Nada que validar: no hay archivos raw pendientes de procesar.")
        return 0

    logger.info("Resumen: errores=%s, advertencias=%s", total_errors, total_warnings)
    if not args.no_summary and report_rows:
        print_monthly_report(report_rows, logger)

    if total_errors:
        return 1
    if total_warnings and args.warnings_as_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
