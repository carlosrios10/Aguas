#!/usr/bin/env python
"""
Reorganiza una entrega de datos (p. ej. EAAB Bogotá) al layout que espera el ETL:

  <dest>/inspecciones/inspecciones_AAAA_MM.txt
  <dest>/consumo/consumo_AAAA_MM.txt
  <dest>/maestro/maestro_AAAA_MM.txt

No transforma columnas ni separadores: solo copia y normaliza el nombre del archivo.
El ETL espera TXT UTF-8 con separador ``|``; si la entrega viene distinto, hay que
convertir en otro paso.

Layouts de entrada (--layout):

  nested (default):  <source>/inspecciones/*.txt, <source>/consumo/*.txt, ...
  flat:            todos los .txt en <source>/; el nombre debe contener el tipo
                   (inspecciones|consumo|maestro) y año-mes detectable.

Ejemplos (desde la raíz del proyecto):

  python scripts/partition_raw_from_delivery.py --source "D:\\entregas\\bogota_mayo" --dry-run
  python scripts/partition_raw_from_delivery.py --source "D:\\entregas\\bogota_mayo"
  python scripts/partition_raw_from_delivery.py --source "D:\\data" --dest "data/raw" --layout flat
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from typing import Iterable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

KNOWN_SOURCES = ("inspecciones", "consumo", "maestro")


def infer_year_month(source: str, path: str) -> tuple[int, int] | None:
    """
    Obtiene (año, mes) desde la ruta o nombre de archivo.
    Si el archivo ya es ``{source}_AAAA_MM.txt``, usa ese patrón; si no, busca AAAA_MM,
    AAAA-MM o AAAAMM (mes 01-12).
    """
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    if ext.lower() != ".txt":
        return None

    pat_strict = re.compile(rf"^{re.escape(source)}_(\d{{4}})_(\d{{2}})\.txt$", re.IGNORECASE)
    m = pat_strict.match(base)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return y, mo

    for regex in (
        r"(\d{4})_(\d{2})",
        r"(\d{4})-(\d{2})",
        r"(?<!\d)(\d{4})(\d{2})(?!\d)",
    ):
        m = re.search(regex, stem)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12:
                return y, mo
    return None


def detect_source_from_flat_name(filename: str) -> str | None:
    """Para --layout flat: elige la fuente cuyo nombre aparece en el basename."""
    base = os.path.basename(filename).lower()
    hit: str | None = None
    for s in KNOWN_SOURCES:
        if s in base:
            if hit is not None:
                # Ambiguo (dos palabras clave)
                return None
            hit = s
    return hit


def iter_nested_files(source_root: str, source: str) -> Iterable[str]:
    sub = os.path.join(source_root, source)
    if not os.path.isdir(sub):
        return
    for name in sorted(os.listdir(sub)):
        path = os.path.join(sub, name)
        if os.path.isfile(path) and name.lower().endswith(".txt"):
            yield path


def iter_flat_files(source_root: str) -> Iterable[str]:
    if not os.path.isdir(source_root):
        return
    for name in sorted(os.listdir(source_root)):
        path = os.path.join(source_root, name)
        if os.path.isfile(path) and name.lower().endswith(".txt"):
            yield path


def partition(
    source_root: str,
    dest_raw: str,
    *,
    layout: str,
    sources: list[str],
    dry_run: bool,
    overwrite: bool,
) -> tuple[int, int, list[str]]:
    """
    Copia archivos a dest_raw. Devuelve (copiados, omitidos, advertencias).
    """
    copied = 0
    skipped = 0
    warnings: list[str] = []

    planned: dict[tuple[str, int, int], str] = {}

    def plan_or_copy(src_path: str, source: str) -> None:
        nonlocal copied, skipped
        ym = infer_year_month(source, src_path)
        if ym is None:
            warnings.append(f"No se pudo inferir AAAA_MM: {src_path}")
            skipped += 1
            return
        y, mo = ym
        key = (source, y, mo)
        dest_name = f"{source}_{y}_{mo:02d}.txt"
        dest_dir = os.path.join(dest_raw, source)
        dest_path = os.path.join(dest_dir, dest_name)

        if key in planned and planned[key] != src_path:
            warnings.append(
                f"Conflicto {dest_name}: ya planeado desde {planned[key]}, también {src_path}"
            )
            skipped += 1
            return
        planned[key] = src_path

        if os.path.exists(dest_path) and not overwrite:
            warnings.append(f"Ya existe (use --overwrite): {dest_path}")
            skipped += 1
            return

        if dry_run:
            print(f"[dry-run] COPY {src_path} -> {dest_path}")
            copied += 1
            return

        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        print(f"OK {src_path} -> {dest_path}")
        copied += 1

    if layout == "nested":
        for source in sources:
            if source not in KNOWN_SOURCES:
                warnings.append(f"Fuente desconocida (omitida): {source}")
                continue
            for src_path in iter_nested_files(source_root, source):
                plan_or_copy(src_path, source)
    elif layout == "flat":
        for src_path in iter_flat_files(source_root):
            source = detect_source_from_flat_name(src_path)
            if source is None:
                warnings.append(
                    f"Layout flat: no se detectó fuente en el nombre (inspecciones|consumo|maestro): {src_path}"
                )
                skipped += 1
                continue
            if source not in sources:
                skipped += 1
                continue
            plan_or_copy(src_path, source)
    else:
        raise ValueError(f"layout inválido: {layout}")

    return copied, skipped, warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copia entrega EAAB/Bogotá a data/raw con nombres ETL (fuente_AAAA_MM.txt)."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Carpeta raíz de la entrega (nested: subcarpetas por fuente; flat: todos los .txt aquí).",
    )
    parser.add_argument(
        "--dest",
        default=os.path.join("data", "raw"),
        help="Carpeta raw destino relativa a la raíz del proyecto o ruta absoluta (default: data/raw).",
    )
    parser.add_argument(
        "--layout",
        choices=("nested", "flat"),
        default="nested",
        help="nested: <source>/inspecciones/*.txt, ... | flat: *.txt en --source con nombre que indique la fuente.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(KNOWN_SOURCES),
        help="Lista separada por comas (default: inspecciones,consumo,maestro).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar qué haría.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir si ya existe el destino.",
    )
    args = parser.parse_args()

    source_root = os.path.abspath(args.source)
    dest = args.dest
    if not os.path.isabs(dest):
        dest = os.path.join(PROJECT_ROOT, dest)
    dest = os.path.abspath(dest)

    if not os.path.isdir(source_root):
        print(f"[ERROR] No existe o no es carpeta: {source_root}", file=sys.stderr)
        return 1

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    copied, skipped, warnings = partition(
        source_root,
        dest,
        layout=args.layout,
        sources=sources,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    for w in warnings:
        print(f"[WARN] {w}", file=sys.stderr)

    mode = "dry-run" if args.dry_run else "hecho"
    print(f"[{mode}] copiados={copied} omitidos={skipped} dest={dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
