"""
ETL mensual: inspecciones y consumo (Querétaro).
Procesamiento incremental: raw xlsx → interim parquet por año/mes.
"""
import logging
import os
import re
import glob
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================================
# ETL MENSUAL - Funciones para procesamiento incremental por mes
# ============================================================================

def get_pending_months(raw_dir, interim_dir, source_name, overwrite=False):
    """
    Compara archivos en raw/ vs parquets en interim/
    Retorna lista de (year, month) pendientes de procesar

    Args:
        raw_dir: directorio con archivos raw de la fuente (ej: data/raw/inspecciones/)
        interim_dir: directorio base con parquets procesados (ej: data/interim/)
        source_name: nombre de la fuente (ej: 'inspecciones', 'consumo')
        overwrite: si True, retorna todos los meses en raw (para reprocesar)

    Returns:
        Lista de tuplas (year, month) pendientes
    """
    raw_files = glob.glob(os.path.join(raw_dir, f"{source_name}_*.xlsx"))
    raw_months = set()

    for f in raw_files:
        basename = os.path.basename(f)
        match = re.search(rf"{source_name}_(\d{{4}})_(\d{{2}})\.xlsx", basename)
        if match:
            raw_months.add((int(match.group(1)), int(match.group(2))))

    if overwrite:
        return sorted(raw_months)

    processed_months = set()
    pattern = os.path.join(interim_dir, source_name, "year=*", "month=*", f"{source_name}.parquet")
    parquet_files = glob.glob(pattern)

    for f in parquet_files:
        match = re.search(r"year=(\d{4})/month=(\d{2})", f.replace("\\", "/"))
        if match:
            processed_months.add((int(match.group(1)), int(match.group(2))))

    return sorted(raw_months - processed_months)


def clean_inspecciones(df):
    """
    Limpieza específica para inspecciones (Querétaro)

    Args:
        df: DataFrame con datos de inspecciones

    Returns:
        DataFrame limpio
    """
    df.columns = df.columns.str.lower()

    df['fecfin_hins'] = pd.to_datetime(df['fecfin_hins'], errors='coerce')
    df = df.dropna(subset=['fecfin_hins']).copy()

    df['year'] = df['fecfin_hins'].dt.year
    df['month'] = df['fecfin_hins'].dt.month
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + '-' +
        df['month'].astype(str).str.zfill(2) + '-01'
    )

    if 'acta_hins' in df.columns:
        df['is_fraud'] = df['acta_hins'].isin([1]).astype(int)
    else:
        df['is_fraud'] = 0

    if 'numcontratoacis_orcoa' in df.columns:
        df = df.rename(columns={'numcontratoacis_orcoa': 'contrato'})

    df = df.dropna(subset=['contrato']).reset_index(drop=True)

    df = df.sort_values(['contrato', 'date', 'is_fraud'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=['contrato', 'date'], keep='first').reset_index(drop=True)

    return df


def clean_consumo(df):
    """
    Limpieza específica para consumo (Querétaro)

    Args:
        df: DataFrame con datos de consumo

    Returns:
        DataFrame limpio
    """
    df.columns = df.columns.str.lower()

    df['anio_grl'] = pd.to_numeric(df['anio_grl'], errors='coerce').astype('Int64')
    df['periodo_grl'] = pd.to_numeric(df['periodo_grl'], errors='coerce').astype('Int64')

    df = df.dropna(subset=['anio_grl', 'periodo_grl']).copy()
    df = df[df['periodo_grl'].between(1, 12)].copy()

    df['date'] = pd.to_datetime(
        df['anio_grl'].astype(str) + '-' +
        df['periodo_grl'].astype(str).str.zfill(2) + '-01'
    )

    if 'contrato_grl' in df.columns:
        df = df.rename(columns={'contrato_grl': 'contrato'})
    if 'consumo_gral' in df.columns:
        df = df.rename(columns={'consumo_gral': 'consumo'})

    df = df.dropna(subset=['contrato']).reset_index(drop=True)

    if 'consumo' in df.columns:
        df['consumo'] = df['consumo'].apply(lambda x: None if pd.isna(x) or x < 0 else x)

    return df


def process_month(raw_dir, interim_dir, source_name, year, month, clean_func, overwrite=False):
    """
    Procesa un mes específico y guarda en interim/

    Args:
        raw_dir: directorio base de raw (ej: data/raw)
        interim_dir: directorio base de interim (ej: data/interim)
        source_name: nombre de la fuente
        year: año
        month: mes
        clean_func: función de limpieza a aplicar
        overwrite: si True, reprocesa aunque ya exista

    Returns:
        True si procesó, False si saltó
    """
    raw_file = os.path.join(raw_dir, source_name, f"{source_name}_{year}_{month:02d}.xlsx")
    output_dir = os.path.join(interim_dir, source_name, f"year={year}", f"month={month:02d}")
    output_file = os.path.join(output_dir, f"{source_name}.parquet")

    if os.path.exists(output_file) and not overwrite:
        logger.info("  [SKIP] %s %s-%02d ya procesado, saltando...", source_name, year, month)
        return False

    if not os.path.exists(raw_file):
        logger.warning("  [WARN] %s no existe, saltando...", raw_file)
        return False

    logger.debug("  [PROC] Procesando %s %s-%02d...", source_name, year, month)
    df = pd.read_excel(raw_file)
    df = clean_func(df)

    os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_file, index=False)

    logger.debug("  [OK] Guardado: %s (%s registros)", output_file, len(df))
    return True


def run_monthly_etl(raw_dir="../../data/raw",
                    interim_dir="../../data/interim",
                    sources=["inspecciones", "consumo"],
                    overwrite=False):
    """
    Ejecuta ETL mensual incremental para todas las fuentes

    Args:
        raw_dir: directorio base de raw
        interim_dir: directorio base de interim
        sources: lista de fuentes a procesar
        overwrite: si True, reprocesa todo aunque ya exista

    Returns:
        Dict con resumen de procesamiento
    """
    clean_funcs = {
        "inspecciones": clean_inspecciones,
        "consumo": clean_consumo
    }

    summary = {}

    for source in sources:
        logger.info("\n%s", "=" * 60)
        logger.info("[%s]", source.upper())
        logger.info("%s", "=" * 60)

        if source not in clean_funcs:
            logger.warning("  [WARN] Funcion de limpieza no definida para '%s', saltando...", source)
            continue

        raw_source_dir = os.path.join(raw_dir, source)
        if not os.path.exists(raw_source_dir):
            logger.warning("  [WARN] Directorio %s no existe, saltando...", raw_source_dir)
            continue

        pending = get_pending_months(raw_source_dir, interim_dir, source, overwrite=overwrite)

        if not pending:
            logger.info("  [OK] No hay meses pendientes")
            summary[source] = {"processed": 0, "skipped": 0, "total": 0}
            continue

        logger.info("  [INFO] %s meses pendientes", len(pending))
        if len(pending) <= 10:
            logger.info("     %s", pending)
        else:
            logger.info("     %s ... %s", pending[:5], pending[-5:])

        processed_count = 0
        skipped_count = 0

        for year, month in tqdm(pending, desc=f"  Procesando {source}"):
            result = process_month(
                raw_dir, interim_dir, source, year, month,
                clean_funcs[source], overwrite
            )
            if result:
                processed_count += 1
            else:
                skipped_count += 1

        summary[source] = {
            "processed": processed_count,
            "skipped": skipped_count,
            "total": len(pending)
        }
        logger.info("  [OK] %s: %s procesados, %s saltados", source, processed_count, skipped_count)

    return summary
