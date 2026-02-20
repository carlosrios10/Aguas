"""
ETL mensual: inspecciones y consumo.
Procesamiento incremental: raw xlsx → interim parquet por año/mes.
Adecuado para EMCALI: inspecciones (contrato, fecha, resultado) y consumo (contrato, ano/mes, consumo, funcion, causa, observacion).
"""
import logging
import os
import re
import glob
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def normalizar_cadena(texto):
    """
    Normaliza una cadena: minúsculas, sin tildes, no alfanuméricos → guión bajo.
    Usado para homogeneizar nombres de columnas.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = re.sub(r"\s+", "_", texto.strip())
    return texto


# ============================================================================
# ETL MENSUAL - Funciones para procesamiento incremental por mes
# ============================================================================

def get_pending_months(raw_dir, interim_dir, source_name, overwrite=False):
    """
    Compara archivos en raw/ vs parquets en interim/
    Retorna lista de (year, month) a procesar.

    Args:
        raw_dir: directorio con archivos raw de la fuente (ej: data/raw/inspecciones/)
        interim_dir: directorio base con parquets procesados (ej: data/interim/)
        source_name: nombre de la fuente (ej: 'inspecciones', 'consumo')
        overwrite: si True, devuelve todos los meses en raw (para reprocesar); si False, solo los pendientes.

    Returns:
        Lista de tuplas (year, month) a procesar
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
    Limpieza para inspecciones (EMCALI).
    Se asume columnas: contrato, fecha, resultado (y opcional observacion).
    - contrato: string, se toma la parte antes del '.' si existe; luego int.
    - fecha: datetime (dayfirst=True) → date (primer día del mes).
    - is_fraud: 1 si resultado == 1, sino 0.
    - Desduplicación por (contrato, date) quedándose con is_fraud máximo.
    """
    df = df.copy()
    df.columns = [normalizar_cadena(c) for c in df.columns]

    df["contrato"] = df["contrato"].astype(str).str.strip().str.split(".").str[0]
    df = df[df["contrato"].str.len() > 0].copy()
    df = df.dropna(subset=["contrato"]).reset_index(drop=True)

    df["date"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"]).copy()

    # Normalizar a primer día del mes
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )

    # resultado puede venir como float (1.0, 0.0)
    df["is_fraud"] = (df["resultado"].astype(float).fillna(-1) == 1).astype(int)

    # Una fila por (contrato, date): la de mayor is_fraud
    df = df.loc[df.groupby(["contrato", "date"])["is_fraud"].idxmax()].reset_index(drop=True)

    # Contrato como int (como en el notebook EMCALI)
    df["contrato"] = df["contrato"].astype(int)

    df["observacion"] = df["observacion"].astype(str)
    return df


def clean_consumo(df):
    """
    Limpieza para consumo (EMCALI). Procesa un mes de datos.
    Se asume columnas: contrato, ano, mes, consumo, funcion, causa, observacion.
    Lógica alineada con notebook 2_Contruccion_dataset_v3.
    """
    df = df.copy()
    df.columns = [normalizar_cadena(c) for c in df.columns]

    df["contrato"] = df["contrato"].astype(str).str.strip()
    df = df[df["contrato"].str.len() > 0].dropna(subset=["contrato"]).reset_index(drop=True)

    df["month"] = pd.to_numeric(df["mes"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    # Consumo: si viene como string con coma (ej. "30,5"), tomar parte entera
    if df["consumo"].dtype == object or (hasattr(df["consumo"].dtype, "name") and df["consumo"].dtype.name == "string"):
        df["consumo"] = df["consumo"].astype(str).str.split(",").str[0]
    df["consumo"] = pd.to_numeric(df["consumo"], errors="coerce")
    df = df[df["consumo"] >= 0].copy()
    df["consumo"] = df["consumo"].fillna(0).astype(int)

    df["causa"] = df["causa"].fillna(0).astype(int)
    df["observacion"] = df["observacion"].fillna(0).astype(int)
    df["funcion"] = df["funcion"].astype(str).str.strip().fillna("")

    # Ordenar y una fila por (contrato, date), como en el notebook
    df = df.sort_values(by=["contrato", "year", "month", "funcion", "consumo"], ascending=[True, True, True, False, False], na_position="last")
    df = df.drop_duplicates(subset=["contrato", "date"], keep="first").reset_index(drop=True)

    df["contrato"] = df["contrato"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _to_str_normalized(serie):
    """
    Convierte a string; preserva NaN. Si el valor es numérico entero (ej. 14.0), devuelve '14'.
    """
    out = pd.Series(index=serie.index, dtype=object)
    mask_na = serie.isna()
    num = pd.to_numeric(serie, errors="coerce")
    mask_whole = num.notna() & (num % 1 == 0)

    out.loc[mask_whole] = num.loc[mask_whole].astype(int).astype(str)
    rest = ~mask_whole & ~mask_na
    out.loc[rest] = serie.loc[rest].astype(str).str.strip().values
    out.loc[mask_na] = np.nan
    return out


def clean_maestro(df):
    """
    Limpieza para maestro (EMCALI). Una foto contrato → atributos.
    Se asume que el Excel trae: contrato, categoria, diametro, estrato, barrio_comuna, ciclo, localidad, medidor.
    """
    df = df.copy()
    df.columns = [normalizar_cadena(c) for c in df.columns]

    df["contrato"] = df["contrato"].astype(str).str.strip()
    df = df[df["contrato"].str.len() > 0].dropna(subset=["contrato"]).reset_index(drop=True)
    df["contrato"] = df["contrato"].astype(int)

    df["categoria"] = _to_str_normalized(df["categoria"]).fillna("sin_dato")
    df["estrato"] = _to_str_normalized(df["estrato"]).fillna("sin_dato")
    df["barrio_comuna"] = _to_str_normalized(df["barrio_comuna"]).fillna("sin_dato")
    df["ciclo"] = _to_str_normalized(df["ciclo"]).fillna("sin_dato")
    df["localidad"] = _to_str_normalized(df["localidad"]).fillna("sin_dato")
    df["diametro"] = pd.to_numeric(df["diametro"], errors="coerce")
    df["medidor"] = _to_str_normalized(df["medidor"]).fillna("sin_dato")

    df = df.drop_duplicates(subset=["contrato"], keep="last").reset_index(drop=True)
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
        logger.info("%s %s-%s ya procesado, saltando.", source_name, year, month)
        return False

    if not os.path.exists(raw_file):
        logger.warning("%s no existe, saltando.", raw_file)
        return False

    logger.debug("Procesando %s %s-%s...", source_name, year, month)
    df = pd.read_excel(raw_file)
    df = clean_func(df)

    os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_file, index=False)

    logger.debug("Guardado: %s (%s registros)", output_file, len(df))
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
        "consumo": clean_consumo,
        "maestro": clean_maestro,
    }

    summary = {}

    for source in sources:
        logger.info("=" * 60)
        logger.info("[%s]", source.upper())
        logger.info("=" * 60)

        if source not in clean_funcs:
            logger.warning("Función de limpieza no definida para '%s', saltando.", source)
            continue

        raw_source_dir = os.path.join(raw_dir, source)
        if not os.path.exists(raw_source_dir):
            logger.warning("Directorio %s no existe, saltando.", raw_source_dir)
            continue

        pending = get_pending_months(raw_source_dir, interim_dir, source, overwrite)

        if not pending:
            logger.info("No hay meses pendientes.")
            summary[source] = {"processed": 0, "skipped": 0, "total_pending": 0}
            continue

        logger.info(
            "%s meses pendientes: %s",
            len(pending),
            pending if len(pending) <= 10 else f"{pending[:5]} ... {pending[-5:]}",
        )

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
            "total_pending": len(pending)
        }
        logger.info("%s: %s procesados, %s saltados", source, processed_count, skipped_count)

    return summary
