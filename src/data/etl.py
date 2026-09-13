"""
ETL mensual CAJ: inspecciones, consumo y maestro.
Procesamiento incremental: raw xlsx → interim parquet por año/mes.
"""
import logging
import os
import re
import glob
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

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
    """Limpieza de inspecciones CAJ."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df.matricula = df.matricula.astype(str)
    df.rename(columns={"matricula": "contrato"}, inplace=True)
    df["data_da_fiscalizacao"] = pd.to_datetime(df["data_da_fiscalizacao"], errors="coerce", dayfirst=True)
    df["date"] = pd.to_datetime(
        df["data_da_fiscalizacao"].dt.year.astype(str) + "-" + df["data_da_fiscalizacao"].dt.month.astype(str)
    )
    df["is_fraud_1"] = df.motivo.isin(["Sim: By-pass", "Sim: LA clandestina", "Sim: Corte ramal violado"]).astype(int)
    df["is_fraud_2"] = df.hidrometro_invertido.isin(["SIM"]).astype(int)
    df["is_fraud_3"] = df.situacao_dos_lacres_cavalete.isin(["Rompido", "Sem lacre"]).astype(int)
    df["is_fraud_4"] = df.situacao_do_hidrometro.isin(
        ["Não está no cavalete", "Danificado - Cliente", "Enviar para análise", "Danificado - Imã"]
    ).astype(int)
    df["is_fraud_5"] = df.situacao_da_la_pelo_fiscal.isin(["Violada"]).astype(int)
    df["is_fraud_6"] = df.situacao_cavalete.isin(["Intervenção no cavalete"]).astype(int)
    df["is_fraud"] = (
        df[["is_fraud_1", "is_fraud_2", "is_fraud_3", "is_fraud_4", "is_fraud_5", "is_fraud_6"]].sum(axis=1) > 0
    ).astype(int)
    df.reset_index(drop=True, inplace=True)
    df = df.loc[df.groupby(["contrato", "date"]).is_fraud.idxmax()]
    return df


def clean_consumo(df):
    """Limpieza de consumo CAJ: numérico con coerce, flags de situacao_la."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df.matricula = df.matricula.astype(str)
    df.mes_fatura = pd.to_datetime(df.mes_fatura)
    df["consumo"] = pd.to_numeric(df["consumo"], errors="coerce")
    df.consumo = df.consumo.astype(float)
    df.situacao_la = df.situacao_la.astype(str)
    df.rename(columns={"matricula": "contrato", "mes_fatura": "date"}, inplace=True)
    df = df.dropna(subset=["consumo"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["contrato", "date"], keep="first").reset_index(drop=True)
    df["flag_ativa"] = np.where(df["situacao_la"].isin(["Ativa"]), 1, 0).astype("uint8")
    df["flag_cancelada"] = np.where(df["situacao_la"].isin(["Cancelada"]), 1, 0).astype("uint8")
    df["flag_c_cavalete"] = np.where(df["situacao_la"].isin(["Cortada Cavalete"]), 1, 0).astype("uint8")
    df["flag_suprimida"] = np.where(df["situacao_la"].isin(["Suprimida"]), 1, 0).astype("uint8")
    return df


def clean_maestro(df):
    """Limpieza de maestro CAJ."""
    df = df.copy()
    df.columns = [unidecode(x.strip().lower()) for x in df.columns]
    df.matricula = df.matricula.astype(str)
    df.rename(columns={"matricula": "contrato"}, inplace=True)
    df.loc[pd.to_numeric(df["tipo_cliente"], errors="coerce").notna(), "tipo_cliente"] = np.nan
    df = df.drop_duplicates(subset=["contrato"])
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
            summary[source] = {"processed": 0, "skipped": 0, "total": 0}
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
            "total": len(pending)
        }
        logger.info("%s: %s procesados, %s saltados", source, processed_count, skipped_count)

    return summary
