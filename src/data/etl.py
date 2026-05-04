"""
ETL mensual EMPAGUA: inspecciones, consumo y maestro.
Procesamiento incremental: raw TXT (pipe ``|``) → interim parquet por año/mes.
Alineado con notebooks/desarrollo/2_Contruccion_dataset_v1 (inspecciones celda 8, consumo 24–28, maestro 53 y 61).

Convención ``data/raw``: ``<fuente>/<fuente>_AAAA_MM.txt``, UTF-8, separador ``|`` (misma forma que ``scripts/format_raw_from_entrega.py`` escribe).
"""
import logging
import os
import re
import glob
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

logger = logging.getLogger(__name__)

# Raw mensual entregado por fuente (convención acordada con clientes / partición desde notebooks/v3)
RAW_SEP = "|"
RAW_ENCODING = "utf-8"

# Resultado de inspección → is_fraud (texto normalizado; notebook celda 8)
RESULTADO_TEXTO_IS_FRAUD = frozenset(
    {"ANOMALO", "FRAUDULENTA", "SERVICIO DIRECTO", "MAL ESTADO"}
)

# desc_categoria → categoría agregada para modelado (notebook celda 61)
MAP_DESC_CATEGORIA = {
    "Residencia": "Residencial",
    "Casa de Alquiler y locales": "Residencial",
    "Locales de cualquier tipo": "Comercial",
    "Restaurante": "Comercial",
    "Bar o discoteca": "Comercial",
    "Car Wash": "Comercial",
    "Lavanderia": "Comercial",
    "Spa o Salon de belleza": "Comercial",
    "Gimnasio": "Comercial",
    "Auto Hotel": "Comercial",
    "Hotel de paso": "Comercial",
    "Centro Comercial": "Comercial",
    "Gobierno": "Institucional",
    "Centros de Estudio Privado": "Institucional",
    "Hospital Privado": "Institucional",
    "Empresa de Seguridad": "Institucional",
    "Purificadora de agua": "Institucional",
    "Fabricas de todo tipo": "Industrial",
    "Edificios de todo tipo": "Otros",
}


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
    raw_files = glob.glob(os.path.join(raw_dir, f"{source_name}_*.txt"))
    raw_months = set()

    for f in raw_files:
        basename = os.path.basename(f)
        match = re.search(rf"{source_name}_(\d{{4}})_(\d{{2}})\.txt", basename)
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
    Exige columnas ``id_servicio``, ``fecha``, ``resultado``, ``id_inspeccion``, ``tiene_sancion``
    (las dos últimas no se transforman aquí; se conservan si vienen en el DataFrame).

    Pasos: nombres de columna strip + lower; ``fecha`` con ``dayfirst=True``; filas sin fecha válida fuera;
    ``id_servicio`` → ``contrato`` (str); ``date`` agregación mensual desde el calendario de ``fecha``;
    ``is_fraud`` = 1 si ``resultado`` coincide **exactamente** con algún texto en
    ``RESULTADO_TEXTO_IS_FRAUD``; una fila por par (``contrato``, ``date``) quedándose la de
    ``is_fraud`` máximo; descarta ``contrato`` vacío.

    Entrada: filas leídas desde raw TXT ``|`` UTF-8.
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()

    required_cols = {"id_servicio", "fecha", "resultado", "id_inspeccion", "tiene_sancion"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Inspecciones EMPAGUA: faltan columnas requeridas {sorted(missing)}")

    df["id_servicio"] = df["id_servicio"].astype(str)
    df = df.dropna(subset=["fecha"]).reset_index(drop=True)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["fecha"]).reset_index(drop=True)

    df.rename(columns={"id_servicio": "contrato"}, inplace=True)

    df["date"] = pd.to_datetime(
        df["fecha"].dt.year.astype(str) + "-" + df["fecha"].dt.month.astype(str)
    )

    df["is_fraud"] = df["resultado"].isin(RESULTADO_TEXTO_IS_FRAUD).astype(int)

    df.reset_index(drop=True, inplace=True)
    df = df.loc[df.groupby(["contrato", "date"])["is_fraud"].idxmax()].reset_index(drop=True)

    df["contrato"] = df["contrato"].astype(str).str.strip()
    df = df[df["contrato"].str.len() > 0].reset_index(drop=True)

    return df


def clean_consumo(df):
    """
    Exige ``id_servicio``, ``fcm_anio``, ``fcm_mes``, ``fcm_m3_fact``, ``m3_fact_tipo``,
    ``cod_problema``, ``estatus``, ``tuvo_cm``.

    Pasos: columnas strip + lower; rename a ``contrato``, ``ano``, ``mes``, ``consumo``;
    ``date`` primer día del mes desde año/mes numéricos (filtra año > 0 y mes 1–12);
    si ``consumo`` viene como texto, toma la parte antes de la primera coma y luego numérico;
    filtra ``consumo`` no nulo y ``>= 0``, castea a entero;
    ``cod_problema`` numérico coerce; ``estatus``, ``m3_fact_tipo``, ``tuvo_cm`` strip + upper;
    ``drop_duplicates(['contrato','date'], keep='first')`` y orden por ``date``.

    Entrada: raw mensual TXT ``|`` UTF-8.
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()

    required_cols = {
        "id_servicio",
        "fcm_anio",
        "fcm_mes",
        "fcm_m3_fact",
        "m3_fact_tipo",
        "cod_problema",
        "estatus",
        "tuvo_cm",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Consumo EMPAGUA: faltan columnas requeridas {sorted(missing)}")

    df.rename(
        columns={
            "id_servicio": "contrato",
            "fcm_anio": "ano",
            "fcm_mes": "mes",
            "fcm_m3_fact": "consumo",
        },
        inplace=True,
    )

    df["contrato"] = df["contrato"].astype(str).str.strip()
    df = df[df["contrato"].str.len() > 0].dropna(subset=["contrato"]).reset_index(drop=True)

    df["month"] = pd.to_numeric(df["mes"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["ano"], errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df = df[(df["year"] > 0) & df["month"].between(1, 12)].copy()

    if df["consumo"].dtype == object or (
        hasattr(df["consumo"].dtype, "name") and df["consumo"].dtype.name == "string"
    ):
        df["consumo"] = df["consumo"].astype(str).str.split(",").str[0]
    df["consumo"] = pd.to_numeric(df["consumo"], errors="coerce")
    df = df[df["consumo"].notna() & (df["consumo"] >= 0)].copy()
    df["consumo"] = df["consumo"].fillna(0).astype(int)

    df["cod_problema"] = pd.to_numeric(df["cod_problema"], errors="coerce")
    df["estatus"] = df["estatus"].astype(str).str.strip().str.upper()
    df["m3_fact_tipo"] = df["m3_fact_tipo"].astype(str).str.strip().str.upper()
    df["tuvo_cm"] = df["tuvo_cm"].astype(str).str.strip().str.upper()

    df = df.drop_duplicates(subset=["contrato", "date"], keep="first").reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_maestro(df):
    """
    Exige ``id_servicio``, ``fecha_medidor``, ``desc_categoria``, ``municipio``, ``colonia``,
    ``zona``, ``tipo``, ``es_digital`` (cabecera en raw como ``id_servicio``; aquí se renombra a ``contrato``).

    Pasos: nombres de columna ``unidecode(strip(lower))``; ``contrato`` str sin vacíos;
    ``fecha_medidor`` a datetime; ``categoria`` desde ``desc_categoria`` vía ``MAP_DESC_CATEGORIA``,
    valores no mapeados → ``Otros``; ``drop_duplicates`` por ``contrato`` con ``keep='last'``.

    Entrada: raw mensual TXT ``|`` UTF-8.
    """
    df = df.copy()
    df.columns = [unidecode(str(x).strip().lower()) for x in df.columns]

    required_cols = {
        "id_servicio",
        "fecha_medidor",
        "desc_categoria",
        "municipio",
        "colonia",
        "zona",
        "tipo",
        "es_digital",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Maestro EMPAGUA: faltan columnas requeridas {sorted(missing)}")

    df["id_servicio"] = df["id_servicio"].astype(str)
    df.rename(columns={"id_servicio": "contrato"}, inplace=True)

    df["contrato"] = df["contrato"].str.strip()
    df = df[df["contrato"].str.len() > 0].dropna(subset=["contrato"]).reset_index(drop=True)

    df["fecha_medidor"] = pd.to_datetime(df["fecha_medidor"], errors="coerce")

    dc = df["desc_categoria"]
    df["categoria"] = "Otros"
    ok = dc.notna()
    df.loc[ok, "categoria"] = (
        dc.loc[ok].astype(str).str.strip().map(MAP_DESC_CATEGORIA).fillna("Otros")
    )

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
    raw_file = os.path.join(raw_dir, source_name, f"{source_name}_{year}_{month:02d}.txt")
    output_dir = os.path.join(interim_dir, source_name, f"year={year}", f"month={month:02d}")
    output_file = os.path.join(output_dir, f"{source_name}.parquet")

    if os.path.exists(output_file) and not overwrite:
        logger.info("%s %s-%s ya procesado, saltando.", source_name, year, month)
        return False

    if not os.path.exists(raw_file):
        logger.warning("%s no existe, saltando.", raw_file)
        return False

    logger.debug("Procesando %s %s-%s...", source_name, year, month)
    df = pd.read_csv(raw_file, sep=RAW_SEP, encoding=RAW_ENCODING, encoding_errors="replace")
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
