"""
ETL mensual **Bogotá** (consumo, inspecciones, maestro).

- **Consumo / maestro / inspecciones:** raw mensual ``.csv`` o ``.txt`` (ver cada ``clean_*``).
"""
import logging
import os
import re
import glob
import pandas as pd
from tqdm import tqdm
from unidecode import unidecode

logger = logging.getLogger(__name__)

RAW_SEP = "|"
RAW_ENCODING = "utf-8"

# Positivos (``is_fraud`` = 1): texto en ``anomaliacausainefectividad`` igual a uno de estos
# (tras normalizar: minúsculas, espacios colapsados). Incluye variantes típicas del dato crudo.
TARGET_1 = (
    "acuerdo de pago",
    "posible anomalía en aparato de medición",
    "posible anomalía en el aparato de medición",
    "posible anomalía en el aparato de medidor",
    "posible anoamlía en el aparato de medición",
    "posible anomalia en el aparato de medicion",
    "posible anomalía en aparato de medidor",
    "posible anomalia en aparato de medicion",
    "medidor en mal estado",
    "posible anomalía en el aparto de medición",
    "posible anomalpia en el aparato de medición",
    "medidor mal instalado",
    "posible anomalái en el aparato de medición",
    "bypass",
    "se recomienda cambio de medidor por la zona",
    "cambio de medidor",
    "cambio de medidor por zona",
    "se recomienda cambio de medior por la zona",
    "cambio por zona",
    "requiere cambio de medidor",
    "cambio por la zona",
    "cambio por el área correspondiente",
    "se recomienda cambio de medior por el área encargada",
    "para cambio por el área correspondiente",
    "se recomienda cambio de medidor por zona",
    "conexión clandestina",
    "acometida con conexión no autorizada",
    'se ubica cola de manguera "1/2 sin conexión',
    "clandestina",
    "hotel con medidor desanclado",
    "posible medidor manipulado",
    "obra sin tpo",
    "obra con tpo",
    "reprogramar retiro en bolsa de seguridad (n/a)",
    "reprogramar cambio de medidor",
    "reprogramar retiro de medidor",
    "reprogramar para recoger medidor",
    'reprogramar taponamiento 1"',
    "retiro del medidor en bolsa de seguridad",
    "se recomienda retiro del medidor por área correspondiente",
    "servicio directo",
    "taponar",
    "taponamiento a manguera no ingresa a predio",
    "totalizadoras",
)

TARGET_1_NORMALIZED = frozenset(
    " ".join(str(t).strip().lower().split()) for t in TARGET_1
)


def _normalize_anomalia_label(x) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().lower().split())


def _first_nonnull(series: pd.Series):
    """Primer valor no nulo de la serie (para agg por grupo)."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    return s.iloc[0]

# ============================================================================
# ETL MENSUAL - Funciones para procesamiento incremental por mes
# ============================================================================

def get_pending_months(raw_dir, interim_dir, source_name, overwrite=False):
    """
    Compara archivos en raw/ vs parquets en interim/
    Retorna lista de (year, month) a procesar.

    Detecta ``{source}_AAAA_MM.csv`` y ``{source}_AAAA_MM.txt``.
    """
    raw_months = set()
    for f in glob.glob(os.path.join(raw_dir, f"{source_name}_*.csv")):
        basename = os.path.basename(f)
        match = re.search(rf"{source_name}_(\d{{4}})_(\d{{2}})\.csv$", basename, re.IGNORECASE)
        if match:
            raw_months.add((int(match.group(1)), int(match.group(2))))
    for f in glob.glob(os.path.join(raw_dir, f"{source_name}_*.txt")):
        basename = os.path.basename(f)
        match = re.search(rf"{source_name}_(\d{{4}})_(\d{{2}})\.txt$", basename, re.IGNORECASE)
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
    Inspecciones **Bogotá** (histórico vigencia mensual).

    Cabeceras: ``[unidecode(x) for x in columns.str.lower()]``; exige ``vigencia_inspeccion``,
    ``ctacontrato`` y ``anomaliacausainefectividad``. ``is_fraud = 1`` si el texto está en
    ``TARGET_1`` (véase ``TARGET_1_NORMALIZED``).
    ``date`` desde ``vigencia_inspeccion`` con ``%Y%m``; una fila por (``contrato``, ``date``) vía ``idxmax``.
    """
    df_ordenes = df.copy()
    df_ordenes.columns = [unidecode(x) for x in df_ordenes.columns.astype(str).str.lower()]

    required = {"vigencia_inspeccion", "ctacontrato", "anomaliacausainefectividad"}
    missing = required - set(df_ordenes.columns)
    if missing:
        raise ValueError(f"Inspecciones Bogotá: faltan columnas requeridas {sorted(missing)}")

    df_ordenes = df_ordenes.dropna(subset=["vigencia_inspeccion"]).reset_index(drop=True)
    df_ordenes["vigencia_inspeccion"] = pd.to_numeric(
        df_ordenes["vigencia_inspeccion"], errors="coerce"
    )
    df_ordenes = df_ordenes.dropna(subset=["vigencia_inspeccion"]).reset_index(drop=True)
    df_ordenes["vigencia_inspeccion"] = (
        df_ordenes["vigencia_inspeccion"].astype(int).astype(str)
    )

    df_ordenes["ctacontrato"] = df_ordenes["ctacontrato"].astype(str)
    df_ordenes.rename(columns={"ctacontrato": "contrato"}, inplace=True)

    _norm = df_ordenes["anomaliacausainefectividad"].map(_normalize_anomalia_label)
    df_ordenes["is_fraud"] = _norm.isin(TARGET_1_NORMALIZED).astype(int)

    df_ordenes["date"] = pd.to_datetime(
        df_ordenes["vigencia_inspeccion"], format="%Y%m", errors="coerce"
    )
    df_ordenes = df_ordenes.dropna(subset=["date"]).reset_index(drop=True)

    df_ordenes = df_ordenes.reset_index(drop=True)
    df_ordenes = df_ordenes.loc[
        df_ordenes.groupby(["contrato", "date"])["is_fraud"].idxmax()
    ].reset_index(drop=True)

    df_ordenes["contrato"] = df_ordenes["contrato"].str.strip()
    df_ordenes = df_ordenes[df_ordenes["contrato"].str.len() > 0].reset_index(drop=True)

    return df_ordenes


def clean_consumo(df):
    """
    Limpieza consumo **Bogotá**.

    Normaliza cabeceras; exige ``ctacontrato``, ``vig``, ``consumo``, ``periodicidad``.
    Dedup por ``(contrato, vig)``: ``consumo`` suma; demás columnas de perfil con ``first`` / ``indicador`` con primer no nulo.
    Filtra ``consumo >= 0`` y ``periodicidad`` equivalente a 2 (bimestral u otra codificación numérica).

    Solo vigencias de **primer semestre calendario** (``month`` 1–6): el ciclo Bogotá bimestral
    no usa meses 7–12 en consumo.
    """
    df_consumo = df.copy()
    df_consumo.columns = [unidecode(str(c).strip().lower()) for c in df_consumo.columns]

    required = {"ctacontrato", "vig", "consumo", "periodicidad"}
    missing = required - set(df_consumo.columns)
    if missing:
        raise ValueError(f"Consumo Bogotá: faltan columnas requeridas {sorted(missing)}")

    df_consumo["ctacontrato"] = df_consumo["ctacontrato"].astype(str)
    df_consumo = df_consumo.dropna(subset=["vig"]).reset_index(drop=True)

    df_consumo["vig"] = pd.to_numeric(df_consumo["vig"], errors="coerce")
    df_consumo = df_consumo.dropna(subset=["vig"]).reset_index(drop=True)
    df_consumo["vig"] = df_consumo["vig"].astype(int).astype(str)

    df_consumo["consumo"] = pd.to_numeric(df_consumo["consumo"], errors="coerce").astype(float)

    df_consumo = df_consumo.rename(columns={"ctacontrato": "contrato"})
    df_consumo = df_consumo[df_consumo["vig"] != "202313"].copy()
    df_consumo = df_consumo[df_consumo["vig"] != "202413"].copy()

    df_consumo["date"] = pd.to_datetime(df_consumo["vig"], format="%Y%m", errors="coerce")
    df_consumo = df_consumo.dropna(subset=["date"]).reset_index(drop=True)
    df_consumo["year"] = df_consumo["date"].dt.year
    df_consumo["month"] = df_consumo["date"].dt.month
    df_consumo = df_consumo.loc[df_consumo["month"] <= 6].copy().reset_index(drop=True)

    df_consumo["contrato"] = df_consumo["contrato"].str.strip()
    df_consumo = df_consumo[df_consumo["contrato"].str.len() > 0].reset_index(drop=True)

    agg_map: dict = {"consumo": "sum"}
    if "indicador" in df_consumo.columns:
        agg_map["indicador"] = _first_nonnull
    for col in (
        "ciclo",
        "poblacion",
        "zona",
        "uso",
        "estrato",
        "periodicidad",
        "codconsumo",
        "lectura1",
        "lectura2",
        "date",
        "year",
        "month",
    ):
        if col in df_consumo.columns and col not in agg_map:
            agg_map[col] = "first"

    df_consumo = (
        df_consumo.groupby(["contrato", "vig"], sort=False)
        .agg(agg_map)
        .reset_index()
    )

    df_consumo = df_consumo[df_consumo["consumo"] >= 0].copy()
    per = pd.to_numeric(df_consumo["periodicidad"], errors="coerce")
    df_consumo = df_consumo[per == 2].copy()

    df_consumo = df_consumo.sort_values(["date", "contrato"]).reset_index(drop=True)
    return df_consumo


def clean_maestro(df):
    """
    Maestro **Bogotá**.

    ``columns = [unidecode(x) for x in columns.str.lower()]``; ``ctacontrato`` como str;
    rename a ``contrato``; descarta contratos vacíos y duplicados por ``contrato`` (``keep='last'``).
    """
    df_maestro = df.copy()
    df_maestro.columns = [unidecode(x) for x in df_maestro.columns.astype(str).str.lower()]

    if "ctacontrato" not in df_maestro.columns:
        raise ValueError(
            f"Maestro Bogotá: falta ctacontrato; columnas: {sorted(df_maestro.columns.tolist())}"
        )

    df_maestro["ctacontrato"] = df_maestro["ctacontrato"].astype(str)
    df_maestro.rename(columns={"ctacontrato": "contrato"}, inplace=True)

    df_maestro["contrato"] = df_maestro["contrato"].str.strip()
    df_maestro = df_maestro[
        df_maestro["contrato"].str.len() > 0
    ].dropna(subset=["contrato"]).reset_index(drop=True)

    df_maestro = df_maestro.drop_duplicates(subset=["contrato"], keep="last").reset_index(drop=True)
    return df_maestro


def _resolve_raw_file(raw_dir, source_name, year, month):
    """Prefiere ``.csv`` (consumo Bogotá); si no, ``.txt`` (legacy)."""
    sub = os.path.join(raw_dir, source_name)
    stem = f"{source_name}_{year}_{month:02d}"
    csv_p = os.path.join(sub, stem + ".csv")
    txt_p = os.path.join(sub, stem + ".txt")
    if os.path.isfile(csv_p):
        return csv_p, "csv"
    if os.path.isfile(txt_p):
        return txt_p, "pipe"
    return None, None


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
        True si guardó parquet con al menos un registro; False si saltó, sin raw, o limpieza vacía.
    """
    raw_file, raw_mode = _resolve_raw_file(raw_dir, source_name, year, month)
    output_dir = os.path.join(interim_dir, source_name, f"year={year}", f"month={month:02d}")
    output_file = os.path.join(output_dir, f"{source_name}.parquet")

    if os.path.exists(output_file) and not overwrite:
        logger.info("%s %s-%s ya procesado, saltando.", source_name, year, month)
        return False

    if raw_file is None:
        logger.warning(
            "No hay raw %s para %s-%02d (.csv ni .txt), saltando.",
            source_name,
            year,
            month,
        )
        return False

    logger.debug("Procesando %s %s-%s desde %s...", source_name, year, month, raw_file)
    if raw_mode == "csv":
        df = pd.read_csv(
            raw_file, encoding=RAW_ENCODING, encoding_errors="replace", low_memory=False
        )
    else:
        df = pd.read_csv(
            raw_file, sep=RAW_SEP, encoding=RAW_ENCODING, encoding_errors="replace"
        )
    df = clean_func(df)

    if df is None or len(df) == 0:
        logger.warning(
            "%s %s-%02d: sin registros tras limpieza; no se escribe parquet.",
            source_name,
            year,
            month,
        )
        if os.path.isfile(output_file):
            os.remove(output_file)
            logger.info("Eliminado parquet previo (evitar datos obsoletos): %s", output_file)
        return False

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
