"""
Dataset wide y features de serie temporal para modelado.

**Flujo de funciones (orden lógico; no cambiar de improviso)**

1. Utilidades de ingeniería: ``llenar_val_vacios_ciclo``, ``compute_change_trend_percentaje_vars``,
   ``compute_constant_consumption_vars``, ``compute_tsfel_consumption_vars`` (y clases tsfel asociadas).
2. Carga interim: ``load_interim_data``, ``get_consumo_date_range``, ``load_maestro_latest``,
   ``get_date_range_for_cutoff``, ``get_fecha_fraud_list``.
3. Reglas de negocio sobre consumo crudo: ``add_consumo_flags``.
4. Wide por mes de corte: ``create_dataset_wide_for_cutoff`` (train e inferencia arman el wide completo
   en una pasada por corte).
5. Punto de entrada train: ``create_train_dataset`` → concat → pasos post-notebook → paso (1) → parquet.
6. Punto de entrada inferencia: ``create_inference_dataset`` → paso (4) completo → merge maestro → paso (1) → parquet.

**Dataset wide Bogotá (notebook de referencia POC)**

Los parquets interim vienen de ``src.data.etl``. La ventana temporal usa ``cant_periodos`` meses (típico 24); el
ancho fijo del pivote es ``NUM_ANTERIOR_COLS`` = 12 columnas ``N_anterior`` (meses calendario ``<= 6`` del
``date_range`` como en el notebook). Todo el módulo es **sólo contrato Bogotá** según notebook POC.

Tras el merge de órdenes en train, ``is_fraud`` se rellena con ``-1`` si no hay orden (igual que el notebook).

Usado desde ``scripts/run_train``, ``scripts/run_inference`` y notebooks POC.
"""
import logging
import os
import re
import glob
from itertools import groupby
import pandas as pd
import numpy as np
import tsfel
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class TsfelVars(BaseEstimator, TransformerMixin):
    def __init__(self, features_names_path=None, num_periodos=12):
        self.num_periodos = num_periodos
        self.features_names_path = features_names_path

    def obtener_cols_anterior(self, num_cols=12):
        return [f'{i}_anterior' for i in range(num_cols, 0, -1)]

    def extra_cols(self, df, domain, cols, window=12):
        cfg = tsfel.get_features_by_domain(domain)
        df_result = tsfel.time_series_features_extractor(cfg, df[cols].values.tolist(), verbose=1, n_jobs=-1)
        df_result['index'] = df.index
        return df_result

    def compute_by_json(self, df, cols, window=12):
        cfg = tsfel.get_features_by_domain(json_path=self.features_names_path)
        df_result = tsfel.time_series_features_extractor(cfg, df[cols].values.tolist(), n_jobs=-1)
        df_result['index'] = df.index
        return df_result

    def crear_all_tsfel(self, df):
        cols_anterior = self.obtener_cols_anterior(self.num_periodos)
        df_result_stat = self.extra_cols(df, "statistical", cols_anterior, window=self.num_periodos)
        df_result_temporal = self.extra_cols(df, "temporal", cols_anterior, window=self.num_periodos)
        df_result_spectral = self.extra_cols(df, "spectral", cols_anterior, window=self.num_periodos)
        return df_result_stat, df_result_temporal, df_result_spectral

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.features_names_path is not None:
            cols_anterior = self.obtener_cols_anterior(self.num_periodos)
            df_tsfel = self.compute_by_json(X, cols_anterior, window=self.num_periodos)
            X = X.merge(df_tsfel, on='index', how='left')
        else:
            df_result_stat, df_result_temporal, df_result_spectral = self.crear_all_tsfel(X)
            df_tsfel = pd.merge(df_result_stat, df_result_temporal, how='inner', on='index')
            df_tsfel = pd.merge(df_tsfel, df_result_spectral, how='inner', on='index')
            X = X.merge(df_tsfel, on='index', how='left')
        return X


class ExtraVars(BaseEstimator, TransformerMixin):
    def __init__(self, num_periodos=3):
        self.num_periodos = num_periodos

    def fit(self, X, y=None):
        return self

    def obtener_cols_anterior(self, num_cols=12):
        return [f'{i}_anterior' for i in range(num_cols, 0, -1)]

    def transform(self, X):
        return self.create_vbles(X)

    def count_cero(self, x):
        return (x == 0.0).sum()

    def count_cero_seguidos(self, x):
        ceros_seguidos = 2
        consumo = x.values
        g = [[k, len(list(v))] for k, v in groupby(consumo)]
        g = [x for x in g if (x[0] == 0.0) & (x[1] >= ceros_seguidos)]
        if any(g):
            return sorted(g, reverse=True, key=lambda x: x[-1])[0][1]
        return 0

    def calc_slope(self, x):
        consumo = list(x.values)
        slope = np.polyfit(range(len(consumo)), consumo, 1)[0]
        return slope

    def create_vbles(self, df_total_super):
        cols_3_anterior = self.obtener_cols_anterior(num_cols=self.num_periodos)
        num_periodos_str = str(self.num_periodos)
        df_total_super.loc[:, 'mean_' + num_periodos_str] = df_total_super[cols_3_anterior].mean(axis=1)
        df_total_super.loc[:, 'cant_ceros_' + num_periodos_str] = df_total_super[cols_3_anterior].apply(
            self.count_cero, axis=1)
        df_total_super.loc[:, 'max_cant_ceros_seg_' + num_periodos_str] = df_total_super[cols_3_anterior].apply(
            self.count_cero_seguidos, axis=1)
        df_total_super.loc[:, 'slope_' + num_periodos_str] = df_total_super[cols_3_anterior].apply(
            self.calc_slope, axis=1)
        df_total_super.loc[:, 'min_cons' + num_periodos_str] = df_total_super[cols_3_anterior].min(axis=1)
        df_total_super.loc[:, 'max_cons' + num_periodos_str] = df_total_super[cols_3_anterior].max(axis=1)
        df_total_super.loc[:, 'std_cons' + num_periodos_str] = df_total_super[cols_3_anterior].std(axis=1)
        df_total_super.loc[:, 'var_cons' + num_periodos_str] = df_total_super[cols_3_anterior].var(axis=1)
        df_total_super.loc[:, 'skew_cons' + num_periodos_str] = df_total_super[cols_3_anterior].skew(axis=1)
        if self.num_periodos > 3:
            df_total_super.loc[:, 'kurt_cons' + num_periodos_str] = df_total_super[cols_3_anterior].kurt(axis=1)
        return df_total_super


class ChangeTrendPercentajeIdentifierWideTransform(BaseEstimator, TransformerMixin):
    def __init__(self, last_base_value, last_eval_value, threshold, is_wide=True):
        self.last_base_value = last_base_value
        self.last_eval_value = last_eval_value
        self.threshold = threshold
        self.is_wide = is_wide

    def convert_wide(self, df):
        df_wide = pd.pivot(df, index=['index'], columns=['date'], values=['consumo']).reset_index()
        df_wide.columns = ['index'] + [str(i) + '_anterior' for i in range(
            self.last_eval_value + self.last_base_value)][::-1]
        return df_wide

    def get_cant_cols(self):
        cols_base = [str(i) + '_anterior' for i in range(
            self.last_eval_value + 1, self.last_base_value + self.last_eval_value + 1)][::-1]
        cols_eval = [str(i) + '_anterior' for i in range(1, self.last_eval_value + 1)][::-1]
        return cols_base, cols_eval

    def compute_trend_percentage_wide(self, X):
        if self.is_wide is False:
            X = self.convert_wide(X)
        cols_base, cols_eval = self.get_cant_cols()
        X['trend_perc'] = 100 * X[cols_eval].mean(axis=1) / (X[cols_base].mean(axis=1) + 0.000001)
        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy = self.compute_trend_percentage_wide(X_copy)
        X_copy['is_fraud_trend_perc'] = (100 - X_copy['trend_perc'] > self.threshold).astype(int)
        return X_copy.is_fraud_trend_perc


class ConstantConsumptionClassifierWide(BaseEstimator, TransformerMixin):
    def __init__(self, min_count_constante):
        self.min_count_constante = min_count_constante

    def fit(self, X, y=None):
        return self

    def len_max_consumo_constante_seg(self, consumo):
        g = [[k, len(list(v))] for k, v in groupby(consumo)]
        g = [x for x in g if (x[1] >= self.min_count_constante)]
        return 1 if any(g) else 0

    def transform(self, X, y=None):
        pred = X.apply(lambda x: self.len_max_consumo_constante_seg(x.values), axis=1)
        return pred


def llenar_val_vacios_ciclo(df, cant_ciclos_validos):
    """Rellena NaNs en columnas de consumo con ffill y bfill."""
    cols_consumo = [f'{i}_anterior' for i in range(cant_ciclos_validos, 0, -1)]
    df.loc[:, cols_consumo] = df.loc[:, cols_consumo].ffill(axis=1).bfill(axis=1)
    return df


def compute_change_trend_percentaje_vars(df, config_caidas):
    """Añade variables de tendencia (trend_perc) según config_caidas [(last_base, last_eval, threshold), ...]."""
    for c in config_caidas:
        last_base_value, last_eval_value, threshold = c
        trend_perc_model = ChangeTrendPercentajeIdentifierWideTransform(
            last_base_value, last_eval_value, threshold)
        df[f'trend_perc_{c[0]}_{c[1]}'] = trend_perc_model.fit_transform(df)
    return df


def compute_constant_consumption_vars(df, config_constantes):
    """Añade variables de consumo constante (constant_<n>) según config_constantes [n, ...]."""
    cols_cons = [str(i) + '_anterior' for i in range(1, 13)][::-1]
    for c in config_constantes:
        const_model = ConstantConsumptionClassifierWide(c)
        df[f'constant_{c}'] = const_model.fit_transform(df[cols_cons])
    return df


def compute_tsfel_consumption_vars(df, cant_periodos):
    """Añade features tsfel + ExtraVars (3, 6, 12 periodos)."""
    pipe_feature_eng_train = Pipeline([
        ("tsfel vars", TsfelVars(features_names_path=None, num_periodos=cant_periodos)),
        ("add vars3", ExtraVars(num_periodos=3)),
        ("add vars6", ExtraVars(num_periodos=6)),
        ("add vars12", ExtraVars(num_periodos=12)),
    ])
    df = pipe_feature_eng_train.fit_transform(df, None)
    return df


# ---------------------------------------------------------------------------
# Carga desde interim y construcción del dataset wide
# ---------------------------------------------------------------------------

# Notebook Bogotá / construcción wide (órdenes, consumo perfil, maestro)
VARS_FOR_ORDENES = [
    "contrato",
    "date",
    "is_fraud",
    "fecha_ejecucion",
    "anomaliacausainefectividad",
    "porqueesparcialobservacion",
    "cuenta_contrato",
    "efectiva",
    "vigencia_inspeccion",
]
VARS_FOR_CONSUMO = [
    "contrato",
    "ciclo",
    "poblacion",
    "zona",
    "uso",
    "estrato",
    "indicador",
    "periodicidad",
    "codconsumo",
    "lectura1",
    "lectura2",
    "vig",
]
VARS_MAESTRO = [
    "contrato",
    "in_sol_sectorial_ac_desc",
    "sg_latitud",
    "sg_longitud",
    "oc_nme_barrio",
    "oc_nme_localidad",
    "md_marca",
    "md_material",
    "md_diametro",
]

# Merge maestro después del concat train (solo columnas disponibles en interim).
MAESTRO_COLUMNS_AFTER_CONCAT = list(dict.fromkeys(VARS_MAESTRO))

# Columnas pivote/ancho estable: primer semestre acotadas a doce (notebook ``12_anterior``…``1_anterior``)
NUM_ANTERIOR_COLS = 12

# Lista de órdenes: default ancla tipo notebook ``2021-01-01`` + cant_periodos.
FECHA_ANCLA_ORDENES_DEFAULT = "2021-01-01"

CONFIG_CAIDAS = [(1, 4, 90), (1, 3, 90), (2, 3, 90), (1, 6, 90), (3, 3, 90), (6, 5, 10), (6, 6, 10), (6, 4, 10), (6, 1, 10), (5, 5, 10)]
CONFIG_CONSTANTES = [8, 9, 10, 3, 4, 5]

# Columnas derivadas por ``add_consumo_flags``
BOGOTA_CONSUMO_FLAG_COLUMNS = (
    "lectura1_flag_fraude",
    "lectura1_flag_tecnico",
    "lectura1_flag_inaccesible",
    "lectura1_flag_admin",
    "lectura1_sin_obs",
    "lectura2_flag_fraude",
    "lectura2_flag_tecnico",
    "lectura2_flag_inaccesible",
    "lectura2_flag_admin",
    "lectura2_sin_obs",
    "codconsumo_flag_c_normal",
    "codconsumo_flag_c_alto",
    "codconsumo_flag_c_bajo",
    "codconsumo_flag_c_avg",
)

# Consumo ya pasado por ``add_consumo_flags``; si falta alguna, error (sin ramas opcionales).
WIDE_ETIQUETADO_REQUIRED_COLUMNS = (
    "contrato",
    "date",
    "consumo",
    "uso",
    "codconsumo",
    "lectura1",
    "lectura2",
    "indicador",
) + BOGOTA_CONSUMO_FLAG_COLUMNS


def _require_columns(df: pd.DataFrame, required: tuple | list, *, context: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{context}: faltan columnas obligatorias {sorted(missing)}.")


def _bogota_normalized_lectura_codes(series: pd.Series) -> pd.Series:
    """Convierte lectura numérica (7, 7.0) a código '7' para isin con grupos de strings."""
    s = pd.Series(series, index=series.index, dtype=object)
    n = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=object)
    num_mask = n.notna()
    if num_mask.any():
        ints = np.rint(n.loc[num_mask].astype(float)).to_numpy(dtype=np.int64)
        out.loc[num_mask] = ints.astype(str)
    str_mask = (~num_mask) & s.notna()
    if str_mask.any():
        out.loc[str_mask] = s.loc[str_mask].astype(str).str.strip()
    return out


def load_interim_data(interim_dir, source, start_date=None, end_date=None):
    """
    Carga solo los parquets cuyo (year, month) está en [start_date, end_date].
    Si ambos son None, carga todos los parquets.
    """
    pattern = os.path.join(interim_dir, source, "year=*", "month=*", f"{source}.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    start_d = pd.to_datetime(start_date) if start_date is not None else None
    end_d = pd.to_datetime(end_date) if end_date is not None else None
    list_df = []
    for f in files:
        match = re.search(r"year=(\d{4})/month=(\d{2})", f.replace("\\", "/"))
        if not match:
            continue
        y, m = int(match.group(1)), int(match.group(2))
        mes_date = pd.Timestamp(year=y, month=m, day=1)
        if start_d is not None and mes_date < start_d:
            continue
        if end_d is not None and mes_date > end_d:
            continue
        df = pd.read_parquet(f)
        if "date" not in df.columns and "year" in df.columns and "month" in df.columns:
            df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")
        list_df.append(df)
    if not list_df:
        return pd.DataFrame()
    out = pd.concat(list_df, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def get_date_range_for_cutoff(cutoff, cant_periodos):
    """Rango [start_date, end_date] de meses a cargar desde interim según cutoff y ventana."""
    cutoff_d = pd.to_datetime(cutoff).replace(day=1)
    start_d = cutoff_d - pd.DateOffset(months=cant_periodos)
    return start_d, cutoff_d


def get_consumo_date_range(interim_dir):
    """
    Devuelve (min_date, max_date) de consumo en interim sin cargar los parquets.
    Inspecciona solo los paths year=*/month=* en interim/consumo/.
    Si no hay archivos, devuelve (None, None).
    """
    pattern = os.path.join(interim_dir, "consumo", "year=*", "month=*", "consumo.parquet")
    files = glob.glob(pattern)
    if not files:
        return None, None
    fechas = []
    for f in files:
        match = re.search(r"year=(\d{4})/month=(\d{2})", f.replace("\\", "/"))
        if match:
            y, m = int(match.group(1)), int(match.group(2))
            fechas.append(pd.Timestamp(year=y, month=m, day=1))
    return min(fechas), max(fechas)


def load_maestro_latest(interim_dir):
    """
    Carga el maestro más reciente desde interim/maestro/ (partición year/month más reciente).
    Retorna DataFrame con al menos contrato, categoria; vacío si no hay archivos.
    """
    pattern = os.path.join(interim_dir, "maestro", "year=*", "month=*", "maestro.parquet")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    best = None
    best_ym = (-1, -1)
    for f in files:
        match = re.search(r"year=(\d{4})/month=(\d{2})", f.replace("\\", "/"))
        if match:
            y, m = int(match.group(1)), int(match.group(2))
            if (y, m) > best_ym:
                best_ym = (y, m)
                best = f
    if best is None:
        return pd.DataFrame()
    return pd.read_parquet(best)


def add_consumo_flags(df):
    """
    Flags de consumo Bogotá. Exige ``lectura1``, ``lectura2`` y ``codconsumo``.

    ``*_sin_obs`` si lectura nula; flags por categorías en ``codconsumo``;
    ``indicador`` opcional (``X``→1; relleno 0). Modifica in-place.
    """
    if df.empty:
        return df
    need = {"lectura1", "lectura2", "codconsumo"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(
            "Consumo (flags): faltan columnas obligatorias "
            + f"{sorted(missing)}. Se requieren lectura1, lectura2 y codconsumo."
        )

    grupo_fraude = ["7", "10", "12"]
    grupo_tecnico = ["4", "5", "6", "8", "9", "11", "14", "15"]
    grupo_inaccesible = ["16", "17", "18", "19", "20", "21"]
    grupo_admin = ["23", "24", "25", "26", "27", "28", "29", "30"]

    lectura1 = _bogota_normalized_lectura_codes(df["lectura1"])
    lectura2 = _bogota_normalized_lectura_codes(df["lectura2"])

    df["lectura1_flag_fraude"] = lectura1.isin(grupo_fraude).astype("int8")
    df["lectura1_flag_tecnico"] = lectura1.isin(grupo_tecnico).astype("int8")
    df["lectura1_flag_inaccesible"] = lectura1.isin(grupo_inaccesible).astype("int8")
    df["lectura1_flag_admin"] = lectura1.isin(grupo_admin).astype("int8")
    df["lectura1_sin_obs"] = lectura1.isna().astype("int8")

    df["lectura2_flag_fraude"] = lectura2.isin(grupo_fraude).astype("int8")
    df["lectura2_flag_tecnico"] = lectura2.isin(grupo_tecnico).astype("int8")
    df["lectura2_flag_inaccesible"] = lectura2.isin(grupo_inaccesible).astype("int8")
    df["lectura2_flag_admin"] = lectura2.isin(grupo_admin).astype("int8")
    df["lectura2_sin_obs"] = lectura2.isna().astype("int8")

    cc = df["codconsumo"]
    df["codconsumo_flag_c_normal"] = cc.eq("Consumo normal").astype("int8")
    df["codconsumo_flag_c_alto"] = cc.eq("Alto consumo").astype("int8")
    df["codconsumo_flag_c_bajo"] = cc.eq("Bajo consumo").astype("int8")
    df["codconsumo_flag_c_avg"] = cc.eq("Cmo prom hist").astype("int8")

    if "indicador" in df.columns:
        df["indicador"] = df["indicador"].replace({"X": 1}).fillna(0).astype("int8")

    return df


def get_fecha_fraud_list(
    df_ordenes,
    df_consumo=None,
    cant_periodos=12,
    cutoff_max=None,
    min_date_consumo=None,
    fecha_ancla_ordenes=None,
):
    """
    Fechas de corte en ``df_ordenes`` con ``date`` >= umbral.

    Notebook Bogotá: ``min_date = fecha_ancla + DateOffset(months=cant_periodos)`` (p. ej. ancla ``2021-01-01``).

    Alternativa empírica: ``min_date = min(consumo) + DateOffset(months=cant_periodos)`` usando
    ``min_date_consumo`` o ``df_consumo``.
    """
    if df_ordenes.empty:
        return []
    if fecha_ancla_ordenes is not None:
        min_date_data = pd.to_datetime(fecha_ancla_ordenes) + pd.DateOffset(months=cant_periodos)
    elif min_date_consumo is not None:
        min_date_data = min_date_consumo + pd.DateOffset(months=cant_periodos)
    elif df_consumo is not None and not df_consumo.empty:
        min_date_data = df_consumo["date"].min() + pd.DateOffset(months=cant_periodos)
    else:
        return []
    fechas = df_ordenes[df_ordenes["date"] >= min_date_data]["date"].drop_duplicates().sort_values()
    fechas = fechas.astype(str).str[:10].unique().tolist()
    if cutoff_max is not None:
        fechas = [f for f in fechas if f <= str(pd.to_datetime(cutoff_max).date())]
    return fechas


def _df_consumo_g_por_uso_notebook(full_ventana: pd.DataFrame, fecha_fraud: pd.Timestamp) -> pd.DataFrame:
    """Agregados consumo_12m / 6m / 3m por ``uso`` (notebook: horizontes últimos 12 y 6 meses)."""
    v = full_ventana.copy()
    out_cols = [
        "uso",
        "consumo_12m_ts_mean",
        "consumo_12m_ts_max",
        "consumo_6m_ts_mean",
        "consumo_6m_ts_max",
        "consumo_3m_ts_mean",
        "consumo_3m_ts_max",
    ]
    if v.empty:
        return pd.DataFrame(columns=out_cols)

    v["uso"] = v["uso"].astype(str).str.strip()
    v.loc[v["uso"].eq("") | v["uso"].isna(), "uso"] = "sin_uso"

    fecha_fraud = pd.to_datetime(fecha_fraud)
    consumo_anual = (
        v.groupby(["uso", "contrato"])["consumo"].sum().groupby(level=0).agg(["mean", "max"])
    )
    consumo_anual.columns = ["consumo_12m_ts_mean", "consumo_12m_ts_max"]
    ix = consumo_anual.index

    d6 = fecha_fraud - pd.DateOffset(months=12)
    sub6 = v[v["date"] >= d6]
    if sub6.empty:
        consumo_6m = pd.DataFrame(
            np.nan, index=ix, columns=["consumo_6m_ts_mean", "consumo_6m_ts_max"]
        )
    else:
        consumo_6m = (
            sub6.groupby(["uso", "contrato"])["consumo"].sum().groupby(level=0).agg(["mean", "max"])
        )
        consumo_6m.columns = ["consumo_6m_ts_mean", "consumo_6m_ts_max"]
        consumo_6m = consumo_6m.reindex(ix)

    d3 = fecha_fraud - pd.DateOffset(months=6)
    sub3 = v[v["date"] >= d3]
    if sub3.empty:
        consumo_3m = pd.DataFrame(
            np.nan, index=ix, columns=["consumo_3m_ts_mean", "consumo_3m_ts_max"]
        )
    else:
        consumo_3m = (
            sub3.groupby(["uso", "contrato"])["consumo"].sum().groupby(level=0).agg(["mean", "max"])
        )
        consumo_3m.columns = ["consumo_3m_ts_mean", "consumo_3m_ts_max"]
        consumo_3m = consumo_3m.reindex(ix)

    return pd.concat([consumo_anual, consumo_6m, consumo_3m], axis=1).reset_index()


def create_dataset_wide_for_cutoff(
    fecha_fraud,
    df_consumo,
    df_ordenes,
    cant_periodos,
    vars_ordenes,
    vars_consumo,
    mode="train",
    max_ctas=None,
):
    """
    Wide por fecha de corte (notebook Bogotá: ventana ``cant_periodos`` meses, pivote primer semestre → 12 ``*_anterior``).

    Al inicio valida columnas en ``df_consumo`` (flags + ``vars_consumo``) y, si ``mode=='train'``, en ``df_ordenes``.
    """
    fecha_fraud = pd.to_datetime(fecha_fraud)
    date_inicial = fecha_fraud - pd.DateOffset(months=int(cant_periodos))

    if df_consumo is None or df_consumo.empty:
        return pd.DataFrame()

    wide_consumo_cols = tuple(
        dict.fromkeys(list(WIDE_ETIQUETADO_REQUIRED_COLUMNS) + list(vars_consumo))
    )
    _require_columns(
        df_consumo,
        wide_consumo_cols,
        context="create_dataset_wide_for_cutoff (df_consumo debe incluir flags y vars_consumo)",
    )
    if mode == "train":
        _require_columns(
            df_ordenes,
            vars_ordenes,
            context="create_dataset_wide_for_cutoff (df_ordenes)",
        )

    df_consumo_ventana = df_consumo[
        (df_consumo["date"] < fecha_fraud) & (df_consumo["date"] >= date_inicial)
    ].copy()
    df_consumo_g = _df_consumo_g_por_uso_notebook(df_consumo_ventana, fecha_fraud)

    df_etiquetado = df_consumo_ventana.copy()
    if mode == "train":
        ctas = df_ordenes[df_ordenes["date"] == fecha_fraud]["contrato"].unique().tolist()
        if not ctas:
            return pd.DataFrame()
        ctas_set = set(ctas)
        contratos_ventana = df_consumo_ventana["contrato"].unique()
        ctas_sin_label = [c for c in contratos_ventana if c not in ctas_set]
        if max_ctas is not None and max_ctas > 0 and len(ctas_sin_label) > 0:
            n_neg = min(max_ctas, len(ctas_sin_label))
            ctas_neg = pd.Series(ctas_sin_label).sample(n=n_neg, random_state=42).tolist()
            ctas_totales = ctas + ctas_neg
        else:
            ctas_totales = ctas
        df_etiquetado = df_etiquetado[df_etiquetado["contrato"].isin(ctas_totales)]

    if df_etiquetado.empty:
        return pd.DataFrame()

    uso_series = df_etiquetado["uso"].astype(str).str.strip()
    df_etiquetado["_uso_join"] = uso_series.replace("", "sin_uso").fillna("sin_uso")

    df_static_vars = df_etiquetado.loc[df_etiquetado.groupby("contrato")["date"].idxmax()]

    agg_map: dict = {
        "_uso_join": pd.Series.nunique,
        "codconsumo": pd.Series.nunique,
        "lectura1": "count",
        "lectura2": "count",
        "indicador": "mean",
    }
    for _fc in BOGOTA_CONSUMO_FLAG_COLUMNS:
        agg_map[_fc] = "mean"
    df_cant = df_etiquetado.groupby("contrato", sort=False).agg(agg_map).reset_index()
    ren = {
        "_uso_join": "cant_uso",
        "codconsumo": "cant_codconsumo",
        "lectura1": "cant_anomalia_1lect",
        "lectura2": "cant_anomalia_2lect",
        "indicador": "mean_indicador",
    }
    ren.update({_fc: "mean_" + _fc for _fc in BOGOTA_CONSUMO_FLAG_COLUMNS})
    df_cant = df_cant.rename(columns=ren)

    def _chg_nb(series):
        s = series.dropna()
        if s.empty:
            return np.nan
        return (s != s.shift()).sum() - 1

    camb_spec = {
        "_uso_join": _chg_nb,
        "codconsumo": _chg_nb,
        "lectura1": _chg_nb,
        "lectura2": _chg_nb,
    }
    df_cc = df_etiquetado.groupby("contrato", sort=False).agg(camb_spec).reset_index()
    df_cc = df_cc.rename(
        columns={
            "_uso_join": "cambios_uso",
            "codconsumo": "cambios_codconsumo",
            "lectura1": "cambios_anomalia_1lect",
            "lectura2": "cambios_anomalia_2lect",
        }
    )
    df_cant = df_cant.merge(df_cc, on="contrato", how="left")
    df_cant = df_cant.merge(df_static_vars[vars_consumo], on="contrato", how="left")

    rango_fechas = pd.date_range(start=date_inicial, end=fecha_fraud, freq="MS", inclusive="left")
    pt = df_etiquetado.pivot_table(index=["contrato"], columns=["date"], values="consumo")
    pt = pt.reindex(columns=rango_fechas)
    # Notebook: sólo columnas con mes calendario <= 6 dentro del rango (primer semestre)
    cols_ps = [c for c in pt.columns if hasattr(c, "month") and pd.Timestamp(c).month <= 6]
    pt_sem = pt.reindex(columns=cols_ps).copy()
    ncol = pt_sem.shape[1]
    if ncol > NUM_ANTERIOR_COLS:
        pt_sem = pt_sem.iloc[:, -NUM_ANTERIOR_COLS:]
    elif ncol < NUM_ANTERIOR_COLS:
        n_pad = NUM_ANTERIOR_COLS - ncol
        pad_df = pd.DataFrame(np.nan, index=pt_sem.index, columns=list(range(n_pad)))
        pt_sem = pd.concat([pad_df, pt_sem], axis=1)
    cols_ant = [f"{x}_anterior" for x in range(NUM_ANTERIOR_COLS, 0, -1)]
    pt_sem.columns = cols_ant
    df_wide = pt_sem.reset_index()
    df_wide["date_fizcalizacion"] = fecha_fraud
    df_wide = df_wide.merge(df_cant, on="contrato", how="left")
    df_wide["uso"] = (
        df_wide["uso"].astype(str).str.strip().replace({"": "sin_uso", "nan": "sin_uso"}).fillna("sin_uso")
    )
    if df_consumo_g.empty:
        raise ValueError(
            "create_dataset_wide_for_cutoff: agregados por uso vacíos (df_consumo_g); "
            "revise ventana de consumo y columnas uso/consumo/date."
        )
    df_wide = df_wide.merge(df_consumo_g, on="uso", how="left")

    df_wide["cant_null"] = df_wide[cols_ant].isnull().sum(axis=1)
    eps = 1e-9
    cols_3 = [f"{x}_anterior" for x in range(3, 0, -1)]
    df_wide["prop_cons_ult3_mean_g"] = df_wide[cols_3].mean(axis=1) / (df_wide["consumo_3m_ts_mean"] + eps)
    df_wide["prop_cons_ult3_max_g"] = df_wide[cols_3].mean(axis=1) / (df_wide["consumo_3m_ts_max"] + eps)
    cols_6 = [f"{x}_anterior" for x in range(6, 0, -1)]
    df_wide["prop_cons_ult6_mean_g"] = df_wide[cols_6].mean(axis=1) / (df_wide["consumo_6m_ts_mean"] + eps)
    df_wide["prop_cons_ult6_max_g"] = df_wide[cols_6].mean(axis=1) / (df_wide["consumo_6m_ts_max"] + eps)
    cols_12w = cols_ant
    df_wide["prop_cons_ult12_mean_g"] = df_wide[cols_12w].mean(axis=1) / (df_wide["consumo_12m_ts_mean"] + eps)
    df_wide["prop_cons_ult12_max_g"] = df_wide[cols_12w].mean(axis=1) / (df_wide["consumo_12m_ts_max"] + eps)

    df_wide["num_mes"] = df_wide["date_fizcalizacion"].dt.month
    df_wide["quarter_anio"] = df_wide["date_fizcalizacion"].dt.quarter
    df_wide["semana_anio"] = df_wide["date_fizcalizacion"].dt.isocalendar().week.astype(int)

    if mode == "train":
        df_wide = df_wide.merge(
            df_ordenes[vars_ordenes],
            left_on=["contrato", "date_fizcalizacion"],
            right_on=["contrato", "date"],
            how="left",
        )
        if "date" in df_wide.columns:
            df_wide = df_wide.drop(columns=["date"])
        if "is_fraud" in df_wide.columns:
            df_wide["is_fraud"] = df_wide["is_fraud"].fillna(-1)

    return df_wide


def create_train_dataset(
    interim_dir,
    processed_dir,
    cant_periodos=24,
    cutoff_max=None,
    max_ctas=None,
    fecha_ancla_ordenes=None,
):
    """
    Train alineado al notebook Bogotá: ``cant_periodos`` meses de ventana, lista de cortes opcionalmente desde
    ``fecha_ancla_ordenes + cant_periodos``; pivote primer semestre con 12 columnas ``*_anterior``;
    merge maestro ``VARS_MAESTRO``. Si ``fecha_ancla_ordenes`` es ``None``, los cortes salen de la fecha
    mínima de consumo en interim y ``cant_periodos``.
    """
    df_ordenes = load_interim_data(interim_dir, "inspecciones")
    if df_ordenes.empty:
        logger.warning("No hay datos en interim para inspecciones.")
        return None

    min_date_consumo, _ = get_consumo_date_range(interim_dir)
    if min_date_consumo is None:
        logger.warning("No hay archivos de consumo en interim.")
        return None

    df_maestro = load_maestro_latest(interim_dir)
    if df_maestro.empty:
        logger.error("No hay maestro en interim. El maestro es una fuente obligatoria; ejecute el ETL para maestro.")
        return None

    logger.info("Maestro cargado (%s contratos). Inspecciones cargadas.", len(df_maestro))
    if fecha_ancla_ordenes is None:
        fecha_list = get_fecha_fraud_list(
            df_ordenes,
            df_consumo=None,
            cant_periodos=cant_periodos,
            cutoff_max=cutoff_max,
            min_date_consumo=min_date_consumo,
        )
    else:
        fecha_list = get_fecha_fraud_list(
            df_ordenes,
            df_consumo=None,
            cant_periodos=cant_periodos,
            cutoff_max=cutoff_max,
            fecha_ancla_ordenes=fecha_ancla_ordenes,
            min_date_consumo=None,
        )
    if not fecha_list:
        logger.warning("No hay fechas de corte válidas.")
        return None

    if cutoff_max is None:
        logger.info("Se procesarán todas las inspecciones que estén en la carpeta (sin límite de fecha).")

    list_df = []
    for fecha_fraud in tqdm(fecha_list, desc="Train dataset"):
        fecha_d = pd.to_datetime(fecha_fraud)
        date_inicial = fecha_d - pd.DateOffset(months=cant_periodos)
        df_consumo_ventana = load_interim_data(
            interim_dir, "consumo", start_date=date_inicial, end_date=fecha_d
        )
        if not df_consumo_ventana.empty:
            add_consumo_flags(df_consumo_ventana)

        df_one = create_dataset_wide_for_cutoff(
            fecha_fraud,
            df_consumo_ventana,
            df_ordenes,
            cant_periodos,
            VARS_FOR_ORDENES,
            VARS_FOR_CONSUMO,
            mode="train",
            max_ctas=max_ctas,
        )
        if not df_one.empty:
            list_df.append(df_one)

    if not list_df:
        return None
    df_wide = pd.concat(list_df, axis=0, ignore_index=True)
    df_wide["date_fizcalizacion"] = pd.to_datetime(df_wide["date_fizcalizacion"])

    cols_m = [c for c in MAESTRO_COLUMNS_AFTER_CONCAT if c in df_maestro.columns]
    if cols_m:
        df_wide = df_wide.merge(df_maestro[cols_m], on="contrato", how="left")
    if "fecha_medidor" in df_wide.columns:
        fecha_medidor = pd.to_datetime(df_wide["fecha_medidor"], errors="coerce")
        df_wide["antiguedad_meses"] = (
            (df_wide["date_fizcalizacion"] - fecha_medidor).dt.days // 30
        )


    df_wide.reset_index(drop=True, inplace=True)
    logger.info("Calculando variables de series de tiempo (tsfel); puede tardar varios minutos...")
    df_wide = llenar_val_vacios_ciclo(df_wide, NUM_ANTERIOR_COLS)
    df_wide = compute_change_trend_percentaje_vars(df_wide, CONFIG_CAIDAS)
    df_wide = compute_constant_consumption_vars(df_wide, CONFIG_CONSTANTES)
    df_wide.reset_index(drop=True, inplace=True)
    df_wide["index"] = range(len(df_wide))
    df_wide = compute_tsfel_consumption_vars(df_wide, NUM_ANTERIOR_COLS)

    out_dir = os.path.join(processed_dir, "train", f"cutoff={pd.to_datetime(cutoff_max or fecha_list[-1]).strftime('%Y-%m-%d')}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "train_wide.parquet")
    df_wide.to_parquet(out_file, index=False)
    logger.info("Dataset de train guardado: %s (%s filas).", out_file, len(df_wide))
    return df_wide


def create_inference_dataset(interim_dir, processed_dir, cutoff, cant_periodos=24, contratos_list=None, columns_filter=None):
    """
    Inferencia: un corte, sin etiqueta. Mismo pipeline genérico que train para ``create_dataset_wide_for_cutoff``
    (wide completo en una pasada), mismas reglas que train (interim Bogotá).

    Merge maestro con ``VARS_MAESTRO``; ``columns_filter`` y ``contratos_list`` opcionales antes de tsfel.
    ``cutoff``: ``YYYY-MM-DD``.
    """
    if cutoff is None or (isinstance(cutoff, str) and not cutoff.strip()):
        raise ValueError("inference.cutoff es obligatorio y no puede estar vacío.")
    cutoff_str = str(cutoff).strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", cutoff_str) or len(cutoff_str) != 10:
        raise ValueError(
            "inference.cutoff debe tener exactamente formato YYYY-MM-DD (ej. 2025-06-01). "
            "No se aceptan otros formatos ni cadenas numéricas largas."
        )
    try:
        pd.to_datetime(cutoff_str, format="%Y-%m-%d")
    except Exception:
        raise ValueError(
            "inference.cutoff no es una fecha válida (ej. mes 01-12, día válido para el mes)."
        )

    start_d, end_d = get_date_range_for_cutoff(cutoff, cant_periodos)
    df_consumo = load_interim_data(interim_dir, "consumo", start_date=start_d, end_date=end_d)
    if df_consumo.empty:
        logger.warning("No hay consumo en interim.")
        return None

    meses_cargados = df_consumo["date"].dt.to_period("M").nunique()
    if meses_cargados < cant_periodos:
        logger.warning(
            "Se requieren %s meses de consumo en el rango [%s, %s], pero solo hay %s meses en interim. "
            "Ejecute el ETL para los meses faltantes.",
            cant_periodos,
            start_d.strftime("%Y-%m-%d"),
            end_d.strftime("%Y-%m-%d"),
            meses_cargados,
        )
        return None

    logger.info("Cargados %s meses de consumo para la ventana del cutoff.", meses_cargados)
    df_maestro = load_maestro_latest(interim_dir)
    if df_maestro.empty:
        logger.error("No hay maestro en interim. El maestro es obligatorio para inferencia; ejecute el ETL para maestro.")
        return None
    logger.info("Maestro cargado (%s contratos).", len(df_maestro))

    add_consumo_flags(df_consumo)

    df_wide = create_dataset_wide_for_cutoff(
        cutoff, df_consumo, pd.DataFrame(), cant_periodos,
        VARS_FOR_ORDENES, VARS_FOR_CONSUMO, mode="inference"
    )

    if df_wide.empty:
        return None

    cols_maestro = [c for c in VARS_MAESTRO if c in df_maestro.columns]
    if cols_maestro:
        df_wide = df_wide.merge(df_maestro[cols_maestro], on="contrato", how="left")
    
    if "fecha_medidor" in df_wide.columns:
        fecha_medidor = pd.to_datetime(df_wide["fecha_medidor"], errors="coerce")
        df_wide["antiguedad_meses"] = (
            (df_wide["date_fizcalizacion"] - fecha_medidor).dt.days // 30
        )

    # Filtrar por columnas del dataset (antes de tsfel, que es costoso)
    if columns_filter and isinstance(columns_filter, dict):
        for col, valores in columns_filter.items():
            if col not in df_wide.columns:
                continue
            vals = [str(v).strip() for v in (valores if isinstance(valores, list) else [valores])]
            df_wide = df_wide[df_wide[col].astype(str).str.strip().isin(vals)]
        if df_wide.empty:
            logger.warning("Dataset de inferencia quedó vacío tras aplicar columns_filter.")
            return None
        logger.info("Filtro aplicado: %s contratos tras filtrar por %s.", len(df_wide), list(columns_filter.keys()))
    if contratos_list is not None:
        df_wide = df_wide[df_wide["contrato"].isin(contratos_list)]
        if df_wide.empty:
            logger.warning("Dataset de inferencia quedó vacío tras aplicar contratos_list.")
            return None

    df_wide.reset_index(drop=True, inplace=True)
    logger.info("Calculando variables de series de tiempo (tsfel); puede tardar varios minutos...")
    df_wide = llenar_val_vacios_ciclo(df_wide, NUM_ANTERIOR_COLS)
    df_wide = compute_change_trend_percentaje_vars(df_wide, CONFIG_CAIDAS)
    df_wide = compute_constant_consumption_vars(df_wide, CONFIG_CONSTANTES)
    df_wide.reset_index(drop=True, inplace=True)
    df_wide["index"] = range(len(df_wide))
    df_wide = compute_tsfel_consumption_vars(df_wide, NUM_ANTERIOR_COLS)

    out_dir = os.path.join(processed_dir, "inference", f"cutoff={pd.to_datetime(cutoff).strftime('%Y-%m-%d')}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "inference_wide.parquet")
    df_wide.to_parquet(out_file, index=False)
    logger.info("Dataset de inferencia guardado: %s (%s filas).", out_file, len(df_wide))
    return df_wide
