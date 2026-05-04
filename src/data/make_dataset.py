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

**Adaptación EMPAGUA**

Los parquets interim deben coincidir con ``src.data.etl`` (contrato, fechas, maestro, consumo). Las listas
``VARS_FOR_*``, ``MAESTRO_COLUMNS_AFTER_CONCAT`` y la lógica de flags son el **contrato de columnas y reglas
EMPAGUA**; el diseño sigue ``notebooks/desarrollo/2_Contruccion_dataset_v1`` como referencia, pero la fuente
de verdad ejecutable es este módulo más el ETL.

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

# --- Contrato de columnas EMPAGUA (salida de etl.py + notebook de referencia) ---
VARS_FOR_ORDENES = [
    "id_inspeccion",
    "contrato",
    "fecha",
    "resultado",
    "tiene_sancion",
    "date",
    "is_fraud",
]
VARS_FOR_CONSUMO = [
    "contrato",
    "id_cuenta",
    "fcm_ciclo",
    "m3_fact_tipo",
    "cod_problema",
    "estatus",
    "tuvo_cm",
    "categoria",
]
# Columnas de maestro para join al wide (incluye categoria si existe en interim)
VARS_MAESTRO = [
    "contrato",
    "municipio",
    "zona",
    "colonia",
    "fecha_medidor",
    "tipo",
    "desc_categoria",
    "es_digital",
    "categoria",
]

# Train tras concat: maestro demográfico sin duplicar ``categoria`` (ya viene del merge consumo–maestro en ETL)
MAESTRO_COLUMNS_AFTER_CONCAT = [
    "contrato",
    "municipio",
    "zona",
    "colonia",
    "fecha_medidor",
    "tipo",
    "desc_categoria",
    "es_digital",
]

CONFIG_CAIDAS = [(1, 4, 90), (1, 3, 90), (2, 3, 90), (1, 6, 90), (3, 3, 90), (6, 5, 10), (6, 6, 10), (6, 4, 10), (6, 1, 10), (5, 5, 10)]
CONFIG_CONSTANTES = [8, 9, 10, 3, 4, 5]


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
    Añade columnas de flags al consumo (notebook 2_Contruccion_dataset_v1 celdas 38–45).
    Requiere cod_problema: grupos de código, fact_flag_estimado (m3_fact_tipo), flag_status_* (estatus).
    Modifica in-place.
    """
    if df.empty:
        return df
    if "cod_problema" not in df.columns:
        return df

    cp = pd.to_numeric(df["cod_problema"], errors="coerce").fillna(-999999).astype(int)
    fraude_codes = [2, 7, 8, 12, 13, 97, 98]
    inacceso_codes = [3, 6, 89, 92]
    servicio_codes = [9, 10, 11]
    lectura_codes = [1, 4, 5] + list(range(75, 97))
    df["flag_fraude"] = np.where(cp.isin(fraude_codes), 1, 0).astype("uint8")
    df["flag_inacceso"] = np.where(cp.isin(inacceso_codes), 1, 0).astype("uint8")
    df["flag_servicio_irregular"] = np.where(cp.isin(servicio_codes), 1, 0).astype("uint8")
    df["flag_lectura_fallida"] = np.where(cp.isin(lectura_codes), 1, 0).astype("uint8")
    df["flag_sin_problema"] = np.where(cp == 0, 1, 0).astype("uint8")
    if "m3_fact_tipo" in df.columns:
        df["fact_flag_estimado"] = df["m3_fact_tipo"].astype(str).str.strip().str.upper().isin(["E"]).astype("uint8")
    if "estatus" in df.columns:
        es = df["estatus"].astype(str).str.strip().str.upper()
        df["flag_status_a"] = np.where(es == "A", 1, 0).astype("uint8")
        df["flag_status_s"] = np.where(es == "S", 1, 0).astype("uint8")
        df["flag_status_b"] = np.where(es == "B", 1, 0).astype("uint8")
    return df


def get_fecha_fraud_list(df_ordenes, df_consumo=None, cant_periodos=12, cutoff_max=None, min_date_consumo=None):
    """
    Fechas de corte válidas: con inspecciones y al menos cant_periodos meses de consumo previo.
    Se puede usar min_date_consumo (timestamp) en lugar de df_consumo para evitar cargar todo el consumo.
    """
    if df_ordenes.empty:
        return []
    if min_date_consumo is not None:
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
    Wide por fecha de corte.

    Train e inference construyen el wide completo en esta función (incluye proporciones y calendario).
    """
    fecha_fraud = pd.to_datetime(fecha_fraud)
    date_inicial = fecha_fraud - pd.DateOffset(months=cant_periodos)

    df_consumo_ventana = df_consumo[
        (df_consumo["date"] < fecha_fraud) & (df_consumo["date"] >= date_inicial)
    ].copy()

    consumo_anual = (
        df_consumo_ventana.groupby(["categoria", "contrato"])["consumo"].sum().groupby(level=0).agg(["mean", "max"])
    )
    consumo_anual.columns = ["consumo_12m_ts_mean", "consumo_12m_ts_max"]
    date_6m = fecha_fraud - pd.DateOffset(months=6)
    consumo_6m = (
        df_consumo_ventana[df_consumo_ventana["date"] >= date_6m]
        .groupby(["categoria", "contrato"])["consumo"].sum().groupby(level=0).agg(["mean", "max"])
    )
    consumo_6m.columns = ["consumo_6m_ts_mean", "consumo_6m_ts_max"]
    date_3m = fecha_fraud - pd.DateOffset(months=3)
    consumo_3m = (
        df_consumo_ventana[df_consumo_ventana["date"] >= date_3m]
        .groupby(["categoria", "contrato"])["consumo"].sum().groupby(level=0).agg(["mean", "max"])
    )
    consumo_3m.columns = ["consumo_3m_ts_mean", "consumo_3m_ts_max"]
    df_consumo_g = pd.concat([consumo_anual, consumo_6m, consumo_3m], axis=1).reset_index()

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

    agg_cols = [c for c in ["m3_fact_tipo", "cod_problema", "estatus"] if c in df_etiquetado.columns]
    df_static_vars = df_etiquetado.loc[df_etiquetado.groupby("contrato")["date"].idxmax()]
    df_cant = df_etiquetado[["contrato"]].drop_duplicates().reset_index(drop=True)
    if agg_cols:
        df_nunique = df_etiquetado.groupby("contrato")[agg_cols].nunique().reset_index()
        df_nunique = df_nunique.rename(columns={
            "m3_fact_tipo": "cant_m3_fact_tipo",
            "cod_problema": "cant_cod_problema",
            "estatus": "cant_estatus",
        })
        df_cant = df_cant.merge(df_nunique, on="contrato", how="left")
    if "tuvo_cm" in df_etiquetado.columns:
        tuvo_cm = (
            df_etiquetado.assign(_tuvo_cm=df_etiquetado["tuvo_cm"].astype(str).str.strip().str.upper())
            .groupby("contrato")["_tuvo_cm"]
            .apply(lambda x: (x == "S").mean())
            .reset_index(name="mean_tuvo_cm")
        )
        df_cant = df_cant.merge(tuvo_cm, on="contrato", how="left")

    flag_cols = [
        c for c in df_etiquetado.columns
        if c.startswith(("fact_flag_", "flag_"))
    ]
    if flag_cols:
        df_flag_means = df_etiquetado.groupby("contrato")[flag_cols].mean().reset_index()
        df_flag_means = df_flag_means.rename(columns={c: "mean_" + c for c in flag_cols})
        df_cant = df_cant.merge(df_flag_means, on="contrato")

    def _cambios(s):
        vals = s.dropna()
        if vals.empty:
            return 0
        return max((vals != vals.shift()).sum() - 1, 0)

    if agg_cols:
        df_cambios = (
            df_etiquetado.groupby("contrato")[agg_cols]
            .apply(lambda g: g.apply(_cambios))
            .reset_index()
        )
        df_cambios = df_cambios.rename(columns={
            "m3_fact_tipo": "cambios_m3_fact_tipo",
            "cod_problema": "cambios_cod_problema",
            "estatus": "cambios_estatus",
        })
        df_cant = df_cant.merge(df_cambios, on="contrato", how="left")
    vars_consumo_exist = [c for c in vars_consumo if c in df_static_vars.columns]
    if vars_consumo_exist:
        df_cant = df_cant.merge(df_static_vars[vars_consumo_exist], on="contrato")

    rango_fechas = pd.date_range(start=date_inicial, end=fecha_fraud, freq="MS", inclusive="left")
    cols_ant = [str(x) + "_anterior" for x in range(cant_periodos, 0, -1)]
    df_wide = df_etiquetado.pivot_table(index=["contrato"], columns=["date"], values="consumo")
    df_wide = df_wide.reindex(columns=rango_fechas)
    df_wide.columns = cols_ant
    df_wide["date_fizcalizacion"] = fecha_fraud
    df_wide = df_wide.reset_index().merge(df_cant, on="contrato", how="left")
    df_wide = df_wide.merge(df_consumo_g, on="categoria", how="left")

    df_wide["cant_null"] = df_wide[cols_ant].isnull().sum(axis=1)
    eps = 1e-9
    cols_3 = [str(x) + "_anterior" for x in range(3, 0, -1)]
    df_wide["prop_cons_ult3_mean_g"] = df_wide[cols_3].mean(axis=1) / (df_wide["consumo_3m_ts_mean"] + eps)
    df_wide["prop_cons_ult3_max_g"] = df_wide[cols_3].mean(axis=1) / (df_wide["consumo_3m_ts_max"] + eps)
    cols_6 = [str(x) + "_anterior" for x in range(6, 0, -1)]
    df_wide["prop_cons_ult6_mean_g"] = df_wide[cols_6].mean(axis=1) / (df_wide["consumo_6m_ts_mean"] + eps)
    df_wide["prop_cons_ult6_max_g"] = df_wide[cols_6].mean(axis=1) / (df_wide["consumo_6m_ts_max"] + eps)
    cols_12 = [str(x) + "_anterior" for x in range(cant_periodos, 0, -1)]
    df_wide["prop_cons_ult12_mean_g"] = df_wide[cols_12].mean(axis=1) / (df_wide["consumo_12m_ts_mean"] + eps)
    df_wide["prop_cons_ult12_max_g"] = df_wide[cols_12].mean(axis=1) / (df_wide["consumo_12m_ts_max"] + eps)

    df_wide["num_mes"] = df_wide["date_fizcalizacion"].dt.month
    df_wide["quarter_anio"] = df_wide["date_fizcalizacion"].dt.quarter
    df_wide["semana_anio"] = df_wide["date_fizcalizacion"].dt.isocalendar().week.astype(int)


    if mode == "train":
        cols_ordenes = [c for c in vars_ordenes if c in df_ordenes.columns]
        df_wide = df_wide.merge(
            df_ordenes[cols_ordenes],
            left_on=["contrato", "date_fizcalizacion"],
            right_on=["contrato", "date"],
            how="left",
        )
        if "date" in df_wide.columns:
            df_wide = df_wide.drop(columns=["date"])
        df_wide["is_fraud"] = df_wide["is_fraud"].fillna(0)

    return df_wide


def create_train_dataset(interim_dir, processed_dir, cant_periodos=12, cutoff_max=None, max_ctas=None):
    """
    Punto de entrada train: respeta el orden de funciones del módulo (véase docstring del paquete).
    Columnas y flags son EMPAGUA (``VARS_*``, ``add_consumo_flags``, interim producido por ``etl.py``).

    Pasos equivalentes a notebooks/desarrollo/2_Contruccion_dataset_v1 — «Construccion data set»:

    1. Lista de fechas de corte; por cada una, chunk wide (consumo + pivot + agregados por categoría).
    2. ``pd.concat`` de chunks.
    3. Merge inspecciones, ``cant_null``, merge maestro (columnas demográficas), proporciones ``prop_*``,
       variables de calendario, ``antiguedad_meses``, ``is_fraud``.fillna(0).
    4. tsfel / tendencias / constantes y guardado parquet.

    max_ctas: tope de contratos sin inspección en el mes (negativos); None = no muestrear negativos.
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
    if "categoria" not in df_maestro.columns:
        logger.error("Maestro sin columna 'categoria'. La fuente maestro debe incluir contrato y categoria.")
        return None

    logger.info("Maestro cargado (%s contratos). Inspecciones cargadas.", len(df_maestro))
    fecha_list = get_fecha_fraud_list(
        df_ordenes, df_consumo=None, cant_periodos=cant_periodos,
        cutoff_max=cutoff_max, min_date_consumo=min_date_consumo
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
            df_consumo_ventana = df_consumo_ventana.merge(
                df_maestro[["contrato", "categoria"]], on="contrato", how="left"
            )
            df_consumo_ventana["categoria"] = df_consumo_ventana["categoria"].fillna("sin_dato")
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
    df_wide = llenar_val_vacios_ciclo(df_wide, cant_periodos)
    df_wide = compute_change_trend_percentaje_vars(df_wide, CONFIG_CAIDAS)
    df_wide = compute_constant_consumption_vars(df_wide, CONFIG_CONSTANTES)
    df_wide.reset_index(drop=True, inplace=True)
    df_wide["index"] = range(len(df_wide))
    df_wide = compute_tsfel_consumption_vars(df_wide, cant_periodos)

    out_dir = os.path.join(processed_dir, "train", f"cutoff={pd.to_datetime(cutoff_max or fecha_list[-1]).strftime('%Y-%m-%d')}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "train_wide.parquet")
    df_wide.to_parquet(out_file, index=False)
    logger.info("Dataset de train guardado: %s (%s filas).", out_file, len(df_wide))
    return df_wide


def create_inference_dataset(interim_dir, processed_dir, cutoff, cant_periodos=12, contratos_list=None, columns_filter=None):
    """
    Inferencia: un corte, sin etiqueta. Mismo pipeline genérico que train para ``create_dataset_wide_for_cutoff``
    (wide completo en una pasada), datos EMPAGUA desde interim (véase docstring del módulo).

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
            "Se requieren %s meses de consumo en el rango [%s, %s], pero solo hay %s meses en interim. Ejecute el ETL para los meses faltantes.",
            cant_periodos, start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d"), meses_cargados,
        )
        return None

    logger.info("Cargados %s meses de consumo para la ventana del cutoff.", meses_cargados)
    df_maestro = load_maestro_latest(interim_dir)
    if df_maestro.empty:
        logger.error("No hay maestro en interim. El maestro es obligatorio para inferencia; ejecute el ETL para maestro.")
        return None
    if "categoria" not in df_maestro.columns:
        logger.error("Maestro sin columna 'categoria'. La fuente maestro debe incluir contrato y categoria.")
        return None
    logger.info("Maestro cargado (%s contratos).", len(df_maestro))

    df_consumo = df_consumo.merge(
        df_maestro[["contrato", "categoria"]], on="contrato", how="left"
    )
    df_consumo["categoria"] = df_consumo["categoria"].fillna("sin_dato")
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
    df_wide = llenar_val_vacios_ciclo(df_wide, cant_periodos)
    df_wide = compute_change_trend_percentaje_vars(df_wide, CONFIG_CAIDAS)
    df_wide = compute_constant_consumption_vars(df_wide, CONFIG_CONSTANTES)
    df_wide.reset_index(drop=True, inplace=True)
    df_wide["index"] = range(len(df_wide))
    df_wide = compute_tsfel_consumption_vars(df_wide, cant_periodos)

    out_dir = os.path.join(processed_dir, "inference", f"cutoff={pd.to_datetime(cutoff).strftime('%Y-%m-%d')}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "inference_wide.parquet")
    df_wide.to_parquet(out_file, index=False)
    logger.info("Dataset de inferencia guardado: %s (%s filas).", out_file, len(df_wide))
    return df_wide
