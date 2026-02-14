"""
Construcción de dataset wide y features de consumo.
Carga desde interim, construcción wide por fecha de corte, ingeniería de variables.
Usado por poc/train, poc/inference y notebooks de desarrollo.
"""
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

VARS_FOR_ORDENES = [
    "contrato", "date", "is_fraud", "solicitud", "instalacion", "resultado",
    "clasificacion_resultado", "causa_de_revision",
]
VARS_FOR_CONSUMO = [
    "contrato", "instalacion", "estado", "categoria", "subcategoria_estrato",
    "localidad", "municipio", "barrio", "microsector",
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


def create_dataset_wide_for_cutoff(fecha_fraud, df_consumo, df_ordenes, cant_periodos, vars_ordenes, vars_consumo, mode="train", max_ctas=None):
    """
    Construye el dataset en formato wide para una fecha de corte.
    mode: 'train' -> solo contratos inspeccionados ese mes, se une target.
          'inference' -> contratos con consumo en la ventana, sin target.
    max_ctas: en mode='train', cantidad máxima de cuentas sin orden a añadir por cutoff (casos negativos).
              Si None, no se añaden. Esas filas tendrán is_fraud=0 y el resto de columnas de órdenes en null.
    Las proporciones (prop_cons_ult*) se calculan con estadísticas de toda la población
    en la ventana [date_inicial, fecha_fraud), no solo con las cuentas con órdenes.
    """
    fecha_fraud = pd.to_datetime(fecha_fraud)
    date_inicial = fecha_fraud - pd.DateOffset(months=cant_periodos)

    # Ventana de consumo: toda la población (para estadísticas globales por categoría)
    df_consumo_ventana = df_consumo[
        (df_consumo["date"] < fecha_fraud) & (df_consumo["date"] >= date_inicial)
    ].copy()

    # Consumo por grupo (categoria): 12m, 6m, 3m sobre toda la población de la ventana
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

    # Cortar por cuentas con órdenes (train) y opcionalmente añadir negativos (max_ctas)
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

    agg_cols = ["categoria", "estado", "determinacion_consumo"]
    df_static_vars = df_etiquetado.loc[df_etiquetado.groupby("contrato")["date"].idxmax()]

    df_cant = (
        df_etiquetado.groupby("contrato")[agg_cols].nunique().reset_index()
    )
    df_cant = df_cant.rename(columns={
        "categoria": "cant_categoria",
        "estado": "cant_estado",
        "determinacion_consumo": "cant_determinacion_consumo",
    })

    def _cambios(s):
        return (s.dropna() != s.dropna().shift()).sum() - 1

    df_cambios = (
        df_etiquetado.groupby("contrato")[agg_cols]
        .apply(lambda g: g.apply(_cambios))
        .reset_index()
    )
    df_cambios = df_cambios.rename(columns={
        "categoria": "cambios_categoria",
        "estado": "cambios_estado",
        "determinacion_consumo": "cambios_determinacion_consumo",
    })
    df_cant = df_cant.merge(df_cambios, on="contrato")
    df_cant = df_cant.merge(df_static_vars[vars_consumo], on="contrato")

    rango_fechas = pd.date_range(start=date_inicial, end=fecha_fraud, freq="MS", inclusive="left")
    cols_ant = [str(x) + "_anterior" for x in range(cant_periodos, 0, -1)]
    df_wide = df_etiquetado.pivot_table(index=["contrato"], columns=["date"], values="consumo")
    df_wide = df_wide.reindex(columns=rango_fechas)
    df_wide.columns = cols_ant
    df_wide["date_fizcalizacion"] = fecha_fraud
    df_wide = df_wide.reset_index().merge(df_cant, on="contrato", how="left")
    df_wide = df_wide.merge(df_consumo_g, on="categoria", how="left")

    df_wide["cant_null"] = df_wide[cols_ant].isnull().sum(axis=1)

    # Proporciones (después del corte por date_inicial): uso de estadísticas de toda la población
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
        df_wide = df_wide.merge(
            df_ordenes[vars_ordenes],
            left_on=["contrato", "date_fizcalizacion"],
            right_on=["contrato", "date"],
            how="left",
        )
        df_wide["is_fraud"] = df_wide["is_fraud"].fillna(0)

    return df_wide


def create_train_dataset(interim_dir, processed_dir, cant_periodos=12, cutoff_max=None, max_ctas=None):
    """
    Carga inspecciones desde interim y, por cada fecha de corte, solo la ventana de consumo necesaria
    (cant_periodos meses), construye dataset wide, aplica ingeniería de variables y guarda en
    processed/train/cutoff=<cutoff_max>/.
    max_ctas: cantidad máxima de cuentas sin orden a añadir por cutoff (casos negativos). Si None, no se añaden.
    """
    df_ordenes = load_interim_data(interim_dir, "inspecciones")
    if df_ordenes.empty:
        print("[WARN] No hay datos en interim para inspecciones.")
        return None

    min_date_consumo, _ = get_consumo_date_range(interim_dir)
    if min_date_consumo is None:
        print("[WARN] No hay archivos de consumo en interim.")
        return None

    fecha_list = get_fecha_fraud_list(
        df_ordenes, df_consumo=None, cant_periodos=cant_periodos,
        cutoff_max=cutoff_max, min_date_consumo=min_date_consumo
    )
    if not fecha_list:
        print("[WARN] No hay fechas de corte válidas.")
        return None

    if cutoff_max is None:
        print("[INFO] Se procesarán todas las inspecciones que estén en la carpeta (sin límite de fecha).")

    list_df = []
    for fecha_fraud in tqdm(fecha_list, desc="Train dataset"):
        fecha_d = pd.to_datetime(fecha_fraud)
        date_inicial = fecha_d - pd.DateOffset(months=cant_periodos)
        df_consumo_ventana = load_interim_data(
            interim_dir, "consumo", start_date=date_inicial, end_date=fecha_d
        )
        df_one = create_dataset_wide_for_cutoff(
            fecha_fraud, df_consumo_ventana, df_ordenes, cant_periodos,
            VARS_FOR_ORDENES, VARS_FOR_CONSUMO, mode="train", max_ctas=max_ctas
        )
        if not df_one.empty:
            list_df.append(df_one)

    if not list_df:
        return None
    df_wide = pd.concat(list_df, axis=0, ignore_index=True)
    df_wide["date_fizcalizacion"] = pd.to_datetime(df_wide["date_fizcalizacion"])

    df_wide.reset_index(drop=True, inplace=True)
    df_wide = llenar_val_vacios_ciclo(df_wide, cant_periodos)
    df_wide = compute_change_trend_percentaje_vars(df_wide, CONFIG_CAIDAS)
    df_wide = compute_constant_consumption_vars(df_wide, CONFIG_CONSTANTES)
    df_wide.reset_index(drop=True, inplace=True)
    df_wide["index"] = range(len(df_wide))
    df_wide = compute_tsfel_consumption_vars(df_wide, cant_periodos)
    df_wide.solicitud = df_wide.solicitud.astype(str)
    
    out_dir = os.path.join(processed_dir, "train", f"cutoff={pd.to_datetime(cutoff_max or fecha_list[-1]).strftime('%Y-%m-%d')}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "train_wide.parquet")
    df_wide.to_parquet(out_file, index=False)
    print(f"[OK] Guardado: {out_file} ({len(df_wide)} filas)")
    return df_wide


def create_inference_dataset(interim_dir, processed_dir, cutoff, cant_periodos=12, contratos_list=None):
    """
    Construye dataset wide para una fecha de corte sin target (para scoring).
    Solo carga meses en [cutoff - cant_periodos, cutoff] desde interim.
    cutoff debe ser no nulo y tener formato de fecha válido (ej. YYYY-MM-DD).
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
        print("[WARN] No hay consumo en interim.")
        return None

    meses_cargados = df_consumo["date"].dt.to_period("M").nunique()
    if meses_cargados < cant_periodos:
        print(
            f"[WARN] Se requieren {cant_periodos} meses de consumo en el rango "
            f"[{start_d.strftime('%Y-%m-%d')}, {end_d.strftime('%Y-%m-%d')}], "
            f"pero solo hay {meses_cargados} meses en interim. Ejecute el ETL para los meses faltantes."
        )
        return None

    df_wide = create_dataset_wide_for_cutoff(
        cutoff, df_consumo, pd.DataFrame(), cant_periodos,
        VARS_FOR_ORDENES, VARS_FOR_CONSUMO, mode="inference"
    )

    if df_wide.empty:
        return None

    df_wide.reset_index(drop=True, inplace=True)
    df_wide = llenar_val_vacios_ciclo(df_wide, cant_periodos)
    df_wide = compute_change_trend_percentaje_vars(df_wide, CONFIG_CAIDAS)
    df_wide = compute_constant_consumption_vars(df_wide, CONFIG_CONSTANTES)
    df_wide.reset_index(drop=True, inplace=True)
    df_wide["index"] = range(len(df_wide))
    df_wide = compute_tsfel_consumption_vars(df_wide, cant_periodos)

    if contratos_list is not None:
        df_wide = df_wide[df_wide["contrato"].isin(contratos_list)]

    out_dir = os.path.join(processed_dir, "inference", f"cutoff={pd.to_datetime(cutoff).strftime('%Y-%m-%d')}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "inference_wide.parquet")
    df_wide.to_parquet(out_file, index=False)
    print(f"[OK] Guardado: {out_file} ({len(df_wide)} filas)")
    return df_wide
