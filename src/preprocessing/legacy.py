"""
Código legacy de preprocessing: TsfelVars, ExtraVars, llenar_val_* y pipeline de feature engineering.
No usado por el flujo poc; la versión activa de TsfelVars/ExtraVars/llenar_val_vacios_ciclo está en src.data.make_dataset.
Mantenido por si notebooks de desarrollo lo referencian.
"""
from itertools import groupby
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import tsfel


class TsfelVars(BaseEstimator, TransformerMixin):
    """Extrae características de series de tiempo con tsfel (versión legacy)."""

    def __init__(self, features_names_path=None, num_periodos=12):
        self.num_periodos = num_periodos
        self.features_names_path = features_names_path

    def obtener_cols_anterior(self, num_cols=12):
        return [f'{i}_anterior' for i in range(num_cols, 0, -1)]

    def extra_cols(self, df, domain, cols, window=12):
        cfg = tsfel.get_features_by_domain(domain)
        df_result = tsfel.time_series_features_extractor(cfg, df[cols].values, n_jobs=-1)
        df_result['index'] = df.index
        return df_result

    def compute_by_json(self, df, cols, window=12):
        cfg = tsfel.get_features_by_domain(json_path=self.features_names_path)
        df_result = tsfel.time_series_features_extractor(cfg, df[cols].values, n_jobs=-1)
        df_result['index'] = df.index
        return df_result

    def crear_all_tsfel(self, df):
        cols_anterior = self.obtener_cols_anterior(self.num_periodos)
        df_result_stat = self.extra_cols(df, "statistical", cols_anterior, window=self.num_periodos)
        df_result_temporal = self.extra_cols(df, "temporal", cols_anterior, window=self.num_periodos)
        self.temp_vars = df_result_temporal.columns.tolist()
        self.temp_vars.remove('index')
        self.stat_vars = df_result_stat.columns.tolist()
        self.stat_vars.remove('index')
        return df_result_stat, df_result_temporal

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.features_names_path is not None:
            cols_anterior = self.obtener_cols_anterior(self.num_periodos)
            df_tsfel = self.compute_by_json(X, cols_anterior, window=self.num_periodos)
            X = X.merge(df_tsfel, on='index', how='left')
        else:
            df_result_stat, df_result_temporal = self.crear_all_tsfel(X)
            df_tsfel = pd.merge(df_result_stat, df_result_temporal, how='inner', on='index')
            X = X.merge(df_tsfel, on='index', how='left')
        return X


class ExtraVars(BaseEstimator, TransformerMixin):
    """Genera variables adicionales de series de tiempo (versión legacy)."""

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


def llenar_val_vacios_ciclo(df, cant_ciclos_validos):
    """Rellena vacíos en columnas de consumo con ffill/bfill (versión legacy)."""
    cols_consumo = [f'{i}_anterior' for i in range(cant_ciclos_validos, 0, -1)]
    df.loc[:, cols_consumo] = df.loc[:, cols_consumo].ffill(axis=1).bfill(axis=1)
    return df


def llenar_val_vacios_str(df, cols, str_value):
    for x in cols:
        df.loc[:, x] = df[x].fillna(str_value)
    return df


def llenar_val_vacios_numeric(df, cols, numeric_value):
    for x in cols:
        df.loc[:, x] = df[x].fillna(numeric_value)
    return df


def build_feature_engeniering_pipeline(f_names_path, num_periodos):
    """Pipeline legacy de feature engineering (firma antigua; usar src.data.make_dataset en su lugar)."""
    pipe_feature_eng_train = Pipeline([
        ("tsfel vars", TsfelVars(features_names_path=f_names_path, num_periodos=num_periodos)),
        ("add vars3", ExtraVars(num_periodos=3)),
        ("add vars6", ExtraVars(num_periodos=6)),
        ("add vars12", ExtraVars(num_periodos=12)),
    ])
    return pipe_feature_eng_train
