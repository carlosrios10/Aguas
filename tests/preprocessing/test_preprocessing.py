"""Tests unitarios para src.preprocessing.preprocessing."""

import numpy as np
import pandas as pd

from src.preprocessing.preprocessing import (
    CardinalityReducer,
    MinMaxScalerRow,
    TeEncoder,
    ToDummy,
    preprocess_model_input,
)


def test_preprocess_model_input_filters_and_string_columns():
    df = pd.DataFrame(
        {
            "cant_null": [3, 7, 3],
            "cant_ceros_12": [5, 5, 10],
            "zona": ["10.0", "1", "1"],
            "municipio": [None, "A", "A"],
            "tipo": ["R", "R", "R"],
            "es_digital": [0, 0, 0],
            "desc_categoria": ["c1", "c1", "c1"],
        }
    )
    out = preprocess_model_input(df)
    assert len(out) == 1
    assert out["cant_null"].iloc[0] == 3
    assert out["zona"].iloc[0] == "10"
    assert out["municipio"].iloc[0] == "sin_dato"


def test_preprocess_model_input_inclusive_thresholds():
    """cant_null <= 6 y cant_ceros_12 <= 9 (límites incluidos)."""
    df = pd.DataFrame(
        {
            "cant_null": [6],
            "cant_ceros_12": [9],
            "zona": ["1"],
            "municipio": ["M"],
            "tipo": ["T"],
            "es_digital": [0],
            "desc_categoria": ["d"],
        }
    )
    out = preprocess_model_input(df)
    assert len(out) == 1


def test_to_dummy_aligns_columns_on_unseen_category():
    X_train = pd.DataFrame({"c": ["a", "b"]})
    t = ToDummy(cols=["c"])
    t.fit(X_train)
    X_test = pd.DataFrame({"c": ["a", "b", "a"]})
    out = t.transform(X_test)
    assert list(out.columns) == list(t.dummy_names)
    assert out.shape[0] == 3
    assert out.filter(like="dummy_c").sum().sum() == 3


def test_te_encoder_smoothing():
    X = pd.DataFrame({"cat": ["a", "a", "b", "b"]})
    y = pd.Series([1, 0, 0, 0])
    te = TeEncoder(cols=["cat"], w=2)
    te.fit(X, y)
    out = te.transform(X.copy())
    assert out.columns.tolist() == ["cat"]
    assert out["cat"].notna().all()
    assert te.mean_global == y.mean()


def test_cardinality_reducer_maps_rare_to_otros():
    # frecuencias: a=0.5, b=0.3, c=0.2 con umbral 0.35 -> solo a
    X = pd.DataFrame({"f": list("aaabbc")})
    r = CardinalityReducer(threshold=0.35)
    r.fit(X)
    out = r.transform(pd.DataFrame({"f": ["a", "b", "c", "d"]}))
    assert out["f"].tolist() == ["a", "otros", "otros", "otros"]


def test_minmax_scaler_row_scales_per_row():
    X = pd.DataFrame([[0.0, 10.0], [4.0, 8.0]])
    s = MinMaxScalerRow()
    out = s.fit_transform(X)
    np.testing.assert_allclose(out[0], [0.0, 1.0], rtol=1e-5)
    np.testing.assert_allclose(out[1], [0.0, 1.0], rtol=1e-5)
