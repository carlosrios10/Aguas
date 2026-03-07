"""
Genera una muestra de datos reales para capacitación y la guarda en
notebooks/capacitacion/datos_sinteticos/.

Punto de partida: inspecciones. Se toman 3 meses consecutivos de inspecciones,
se eligen n_inspecciones (contratos distintos) con ratio_positivos (ej. 20%) de
casos positivos. Consumo: desde (fecha mínima de inspecciones - 12 meses) hasta
(fecha máxima de inspecciones). Maestro: solo esos contratos.
Los contratos se anonimizan como c0001, c0002, ... por defecto.

Uso (desde la raíz del proyecto):
  python scripts/run_muestra_capacitacion.py
  python scripts/run_muestra_capacitacion.py --n-inspecciones 1500 --ratio-positivos 0.25
  python scripts/run_muestra_capacitacion.py --no-anonimizar
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# Raíz del proyecto para imports y rutas de salida
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config, get_paths
from src.data.make_dataset import load_interim_data, load_maestro_latest

# Parámetros por defecto (muestra para capacitación)
DEFAULT_N_INSPECCIONES = 1000   # inspecciones = contratos distintos
DEFAULT_MESES_VENTANA = 3       # 3 meses consecutivos de inspecciones
DEFAULT_MESES_CONSUMO_ATRAS = 12  # consumo: 12 meses hacia atrás desde fecha mínima inspección
DEFAULT_RATIO_POSITIVOS = 0.20  # 20% de las inspecciones con resultado positivo (fraude)
OUTPUT_SUBDIR = "notebooks/capacitacion/datos_sinteticos"


def main():
    parser = argparse.ArgumentParser(description="Genera muestra real para capacitación (anclada en inspecciones).")
    parser.add_argument("--n-inspecciones", type=int, default=DEFAULT_N_INSPECCIONES, help=f"Inspecciones (contratos distintos) a tomar (default: {DEFAULT_N_INSPECCIONES})")
    parser.add_argument("--meses-ventana", type=int, default=DEFAULT_MESES_VENTANA, help=f"Meses consecutivos de inspecciones (default: {DEFAULT_MESES_VENTANA})")
    parser.add_argument("--meses-consumo-atras", type=int, default=DEFAULT_MESES_CONSUMO_ATRAS, help=f"Meses de consumo hacia atrás desde fecha mínima inspección (default: {DEFAULT_MESES_CONSUMO_ATRAS})")
    parser.add_argument("--ratio-positivos", type=float, default=DEFAULT_RATIO_POSITIVOS, help=f"Proporción de inspecciones positivas/fraude (default: {DEFAULT_RATIO_POSITIVOS})")
    parser.add_argument("--config", default="config.yaml", help="Archivo de config en config/")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("--no-anonimizar", action="store_true", help="No anonimizar contratos (mantener IDs reales)")
    parser.add_argument("--factor-consumo-negativos", type=float, default=1.0,
                        help="Factor por el que se multiplica el consumo de contratos sin fraude (didáctico: ej. 1.15 para que no fraude consuma más que fraude). Default: 1.0 = sin cambio.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_paths(cfg)
    interim_dir = paths["interim"]
    out_dir = os.path.join(PROJECT_ROOT, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Cargar inspecciones (punto de partida)
    df_inspecciones_full = load_interim_data(interim_dir, "inspecciones")
    if df_inspecciones_full.empty or "date" not in df_inspecciones_full.columns:
        print("[ERROR] No hay inspecciones en interim. Ejecute el ETL (inspecciones) primero.", file=sys.stderr)
        sys.exit(1)
    df_inspecciones_full["date"] = pd.to_datetime(df_inspecciones_full["date"])

    # 2) Tres meses consecutivos: tomar los 3 meses más recientes con datos
    fechas_unicas = df_inspecciones_full["date"].dt.to_period("M").unique()
    if len(fechas_unicas) < args.meses_ventana:
        print("[ERROR] No hay suficientes meses de inspecciones. Se requieren al menos", args.meses_ventana, file=sys.stderr)
        sys.exit(1)
    fechas_ordenadas = sorted(fechas_unicas)
    ventana_periodos = fechas_ordenadas[-args.meses_ventana:]
    ventana_start = ventana_periodos[0].to_timestamp()
    ventana_end = ventana_periodos[-1].to_timestamp() + pd.offsets.MonthEnd(0)

    df_ventana = df_inspecciones_full[
        (df_inspecciones_full["date"] >= ventana_start) &
        (df_inspecciones_full["date"] <= ventana_end)
    ].copy()
    # Una fila por (contrato, date): quedarse con is_fraud máximo (positivo prevalece)
    df_ventana = df_ventana.loc[df_ventana.groupby(["contrato", "date"])["is_fraud"].idxmax()].reset_index(drop=True)
    # Una fila por contrato en la ventana: la de mayor is_fraud (si tuvo positivo, esa)
    df_uno_por_contrato = df_ventana.loc[df_ventana.groupby("contrato")["is_fraud"].idxmax()].reset_index(drop=True)

    pool_positivos = df_uno_por_contrato[df_uno_por_contrato["is_fraud"] == 1]["contrato"].tolist()
    pool_negativos = df_uno_por_contrato[df_uno_por_contrato["is_fraud"] == 0]["contrato"].tolist()
    rng = np.random.default_rng(args.seed)
    n_pos = max(0, round(args.n_inspecciones * args.ratio_positivos))
    n_pos = min(n_pos, len(pool_positivos))
    n_neg = min(args.n_inspecciones - n_pos, len(pool_negativos))
    sel_pos = rng.choice(pool_positivos, size=n_pos, replace=False).tolist() if n_pos > 0 and pool_positivos else []
    sel_neg = rng.choice(pool_negativos, size=n_neg, replace=False).tolist() if n_neg > 0 and pool_negativos else []
    set_contratos = set(sel_pos) | set(sel_neg)
    if not set_contratos:
        print("[ERROR] No se pudo seleccionar inspecciones.", file=sys.stderr)
        sys.exit(1)

    # Mapeo anonimización: contrato real -> c0001, c0002, ...
    contratos_ordenados = sorted(set_contratos)
    anon_map = {c: f"c{i:04d}" for i, c in enumerate(contratos_ordenados, 1)}

    # Una fila de inspección por contrato seleccionado (la que está en la ventana)
    df_insp_ventana = df_ventana[df_ventana["contrato"].isin(set_contratos)].copy()
    df_insp_ventana = df_insp_ventana.loc[df_insp_ventana.groupby("contrato")["is_fraud"].idxmax()].reset_index(drop=True)
    df_insp_out = df_insp_ventana[["contrato", "date", "is_fraud"]].copy()
    df_insp_out["fecha_inspeccion"] = pd.to_datetime(df_insp_out["date"]).dt.strftime("%Y-%m-%d")
    df_insp_out["resultado"] = df_insp_out["is_fraud"].astype(int)
    df_insp_out = df_insp_out[["contrato", "fecha_inspeccion", "resultado"]]

    fecha_min_inspeccion = df_insp_ventana["date"].min()
    fecha_max_inspeccion = df_insp_ventana["date"].max()
    consumo_desde = (pd.to_datetime(fecha_min_inspeccion) - pd.DateOffset(months=args.meses_consumo_atras)).replace(day=1)
    consumo_hasta = pd.to_datetime(fecha_max_inspeccion).replace(day=1)

    # 3) Maestro (para merge con consumo)
    df_maestro = load_maestro_latest(interim_dir)
    if df_maestro.empty:
        print("[ERROR] No hay maestro en interim. Ejecute el ETL (maestro) primero.", file=sys.stderr)
        sys.exit(1)

    # 4) Consumo: desde (fecha_min_inspeccion - 12 meses) hasta fecha_max_inspeccion, solo esos contratos
    df_consumo = load_interim_data(interim_dir, "consumo", start_date=consumo_desde, end_date=consumo_hasta)
    if df_consumo.empty:
        print("[ERROR] No hay consumo en el rango [fecha_min_inspeccion - 12m, fecha_max_inspeccion].", file=sys.stderr)
        sys.exit(1)
    df_consumo_sel = df_consumo[df_consumo["contrato"].isin(set_contratos)].copy()
    cols_maestro = ["contrato", "categoria"]
    if "medidor" in df_maestro.columns:
        cols_maestro.append("medidor")
    if "localidad" in df_maestro.columns:
        cols_maestro.append("localidad")
    df_m = df_maestro[cols_maestro].drop_duplicates(subset=["contrato"], keep="last")
    df_consumo_sel = df_consumo_sel.merge(df_m, on="contrato", how="left")
    df_consumo_sel["categoria"] = df_consumo_sel["categoria"].fillna("sin_dato").astype(str)
    df_consumo_sel["tipo_medidor"] = df_consumo_sel["medidor"].fillna("sin_dato").astype(str) if "medidor" in df_consumo_sel.columns else "sin_dato"
    df_consumo_sel["zona"] = df_consumo_sel["localidad"].fillna("sin_dato").astype(str) if "localidad" in df_consumo_sel.columns else "sin_dato"
    df_consumo_sel["fecha"] = pd.to_datetime(df_consumo_sel["date"]).dt.strftime("%Y-%m-%d")
    # Opcional (didáctico): aumentar consumo de negativos para que quede leve mayor que positivos
    if args.factor_consumo_negativos != 1.0:
        mask_neg = df_consumo_sel["contrato"].isin(sel_neg)
        df_consumo_sel = df_consumo_sel.copy()
        df_consumo_sel.loc[mask_neg, "consumo"] = df_consumo_sel.loc[mask_neg, "consumo"] * args.factor_consumo_negativos
    df_consumo_out = df_consumo_sel[["contrato", "fecha", "consumo", "categoria", "tipo_medidor", "zona"]]

    # Maestro muestra
    df_maestro_sel = df_maestro[df_maestro["contrato"].isin(set_contratos)].drop_duplicates(subset=["contrato"], keep="last").copy()
    df_maestro_sel["categoria"] = df_maestro_sel["categoria"].fillna("sin_dato").astype(str)
    df_maestro_sel["tipo_medidor"] = df_maestro_sel["medidor"].fillna("sin_dato").astype(str) if "medidor" in df_maestro_sel.columns else "sin_dato"
    df_maestro_sel["zona"] = df_maestro_sel["localidad"].fillna("sin_dato").astype(str) if "localidad" in df_maestro_sel.columns else "sin_dato"
    df_maestro_out = df_maestro_sel[["contrato", "categoria", "tipo_medidor", "zona"]]

    # Aplicar anonimización si está activa
    if not args.no_anonimizar:
        df_insp_out = df_insp_out.copy()
        df_insp_out["contrato"] = df_insp_out["contrato"].map(anon_map)
        df_consumo_out = df_consumo_out.copy()
        df_consumo_out["contrato"] = df_consumo_out["contrato"].map(anon_map)
        df_maestro_out = df_maestro_out.copy()
        df_maestro_out["contrato"] = df_maestro_out["contrato"].map(anon_map)

    # Guardar
    path_ins = os.path.join(out_dir, "inspecciones_muestra.csv")
    path_cons = os.path.join(out_dir, "consumo_muestra.csv")
    path_mast = os.path.join(out_dir, "maestro_muestra.csv")
    df_insp_out.to_csv(path_ins, index=False)
    df_consumo_out.to_csv(path_cons, index=False)
    df_maestro_out.to_csv(path_mast, index=False)

    n_positivos_real = (df_insp_out["resultado"] == 1).sum()
    print("Muestra generada en:", out_dir)
    print("  - Ventana inspecciones:   ", ventana_start.strftime("%Y-%m"), "a", ventana_end.strftime("%Y-%m"), f"({args.meses_ventana} meses)")
    print("  - inspecciones_muestra.csv:", len(df_insp_out), "filas (contratos distintos)")
    print("  - consumo_muestra.csv:     ", len(df_consumo_out), "filas")
    print("  - maestro_muestra.csv:    ", len(df_maestro_out), "filas")
    print("  - Casos positivos:        ", n_positivos_real, f"({100*n_positivos_real/len(df_insp_out):.1f}% de inspecciones)")
    if args.factor_consumo_negativos != 1.0:
        print("  - Consumo negativos escalado por factor:", args.factor_consumo_negativos, "(didáctico)")
    if not args.no_anonimizar:
        print("  Contratos anonimizados: c0001, c0002, ...")


if __name__ == "__main__":
    main()
