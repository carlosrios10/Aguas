# Querétaro POC

POC de pipeline de ML para detección de anomalías/consumo (Querétaro): ETL, construcción de dataset wide, entrenamiento con LightGBM (con calibración isotónica), e inferencia.

**Repositorio:** [https://github.com/carlosrios10/Aguas](https://github.com/carlosrios10/Aguas)

## Descripción

El proyecto procesa datos de inspecciones y consumo, construye un dataset en formato wide con features de series de tiempo (tsfel, tendencias, consumo constante, etc.), entrena un modelo LGBM, lo calibra (isotonic, cv=5) y permite ejecutar inferencia con el modelo calibrado. El flujo puede ejecutarse con **notebooks** en `poc/` o con **scripts** en `scripts/`; la configuración central está en **config/config.yaml**.

- **Manual de usuario:** [docs/manual_usuario.md](docs/manual_usuario.md)  
- **Configuración del entorno:** [docs/setup.md](docs/setup.md)

## Estructura del proyecto

```
proyecto/
├── config/
│   └── config.yaml             # Configuración central (paths, inference, train, etl, log_level)
├── poc/                        # Pipeline con notebooks (ejecutar en orden)
│   ├── 1_etl.ipynb             # Paso 1: ETL mensual (raw → interim)
│   ├── train.ipynb             # Paso 2: Dataset train + LGBM + calibración → lgbm_model_cal.pkl
│   └── inference.ipynb         # Paso 3: Dataset inferencia + scoring con modelo calibrado
├── scripts/                    # Alternativa por línea de comandos (usa config/config.yaml)
│   ├── run_etl.py              # ETL mensual
│   ├── run_train.py            # Train + calibración → models/lgbm_model_cal.pkl
│   └── run_inference.py        # Inferencia con modelo calibrado
├── src/
│   ├── config.py               # load_config, get_paths
│   ├── data/
│   │   ├── etl.py              # ETL mensual (inspecciones, consumo)
│   │   └── make_dataset.py     # Dataset wide, create_train/inference_dataset, columns_filter
│   ├── modeling/
│   │   ├── supervised_models.py
│   │   ├── helpers.py
│   │   └── legacy/
│   └── preprocessing/
├── data/                       # No versionado
│   ├── raw/                    # Entrada ETL
│   ├── interim/                # Salida ETL (parquets por año/mes)
│   ├── processed/             # Datasets wide (train, inference)
│   ├── predictions/            # CSV de scores por inferencia
│   └── logs/                   # Logs de run_etl, run_train, run_inference
├── models/                     # No versionado
│   ├── features.pkl
│   ├── hyperparams.pkl
│   └── lgbm_model_cal.pkl      # Modelo calibrado (salida de train)
├── docs/
│   ├── manual_usuario.md       # Manual de usuario
│   └── setup.md                # Configuración del entorno
├── requirements.txt
└── README.md
```

## Configuración

Toda la configuración (rutas, fechas de corte, filtros, nivel de log) está en **config/config.yaml**. Los notebooks en `poc/` y los scripts en `scripts/` leen esta config. Para una actualización mensual típica basta con editar **inference.cutoff** (mes a predecir, formato YYYY-MM-DD).

- **paths**: raw, interim, processed, models, predictions, logs  
- **inference**: cutoff, cant_periodos, contratos_list, columns_filter  
- **train**: cutoff, cant_periodos, sam_th, param_imb_method, preprocesor_num  
- **etl**: sources (inspecciones, consumo), overwrite  
- **log_level**: DEBUG, INFO, WARNING, ERROR  

Ver [docs/manual_usuario.md](docs/manual_usuario.md) para detalle de cada parámetro.

## Cómo ejecutar

### Prerrequisitos

1. **Entorno Python**  
   [docs/setup.md](docs/setup.md) y `pip install -r requirements.txt`.

2. **Datos**  
   En `data/raw/` la estructura que espera el ETL (inspecciones y consumo por mes). En `models/` deben existir `features.pkl` y `hyperparams.pkl` para entrenar.

### Opción A: Scripts (recomendado para producción)

Desde la raíz del proyecto:

```bash
python scripts/run_etl.py
python scripts/run_train.py
python scripts/run_inference.py
```

Opciones útiles: `run_etl.py --overwrite`, `run_etl.py --config otro.yaml`, `run_train.py --config otro.yaml`, `run_inference.py --config otro.yaml`. Los logs se escriben en `data/logs/` (etl_*.log, train_*.log, inference_*.log).

### Opción B: Notebooks

Ejecutar en orden desde la raíz (kernel con raíz del proyecto):

1. `poc/1_etl.ipynb` — ETL  
2. `poc/train.ipynb` — Dataset train + LGBM + calibración → `lgbm_model_cal.pkl`  
3. `poc/inference.ipynb` — Dataset inferencia + scoring con modelo calibrado  

Los notebooks leen la config desde **config/config.yaml**.

### Resumen de inputs y outputs

| Paso     | Inputs principales | Outputs principales |
|----------|--------------------|----------------------|
| ETL      | `data/raw/`, config (paths, etl) | `data/interim/`, opc. `data/logs/etl_*.log` |
| Train    | `data/interim/`, `models/features.pkl`, `models/hyperparams.pkl`, config (train) | `data/processed/train/.../train_wide.parquet`, `models/lgbm_model_cal.pkl`, opc. `data/logs/train_*.log` |
| Inferencia | `data/interim/`, `models/lgbm_model_cal.pkl`, `models/features.pkl`, config (inference) | `data/processed/inference/.../`, `data/predictions/scores_<CUTOFF>.csv`, opc. `data/logs/inference_*.log` |

## Parámetros clave

- **inference.cutoff** — Mes a predecir (YYYY-MM-DD). Define el dataset de inferencia y el nombre del CSV (`scores_<CUTOFF>.csv`).
- **train.cutoff** — Fecha tope de inspecciones para train; se usan todos los meses con inspecciones hasta esa fecha. `null` = sin límite.
- **inference.columns_filter** — Opcional: filtrar por columnas antes de tsfel (ej. `{ ciclo: ["14"] }`). `null` = sin filtro.
- **cant_periodos** — Ventana de meses de consumo hacia atrás (p. ej. 12).

Más detalles en [docs/manual_usuario.md](docs/manual_usuario.md).
