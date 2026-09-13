# CAJ POC (AquaData)

POC de pipeline de ML para detección de anomalías/fraude en consumo: ETL, construcción de dataset wide, entrenamiento con LightGBM e inferencia.

**Repositorio:** [https://github.com/carlosrios10/Aguas](https://github.com/carlosrios10/Aguas)

## Descripción

El proyecto procesa datos de inspecciones y consumo, construye un dataset en formato wide con features de series de tiempo (tsfel, tendencias, consumo constante, etc.), entrena un modelo LGBM y permite ejecutar inferencia. La **configuración** está centralizada en `config/config.yaml`. El flujo se puede ejecutar con **notebooks** en `poc/` o con **scripts** en `scripts/`.

## Estructura del proyecto

```
proyecto/
├── config/
│   ├── config.yaml              # Configuración: paths, etl, train, inference, log_level
│   └── tsfel_config_consumo.json  # Features tsfel que entran al dataset wide
├── scripts/                     # Scripts ejecutables (alternativa a notebooks)
│   ├── run_etl.py               # ETL: raw → interim (inspecciones, consumo, maestro)
│   ├── run_train.py             # Dataset train + entrenamiento LGBM
│   └── run_inference.py         # Dataset inferencia + scoring
├── poc/                         # Pipeline en notebooks (ejecutar en orden)
│   ├── etl.ipynb                # Paso 1: ETL mensual (raw → interim)
│   ├── train.ipynb              # Paso 2: Dataset train + entrenamiento LGBM
│   └── inference.ipynb          # Paso 3: Dataset inferencia + scoring (incl. columns_filter)
├── src/
│   ├── data/                    # ETL y construcción de dataset
│   │   ├── etl.py               # ETL mensual (inspecciones, consumo, maestro)
│   │   └── make_dataset.py      # Dataset wide, features, create_train/inference_dataset
│   ├── modeling/                # Modelo y utilidades
│   │   ├── supervised_models.py # LGBMModel, get_preprocesor
│   │   └── helpers.py           # save_model
│   └── preprocessing/
│       └── preprocessing.py     # preprocess_model_input, TeEncoder, CardinalityReducer
├── data/                        # Datos (no versionados; ver docs)
│   ├── raw/                     # Entrada del ETL (inspecciones, consumo, maestro)
│   ├── interim/                 # Salida ETL (parquets por año/mes)
│   ├── processed/               # Datasets wide (train, inference por cutoff)
│   ├── predictions/             # CSV de scores por inferencia
│   └── logs/                    # Logs de ejecución (etl_*.log, inference_*.log, train_*.log)
├── models/                      # Artefactos (no versionados)
│   ├── features.pkl             # Columnas del modelo (se entrega; no se genera acá)
│   ├── hyperparams.pkl          # Hiperparámetros LGBM (se entrega; no se genera acá)
│   └── lgbm_model.pkl           # Modelo entrenado (salida de run_train.py)
├── docs/                        # Documentación
│   ├── manual_usuario.md        # Manual para ejecución mensual (ETL + inferencia)
│   ├── guia_etl_paso_a_paso.md  # Cómo armar los Excel raw y correr el ETL
│   ├── guia_train_paso_a_paso.md  # Cómo entrenar (pkl entregados → lgbm_model.pkl)
│   ├── guia_inferencia_paso_a_paso.md  # Guía paso a paso inferencia (personas no técnicas)
│   └── setup.md                 # Configuración del entorno
├── requirements.txt
└── README.md
```

- **config/**: `config.yaml` define rutas (`paths`, incl. `logs`), nivel de log (`log_level`), opciones ETL (`etl`), parámetros de train e inferencia. Los notebooks y scripts leen esta config.
- **scripts/**: `run_etl.py`, `run_train.py`, `run_inference.py` ejecutan el pipeline desde línea de comandos (usan `config/config.yaml` por defecto; `--config otro.yaml` para otro archivo). Si está definido `paths.logs`, cada ejecución escribe un archivo de log en `data/logs/`.
- **poc/**: notebooks equivalentes a los scripts para desarrollo y exploración.
- **src/**: código reutilizable (ETL, dataset, modelo, preprocesado).
- **data/** y **models/**: no se suben a Git. `features.pkl` y `hyperparams.pkl` se entregan y se copian a `models/` antes del train. `lgbm_model.pkl` lo genera `run_train.py`.

## Cómo ejecutar

Para uso operativo mensual (solo editar config e ejecutar ETL e inferencia), ver **[Manual de usuario](docs/manual_usuario.md)**.

### Prerrequisitos

1. **Entorno Python**  
   Crear y activar el entorno virtual e instalar dependencias. En este proyecto el entorno estándar se llama **`qenv`** (ver pasos detallados en [docs/setup.md](docs/setup.md)):

   ```bash
   python -m venv qenv
   .\qenv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. **Datos de entrada**  
   En `data/raw/` debe estar la estructura que espera el ETL: inspecciones, consumo y maestro (carpetas `inspecciones/`, `consumo/`, `maestro/` con archivos `*_AAAA_MM.xlsx`). Ver `config/config.yaml` → `etl.sources` y [docs/manual_usuario.md](docs/manual_usuario.md).

3. **Artefactos para train**  
   Antes de entrenar, copiar en `models/` los archivos entregados `features.pkl` y `hyperparams.pkl`. No se generan en este repositorio. `run_train.py` los usa y escribe `models/lgbm_model.pkl`. La inferencia mensual usa ese modelo y `features.pkl`.

### Configuración (`config/config.yaml`)

Todos los parámetros editables están en un solo archivo:

- **paths**: rutas a `raw`, `interim`, `processed`, `models`, `predictions`, `logs` y `tsfel_config` (relativas a la raíz).
- **log_level**: nivel de logging (`INFO`, `DEBUG`, `WARNING`, `ERROR`). Aplica a ETL, train e inferencia cuando se ejecutan por script.
- **etl**: `sources` (inspecciones, consumo, maestro), `overwrite` (reprocesar todo o solo pendientes).
- **train**: `cutoff`, `cant_periodos`, `max_ctas_neg`, sampling (`sam_th`, `param_imb_method`), `preprocesor_num`.
- **inference**: `cutoff`, `cant_periodos`, `contratos_list` (null = contratos del maestro con consumo en la ventana), `columns_filter` (opcional; columnas que ya existen antes de tsfel, por ejemplo `tipo_cliente` o `marca`), `output_columns` (columnas extra del CSV; el script siempre agrega `score` al final; null = solo contrato y score).

Los notebooks y los scripts leen esta config; los scripts permiten usar otro archivo con `--config otro.yaml`.

### Ejecución por scripts (desde la raíz del proyecto)

```bash
python scripts/run_etl.py                    # ETL (usa paths y etl.* de config)
python scripts/run_etl.py --overwrite         # Reprocesar todo
python scripts/run_etl.py --config prod.yaml

python scripts/run_train.py                   # Train (usa paths y train.* de config)
python scripts/run_train.py --config prod.yaml

python scripts/run_inference.py               # Inferencia (usa paths e inference.* de config)
python scripts/run_inference.py --config prod.yaml
```

### Ejecución por notebooks

Ejecutar los notebooks **desde la raíz del proyecto** (o con el kernel configurado con la raíz como directorio de trabajo) para que los imports `from src...` y la carga de config funcionen.

| Orden | Notebook        | Qué hace |
|-------|-----------------|----------|
| 1     | `etl.ipynb`     | ETL: lee `data/raw/`, procesa y escribe en `data/interim/` (parquets por año/mes). Usa `config` para paths y `etl.sources`/`etl.overwrite`. |
| 2     | `train.ipynb`   | Lee `data/interim/`, construye dataset wide de **train** (fechas de corte ≤ cutoff de config), entrena LGBM y guarda `models/lgbm_model.pkl`. Usa `models/features.pkl` y `models/hyperparams.pkl`. |
| 3     | `inference.ipynb` | Construye dataset wide de **inferencia** para el cutoff de config, carga modelo y features, guarda `data/predictions/scores_<CUTOFF>_<AAAAMMDD>_<HHMMSS>.csv`. |

### Parámetros importantes (en `config/config.yaml`)

- **train.cutoff**  
  `null` usa todas las inspecciones válidas. Una fecha (YYYY-MM-DD) es tope: solo meses con inspección menores o iguales a esa fecha.

- **inference.cutoff**  
  Mes a predecir; ese mes no entra en el análisis (solo consumo anterior). El CSV de salida se nombra `scores_<CUTOFF>_<AAAAMMDD>_<HHMMSS>.csv` (fecha y hora de ejecución).

- **inference.columns_filter** (opcional)  
  Filtro extra del dataset de inferencia (ej. `{ tipo_cliente: ["Comercial"] }`). Se aplica antes de tsfel, y solo a columnas que ya existen (no `ciudad_sector`). `null` no agrega filtro: el universo sigue siendo los contratos del maestro con consumo en la ventana.

- **cant_periodos** (train e inference)  
  Ventana de meses de consumo hacia atrás (p. ej. 12).

### Resumen de inputs y outputs

| Paso       | Inputs principales                                                                 | Outputs principales |
|------------|-------------------------------------------------------------------------------------|----------------------|
| ETL        | `data/raw/` (inspecciones, consumo, maestro), `config` (paths, etl)                  | `data/interim/`, opcionalmente `data/logs/etl_*.log` |
| Train      | `data/interim/`, `models/features.pkl`, `models/hyperparams.pkl`, `config` (paths, train) | `data/processed/train/.../`, `models/lgbm_model.pkl`, opc. `data/logs/train_*.log` |
| Inferencia | `data/interim/`, `models/lgbm_model.pkl`, `models/features.pkl`, `config` (paths, inference, columns_filter) | `data/processed/inference/.../`, `data/predictions/scores_<CUTOFF>_<timestamp>.csv`, opc. `data/logs/inference_*.log` |
