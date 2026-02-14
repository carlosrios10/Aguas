# AR-EPM POC

POC de pipeline de ML para detección de anomalías/fraude en consumo: ETL, construcción de dataset wide, entrenamiento con LightGBM e inferencia.

**Repositorio:** [https://github.com/carlosrios10/Aguas](https://github.com/carlosrios10/Aguas)

## Descripción

El proyecto procesa datos de inspecciones y consumo, construye un dataset en formato wide con features de series de tiempo (tsfel, tendencias, consumo constante, etc.), entrena un modelo LGBM y permite ejecutar inferencia. La **configuración** está centralizada en `config/config.yaml`. El flujo se puede ejecutar con **notebooks** en `poc/` o con **scripts** en `scripts/`.

## Estructura del proyecto

```
ar-epm_poc/
├── config/
│   └── config.yaml              # Configuración: paths, etl, train, inference
├── scripts/                     # Scripts ejecutables (alternativa a notebooks)
│   ├── run_etl.py               # ETL: raw → interim
│   ├── run_train.py             # Dataset train + entrenamiento LGBM
│   └── run_inference.py         # Dataset inferencia + scoring
├── poc/                         # Pipeline en notebooks (ejecutar en orden)
│   ├── 1_etl.ipynb             # Paso 1: ETL mensual (raw → interim)
│   ├── train.ipynb              # Paso 2: Dataset train + entrenamiento LGBM
│   └── inference.ipynb          # Paso 3: Dataset inferencia + scoring
├── src/
│   ├── data/                    # ETL y construcción de dataset
│   │   ├── etl.py               # ETL mensual (inspecciones, consumo)
│   │   └── make_dataset.py      # Dataset wide, features, create_train/inference_dataset
│   ├── modeling/                # Modelo y utilidades
│   │   ├── supervised_models.py # LGBMModel, get_preprocesor
│   │   ├── helpers.py           # save_model, etc.
│   │   └── legacy/              # Código legacy (no usado en poc)
│   └── preprocessing/           # Preprocesado para el modelo
│       ├── preprocessing.py     # ToDummy, TeEncoder, CardinalityReducer, MinMaxScalerRow
│       └── legacy.py            # Código legacy (no usado en poc)
├── data/                        # Datos (no versionados; ver docs)
│   ├── raw/                     # Entrada del ETL
│   ├── interim/                 # Salida ETL (consumo, inspecciones por mes)
│   ├── processed/               # Datasets wide (train, inference por cutoff)
│   └── predictions/             # CSV de scores por inferencia
├── models/                      # Modelos y artefactos (no versionados)
│   ├── features.pkl            # Lista de columnas para el modelo (desarrollo)
│   ├── hyperparams.pkl          # Hiperparámetros LGBM (desarrollo)
│   └── lgbm_model.pkl           # Modelo entrenado (salida del train)
├── docs/                        # Documentación (p. ej. setup.md)
├── requirements.txt
└── README.md
```

- **config/**: `config.yaml` define rutas (`paths`), opciones ETL (`etl`), parámetros de train e inferencia. Los notebooks y scripts leen esta config.
- **scripts/**: `run_etl.py`, `run_train.py`, `run_inference.py` ejecutan el pipeline desde línea de comandos (usan `config/config.yaml` por defecto; se puede pasar `--config otro.yaml`).
- **poc/**: notebooks equivalentes a los scripts para desarrollo y exploración.
- **src/**: código reutilizable (ETL, dataset, modelo, preprocesado).
- **data/** y **models/**: no se suben a Git; en otra máquina se copian o generan (ver `docs/`).

## Cómo ejecutar

Para uso operativo mensual (solo editar config e ejecutar ETL e inferencia), ver **[Manual de usuario](docs/manual_usuario.md)**.

### Prerrequisitos

1. **Entorno Python**  
   Crear y activar el entorno virtual e instalar dependencias. Pasos detallados en [docs/setup.md](docs/setup.md).

   ```bash
   pip install -r requirements.txt
   ```

2. **Datos de entrada**  
   En `data/raw/` debe estar la estructura que espera el ETL (archivos de inspecciones y consumo según lo definido en `1_etl.ipynb`). Si clonas sin datos, copiar o generar según `docs/`.

3. **Artefactos para train**  
   El notebook de train espera en `models/` los archivos `features.pkl` y `hyperparams.pkl`. Si no existen, hay que crearlos antes (proceso de selección de features e hiperparámetros).

### Configuración (`config/config.yaml`)

Todos los parámetros editables están en un solo archivo:

- **paths**: rutas a `raw`, `interim`, `processed`, `models`, `predictions` (relativas a la raíz del proyecto).
- **etl**: `sources` (inspecciones, consumo), `overwrite` (reprocesar todo o solo pendientes).
- **train**: `cutoff`, `cant_periodos`, `max_ctas_neg`, sampling (`sam_th`, `param_imb_method`), `preprocesor_num`.
- **inference**: `cutoff`, `cant_periodos`, `contratos_list` (null = todos los contratos con consumo).

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
| 1     | `1_etl.ipynb`   | ETL: lee `data/raw/`, procesa y escribe en `data/interim/` (parquets por año/mes). Usa `config` para paths y `etl.sources`/`etl.overwrite`. |
| 2     | `train.ipynb`   | Lee `data/interim/`, construye dataset wide de **train** (fechas de corte ≤ cutoff de config), entrena LGBM y guarda `models/lgbm_model.pkl`. Usa `models/features.pkl` y `models/hyperparams.pkl`. |
| 3     | `inference.ipynb` | Construye dataset wide de **inferencia** para el cutoff de config, carga modelo y features, guarda `data/predictions/scores_<CUTOFF>.csv`. |

### Parámetros importantes (en `config/config.yaml`)

- **train.cutoff**  
  Fecha tope de inspecciones: se usan todos los meses con inspecciones hasta esa fecha para el dataset de train.

- **inference.cutoff**  
  Mes a predecir; ese mes no entra en el análisis (solo consumo anterior). Define el nombre del CSV de salida (`scores_<CUTOFF>.csv`).

- **cant_periodos** (train e inference)  
  Ventana de meses de consumo hacia atrás (p. ej. 12).

### Resumen de inputs y outputs

| Paso     | Inputs principales                                              | Outputs principales |
|----------|------------------------------------------------------------------|---------------------|
| ETL      | `data/raw/`, `config` (paths, etl)                               | `data/interim/`     |
| Train    | `data/interim/`, `models/features.pkl`, `models/hyperparams.pkl`, `config` (paths, train) | `data/processed/train/.../`, `models/lgbm_model.pkl` |
| Inferencia | `data/interim/`, `models/lgbm_model.pkl`, `models/features.pkl`, `config` (paths, inference) | `data/processed/inference/.../`, `data/predictions/scores_<CUTOFF>.csv` |
