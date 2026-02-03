# Querétaro POC

POC de pipeline de ML para detección de anomalías/consumo (Querétaro): ETL, construcción de dataset wide, entrenamiento con LightGBM e inferencia.

**Repositorio:** [https://github.com/carlosrios10/Aguas](https://github.com/carlosrios10/Aguas)

## Descripción

El proyecto procesa datos de inspecciones y consumo, construye un dataset en formato wide con features de series de tiempo (tsfel, tendencias, consumo constante, etc.), entrena un modelo LGBM y permite ejecutar inferencia. El flujo se ejecuta con **tres notebooks** en `poc/`.

## Estructura del proyecto

```
queretaro_poc/
├── poc/                         # Pipeline principal (3 notebooks, ejecutar en orden)
│   ├── 1_etl.ipynb              # Paso 1: ETL mensual (raw → interim)
│   ├── train.ipynb              # Paso 2: Dataset train + entrenamiento LGBM + calibración
│   └── inference.ipynb         # Paso 3: Dataset inferencia + scoring
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
│   └── lgbm_model_cal.pkl       # Modelo calibrado (salida del train)
├── docs/                        # Documentación (p. ej. setup.md)
├── requirements.txt
└── README.md
```

- **poc/**: tres notebooks que orquestan el flujo completo (1_etl → train → inference).
- **src/**: código reutilizable (ETL, dataset, modelo, preprocesado).
- **data/** y **models/**: no se suben a Git; en otra máquina se copian o generan (ver `docs/`).

## Cómo ejecutar

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

### Orden de ejecución

Ejecutar los notebooks **desde la raíz del proyecto** (o con el kernel configurado para que la raíz sea el directorio del proyecto) para que los imports `from src...` y las rutas relativas (`../data/`, `../models/`) resuelvan correctamente.

| Orden | Notebook        | Qué hace |
|-------|-----------------|----------|
| 1     | `1_etl.ipynb`   | Lee datos de `data/raw/`, los procesa y escribe en `data/interim/` (parquets por año/mes: consumo, inspecciones). |
| 2     | `train.ipynb`   | Lee `data/interim/`, construye el dataset wide de **train** (todas las fechas de corte con inspecciones ≤ CUTOFF), entrena LGBM, calibra y guarda `models/lgbm_model_cal.pkl`. Usa `models/features.pkl` y `models/hyperparams.pkl`. |
| 3     | `inference.ipynb` | Lee `data/interim/`, construye el dataset wide de **inferencia** para un CUTOFF (mes a predecir), carga el modelo y features, calcula scores y guarda `data/predictions/scores_<CUTOFF>.csv`. |

### Parámetros importantes

- **CUTOFF (train)**  
  Fecha tope de inspecciones: se usan **todos** los meses con inspecciones hasta esa fecha para armar el dataset de train (no solo ese mes). Se configura en el notebook de train.

- **CUTOFF (inference)**  
  Mes que se quiere predecir; ese mes **no** entra en el análisis (solo se usa consumo anterior). Define también el nombre del CSV de salida (`scores_<CUTOFF>.csv`).

- **CANT_PERIODOS**  
  Ventana de meses de consumo hacia atrás (p. ej. 12). Común en train e inferencia.

### Resumen de inputs y outputs

| Notebook       | Inputs principales                          | Outputs principales |
|----------------|---------------------------------------------|----------------------|
| `1_etl.ipynb`  | `data/raw/`                                 | `data/interim/`      |
| `train.ipynb`  | `data/interim/`, `models/features.pkl`, `models/hyperparams.pkl` | `data/processed/train/.../train_wide.parquet`, `models/lgbm_model_cal.pkl` |
| `inference.ipynb` | `data/interim/`, `models/lgbm_model_cal.pkl`, `models/features.pkl` | `data/processed/inference/.../inference_wide.parquet`, `data/predictions/scores_<CUTOFF>.csv` |
