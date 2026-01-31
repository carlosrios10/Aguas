# Querétaro POC

POC de pipeline de ML para detección de anomalías/consumo (Querétaro): ETL, construcción de dataset wide, entrenamiento con LightGBM e inferencia.

## Descripción

El proyecto procesa datos de inspecciones y consumo, construye un dataset en formato wide con features de series de tiempo (tsfel, tendencias, consumo constante, etc.), entrena un modelo LGBM y permite ejecutar inferencia. El flujo está organizado en cuatro pasos ejecutables desde notebooks en `poc/`.

## Estructura del proyecto

```
queretaro_poc/
├── poc/                         # Pipeline principal (ejecutar en orden)
│   ├── 1_etl.ipynb              # ETL mensual: raw → interim
│   ├── 2_dataset_creation.ipynb # Dataset wide + features
│   ├── 3_train.ipynb            # Entrenamiento LGBM
│   └── 4_inference.ipynb        # Inferencia con modelo guardado
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
│       └── legacy.py           # Código legacy (no usado en poc)
├── data/                        # Datos (no versionados; ver docs)
├── models/                      # Modelos guardados (.pkl, no versionados)
├── docs/                        # Documentación (p. ej. creación de entorno)
├── requirements.txt
└── README.md
```

- **poc/**: notebooks que orquestan el flujo; se ejecutan en orden 1 → 2 → 3 → 4.
- **src/**: código reutilizable (ETL, dataset, modelo, preprocesado).
- **data/** y **models/**: no se suben a Git; en otra máquina se copian o generan (ver `docs/`).

## Cómo ejecutar

1. **Entorno:** Crear y activar el entorno virtual e instalar dependencias. Pasos detallados en [docs/setup.md](docs/setup.md).
2. **Datos:** Tener en `data/` la estructura esperada (raw/interim según indique el ETL). Si clonas sin datos, copiar o generar según `docs/`.
3. **Pipeline:** Abrir y ejecutar los notebooks de `poc/` en orden:
   - `1_etl.ipynb` → ETL mensual.
   - `2_dataset_creation.ipynb` → Construcción del dataset y features.
   - `3_train.ipynb` → Entrenar y guardar el modelo.
   - `4_inference.ipynb` → Cargar modelo y generar predicciones.

Ejecutar los notebooks desde la raíz del proyecto para que los imports `from src...` resuelvan correctamente.
