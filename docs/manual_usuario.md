# Manual de usuario — Querétaro POC

Este documento describe cómo usar el pipeline de ML (ETL, entrenamiento con calibración e inferencia) mediante **configuración** y **scripts** o **notebooks**.

---

## 1. Configuración central: config/config.yaml

Toda la configuración del pipeline está en **config/config.yaml**. No es necesario modificar código para cambiar rutas, fechas de corte o filtros.

### 1.1 Actualización mensual (inferencia)

Para lanzar una inferencia del mes siguiente, suele bastar con editar una línea:

```yaml
inference:
  cutoff: "2026-04-01"   # Mes a predecir (YYYY-MM-DD)
```

El resto de la sección puede dejarse como está:

```yaml
inference:
  cutoff: "2026-03-01"
  cant_periodos: 12
  contratos_list: null   # null = todos los contratos con consumo en la ventana
  columns_filter: null   # opcional: ej. { ciclo: ["14"] }; null = sin filtro
```

### 1.2 Rutas (paths)

Rutas relativas a la raíz del proyecto. No incluir la raíz; el código la resuelve.

```yaml
paths:
  raw: "data/raw"
  interim: "data/interim"
  processed: "data/processed"
  models: "models"
  predictions: "data/predictions"
  logs: "data/logs"
```

- **logs**: donde se guardan los archivos de log al ejecutar los scripts (`etl_*.log`, `train_*.log`, `inference_*.log`). Si se omite, los scripts solo escriben en consola.

### 1.3 Nivel de log (log_level)

Valores: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Afecta a consola y a los archivos en `data/logs/`.

```yaml
log_level: "INFO"
```

### 1.4 ETL (etl)

```yaml
etl:
  sources:
    - inspecciones
    - consumo
  overwrite: false   # true = reprocesar todo; también puede usarse --overwrite en run_etl.py
```

- **sources**: fuentes a procesar (en este POC solo inspecciones y consumo).  
- **overwrite**: si es `true`, se reprocesan todos los meses aunque ya existan en `data/interim/`. En línea de comandos, `--overwrite` tiene prioridad.

### 1.5 Train (train)

```yaml
train:
  cutoff: null       # null = todas las inspecciones disponibles (sin límite de fecha)
  cant_periodos: 12
  max_ctas_neg: 1500
  sam_th: 0.5
  param_imb_method: "over"
  preprocesor_num: 1
```

- **cutoff**: fecha tope de inspecciones para armar el dataset de train. Se usan **todos** los meses con inspecciones hasta esa fecha. `null` = sin límite.  
- **cant_periodos**: ventana de meses de consumo hacia atrás por fecha de corte.  
- El resto son parámetros del modelo/preprocesador (sampling, desbalance, etc.).

### 1.6 Inferencia (inference)

- **cutoff** (obligatorio): mes a predecir, formato **YYYY-MM-DD**. Define el dataset de inferencia y el nombre del CSV de salida: `scores_<CUTOFF>.csv`.  
- **cant_periodos**: ventana de meses de consumo (ej. 12).  
- **contratos_list**: `null` = se scorean todos los contratos con consumo en la ventana; si es una lista, solo esos contratos.  
- **columns_filter** (opcional): diccionario `columna: [valores]` para filtrar el dataset **antes** de calcular features (tsfel). Ejemplo: `{ ciclo: ["14"] }`. Si es `null`, no se aplica filtro por columnas.

---

## 2. Ejecución por scripts

Desde la **raíz del proyecto**:

```bash
# 1. ETL (raw → interim)
python scripts/run_etl.py

# 2. Entrenamiento + calibración (interim → modelo calibrado)
python scripts/run_train.py

# 3. Inferencia (interim + modelo calibrado → scores CSV)
python scripts/run_inference.py
```

### 2.1 Opciones de run_etl.py

- `--config nombre.yaml` — Archivo de config en `config/` (por defecto `config.yaml`).  
- `--overwrite` — Reprocesar todos los meses aunque ya existan en interim.  
- `--raw-dir`, `--interim-dir` — Sobrescribir rutas de raw e interim (opcional).

### 2.2 Opciones de run_train.py y run_inference.py

- `--config nombre.yaml` — Usar otro archivo de config en `config/`.

### 2.3 Logs

Si en **config.yaml** está definido `paths.logs`, cada script escribe un archivo en `data/logs/`:

- `etl_YYYYMMDD_HHMMSS.log`  
- `train_YYYYMMDD_HHMMSS.log`  
- `inference_YYYYMMDD_HHMMSS.log`  

El nivel de detalle se controla con **log_level** en el YAML.

---

## 3. Ejecución por notebooks

Los notebooks en **poc/** reproducen el mismo flujo y leen la config desde **config/config.yaml**:

1. **poc/1_etl.ipynb** — ETL: rutas y fuentes desde `paths` y `etl`.  
2. **poc/train.ipynb** — Dataset train, LGBM, calibración (isotonic, cv=5), guardado de `lgbm_model_cal.pkl`; parámetros desde `train` y `paths`.  
3. **poc/inference.ipynb** — Dataset inferencia, carga de `lgbm_model_cal.pkl` y `features.pkl`, scoring y guardado de `scores_<CUTOFF>.csv`; parámetros desde `inference` y `paths`, incluyendo **columns_filter**.

Ejecutar los notebooks con el **kernel apuntando a la raíz del proyecto** para que `from src...` y las rutas resuelvan bien.

---

## 4. Modelo calibrado

El entrenamiento (notebook o script) hace:

1. Entrenar el LGBM con el dataset de train.  
2. Calibrar con **CalibratedClassifierCV** (método isotonic, cv=5) sobre el mismo train.  
3. Guardar solo el **modelo calibrado** en `models/lgbm_model_cal.pkl`.

La inferencia (notebook o script) usa siempre **lgbm_model_cal.pkl** para calcular los scores P(riesgo). No se usa un modelo sin calibrar en producción.

---

## 5. Errores frecuentes y soluciones

- **"No se pudo cargar la config"**  
  Comprobar que existe `config/config.yaml` y que se ejecuta desde la raíz del proyecto (o que el path al config es correcto con `--config`).

- **"inference.cutoff es obligatorio" / "formato YYYY-MM-DD"**  
  En config, `inference.cutoff` debe ser una fecha válida con exactamente formato YYYY-MM-DD (por ejemplo `2026-03-01`).

- **"No se pudo crear el dataset de train"**  
  Verificar que en `data/interim/` hay datos de inspecciones y consumo (ejecutar antes el ETL).

- **"No se pudo crear el dataset de inferencia"**  
  Verificar que hay consumo en interim para la ventana del cutoff y que `inference.cutoff` tiene formato YYYY-MM-DD.

- **"Dataset de inferencia quedó vacío tras aplicar columns_filter"**  
  Los valores en `inference.columns_filter` no coinciden con ningún registro (p. ej. valores de ciclo o localidad que no existen). Revisar valores o usar `columns_filter: null` para procesar todos.

- **"Columnas faltantes en train_wide / inference_wide"**  
  El `features.pkl` usado no coincide con las columnas del dataset. Entrenar de nuevo o alinear la versión de `features.pkl` con el dataset generado.

- **No existe `lgbm_model_cal.pkl`**  
  Ejecutar antes el paso de train (notebook o `python scripts/run_train.py`). La inferencia siempre usa el modelo calibrado en `models/lgbm_model_cal.pkl`.

---

## 6. Resumen rápido

| Qué quiero hacer        | Dónde se configura              | Comando / pasos |
|-------------------------|----------------------------------|------------------|
| Cambiar mes a predecir  | `config/config.yaml` → inference.cutoff | Editar YAML y ejecutar inferencia (script o notebook). |
| Cambiar rutas           | `config/config.yaml` → paths    | Editar YAML. |
| Filtrar por ciclo/otra columna | inference.columns_filter       | Ej. `columns_filter: { ciclo: ["14"] }`. |
| Reprocesar todo el ETL  | Script: `--overwrite` o etl.overwrite: true | `python scripts/run_etl.py --overwrite` |
| Ver más detalle en logs | log_level: "DEBUG"              | Editar YAML y volver a ejecutar el script. |

Para instalación del entorno y dependencias, ver [setup.md](setup.md).
