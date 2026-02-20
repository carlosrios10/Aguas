# Manual de usuario – Scoring mensual (Inferencia)

Este manual está dirigido a la persona que ejecuta el proceso de **scoring (inferencia)** cada mes para obtener puntajes de riesgo de fraude/anomalía en el consumo de agua. No es necesario conocer el código; solo editar un archivo de configuración y ejecutar uno o dos comandos.

---

## 1. Qué hace el proceso

El proceso de **inferencia** usa el modelo ya entrenado para calcular un **puntaje de riesgo por contrato** en el mes indicado. El resultado es un archivo CSV con los scores.

Cada mes usted solo debe:
1. Indicar el mes a predecir en la configuración
2. Ejecutar el ETL si hay datos nuevos
3. Ejecutar la inferencia

---

## 2. Requisitos previos

- **Entorno Python** configurado (entorno virtual activado, dependencias instaladas con `pip install -r requirements.txt`).
- **Datos de entrada** en `data/raw/`:
  - `data/raw/inspecciones/inspecciones_AAAA_MM.xlsx`
  - `data/raw/consumo/consumo_AAAA_MM.xlsx`
  
  (Sustituya AAAA por el año y MM por el mes, por ejemplo `consumo_2026_03.xlsx`).
- **Modelo entrenado** en `models/`:
  - `lgbm_model_cal.pkl` (modelo calibrado)
  - `features.pkl` (lista de variables)

---

## 3. Configuración: indicar el mes a predecir

Antes de ejecutar, edite el archivo **`config/config.yaml`**:

```yaml
inference:
  cutoff: "2026-03-01"      # Mes a predecir (formato YYYY-MM-DD)
  cant_periodos: 12         # Meses de historial (no cambiar)
  contratos_list: null      # null = todos los contratos
  columns_filter: null      # Filtro opcional (ver abajo)
```

### Parámetros importantes

| Parámetro | Descripción |
|-----------|-------------|
| `cutoff` | **Obligatorio.** Mes a predecir en formato `"YYYY-MM-DD"`. Use siempre comillas. |
| `columns_filter` | Opcional. Permite filtrar contratos por columna. Ejemplo: `{colonia_grl: ["CENTRO", "NORTE"]}` |

---

## 4. Cómo ejecutar

### Opción A: Doble clic (Windows)

En la carpeta `docs/` hay archivos `.bat` que puede ejecutar con doble clic:

| Archivo | Qué hace |
|---------|----------|
| `run_etl.bat` | Ejecuta el ETL |
| `run_inference.bat` | Ejecuta la inferencia |

### Opción B: Línea de comandos

Abra una terminal en la **raíz del proyecto**.

**ETL** (solo si hay datos nuevos en `data/raw/`):
```bash
python scripts/run_etl.py
```

Para reprocesar todos los meses desde cero:
```bash
python scripts/run_etl.py --overwrite
```

**Inferencia**:
```bash
python scripts/run_inference.py
```

### Opción C: Jupyter Lab (notebooks interactivos)

Si prefiere ejecutar paso a paso de forma interactiva, puede usar Jupyter Lab con los notebooks en la carpeta `poc/`:

1. Inicie Jupyter Lab desde la raíz del proyecto:
   ```bash
   jupyter lab
   ```

2. Abra el notebook correspondiente:

| Notebook | Propósito |
|----------|-----------|
| `poc/1_etl.ipynb` | ETL paso a paso |
| `poc/inference.ipynb` | Inferencia interactiva |

3. Ejecute las celdas en orden (Shift+Enter).

Esta opción es útil para ver el proceso en detalle o depurar problemas.

---

## 5. Resultado

Al finalizar sin errores, el archivo de scores se guarda en:

```
data/predictions/scores_<CUTOFF>.csv
```

Por ejemplo: `scores_2026-03-01.csv`

El CSV contiene:
| Columna | Descripción |
|---------|-------------|
| `contrato` | Número de contrato |
| `score` | Probabilidad de riesgo (0 a 1). Mayor = más riesgo. |

---

## 6. Logs

Los logs de cada ejecución se guardan en `data/logs/`:
- `etl_YYYYMMDD_HHMMSS.log`
- `inference_YYYYMMDD_HHMMSS.log`

Revise estos archivos si necesita diagnosticar un problema.

---

## 7. Si algo falla

| Error | Solución |
|-------|----------|
| "No se pudo cargar la config" | Verifique que `config/config.yaml` existe y que `inference.cutoff` tiene formato `"YYYY-MM-DD"` con comillas. |
| "inference.cutoff es obligatorio" | El cutoff está vacío. Ponga una fecha válida. |
| "No hay consumo en interim" | Ejecute el ETL primero. Debe haber datos en `data/interim/consumo/` para los meses requeridos. |
| "Menos de X meses de consumo" | No hay suficiente historial. Cargue más meses en `data/raw/` y ejecute el ETL. |
| "Columnas faltantes" | El modelo espera variables que no están en el dataset. Contacte al equipo técnico. |
| Error al cargar el modelo | Verifique que existan `models/lgbm_model_cal.pkl` y `models/features.pkl`. |

Si el error no le resulta claro, revise el log en `data/logs/` y contacte al equipo técnico.

---

## 8. Nota sobre entrenamiento

El **entrenamiento** del modelo (`run_train.py`) NO forma parte del flujo mensual. Se ejecuta cuando el equipo de análisis decide reentrenar. Usted solo debe ejecutar **ETL** (si hay datos nuevos) e **inferencia** cada mes.

---

## Resumen rápido

| Paso | Acción |
|------|--------|
| 1 | Editar `config/config.yaml` → poner `inference.cutoff: "YYYY-MM-DD"` |
| 2 | Colocar datos nuevos en `data/raw/` (si aplica) |
| 3 | Ejecutar ETL: doble clic en `docs/run_etl.bat` o `python scripts/run_etl.py` |
| 4 | Ejecutar inferencia: doble clic en `docs/run_inference.bat` o `python scripts/run_inference.py` |
| 5 | Revisar resultado en `data/predictions/scores_<CUTOFF>.csv` |
