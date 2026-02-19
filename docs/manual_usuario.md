# Manual de usuario – Pipeline de scoring mensual

Este manual está dirigido a la persona que ejecuta el proceso de **scoring (inferencia)** cada mes para obtener puntajes de riesgo de fraude/anomalía en el consumo de agua. No es necesario conocer el código; solo editar un archivo de configuración y ejecutar dos comandos (o uno, si no hay datos nuevos).

---

## 1. Qué hace el proceso

- **ETL:** Toma los archivos de inspecciones y consumo que usted coloca en `data/raw/`, los limpia y guarda en `data/interim/`.
- **Inferencia (scoring):** Usa el modelo ya entrenado y el mes que usted indique para calcular un puntaje de riesgo por contrato. El resultado es un archivo CSV con los scores.

Cada mes usted solo debe **indicar el mes a predecir** y ejecutar el proceso. El resto de la configuración suele no cambiar.

---

## 2. Requisitos previos

- **Entorno Python** ya configurado en su máquina (entorno virtual activado, dependencias instaladas con `pip install -r requirements.txt`). Si no lo tiene, solicite apoyo al equipo técnico o consulte [docs/setup.md](setup.md) si existe.
- **Datos de entrada:** En la carpeta `data/raw/` deben estar los archivos según el formato esperado:
  - `data/raw/inspecciones/inspecciones_AAAA_MM.xlsx`
  - `data/raw/consumo/consumo_AAAA_MM.xlsx`
  - `data/raw/maestro/maestro_AAAA_MM.xlsx` (obligatorio para inferencia; el ETL debe haberlo procesado)
  (sustituya AAAA por el año y MM por el mes, por ejemplo 2026 y 03).
- **Modelo en uso:** En la carpeta `models/` deben existir los archivos `lgbm_model.pkl` y `features.pkl` (proporcionados o generados por el equipo que entrena el modelo).

---

## 3. Paso obligatorio cada mes: indicar el mes a predecir

Antes de ejecutar la inferencia, debe editar el archivo de configuración para indicar **qué mes se va a predecir**.

1. Abra el archivo **`config/config.yaml`** (en la raíz del proyecto).
2. Al inicio del archivo verá la sección **inference** y el parámetro **cutoff**.
3. Cambie el valor de **cutoff** al mes que desea predecir, en formato **YYYY-MM-DD** (año-mes-día; el día puede ser 01).

**Ejemplo:** Para predecir marzo de 2026:

```yaml
inference:
  cutoff: "2026-03-01"
  cant_periodos: 12
  contratos_list: null
  columns_filter: null   # opcional: filtrar por columnas del dataset (ej: { ciclo: ["14","16"] })
```

- Use siempre comillas alrededor de la fecha.
- El formato debe ser exactamente **YYYY-MM-DD** (por ejemplo `"2026-04-01"` para abril de 2026).
- **columns_filter** (opcional): si desea restringir la inferencia a ciertos valores (por ejemplo solo algunos ciclos o localidades), indique un diccionario `columna: [valores]`. Si es `null`, se procesan todos los contratos con consumo.
- En **paths** puede aparecer **logs: "data/logs"**; ahí se guardan archivos de log de cada ejecución (ETL, inferencia, entrenamiento). Opcionalmente puede configurarse **log_level: "INFO"** (o "DEBUG", "WARNING", "ERROR") en el YAML.
- No modifique el resto del archivo a menos que le indiquen lo contrario.

---

## 4. Cómo ejecutar (desde la raíz del proyecto)

Abra una terminal y sitúese en la **raíz del proyecto** (la carpeta que contiene `config/`, `scripts/`, `data/`, etc.).

### 4.1. ETL (solo si hay datos nuevos)

Si cargó archivos nuevos en `data/raw/` (inspecciones o consumo del mes), ejecute primero el ETL:

```bash
python scripts/run_etl.py
```

Si desea reprocesar todos los meses desde cero (no solo los nuevos):

```bash
python scripts/run_etl.py --overwrite
```

Si no hay archivos nuevos en `data/raw/`, puede **saltar este paso** y pasar directo a la inferencia.

### 4.2. Inferencia (scoring)

Ejecute:

```bash
python scripts/run_inference.py
```

El script leerá el **cutoff** que configuró en `config/config.yaml` y generará el archivo de scores para ese mes.

---

## 5. Dónde está el resultado

Al finalizar la inferencia sin errores, el archivo de resultados se guarda en:

**`data/predictions/scores_<CUTOFF>.csv`**

Donde `<CUTOFF>` es la fecha que puso en `inference.cutoff` (por ejemplo `scores_2026-03-01.csv`). Ese CSV contiene, por contrato, el puntaje de riesgo (probabilidad de fraude/anomalía) según el modelo.

Si en la config está definido **paths.logs** (por ejemplo `data/logs`), cada ejecución de ETL, inferencia o entrenamiento genera además un archivo de log con fecha y hora (por ejemplo `inference_20260218_120000.log`) para facilitar la depuración.

---

## 6. Si algo falla

- **Error al cargar la config:** Compruebe que `config/config.yaml` existe y que la fecha en `inference.cutoff` tiene formato **YYYY-MM-DD** y está entre comillas.
- **Error “inference.cutoff es obligatorio”:** No deje el cutoff vacío ni en blanco. Ponga una fecha válida, por ejemplo `"2026-03-01"`.
- **Error por “meses cargados” o datos insuficientes:** Asegúrese de haber ejecutado el ETL para los meses que necesita la ventana de consumo (por defecto 12 meses hacia atrás desde el cutoff). Debe haber archivos en `data/interim/consumo/` y `data/interim/maestro/` para que la inferencia funcione.
- **Error “No hay maestro en interim”:** La inferencia requiere el maestro procesado. Ejecute el ETL (incluyendo la fuente **maestro** en `config.yaml` → `etl.sources`) y asegúrese de tener al menos un archivo en `data/raw/maestro/` (por ejemplo `maestro_AAAA_MM.xlsx`).
- **“Dataset de inferencia quedó vacío tras aplicar columns_filter”:** El filtro definido en `inference.columns_filter` no coincide con ningún contrato (por ejemplo valores de ciclo o localidad que no existen en los datos). Revise los valores en el maestro o deje `columns_filter: null` para procesar todos.
- **Error al cargar el modelo:** Verifique que en `models/` existan `lgbm_model.pkl` y `features.pkl`. Si faltan, debe proporcionarlos el equipo que entrena el modelo.

Si el mensaje de error no le resulta claro, revise los archivos en **`data/logs/`** (si existen); allí se guarda el detalle de cada ejecución. Anote el texto completo del error y contacte al equipo técnico si es necesario.

---

## 7. Entrenamiento del modelo (no es mensual)

El **entrenamiento** del modelo no forma parte del flujo mensual del usuario. Se ejecuta cuando el equipo de análisis decide reentrenar (por ejemplo con más datos o nuevos parámetros), mediante `python scripts/run_train.py` o el notebook `poc/train.ipynb`. Usted solo debe ejecutar **ETL** (si hay datos nuevos) e **inferencia** cada mes, después de actualizar **inference.cutoff** en `config/config.yaml`.

---

## Resumen rápido

| Qué hacer cada mes | Dónde / Cómo |
|-------------------|--------------|
| 1. Indicar el mes a predecir | Editar `config/config.yaml` → `inference.cutoff` (formato "YYYY-MM-DD") |
| 2. Cargar datos nuevos (si aplica) | Colocar archivos en `data/raw/inspecciones/` y `data/raw/consumo/` |
| 3. Ejecutar ETL (si hay datos nuevos) | `python scripts/run_etl.py` |
| 4. Ejecutar inferencia | `python scripts/run_inference.py` |
| 5. Revisar resultados | Abrir `data/predictions/scores_<CUTOFF>.csv` |
