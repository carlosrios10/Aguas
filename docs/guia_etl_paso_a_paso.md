# Guía paso a paso: Cómo preparar los archivos raw y ejecutar el ETL

Esta guía está pensada para quien debe **generar o colocar los archivos de entrada** del proyecto y ejecutar el **ETL** (extracción y limpieza). El ETL lee archivos Excel desde `data/raw/`, los normaliza y guarda el resultado en `data/interim/`. Para el flujo mensual completo (ETL + inferencia), consulte [manual_usuario.md](manual_usuario.md).

---

## ¿Qué hace el ETL?

- **Lee** archivos Excel (`.xlsx`) desde `data/raw/` organizados por fuente y mes.
- **Limpia y normaliza** los datos (nombres de columnas en minúsculas, tipos de dato, fechas al primer día del mes, deduplicación).
- **Escribe** en `data/interim/` archivos Parquet por año/mes (por ejemplo `data/interim/consumo/year=2025/month=03/consumo.parquet`).

Solo se procesan los **meses pendientes**: si un archivo ya tiene su Parquet en `interim/`, se omite salvo que se ejecute con `--overwrite`.

---

## Estructura de carpetas en `data/raw/`

Debe existir una carpeta por cada **fuente** y, dentro de ella, archivos con el nombre exacto:

```text
data/raw/
├── inspecciones/
│   ├── inspecciones_2025_01.xlsx
│   ├── inspecciones_2025_02.xlsx
│   └── ...
└── consumo/
    ├── consumo_2025_01.xlsx
    ├── consumo_2025_02.xlsx
    └── ...
```

**Nombre del archivo:** `{fuente}_{AAAA}_{MM}.xlsx`  
- **fuente:** `inspecciones` o `consumo`  
- **AAAA:** año con 4 dígitos (ej. 2025)  
- **MM:** mes con 2 dígitos (ej. 01, 02, 12)

---

## 1. Archivos de inspecciones

### Columnas requeridas (según el ETL actual)

El ETL espera, tras normalizar los nombres de columnas (minúsculas, sin tildes, espacios → `_`), al menos estas columnas en los Excel de inspecciones:

| Columna en el Excel (normalizada) | Descripción |
|-----------------------------------|-------------|
| **servicio_suscrito**            | Identificador del servicio/contrato. Se convertirá a `contrato`. Debe tener 9 caracteres después de limpiar. |
| **fecha**                        | Periodo de la inspección en formato **YYYYMM** (por ejemplo `202501` para enero 2025). El ETL lo convierte a fecha `YYYY-MM-01`. |
| **clasificacion_resultado**      | Clasificación del resultado (`Fraude`, `Anomalía`, `Sin definición`, etc.). Se usa para crear el campo `is_fraud`. |

Otras columnas (como `solicitud`) son opcionales: si existen, se conservan; si no, el ETL igual funciona.

**Reglas clave en el ETL:**

- Los nombres de columna se normalizan (mayúsculas, tildes y espacios se corrigen).
- `servicio_suscrito` → `contrato`.  
  - Se filtran filas donde `contrato` tenga exactamente 9 caracteres y no sea nulo.
- `fecha` se interpreta con formato `%Y%m`.  
  - Si no cumple ese formato, se descarta la fila.
- `clasificacion_resultado`:
  - Se descartan filas con `clasificacion_resultado == "Sin definición"`.
  - Se crea `is_fraud = 1` si `clasificacion_resultado` es `"Fraude"` o `"Anomalía"`, 0 en otro caso.
- Si hay varias filas para el mismo `(contrato, mes)`, el ETL se queda con la de mayor `is_fraud` (si hay algún fraude/anomalía en el mes, el contrato queda marcado como 1 para ese mes).

### Ejemplo de contenido (inspecciones_2025_01.xlsx)

```text
servicio_suscrito | fecha  | clasificacion_resultado | solicitud (opcional) | ...
------------------|--------|-------------------------|----------------------|---
000123456         | 202501 | Sin definición          | 1234                 | ...
000123456         | 202501 | Fraude                  | 5678                 | ...
000234567         | 202501 | Anomalía                | 9012                 | ...
```

Tras el ETL:

- `contrato` será `"000123456"`, `"000234567"`, etc.
- `date` será `2025-01-01` para todas las filas de enero 2025.
- `is_fraud` será 1 si hay alguna fila con Fraude o Anomalía para ese contrato y mes.

---

## 2. Archivos de consumo

### Columnas requeridas (según el ETL actual)

El ETL espera, tras normalizar los nombres de columnas, al menos:

| Columna en el Excel (normalizada) | Descripción |
|-----------------------------------|-------------|
| **niu**                          | Identificador del contrato. Se convertirá a `contrato`. |
| **fecha_mes**                    | Periodo de consumo en formato **YYYYMM** (ej. `202501`). Se convierte a fecha `YYYY-MM-01`. |
| **consumo**                      | Consumo del periodo. Valores negativos se descartan. |

Otras columnas que el ETL usa si existen:

- `instalacion` (texto): se limpia y mantiene.
- `subcategoria_estrato` (texto): se limpia y mantiene.
- `localidad`, `municipio` (texto): se limpian y mantienen.
- `barrio` (texto): se rellena con `"sin_dato"` si falta.
- `determinacion_consumo` (texto): si existe, se limpia y mantiene.

**Reglas clave en el ETL:**

- Los nombres de columna se normalizan.
- `niu` → `contrato`.
- `fecha_mes` se interpreta con formato `%Y%m`. Filas con fecha inválida se descartan.
- Se crea una columna `date` con el primer día del mes (`YYYY-MM-01`).
- `consumo`:
  - Si es NaN o < 0, se convierte a `None` (se considerará como faltante).
- Se ordena por `date` (y eventualmente otras columnas) para dejar los registros limpios y en orden temporal.

### Ejemplo de contenido (consumo_2025_01.xlsx)

```text
niu       | fecha_mes | consumo | localidad | municipio | barrio        | determinacion_consumo
----------|-----------|---------|-----------|-----------|---------------|----------------------
000123456 | 202501    | 25      | NORTE     | CIUDAD    | CENTRO        | NORMAL
000123456 | 202501    | -5      | NORTE     | CIUDAD    | CENTRO        | NORMAL
000234567 | 202501    | 32      | SUR       | CIUDAD    | GRANADA       | NORMAL
000345678 | 202501    | 18      | ESTE      | CIUDAD    | sin_dato      | NORMAL
```

- El valor -5 se considera inválido y se descarta.
- Se mantiene una fila por `(contrato, mes)` tras la limpieza y ordenamiento, con el consumo válido.

---

## 3. Resumen: generar los archivos raw

1. **Crear las carpetas** `data/raw/inspecciones/` y `data/raw/consumo/` (si no existen).
2. **Exportar o generar** los Excel con las columnas indicadas para cada fuente:
   - Inspecciones: `servicio_suscrito`, `fecha` (YYYYMM), `clasificacion_resultado` (y opcionales).
   - Consumo: `niu`, `fecha_mes` (YYYYMM), `consumo` (y opcionales).
3. **Nombrar cada archivo** exactamente: `inspecciones_AAAA_MM.xlsx` y `consumo_AAAA_MM.xlsx`.
4. **Colocar cada archivo** en la carpeta de su fuente.
5. Revisar que **fechas y números** estén en formato coherente (fechas tipo `202501`, consumo numérico, textos sin caracteres extraños).

---

## 4. Ejecutar el ETL (desde la raíz del proyecto)

Con los archivos raw ya en su sitio:

1. Abra una terminal en la **raíz del proyecto** (donde están `config/`, `scripts/`, `data/`).
2. (Opcional) Active el entorno virtual si lo usa.
3. Ejecute:

```bash
python scripts/run_etl.py
```

- El ETL lee la configuración en `config/config.yaml` (rutas en `paths`, fuentes en `etl.sources`).
- Solo procesa **meses que aún no tienen Parquet** en `data/interim/`.

Para **reprocesar todos los meses** (sobrescribir lo que ya está en interim):

```bash
python scripts/run_etl.py --overwrite
```

Para usar otro archivo de configuración:

```bash
python scripts/run_etl.py --config otro.yaml
```

---

## 5. Ejecución alternativa: desde el notebook (poc/1_etl.ipynb)

Si prefiere ejecutar el ETL de forma **interactiva** (celda a celda, viendo el resumen de lo procesado), puede usar el notebook en la carpeta `poc/`:

1. Abra la terminal en la **raíz del proyecto** (donde están `config/`, `scripts/`, `poc/`, `data/`).
2. (Opcional) Active el entorno virtual si lo usa (por ejemplo: `.\qenv\Scripts\activate` en Windows).
3. Inicie Jupyter Lab: `jupyter lab` (o abra el proyecto en VS Code y abra el notebook desde ahí).
4. En el panel de archivos, entre en la carpeta **`poc`** y abra **`1_etl.ipynb`**.
5. Asegúrese de que el **kernel** usa el entorno del proyecto (donde están instaladas las dependencias).
6. Ejecute las celdas en orden (**Run All** o celda por celda con Shift+Enter).

El notebook carga la configuración desde **`config/config.yaml`** (rutas, `etl.sources`, `etl.overwrite`) y llama a la misma lógica que el script; el resultado es el mismo: los Parquet se escriben en `data/interim/` por fuente y año/mes. No es necesario duplicar parámetros: basta con tener los archivos raw en su sitio y la config correcta.

**Resumen:** (1) Archivos raw en `data/raw/` según esta guía. (2) Abrir `poc/1_etl.ipynb`. (3) Ejecutar todas las celdas. (4) Revisar los Parquet en `data/interim/`.

---

## 6. Dónde está el resultado

Tras una ejecución correcta:

- **Inspecciones:** `data/interim/inspecciones/year=AAAA/month=MM/inspecciones.parquet`
- **Consumo:** `data/interim/consumo/year=AAAA/month=MM/consumo.parquet`

Si en `config/config.yaml` está definido `paths.logs` (por ejemplo `data/logs`), cada ejecución del ETL escribe además un archivo de log (ej. `etl_20250214_120000.log`) para depuración.

---

## 7. Si algo falla

- **"Directorio ... no existe, saltando"**  
  Cree la carpeta correspondiente en `data/raw/` (por ejemplo `data/raw/inspecciones/`) y coloque ahí los archivos con el nombre `{fuente}_{AAAA}_{MM}.xlsx`.

- **Error al leer el Excel o columna no encontrada**  
  Compruebe que el archivo tiene las **columnas requeridas** con nombres que, una vez normalizados (minúsculas, sin tildes, espacios → `_`), coincidan con los de esta guía (p. ej. `contrato`, `fecha`, `resultado` para inspecciones; `ano`, `mes` para consumo).

- **"No hay meses pendientes"**  
  Todos los archivos que hay en `data/raw/` para esa fuente ya tienen su Parquet en `data/interim/`. Para volver a procesarlos, use `python scripts/run_etl.py --overwrite`.

Para más detalle sobre el flujo mensual y la inferencia, consulte [manual_usuario.md](manual_usuario.md).
