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

```
data/raw/
├── inspecciones/
│   ├── inspecciones_2025_01.xlsx
│   ├── inspecciones_2025_02.xlsx
│   └── ...
├── consumo/
│   ├── consumo_2025_01.xlsx
│   ├── consumo_2025_02.xlsx
│   └── ...
└── maestro/
    ├── maestro_2025_01.xlsx
    └── ...
```

**Nombre del archivo:** `{fuente}_{AAAA}_{MM}.xlsx`  
- **fuente:** `inspecciones`, `consumo` o `maestro`  
- **AAAA:** año con 4 dígitos (ej. 2025)  
- **MM:** mes con 2 dígitos (ej. 01, 02, 12)

Si en `config/config.yaml` la sección `etl.sources` incluye solo `inspecciones` y `consumo`, no es obligatorio tener `maestro/` para el ETL de esas fuentes; pero **para inferencia el maestro es obligatorio** (debe estar en `raw` y haberse procesado con el ETL).

---

## 1. Archivos de inspecciones

### Columnas requeridas

| Columna    | Tipo esperado | Descripción |
|------------|----------------|-------------|
| **contrato** | Texto o número | Identificador del contrato. Puede incluir punto (ej. `12345.0`); el ETL toma la parte entera. |
| **fecha**    | Fecha          | Fecha de la inspección. Se normaliza al **primer día del mes**. |
| **resultado**| Número (0 o 1) | Resultado de la inspección: **1** = fraude/irregularidad, **0** (u otro) = sin fraude. |
| **observacion** | Opcional   | Texto; puede estar vacío. |

Los nombres de columna pueden estar en mayúsculas o con tildes en el Excel; el ETL los normaliza (minúsculas, sin tildes, espacios → guión bajo).

### Ejemplo de contenido (inspecciones_2025_01.xlsx)

| contrato | fecha       | resultado | observacion   |
|----------|-------------|-----------|---------------|
| 100001   | 15/01/2025  | 0         | Sin novedad   |
| 100002   | 20/01/2025  | 1         | Medidor alterado |
| 100003   | 10/01/2025  | 0         |               |

- Una fila por inspección. Si un mismo contrato aparece varias veces en el mismo mes, el ETL deja una fila por (contrato, mes) quedándose con el máximo de resultado (1 prevalece sobre 0).

---

## 2. Archivos de consumo

### Columnas requeridas

| Columna     | Tipo esperado | Descripción |
|-------------|----------------|-------------|
| **contrato** | Texto o número | Identificador del contrato. |
| **ano**      | Número         | Año del periodo de consumo (4 dígitos, ej. 2025). |
| **mes**      | Número         | Mes (1 a 12). |
| **consumo**  | Número         | Consumo medido (entero o decimal; si es texto con coma, se toma la parte entera). Valores negativos se eliminan. |
| **funcion**  | Texto          | Clasificación del registro; puede estar vacío. |
| **causa**    | Número o texto | Código causa; se rellena con 0 si falta. |
| **observacion** | Número o texto | Código observación; se rellena con 0 si falta. |

### Ejemplo de contenido (consumo_2025_01.xlsx)

| contrato | ano | mes | consumo | funcion | causa | observacion |
|----------|-----|-----|---------|---------|-------|-------------|
| 100001   | 2025 | 1  | 25      | NORMAL  | 0     | 0           |
| 100001   | 2025 | 1  | 10      | AJUSTE  | 0     | 0           |
| 100002   | 2025 | 1  | 32      | NORMAL  | 0     | 0           |
| 100003   | 2025 | 1  | 18      | NORMAL  | 0     | 0           |

- El ETL ordena por (contrato, año, mes, funcion, consumo) y deja **una fila por (contrato, mes)** (se mantiene la primera según ese orden). Así se evitan duplicados por contrato y periodo.

---

## 3. Archivos de maestro

### Columnas requeridas

| Columna        | Tipo esperado | Descripción |
|----------------|----------------|-------------|
| **contrato**   | Texto o número | Identificador del contrato (único por fila). |
| **categoria**  | Texto          | Categoría del contrato (ej. residencial, comercial). Obligatorio para el modelo; se rellena "sin_dato" si falta. |
| **diametro**   | Número         | Diámetro; puede ser decimal. |
| **estrato**    | Texto          | Estrato; se rellena "sin_dato" si falta. |
| **barrio_comuna** | Texto       | Barrio o comuna. Si en el Excel se llama "barrio comuna" (con espacio), el ETL lo normaliza a `barrio_comuna`. |
| **ciclo**      | Texto          | Ciclo de facturación. |
| **localidad**  | Texto          | Localidad. |
| **medidor**    | Texto          | Tipo de medidor; se rellena "sin_dato" si falta. |

Si alguna de estas columnas no existe en el Excel, el ETL puede fallar o rellenar con valor por defecto según el código; es recomendable incluir al menos **contrato** y **categoria**.

### Ejemplo de contenido (maestro_2025_01.xlsx)

| contrato | categoria   | diametro | estrato | barrio_comuna | ciclo | localidad | medidor |
|----------|-------------|----------|---------|---------------|-------|-----------|---------|
| 100001   | residencial | 19       | 2       | Centro        | 14    | Norte     | M1      |
| 100002   | comercial   | 25       | 3       | Granada       | 14    | Sur       | M2      |
| 100003   | residencial | 19       | 1       | Paso del Comercio | 16 | Este      | M1      |

- Debe haber **una fila por contrato**. Si un contrato aparece más de una vez, el ETL se queda con la última fila.

---

## 4. Resumen: generar los archivos raw

1. **Crear las carpetas** `data/raw/inspecciones/`, `data/raw/consumo/`, `data/raw/maestro/` (si no existen).
2. **Exportar o generar** los Excel con las columnas indicadas para cada fuente.
3. **Nombrar cada archivo** exactamente: `inspecciones_AAAA_MM.xlsx`, `consumo_AAAA_MM.xlsx`, `maestro_AAAA_MM.xlsx`.
4. **Colocar cada archivo** en la carpeta de su fuente.
5. Revisar que **fechas y números** estén en formato coherente (fechas reconocibles por Excel, consumo numérico, resultado 0/1 en inspecciones).

---

## 5. Ejecutar el ETL (desde la raíz del proyecto)

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

## 6. Ejecución alternativa: desde el notebook (poc/1_etl.ipynb)

Si prefiere ejecutar el ETL de forma **interactiva** (celda a celda, viendo el resumen de lo procesado), puede usar el notebook en la carpeta `poc/`:

1. Abra la terminal en la **raíz del proyecto** (donde están `config/`, `scripts/`, `poc/`, `data/`).
2. (Opcional) Active el entorno virtual si lo usa (por ejemplo: `.\.venv\Scripts\activate` en Windows).
3. Inicie Jupyter Lab: `jupyter lab` (o abra el proyecto en VS Code y abra el notebook desde ahí).
4. En el panel de archivos, entre en la carpeta **`poc`** y abra **`1_etl.ipynb`**.
5. Asegúrese de que el **kernel** usa el entorno del proyecto (donde están instaladas las dependencias).
6. Ejecute las celdas en orden (**Run All** o celda por celda con Shift+Enter).

El notebook carga la configuración desde **`config/config.yaml`** (rutas, `etl.sources`, `etl.overwrite`) y llama a la misma lógica que el script; el resultado es el mismo: los Parquet se escriben en `data/interim/` por fuente y año/mes. No es necesario duplicar parámetros: basta con tener los archivos raw en su sitio y la config correcta.

**Resumen:** (1) Archivos raw en `data/raw/` según esta guía. (2) Abrir `poc/1_etl.ipynb`. (3) Ejecutar todas las celdas. (4) Revisar los Parquet en `data/interim/`.

---

## 7. Dónde está el resultado

Tras una ejecución correcta:

- **Inspecciones:** `data/interim/inspecciones/year=AAAA/month=MM/inspecciones.parquet`
- **Consumo:** `data/interim/consumo/year=AAAA/month=MM/consumo.parquet`
- **Maestro:** `data/interim/maestro/year=AAAA/month=MM/maestro.parquet`

Si en `config/config.yaml` está definido `paths.logs` (por ejemplo `data/logs`), cada ejecución del ETL escribe además un archivo de log (ej. `etl_20250214_120000.log`) para depuración.

---

## 8. Si algo falla

- **"Directorio ... no existe, saltando"**  
  Cree la carpeta correspondiente en `data/raw/` (por ejemplo `data/raw/inspecciones/`) y coloque ahí los archivos con el nombre `{fuente}_{AAAA}_{MM}.xlsx`.

- **Error al leer el Excel o columna no encontrada**  
  Compruebe que el archivo tiene las **columnas requeridas** con nombres que, una vez normalizados (minúsculas, sin tildes, espacios → `_`), coincidan con los de esta guía (p. ej. `contrato`, `fecha`, `resultado` para inspecciones; `ano`, `mes` para consumo).

- **"No hay meses pendientes"**  
  Todos los archivos que hay en `data/raw/` para esa fuente ya tienen su Parquet en `data/interim/`. Para volver a procesarlos, use `python scripts/run_etl.py --overwrite`.

- **Maestro obligatorio para inferencia**  
  La inferencia necesita el maestro en `data/interim/maestro/`. Asegúrese de tener al menos un archivo en `data/raw/maestro/` (p. ej. `maestro_2025_01.xlsx`) y de que la fuente `maestro` esté en `etl.sources` en `config/config.yaml`, y ejecute el ETL.

Para más detalle sobre el flujo mensual y la inferencia, consulte [manual_usuario.md](manual_usuario.md).
