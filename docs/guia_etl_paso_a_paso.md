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

Los nombres pueden ir en mayúsculas. El ETL hace `strip` y minúsculas. No convierte espacios a guión bajo: el nombre, en minúsculas, tiene que coincidir con el de esta guía. Si falta una columna, el ETL falla.

## 1. Archivos de inspecciones

### Columnas requeridas

| Columna | Tipo esperado | Descripción |
|---------|----------------|-------------|
| **matricula** | Texto | Identificador del contrato (ej. `1203639-0`). El ETL lo renombra a `contrato`. |
| **data_da_fiscalizacao** | Fecha | Fecha de la fiscalización. El ETL la lleva al primer día del mes. |
| **motivo** | Texto | Se usa para marcar fraude (by-pass, LA clandestina, corte ramal violado). |
| **hidrometro_invertido** | Texto | Fraude si el valor es `SIM`. |
| **situacao_dos_lacres_cavalete** | Texto | Fraude si es `Rompido` o `Sem lacre`. |
| **situacao_do_hidrometro** | Texto | Fraude si el hidrómetro no está en el cavalete, está dañado o hay que enviarlo a análisis. |
| **situacao_da_la_pelo_fiscal** | Texto | Fraude si es `Violada` (`is_fraud_5`). |
| **situacao_cavalete** | Texto | Fraude si es `Intervenção no cavalete` (`is_fraud_6`). |

### Ejemplo de contenido (inspecciones_2025_01.xlsx)

| matricula | data_da_fiscalizacao | motivo | hidrometro_invertido | situacao_dos_lacres_cavalete | situacao_do_hidrometro | situacao_da_la_pelo_fiscal | situacao_cavalete |
|-----------|----------------------|--------|----------------------|------------------------------|------------------------|----------------------------|-------------------|
| 1203639-0 | 15/01/2025 | Não | NÃO | Intacto | Normal | Normal | Normal |
| 1203640-0 | 20/01/2025 | Sim: By-pass | NÃO | Rompido | Normal | Normal | Normal |

- Si el mismo contrato aparece más de una vez en el mismo mes, el ETL deja una fila y se queda con el mayor `is_fraud` (1 prevalece sobre 0).

---

## 2. Archivos de consumo

### Columnas requeridas

| Columna | Tipo esperado | Descripción |
|---------|----------------|-------------|
| **matricula** | Texto | Identificador del contrato. El ETL lo renombra a `contrato`. |
| **mes_fatura** | Fecha | Mes de factura. El ETL lo renombra a `date`. |
| **consumo** | Número | Consumo. Textos no numéricos (por ejemplo `-`) se descartan. |
| **situacao_la** | Texto | Situación de la conexión. El ETL arma flags para `Ativa`, `Cancelada`, `Cortada Cavalete` y `Suprimida`. |

### Ejemplo de contenido (consumo_2025_01.xlsx)

| matricula | mes_fatura | consumo | situacao_la |
|-----------|------------|---------|-------------|
| 1203639-0 | 2025-01-01 | 25 | Ativa |
| 1203640-0 | 2025-01-01 | 10 | Cancelada |
| 1203641-0 | 2025-01-01 | 32 | Cortada Cavalete |

- Filas sin consumo numérico se eliminan. Si hay más de una fila por contrato y mes, se queda la primera.

---

## 3. Archivos de maestro

### Columnas requeridas

| Columna | Tipo esperado | Descripción |
|---------|----------------|-------------|
| **matricula** | Texto | Identificador del contrato. El ETL lo renombra a `contrato`. |
| **localizacao** | Texto | Localización. El preprocesamiento arma `ciudad_sector` con los tres primeros segmentos separados por punto. |
| **marca** | Texto | Marca del medidor. |
| **tipo_instalacao** | Texto | Tipo de instalación. |
| **tipo_cliente** | Texto | Tipo de cliente. Si el valor es numérico, el ETL lo deja nulo. |

### Ejemplo de contenido (maestro_2025_01.xlsx)

| matricula | localizacao | marca | tipo_instalacao | tipo_cliente |
|-----------|-------------|-------|-----------------|--------------|
| 1203639-0 | 01.01.03.001 | LAO | Cavalete | Residencial |
| 1203640-0 | 01.02.01.010 | SAGA | Cavalete | Comercial |

- El maestro no tiene fecha propia: el archivo mensual es una foto. Si un contrato aparece más de una vez, el ETL se queda con la primera fila.

---

## 4. Resumen: generar los archivos raw

1. **Crear las carpetas** `data/raw/inspecciones/`, `data/raw/consumo/`, `data/raw/maestro/` (si no existen).
2. **Exportar o generar** los Excel con las columnas indicadas para cada fuente.
3. **Nombrar cada archivo** exactamente: `inspecciones_AAAA_MM.xlsx`, `consumo_AAAA_MM.xlsx`, `maestro_AAAA_MM.xlsx`.
4. **Colocar cada archivo** en la carpeta de su fuente.
5. Revisar que **fechas y números** estén en formato coherente (fechas reconocibles, consumo numérico, `matricula` como texto).

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

## 6. Ejecución alternativa: desde el notebook (poc/etl.ipynb)

Si prefiere ejecutar el ETL de forma **interactiva** (celda a celda, viendo el resumen de lo procesado), puede usar el notebook en la carpeta `poc/`:

1. Abra la terminal en la **raíz del proyecto** (donde están `config/`, `scripts/`, `poc/`, `data/`).
2. (Opcional) Active el entorno virtual si lo usa (en este proyecto: `.\qenv\Scripts\activate` en Windows).
3. Inicie Jupyter Lab: `jupyter lab` (o abra el proyecto en VS Code y abra el notebook desde ahí).
4. En el panel de archivos, entre en la carpeta **`poc`** y abra **`etl.ipynb`**.
5. Asegúrese de que el **kernel** usa el entorno del proyecto (donde están instaladas las dependencias).
6. Ejecute las celdas en orden (**Run All** o celda por celda con Shift+Enter).

El notebook carga la configuración desde **`config/config.yaml`** (rutas, `etl.sources`, `etl.overwrite`) y llama a la misma lógica que el script; el resultado es el mismo: los Parquet se escriben en `data/interim/` por fuente y año/mes. No es necesario duplicar parámetros: basta con tener los archivos raw en su sitio y la config correcta.

**Resumen:** (1) Archivos raw en `data/raw/` según esta guía. (2) Abrir `poc/etl.ipynb`. (3) Ejecutar todas las celdas. (4) Revisar los Parquet en `data/interim/`.

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
  Compruebe que el archivo tiene las **columnas requeridas** de esta guía. El ETL solo hace strip y minúsculas (en maestro, también quita tildes). Un espacio en el nombre no se convierte en `_`. Ejemplos: `matricula` y `data_da_fiscalizacao` en inspecciones; `mes_fatura` y `situacao_la` en consumo.

- **"No hay meses pendientes"**  
  Todos los archivos que hay en `data/raw/` para esa fuente ya tienen su Parquet en `data/interim/`. Para volver a procesarlos, use `python scripts/run_etl.py --overwrite`.

- **Maestro obligatorio para inferencia**  
  La inferencia necesita el maestro en `data/interim/maestro/`. Asegúrese de tener al menos un archivo en `data/raw/maestro/` (p. ej. `maestro_2025_01.xlsx`) y de que la fuente `maestro` esté en `etl.sources` en `config/config.yaml`, y ejecute el ETL.

Para más detalle sobre el flujo mensual y la inferencia, consulte [manual_usuario.md](manual_usuario.md).
