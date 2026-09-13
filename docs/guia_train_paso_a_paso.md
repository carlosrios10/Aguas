# Guía paso a paso: Cómo entrenar el modelo

Esta guía explica cómo ejecutar el **entrenamiento** una vez, en la sesión de entrega o cuando se decida reentrenar. No forma parte del uso mensual.

El uso de cada mes (ETL si hay archivos nuevos, y luego inferencia) está en [manual_usuario.md](manual_usuario.md). Antes del train tiene que haber corrido el ETL: [guia_etl_paso_a_paso.md](guia_etl_paso_a_paso.md). Después del train, los puntajes se obtienen con [guia_inferencia_paso_a_paso.md](guia_inferencia_paso_a_paso.md).

---

## ¿Qué hace el entrenamiento?

- Lee inspecciones, consumo y maestro ya procesados en `data/interim/`.
- Arma un dataset wide (ventana de consumo hacia atrás, por defecto 12 meses) y lo guarda en `data/processed/train/`.
- Usa dos archivos que se entregan en la sesión: `models/features.pkl` (columnas del modelo) y `models/hyperparams.pkl` (hiperparámetros). **No se generan en este repositorio.**
- Entrena LightGBM y escribe `models/lgbm_model.pkl`. Si ese archivo ya existía, lo reemplaza.

**No hace falta traer un modelo ya entrenado.** El que se usa en inferencia es el que produce este paso, con los datos de esta instalación.

---

## ¿Qué necesito antes de empezar?

1. **ETL ya ejecutado.**  
   Deben existir datos en `data/interim/inspecciones/`, `data/interim/consumo/` y `data/interim/maestro/`. Si alguna de esas carpetas está vacía, el train no arranca. Ver [guia_etl_paso_a_paso.md](guia_etl_paso_a_paso.md).

2. **Los dos archivos entregados, copiados en `models/`.**  
   Crear la carpeta `models/` en la raíz del proyecto si no existe, y copiar ahí:
   - `features.pkl`
   - `hyperparams.pkl`  
   Sin esos dos archivos el train falla. No hace falta `lgbm_model.pkl` antes de ejecutar.

3. **Ventana de consumo.**  
   Con `cant_periodos: 12`, un mes de inspección solo entra al train si hay al menos 12 meses de consumo anteriores. Si el consumo empieza en enero de 2022, la primera inspección usable es enero de 2023.

4. **Entorno listo.**  
   Python y dependencias instalados (`pip install -r requirements.txt`). Ver [setup.md](setup.md).

---

## Paso 1: Revisar el corte de train

1. Abre `config/config.yaml`.
2. Busca la sección **`train`** (no `inference`). Verás algo como:

   ```yaml
   train:
     cutoff: null
     cant_periodos: 12
     max_ctas_neg: 500
     sam_th: 0.6
     param_imb_method: "over"
     preprocesor_num: 1
   ```

3. **`cutoff`** es lo único que suele cambiarse en esta sesión:
   - `null`: usa todas las inspecciones que tengan 12 meses de consumo previo.
   - Una fecha `"YYYY-MM-DD"` (día 01): solo meses de inspección menores o iguales a esa fecha.

4. No cambies `cant_periodos`, `max_ctas_neg`, `sam_th`, `param_imb_method` ni `preprocesor_num` si no te lo indican. Esos valores tienen que coincidir con los `features.pkl` e `hyperparams.pkl` entregados.

5. `inference.cutoff` no lo usa el train. Se configura después, al puntuar un mes.

Guarda el archivo.

---

## Paso 2: Abrir la terminal en la raíz del proyecto

La terminal tiene que estar en la carpeta que contiene `config`, `scripts`, `data` y `models`.

- En Windows puedes abrir PowerShell o CMD desde esa carpeta, o usar `cd` hasta esa ruta.
- Ejemplo: `cd C:\ruta\al\proyecto`.

---

## Paso 3: Activar el entorno (si te lo indicaron)

Si el equipo te dijo que uses un entorno virtual, en este proyecto el nombre habitual es **`qenv`**:

- **Windows (PowerShell):**

  ```text
  .\qenv\Scripts\activate
  ```

- Verás `(qenv)` al inicio de la línea.

Si no usas entorno virtual, omite este paso.

---

## Paso 4: Ejecutar el train

En la misma terminal:

```text
python scripts/run_train.py
```

- Puede tardar bastante (construcción del dataset y variables de series de tiempo). No cierres la terminal.
- En pantalla verás `=== ENTRENAMIENTO ===`, el `CUTOFF` y los pasos 1/5 a 5/5.
- Si todo va bien, el último mensaje es **`Entrenamiento completado. Modelo guardado:`** seguido de la ruta de `lgbm_model.pkl`.
- También aparece `Dataset listo: N filas, target mean = ...`. Anota esas dos cifras: sirven para confirmar que el entrenamiento vio datos.

Si aparece un error, no vuelvas a lanzarlo cambiando parámetros. Anota el mensaje (o el log) y revísalo con la sección [Si algo falla](#si-algo-falla).

---

## Paso 5: Dónde está el resultado

| Qué | Dónde |
|-----|--------|
| Modelo para inferencia | `models/lgbm_model.pkl` |
| Dataset usado para entrenar | `data/processed/train/cutoff=<fecha>/train_wide.parquet` |
| Log de la ejecución | `data/logs/train_<AAAAMMDD>_<HHMMSS>.log` |

Si `train.cutoff` es `null`, la carpeta `cutoff=` usa la fecha de la última inspección que entró al train, no la palabra `null`.

A partir de aquí el flujo mensual es ETL (si hay archivos nuevos) e inferencia. El modelo no se vuelve a entrenar cada mes.

---

## Resumen rápido

| Paso | Qué hacer |
|------|-----------|
| 0 | ETL hecho (`data/interim/` con inspecciones, consumo y maestro). |
| 1 | Copiar `features.pkl` y `hyperparams.pkl` en `models/`. |
| 2 | En `config/config.yaml` → `train.cutoff`: `null` (todo) o una fecha tope `"YYYY-MM-01"`. |
| 3 | Terminal en la raíz del proyecto. Activar el entorno si aplica. |
| 4 | `python scripts/run_train.py` |
| 5 | Confirmar `models/lgbm_model.pkl` y el mensaje de entrenamiento completado. |

---

## Ejecución alternativa: notebook

El resultado es el mismo que el script. Hay que hacer antes el paso de copiar los `.pkl` y revisar `train.cutoff`.

1. Terminal en la raíz del proyecto. Activar el entorno si aplica (`.\qenv\Scripts\activate`).
2. Ejecutar `jupyter lab` y no cerrar esa terminal.
3. Abrir `poc/train.ipynb`.
4. **Run → Run All Cells** (o celda por celda con Shift+Enter).
5. Al final debe aparecer el mensaje de modelo guardado en `models/lgbm_model.pkl`.

---

## Si algo falla

- **No se pudo cargar la config.**  
  `config/config.yaml` tiene que existir. Si `train.cutoff` es una fecha, va entre comillas y en formato `YYYY-MM-DD` (por ejemplo `"2025-09-01"`). `null` se escribe sin comillas.

- **No se pudo crear el dataset de train** / **No hay datos en interim para inspecciones** / **No hay archivos de consumo** / **No hay maestro en interim.**  
  El ETL no corrió, o no corrió para esa fuente. Vuelve a [guia_etl_paso_a_paso.md](guia_etl_paso_a_paso.md) y confirma que existen parquets en las tres carpetas de `data/interim/`.

- **No hay fechas de corte válidas.**  
  Hay inspecciones, pero ninguna tiene los 12 meses de consumo previos (`cant_periodos`). Hace falta consumo más antiguo, o bajar el tope no ayuda: el tope solo recorta meses, no acorta la ventana.

- **No encuentra `features.pkl` o `hyperparams.pkl`.**  
  Esos archivos se entregan en la sesión. Tienen que estar en `models/`, con ese nombre exacto. El train no los crea.

- **Columnas faltantes en train_wide.**  
  `features.pkl` no coincide con el dataset que acaba de construirse (otra versión de features, o el ETL no dejó las columnas que el modelo espera). No sigas a inferencia con ese resultado.

- **Quiero repetir el train.**  
  Vuelve a ejecutar `python scripts/run_train.py`. Sobrescribe `models/lgbm_model.pkl`. La inferencia siguiente usa ese archivo nuevo.

---

**Documento:** Guía de entrenamiento paso a paso.  
**Proyecto:** AquaData (caj_poc). Después de este paso, ver [guia_inferencia_paso_a_paso.md](guia_inferencia_paso_a_paso.md).
