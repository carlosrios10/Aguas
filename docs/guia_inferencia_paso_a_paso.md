# Guía paso a paso: Cómo obtener la lista de puntajes de riesgo (inferencia)

Esta guía está pensada para **personas no técnicas**. Explica cómo ejecutar el proceso que genera el archivo con los **puntajes de riesgo** de cada contrato, para poder **priorizar inspecciones** (empezar por los de mayor riesgo).

---

## ¿Qué es la “inferencia” y para qué sirve?

- **Inferencia** es el proceso que usa el modelo ya entrenado para **asignar un puntaje (score)** a cada contrato.
- Ese puntaje indica la **probabilidad de riesgo** (por ejemplo, de fraude o anomalía): un valor entre 0 y 1.
- **Uso práctico:** ordenar los contratos de mayor a menor puntaje y planificar las inspecciones empezando por los de mayor riesgo.

**No hace falta entender cómo funciona el modelo.** Solo hay que seguir los pasos para obtener el archivo de resultados.

---

## ¿Qué necesito antes de empezar?

1. **Datos preparados**  
   Los archivos de consumo e inspecciones deben estar en las carpetas que usa el proyecto (normalmente alguien del equipo técnico ya habrá ejecutado el ETL).

2. **Modelo entrenado**  
   Debe existir un modelo guardado (por ejemplo, el equipo habrá ejecutado antes el proceso de entrenamiento). Si no, la inferencia fallará y tendrás que pedir que entrenen el modelo primero.

3. **Entorno listo**  
   El proyecto debe estar instalado en tu PC o en el servidor donde lo ejecutes (Python, dependencias). Si no estás seguro, pide ayuda al equipo técnico para la primera vez.

---

## Paso 1: Indicar el mes que quieres puntuar

El sistema necesita saber **para qué mes** quieres los puntajes (por ejemplo, enero 2025).

1. Abre la carpeta del proyecto en tu computadora.
2. Entra en la carpeta **`config`**.
3. Abre el archivo **`config.yaml`** con un editor de texto (Bloc de notas, Notepad++, VS Code, etc.).
4. Busca la línea que dice **`cutoff`** dentro de la sección **`inference`**. Verás algo como:
   ```yaml
   inference:
     cutoff: "2025-01-01"  # mes a predecir
   ```
5. **Cambia solo la fecha** por el mes que quieras. Usa siempre el formato **año-mes-día**, y el día **01**:
   - Enero 2025 → `"2025-01-01"`
   - Febrero 2025 → `"2025-02-01"`
   - Marzo 2025 → `"2025-03-01"`
6. Guarda el archivo y ciérralo.

**Resumen:** Solo tocas la fecha en `cutoff`. No modifiques el resto del archivo si no te han indicado lo contrario.

---

## Paso 2: Abrir la terminal (línea de comandos)

- **En Windows:**  
  - Puedes abrir **PowerShell** o **Símbolo del sistema** (CMD).  
  - O en el Explorador de archivos, escribir en la barra de direcciones `cmd` o `powershell` y pulsar Enter (se abrirá en esa carpeta).

- **Importante:** Tienes que estar en la **carpeta raíz del proyecto** (la que contiene las carpetas `config`, `scripts`, `data`, etc.).  
  - Si abriste la terminal desde otra carpeta, escribe algo como (ajusta la ruta a tu PC):
    ```text
    cd C:\ruta\donde\esta\el\proyecto\queretaro_poc
    ```
  - y pulsa Enter.

---

## Paso 3: Activar el entorno (si te lo han indicado)

Si el equipo te dijo que uses un “entorno virtual” (venv):

- **Windows (PowerShell):**
  ```text
  .\qenv\Scripts\activate
  ```
  (o el nombre de la carpeta del entorno que te hayan dado).

- Verás que aparece el nombre del entorno al inicio de la línea; entonces ya puedes pasar al paso 4.

Si no usas entorno virtual, omite este paso.

---

## Paso 4: Ejecutar el proceso de inferencia

En la misma terminal, escribe exactamente:

```text
python scripts/run_inference.py
```

y pulsa **Enter**.

- El proceso puede tardar unos minutos según la cantidad de datos.
- Verás mensajes en pantalla indicando el avance (por ejemplo: “Paso 1/5”, “Paso 2/5”, etc.).
- Si todo va bien, al final verás un mensaje indicando que se guardó el archivo de resultados.

**Si aparece un error:** anota el mensaje completo o haz una captura de pantalla y compártela con el equipo técnico. No cambies otras cosas en el proyecto sin su guía.

---

## Paso 5: Dónde está el archivo de resultados

Al terminar correctamente, el sistema guarda un archivo con los puntajes:

- **Carpeta:** `data/predictions/`
- **Nombre del archivo:** `scores_AAAA-MM-DD.csv`  
  La fecha es la que pusiste en `cutoff`. Por ejemplo, si usaste `"2025-01-01"`, el archivo se llamará **`scores_2025-01-01.csv`**.

Puedes abrirlo con **Excel** o **LibreOffice Calc** (o cualquier hoja de cálculo).

---

## Paso 6: Qué contiene el archivo y cómo usarlo

El archivo tiene al menos dos columnas (**contrato** y **score**). El equipo técnico puede configurar columnas adicionales (por ejemplo colonia, municipio, localidad, tipo de servicio) en la configuración del proyecto; en ese caso aparecerán también en el CSV.

| Columna   | Significado |
|----------|-------------|
| **contrato** | Identificador del contrato. |
| **score**    | Puntaje de riesgo (entre 0 y 1). **Mayor valor = mayor riesgo.** |

**Uso recomendado:**

1. Ordenar la tabla por la columna **score** de **mayor a menor**.
2. Los contratos que quedan arriba son los de **mayor riesgo**.
3. Esos son los que conviene **priorizar para inspección** (según la capacidad del equipo y la política de la empresa).

El puntaje es una **herramienta de apoyo a la decisión**, no reemplaza el criterio del responsable: se usa para ordenar y priorizar, no como “sí o no” automático.

---

## Resumen rápido

| Paso | Qué hacer |
|------|-----------|
| 1 | En `config/config.yaml`, cambiar **solo** la fecha `cutoff` al mes que quieras (formato AAAA-MM-01). |
| 2 | Abrir la terminal en la carpeta raíz del proyecto. |
| 3 | (Opcional) Activar el entorno virtual si te lo indicaron. |
| 4 | Ejecutar: `python scripts/run_inference.py` |
| 5 | Buscar el archivo en `data/predictions/scores_AAAA-MM-DD.csv`. |
| 6 | Abrirlo en Excel, ordenar por **score** de mayor a menor y usar esa lista para priorizar inspecciones. |

---

## Si algo falla

- **“No se pudo cargar la config”**  
  Revisa que el archivo `config/config.yaml` exista y que la fecha en `cutoff` esté entre comillas y con formato `AAAA-MM-DD` (por ejemplo `"2025-01-01"`).

- **“No se pudo crear el dataset”**  
  Puede que falten datos en las carpetas que usa el ETL. Pasa el mensaje de error al equipo técnico para que revisen las carpetas `data/raw` y `data/interim`.

- **“Columnas faltantes” o “modelo no encontrado”**  
  Normalmente significa que el modelo aún no se ha entrenado o que hay que volver a entrenar. Contacta al equipo técnico.

- **No encuentro la carpeta `data/predictions`**  
  Puede estar dentro de la carpeta del proyecto con otro nombre si lo han configurado distinto. Revisa en `config/config.yaml` la ruta que pone en `paths` → `predictions`; esa es la carpeta donde se guardan los CSV.

---

**Documento:** Guía de inferencia paso a paso (personas no técnicas)  
**Proyecto:** AquaData – queretaro_poc
