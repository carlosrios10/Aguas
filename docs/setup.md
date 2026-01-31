# Creación del entorno

Pasos para configurar el entorno (basados en *Creacion_Entorno.pdf*).

## Entornos donde se ejecutó el proyecto

- **AWS** — Instancia ml.m5.2xlarge  
  - Sistema operativo: Ubuntu 22.04.5 LTS (Jammy Jellyfish)  
  - Versión base: Debian  

- **Windows 11 Pro** — 32 GB RAM, Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz 2.59 GHz  

---

## Preparación del entorno en Windows

### Instalación de Pyenv

Para gestionar versiones de Python en Windows se recomienda usar **Pyenv**.  
Instalación: [https://github.com/pyenv-win/pyenv-win](https://github.com/pyenv-win/pyenv-win)

### Instalación de una versión de Python

Con Pyenv instalado, instala la versión de Python requerida:

```bash
pyenv install 3.10.11
```

### Creación de un entorno virtual con venv

**3.1 Seleccionar la versión de Python correcta**

Antes de crear el entorno virtual, asegúrate de que Pyenv use la versión instalada:

```bash
pyenv global 3.10.11
```

**3.2 Navegar al directorio del proyecto**

Ubícate en la raíz del proyecto (donde está `requirements.txt`):

```bash
cd C:\ruta\a\tu\proyecto
```

Si acabas de clonar el repo:

```bash
git clone https://github.com/TU_USUARIO/queretaro_poc.git
cd queretaro_poc
```

**3.3 Crear el entorno virtual**

```bash
python -m venv qenv
```

**3.4 Activar el entorno virtual**

En Windows (PowerShell o CMD):

```bash
.\qenv\Scripts\activate
```

**Verificación:** Al activar, deberías ver `(qenv)` al inicio de la línea de comandos.

### Instalación de dependencias

Con el entorno virtual activado:

```bash
pip install -r requirements.txt
```

### Lanzar Jupyter Lab

```bash
jupyter lab
```

Se abrirá una pestaña en el navegador con la interfaz de Jupyter Lab.

### Comprobar que todo funciona

Ejecutá un notebook de prueba para confirmar que las librerías funcionan correctamente:

- **Validacion_librerias.ipynb** (en la raíz del proyecto)

O, para probar el pipeline:

- Un notebook de `poc/` (por ejemplo `poc/1_etl.ipynb`) y una celda con `from src.data import etl`.

---

## Datos y modelos (no versionados)

Las carpetas `data/` y `models/` no se suben a Git. En una máquina nueva:

- **data/:** Copiar desde la máquina original o generar ejecutando el ETL si tenés los archivos raw.
- **models/:** Se crean al ejecutar `poc/3_train.ipynb`; si ya tenés un modelo, copiá los `.pkl` a `models/`.
