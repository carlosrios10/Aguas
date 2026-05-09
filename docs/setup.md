# Configuración del entorno Python

Guía para preparar el entorno de desarrollo y ejecución del pipeline (ETL, entrenamiento, inferencia). Úsala en máquinas locales o en servidores (por ejemplo, despliegue en EAAB).

## Requisitos

| Componente | Detalle |
|------------|---------|
| **Python** | **3.10.x** (recomendado **3.10.11**). El proyecto se probó con esa línea; versiones 3.11+ pueden funcionar pero no están garantizadas con las versiones fijadas en `requirements.txt`. |
| **Git** | Para clonar el repositorio. |
| **Espacio** | Reserva espacio para `data/` (raw, interim, processed) y `models/`; no van en Git. |

Raíz del proyecto: carpeta donde están `requirements.txt`, `config/`, `src/` y `scripts/`.

---

## Opción A — Windows con Pyenv-win (recomendado si gestionás varias versiones)

### 1. Instalar Pyenv-win

Instrucciones oficiales: [pyenv-win](https://github.com/pyenv-win/pyenv-win).

### 2. Instalar Python 3.10.11

```powershell
pyenv install 3.10.11
pyenv local 3.10.11
```

Si no usás `pyenv local`, podés fijar la versión global con `pyenv global 3.10.11`.

### 3. Crear y activar el entorno virtual

Desde la raíz del repo:

```powershell
python -m venv qenv
.\qenv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación por política de ejecución, en una sesión con permisos adecuados:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Alternativa en **CMD**:

```cmd
qenv\Scripts\activate.bat
```

Deberías ver el prefijo `(qenv)` en el prompt.

### 4. Instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. (Opcional) Herramientas de desarrollo y tests

```powershell
pip install -r requirements-dev.txt
```

---

## Opción B — Windows sin Pyenv (Python desde python.org)

1. Descargá e instalá [Python 3.10.11](https://www.python.org/downloads/release/python-31011/) (marca “Add python.exe to PATH” si el instalador lo ofrece).
2. En la raíz del proyecto:

```powershell
py -3.10 -m venv qenv
.\qenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si `py -3.10` no está disponible, usá `python -m venv qenv` siempre que `python --version` muestre 3.10.x.

---

## Linux / Ubuntu (servidor o WSL)

Ejemplo con el paquete `python3.10-venv` del sistema o Python 3.10 instalado:

```bash
cd /ruta/al/proyecto
python3.10 -m venv qenv
source qenv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Opcional:

```bash
pip install -r requirements-dev.txt
```

---

## Jupyter Lab

Con el entorno activado:

```bash
jupyter lab
```

Abrí los notebooks desde la raíz del proyecto (o configurá el kernel con esa raíz como directorio de trabajo) para que funcionen los imports `from src...` y la carga de `config/config.yaml`.

---

## Comprobar el entorno

| Comando | Qué valida |
|---------|------------|
| `python --version` | Debe mostrar **Python 3.10.x**. |
| `pip check` | Dependencias sin conflictos declarados. |
| `python -c "import lightgbm, pandas, tsfel; print('ok')"` | Imports críticos del pipeline. |
| `pytest` | Solo si instalaste `requirements-dev.txt`; ejecuta la suite en `tests/`. |

Prueba manual adicional:

- **Validacion_librerias.ipynb** (raíz del proyecto), o
- Una celda en `poc/1_etl.ipynb` con `from src.data import etl`.

---

## Datos y modelos (no versionados)

Las carpetas `data/` y `models/` no se suben a Git. En una máquina nueva:

- **data/**: copiá desde el origen acordado o generá ejecutando el ETL si tenés los archivos raw (`data/raw/` según `config/config.yaml` → `etl.sources`).
- **data/logs/**: se crea al ejecutar los scripts si en la config está definido `paths.logs`.
- **models/**: resultado de `scripts/run_train.py` o `poc/train.ipynb`; o copiá los `.pkl` (`lgbm_model.pkl`, `features.pkl`, `hyperparams.pkl`) si ya existen.

---

## Referencias de entornos donde se ejecutó el proyecto

- **AWS** — Instancia `ml.m5.2xlarge`, Ubuntu 22.04.5 LTS.
- **Windows 11 Pro** — 32 GB RAM, Intel Core i7-6700HQ.

Si documentás un entorno estándar para EAAB (VM, VDI, servidor interno), conviene añadir una fila aquí con SO, Python y política de red (proxy, mirrors de pip) para el equipo.
