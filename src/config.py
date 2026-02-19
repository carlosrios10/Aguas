"""
Carga de configuración centralizada desde config/config.yaml.

Uso en notebooks o scripts:
  from src.config import load_config, get_paths
  cfg = load_config()
  paths = get_paths(cfg)
  INTERIM_DIR = paths["interim"]
  CUTOFF = cfg["train"]["cutoff"]
"""
import os
import yaml

# Raíz del proyecto (carpeta que contiene config/, src/, data/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_name="config.yaml"):
    """Carga el YAML de config. Por defecto busca config/config.yaml en la raíz del proyecto."""
    path = os.path.join(PROJECT_ROOT, "config", config_name)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths(cfg=None):
    """Devuelve un dict de rutas absolutas a partir de cfg['paths']. Si cfg es None, carga la config."""
    if cfg is None:
        cfg = load_config()
    return {
        k: os.path.join(PROJECT_ROOT, v)
        for k, v in cfg["paths"].items()
    }
