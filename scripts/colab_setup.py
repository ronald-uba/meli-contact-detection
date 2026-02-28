"""
colab_setup.py
--------------
Setup automático del entorno en Google Colab.
Montar Drive, clonar/actualizar el repo, instalar dependencias.

Uso (primera celda del notebook):
    exec(open("/content/drive/MyDrive/contact-detection/scripts/colab_setup.py").read())
"""

import os
import subprocess
import sys


# ── 1. Montar Google Drive ────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

# ── 2. Paths en Drive ─────────────────────────────────────────────────────────
DRIVE_ROOT  = "/content/drive/MyDrive/contact-detection"
DATA_DIR    = f"{DRIVE_ROOT}/data"
POOLS_DIR   = f"{DATA_DIR}/pools"
SPLITS_DIR  = f"{DATA_DIR}/splits"
OUTPUTS_DIR = f"{DRIVE_ROOT}/outputs"
LOGS_DIR    = f"{DRIVE_ROOT}/logs"

for d in [DATA_DIR, POOLS_DIR, SPLITS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 3. Clonar o actualizar repo ───────────────────────────────────────────────
REPO_URL = "https://github.com/ronald-uba/meli-contact-detection.git"
REPO_DIR = "/content/meli-contact-detection"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print(f"✅ Repo clonado en {REPO_DIR}")
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
    print(f"✅ Repo actualizado en {REPO_DIR}")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# ── 4. Instalar dependencias ──────────────────────────────────────────────────
subprocess.run([
    "pip", "install", "-q",
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    "transformers", "datasets", "peft", "trl", "pyyaml",
], check=True)
print("✅ Dependencias instaladas")

# ── 5. Exportar paths como variables de entorno ───────────────────────────────
os.environ["DRIVE_ROOT"]  = DRIVE_ROOT
os.environ["DATA_DIR"]    = DATA_DIR
os.environ["POOLS_DIR"]   = POOLS_DIR
os.environ["SPLITS_DIR"]  = SPLITS_DIR
os.environ["OUTPUTS_DIR"] = OUTPUTS_DIR
os.environ["LOGS_DIR"]    = LOGS_DIR

print(f"""
Setup completo:
  DRIVE_ROOT  = {DRIVE_ROOT}
  POOLS_DIR   = {POOLS_DIR}
  SPLITS_DIR  = {SPLITS_DIR}
  OUTPUTS_DIR = {OUTPUTS_DIR}
  REPO_DIR    = {REPO_DIR}
""")
