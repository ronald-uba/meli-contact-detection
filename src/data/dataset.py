"""
dataset.py
----------
Carga un CSV de listings y construye un HuggingFace Dataset multimodal.
Cada ejemplo incluye rutas a imágenes descargadas, texto del prompt y la respuesta esperada.
"""

import json
import os
import re
import requests
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset


PROMPT_TEMPLATE = """Analizá las imágenes y el texto de esta publicación de Mercado Libre.
Determiná si contiene datos de contacto como teléfonos, WhatsApp, URLs u otros medios de contacto directo.

Título: {title}

Descripción: {description}

Respondé SOLO con un JSON válido:
{{"resultado": "DC-adrede" | "DC-involuntario" | "DC-negativo", "explicacion": "..."}}"""


def download_images(
    urls: list[str],
    item_id: str,
    img_dir: Path,
    max_images: int = 10,
) -> list[str]:
    """Descarga hasta max_images imágenes y retorna sus paths locales."""
    img_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, url in enumerate(urls[:max_images]):
        ext = url.split(".")[-1].split("?")[0] or "jpg"
        dest = img_dir / f"{item_id}_{i}.{ext}"
        if not dest.exists():
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                dest.write_bytes(r.content)
            except Exception as e:
                print(f"⚠️  Error descargando {url}: {e}")
                continue
        paths.append(str(dest))
    return paths


def build_prompt(row: pd.Series, prompt_max_chars: int = 2500) -> str:
    title = str(row.get("ITE_ITEM_TITLE", ""))
    desc  = str(row.get("ITE_ITEM_DESCRIPTION", ""))[:prompt_max_chars]
    return PROMPT_TEMPLATE.format(title=title, description=desc)


def label_to_answer(result: int, explanation: str = "") -> str:
    label = "DC-adrede" if result == 1 else "DC-negativo"
    return json.dumps({"resultado": label, "explicacion": explanation}, ensure_ascii=False)


def csv_to_dataset(
    csv_path: str,
    img_dir: str,
    max_images: int = 10,
    prompt_max_chars: int = 2500,
    limit: Optional[int] = None,
) -> Dataset:
    """
    Lee un CSV de listings y construye un Dataset HF listo para training.

    Args:
        csv_path: Path al CSV con columnas item_id, RESULT, etc.
        img_dir: Directorio donde guardar las imágenes descargadas.
        max_images: Máximo de imágenes por listing.
        prompt_max_chars: Truncado del texto de descripción en el prompt.
        limit: Si se especifica, usa solo las primeras N filas (para debugging).
    """
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    img_dir = Path(img_dir)
    records = []

    for _, row in df.iterrows():
        item_id = str(row["item_id"])

        # Parsear URLs de imágenes
        try:
            pics = json.loads(row.get("pictures", "[]"))
            urls = [p["url"] for p in sorted(pics, key=lambda x: x.get("PIC_NRO", 0))]
        except Exception:
            urls = []

        paths = download_images(urls, item_id, img_dir, max_images)
        if not paths:
            continue

        records.append({
            "item_id":    item_id,
            "image_path": paths,
            "prompt_text": build_prompt(row, prompt_max_chars),
            "answer":     label_to_answer(int(row["RESULT"]), str(row.get("EXPLANATION", ""))),
            "label":      int(row["RESULT"]),
        })

    return Dataset.from_list(records)
