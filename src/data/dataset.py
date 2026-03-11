"""
dataset.py
----------
Carga un CSV de listings MeLi y construye un HuggingFace Dataset multimodal.
Cada ejemplo incluye rutas a imágenes descargadas, texto del prompt y la respuesta esperada.
"""

import ast
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from datasets import Dataset
from PIL import Image

from src.data.csv_reader import read_pool_csv


# ── Constantes ────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "image/*,*/*;q=0.8",
    "Referer": "https://www.mercadolibre.com/",
}

PROMPT_TEMPLATE = (
    "You are detecting contact data or evasion signals in marketplace listings.\n"
    "Return ONLY valid JSON (no markdown, no code fences, no extra text).\n"
    "Keys: has_contact_data (0 or 1), source_field (list of strings), reason_short (string).\n\n"
    "Mark has_contact_data=1 if the listing contains any of the following:\n"
    "- Phone numbers — including obfuscated forms: digits replaced by words, Roman numerals, "
    "objects, fruits, or measurements; spacing/dashes between digits; e.g. '1-1 cincuenta-cinco "
    "treinta' or 'once doce trece'. Also includes lottery/charada codes where animals or objects "
    "represent digits (e.g. Argentine quiniela: 'caballo gato toro' = 1 5 7; Brazilian jogo do "
    "bicho animal sequences). Includes WhatsApp or any messaging number.\n"
    "- Emails or messaging platform user IDs (Skype, Discord, Telegram, etc.).\n"
    "- External URLs or domains — including obfuscated/non-clickable forms such as "
    "'empresa . com . ar', 'usemodeladores com br', 'instagram dot com slash store1', "
    "or any domain with LATAM TLDs (.ar, .uy, .cl, .ec, .mx, .br, .co, .pe, .ve, etc.) "
    "not belonging to Mercado Libre, Mercado Pago, or Mercado Shops.\n"
    "- Social media handles — ONLY if an explicit handle, username, or direct link is provided "
    "(e.g. @store, instagram.com/store123, 'IG store123', 'FB companyname'). "
    "Exception: treat 'Follow us / search us on social media: <name>' as contact data.\n"
    "- Physical addresses or explicit invitations to contact outside Mercado Libre.\n\n"
    "Do NOT mark as contact data:\n"
    "- Links to Mercado Libre, Mercado Livre, Mercado Pago, Mercado Shops, or mlstatic.com.\n"
    "- Numeric codes with known technical formats: SKU, GTIN, EAN, ISBN, chassis/VIN, "
    "barcode, ZIP/CEP, product model/size codes, serial numbers, internal references, "
    "tracking numbers.\n"
    "- Brand names, logos, or watermarks visible on product images.\n"
    "- Products that ARE templates (invitations, business cards, labels) — "
    "placeholder contact in the design is not evasion.\n"
    "- Generic 'visit our page / channel' mentions without a handle, username, or link.\n"
    "- Generic delivery mentions without an external contact channel.\n\n"
    "Title: {title}\n"
    "Description: {description}\n"
    "Attributes: {attributes}\n"
)


# ── Helpers de parseo ─────────────────────────────────────────────────────────

def _safe_json(s) -> Optional[object]:
    """Parsea string/dict/list sin lanzar excepciones."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    if isinstance(s, (dict, list)):
        return s
    s = str(s).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _attributes_to_text(attributes_cell, max_pairs: int = 30) -> str:
    """Convierte la columna 'attributes' a un string 'name: val | name: val ...'"""
    obj = _safe_json(attributes_cell)
    if not isinstance(obj, list):
        return ""
    pairs = []
    for a in obj[:max_pairs]:
        if not isinstance(a, dict):
            continue
        name = a.get("attribute_name") or a.get("attribute_id")
        val  = a.get("value_name") or a.get("value_id")
        if name and val:
            pairs.append(f"{name}: {val}")
    return " | ".join(pairs)


def _shorten_explanation(explanation_cell, max_chars: int = 220) -> str:
    obj = _safe_json(explanation_cell)
    if isinstance(obj, list) and obj:
        s = str(obj[0])
    elif obj is None:
        s = str(explanation_cell) if explanation_cell is not None else ""
    else:
        s = str(obj)
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:max_chars] + "...") if len(s) > max_chars else s


def _parse_source_field(sf) -> list[str]:
    sf = "" if sf is None or (isinstance(sf, float) and pd.isna(sf)) else str(sf)
    parts = re.split(r"[,\+;/]| and ", sf)
    out = [p.strip().title() for p in parts if p.strip()]
    return out if out else ["Unknown"]


# ── Selección de URLs de imágenes ─────────────────────────────────────────────

def _extract_urls_from_pictures(pictures_cell) -> list[str]:
    """Parsea la columna 'pictures' y retorna todas las URLs ordenadas por PIC_NRO."""
    obj = _safe_json(pictures_cell)
    if obj is None:
        return []
    if isinstance(obj, dict) and "pictures" in obj:
        pics = obj["pictures"]
    elif isinstance(obj, list):
        pics = obj
    else:
        return []
    if not isinstance(pics, list):
        return []

    def pic_key(p):
        try:
            return int(p.get("PIC_NRO", 10**9))
        except Exception:
            return 10**9

    urls = []
    for p in sorted(pics, key=lambda p: pic_key(p) if isinstance(p, dict) else 10**9):
        if not isinstance(p, dict):
            continue
        u = p.get("link_pic_id") or p.get("url")
        if isinstance(u, str) and u.startswith("http"):
            urls.append(u)
    return urls


def _extract_urls_from_pictures_dp(pictures_dp_cell) -> list[str]:
    """Parsea la columna 'pictures_dp' (imágenes con DC detectado)."""
    obj = _safe_json(pictures_dp_cell)
    if isinstance(obj, list):
        return [u for u in obj if isinstance(u, str) and u.startswith("http")]
    return []


def _select_keep_first_last(urls: list[str], k: int = 10, seed: int = 42) -> list[str]:
    """
    Retorna hasta k URLs.
    Si len(urls) > k: garantiza primera y última; muestrea k-2 del medio.
    """
    urls = [u for u in urls if isinstance(u, str) and u.startswith("http")]
    if len(urls) <= k:
        return urls
    first, last, middle = urls[0], urls[-1], urls[1:-1]
    rnd = random.Random(seed)
    need = k - 2
    picked_mid = rnd.sample(middle, min(len(middle), need))
    mid_set = set(picked_mid)
    picked_mid_ordered = [u for u in middle if u in mid_set]
    return [first] + picked_mid_ordered + [last]


def pick_image_urls(row: pd.Series, max_images: int = 10, seed: int = 42) -> list[str]:
    """
    Selecciona hasta max_images URLs para un listing.
    - Positivos (RESULT=1): usa pictures_dp si existe, si no pictures.
    - Negativos (RESULT=0): usa pictures.
    Aplica estrategia first+last para listings con muchas imágenes.
    """
    is_positive = int(row.get("RESULT", 0)) == 1
    urls = []

    if is_positive:
        dp_urls = _extract_urls_from_pictures_dp(row.get("pictures_dp"))
        urls = dp_urls if dp_urls else _extract_urls_from_pictures(row.get("pictures"))
    else:
        urls = _extract_urls_from_pictures(row.get("pictures"))

    return _select_keep_first_last(urls, k=max_images, seed=seed)


# ── Descarga y guardado de imágenes ──────────────────────────────────────────

def _download_image(url: str, timeout: int = 25) -> Optional[Image.Image]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def _save_resized(img: Image.Image, out_path: Path, max_side: int = 512, quality: int = 90) -> None:
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    nw, nh = int(w * scale), int(h * scale)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=quality, optimize=True)


# ── Builder principal ─────────────────────────────────────────────────────────

def _download_one(args: tuple) -> Optional[str]:
    """Descarga y guarda una imagen. Retorna el path local o None si falla."""
    url, dest, img_max_side = args
    try:
        if dest.exists():
            return str(dest)
    except OSError:
        pass  # Drive FUSE I/O error — intentar descargar igual
    img = _download_image(url)
    if img is None:
        return None
    try:
        _save_resized(img, dest, max_side=img_max_side)
    except OSError:
        return None
    return str(dest)


def csv_to_dataset(
    csv_path: str,
    img_dir: str,
    max_images: int = 10,
    img_max_side: int = 512,
    prompt_max_chars: int = 2500,
    seed: int = 42,
    limit: Optional[int] = None,
    n_download_workers: int = 16,
    verbose: bool = True,
) -> Dataset:
    """
    Lee un CSV de listings MeLi y construye un Dataset HF listo para training.

    Args:
        csv_path            : Path al CSV con columnas item_id, RESULT, pictures, etc.
        img_dir             : Directorio donde guardar las imágenes (persistente en Drive).
        max_images          : Máximo de imágenes por listing.
        img_max_side        : Lado máximo al que redimensionar (píxeles).
        prompt_max_chars    : Truncado de la descripción en el prompt.
        seed                : Semilla para selección de imágenes.
        limit               : Si se especifica, usa solo las primeras N filas (debugging).
        n_download_workers  : Threads para descarga paralela de imágenes.
        verbose             : Imprimir progreso.
    """
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    img_dir = Path(img_dir)

    # Columnas de auditoría presentes en los CSVs reales (no se usan en training)
    # RESULT_4_1_MINI, RESULT_4_1, tiene_41 → se ignoran, el label es RESULT
    # EXPLANATION ya contiene la explicación del mejor modelo disponible (4.1 > Mini)

    # ── Fase 1: recopilar todas las tareas de descarga ────────────────────────
    # tasks[item_id] = [(url, dest, img_max_side), ...]
    tasks_by_item: dict[str, list[tuple]] = {}
    rows_by_item: dict[str, pd.Series] = {}

    n_rows = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        item_id = str(row["item_id"])
        urls    = pick_image_urls(row, max_images=max_images, seed=seed)
        tasks   = [
            (url, img_dir / item_id / f"img_{i:02d}.jpg", img_max_side)
            for i, url in enumerate(urls, start=1)
        ]
        tasks_by_item[item_id] = tasks
        rows_by_item[item_id]  = row
        if verbose and (i + 1) % 10_000 == 0:
            print(f"  Fase 1: {i + 1:,}/{n_rows:,} filas procesadas...")

    if verbose:
        print(f"  Fase 1 completa: {len(tasks_by_item):,} items, {sum(len(t) for t in tasks_by_item.values()):,} tareas")

    # ── Fase 2: descargar en paralelo ─────────────────────────────────────────
    all_tasks = [(url, dest, ms) for tasks in tasks_by_item.values() for url, dest, ms in tasks]
    n_total   = len(all_tasks)

    # Chequeo de caché en paralelo (Drive FUSE: ~8ms/stat × 660k = 90min secuencial → <2min paralelo)
    # Usa 32 workers para maximizar concurrencia en operaciones I/O-bound
    _cache_workers = max(32, n_download_workers)
    if verbose:
        print(f"  Chequeando caché ({n_total:,} paths, {_cache_workers} workers)...")

    def _exists(dest):
        try:
            return dest.exists()
        except OSError:
            return False

    with ThreadPoolExecutor(max_workers=_cache_workers) as _pool:
        _exist_flags = list(_pool.map(_exists, [dest for _, dest, _ in all_tasks]))
    n_cached = sum(_exist_flags)

    if verbose:
        print(f"  Imágenes a descargar : {n_total - n_cached:,}  (en caché: {n_cached:,})")

    results_by_dest: dict[str, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=n_download_workers) as pool:
        futures = {pool.submit(_download_one, t): t[1] for t in all_tasks}
        done = 0
        for fut in as_completed(futures):
            dest = futures[fut]
            results_by_dest[str(dest)] = fut.result()
            done += 1
            if verbose and done % 1000 == 0:
                print(f"    {done:,}/{n_total:,} imágenes procesadas...")

    # ── Fase 3: construir registros ────────────────────────────────────────────
    records = []
    n_no_img = 0

    for item_id, tasks in tasks_by_item.items():
        local_paths = [
            results_by_dest[str(dest)]
            for _, dest, _ in tasks
            if results_by_dest.get(str(dest)) is not None
        ]

        if not local_paths:
            n_no_img += 1
            continue

        row   = rows_by_item[item_id]
        title = str(row.get("ITE_ITEM_TITLE", "") or "")
        desc  = str(row.get("ITE_ITEM_DESCRIPTION", "") or "")[:prompt_max_chars]
        attrs = _attributes_to_text(row.get("attributes"))

        prompt_text = PROMPT_TEMPLATE.format(
            title=title, description=desc, attributes=attrs
        )

        y = int(row["RESULT"])
        sf_list = _parse_source_field(row.get("SOURCE_FIELD"))
        reason  = _shorten_explanation(row.get("EXPLANATION", ""))

        answer = json.dumps({
            "has_contact_data": y,
            "source_field":     sf_list,
            "reason_short":     reason or ("Contact data detected." if y == 1 else "No contact data detected."),
        }, ensure_ascii=False)

        records.append({
            "item_id":     item_id,
            "image_path":  local_paths,
            "prompt_text": prompt_text,
            "answer":      answer,
            "label":       y,
        })

    if verbose:
        print(f"  Ejemplos construidos : {len(records):,}")
        print(f"  Sin imágenes (skip)  : {n_no_img:,}")

    return Dataset.from_list(records)
