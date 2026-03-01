"""
prefetch_images.py
------------------
Pre-descarga imágenes de los splits a Google Drive desde Colab.

Características:
  - Saltea imágenes ya descargadas (reanudable si se interrumpe)
  - Reintentos con backoff exponencial por imagen
  - Descargas paralelas con ThreadPoolExecutor
  - Barra de progreso por split
  - Log de URLs fallidas en output_dir/failed_{split}.txt

Uso desde Colab:
    %run /content/meli-contact-detection/scripts/prefetch_images.py \\
        --splits_dir  /content/drive/MyDrive/contact-detection/data/splits \\
        --output_dir  /content/drive/MyDrive/contact-detection/data/images \\
        --splits train val \\
        --workers 8
"""

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import pick_image_urls

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "image/*,*/*;q=0.8",
    "Referer": "https://www.mercadolibre.com/",
}


# ── Descarga individual con reintentos ────────────────────────────────────────

def _download_one(url: str, dest: Path, max_retries: int, max_side: int) -> bool:
    """Descarga y guarda una imagen. Retorna True si tuvo éxito."""
    if dest.exists():
        return True

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")

            w, h = img.size
            scale = min(max_side / max(w, h), 1.0)
            nw, nh = int(w * scale), int(h * scale)
            if (nw, nh) != (w, h):
                img = img.resize((nw, nh), Image.LANCZOS)

            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest, format="JPEG", quality=90, optimize=True)
            return True

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.random())

    return False


# ── Descarga de un split ──────────────────────────────────────────────────────

def prefetch_split(
    csv_path: Path,
    output_dir: Path,
    split_name: str,
    workers: int,
    max_retries: int,
    max_side: int,
    seed: int,
) -> None:
    df = pd.read_csv(csv_path)
    print(f"\n[{split_name}] {len(df):,} filas")

    # Construir lista completa de (url, dest)
    tasks = []
    for _, row in df.iterrows():
        item_id = str(row["item_id"])
        urls = pick_image_urls(row, max_images=10, seed=seed)
        for i, url in enumerate(urls, start=1):
            dest = output_dir / split_name / item_id / f"img_{i:02d}.jpg"
            tasks.append((url, dest))

    already = sum(1 for _, dest in tasks if dest.exists())
    pending = [(url, dest) for url, dest in tasks if not dest.exists()]

    print(f"  Total imágenes  : {len(tasks):,}")
    print(f"  Ya en Drive     : {already:,}")
    print(f"  Por descargar   : {len(pending):,}")

    if not pending:
        print("  ✓ Nada que descargar.")
        return

    failed_urls = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, url, dest, max_retries, max_side): url
            for url, dest in pending
        }
        with tqdm(total=len(pending), desc=split_name, unit="img") as pbar:
            for future in as_completed(futures):
                url = futures[future]
                if not future.result():
                    failed_urls.append(url)
                pbar.update(1)

    ok = len(pending) - len(failed_urls)
    print(f"  ✓ Descargadas   : {ok:,}")
    if failed_urls:
        print(f"  ✗ Fallidas      : {len(failed_urls):,}")
        log = output_dir / f"failed_{split_name}.txt"
        log.write_text("\n".join(failed_urls))
        print(f"  Log guardado en : {log}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-descarga imágenes de splits a Drive.")
    parser.add_argument("--splits_dir",  required=True, help="Dir con train.csv / val.csv")
    parser.add_argument("--output_dir",  required=True, help="Dir destino de imágenes en Drive")
    parser.add_argument("--splits",      nargs="+", default=["train", "val"])
    parser.add_argument("--workers",     type=int, default=8,  help="Threads paralelos")
    parser.add_argument("--max_retries", type=int, default=3,  help="Reintentos por imagen")
    parser.add_argument("--max_side",    type=int, default=512, help="Lado máximo en px")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)

    for split in args.splits:
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"⚠️  No encontrado: {csv_path}")
            continue
        prefetch_split(
            csv_path=csv_path,
            output_dir=output_dir,
            split_name=split,
            workers=args.workers,
            max_retries=args.max_retries,
            max_side=args.max_side,
            seed=args.seed,
        )

    print("\n✅ Pre-descarga completada.")


if __name__ == "__main__":
    main()
