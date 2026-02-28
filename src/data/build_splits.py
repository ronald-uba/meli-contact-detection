"""
build_splits.py
---------------
Genera train.csv, val.csv y test.csv a partir de los tres pools de datos,
siguiendo el esquema de mezcla definido en configs/dataset.yaml.

Uso:
    python src/data/build_splits.py \
        --config configs/dataset.yaml \
        --pools_dir /content/drive/MyDrive/contact-detection/data/pools \
        --output_dir /content/drive/MyDrive/contact-detection/data/splits
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def sample_stratified(df: pd.DataFrame, n: int, label_col: str, seed: int) -> pd.DataFrame:
    """Muestrea n filas preservando la proporción de clases."""
    return (
        df.groupby(label_col, group_keys=False)
          .apply(lambda g: g.sample(frac=n / len(df), random_state=seed))
          .sample(frac=1, random_state=seed)  # shuffle
          .reset_index(drop=True)
    )


def build_splits(config_path: str, pools_dir: str, output_dir: str) -> None:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)

    pools_dir = Path(pools_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar pools ──────────────────────────────────────────────────────────
    print("Cargando pools...")
    raw = pd.read_csv(pools_dir / cfg["pools"]["raw"]["file"])
    pos = pd.read_csv(pools_dir / cfg["pools"]["positive"]["file"])
    hard_neg = pd.read_csv(pools_dir / cfg["pools"]["hard_negative"]["file"])

    label_col = "RESULT"

    # ── Muestrar raw pool (estratificado para preservar 1.5% DC) ─────────────
    raw_cfg = cfg["pools"]["raw"]
    raw_train = sample_stratified(raw, raw_cfg["train"], label_col, seed)
    raw_val   = sample_stratified(
        raw.drop(raw_train.index), raw_cfg["val"], label_col, seed
    )
    raw_test  = sample_stratified(
        raw.drop(raw_train.index).drop(raw_val.index), raw_cfg["test"], label_col, seed
    )

    # ── Muestrar positivos ────────────────────────────────────────────────────
    pos_cfg = cfg["pools"]["positive"]
    pos_train = pos.sample(n=pos_cfg["train"], random_state=seed)
    pos_val   = pos.drop(pos_train.index).sample(n=pos_cfg["val"], random_state=seed)

    # ── Muestrar hard negatives ───────────────────────────────────────────────
    hn_cfg = cfg["pools"]["hard_negative"]
    hn_train = hard_neg.sample(n=hn_cfg["train"], random_state=seed)
    hn_val   = hard_neg.drop(hn_train.index).sample(n=hn_cfg["val"], random_state=seed)

    # ── Concatenar y shufflear ────────────────────────────────────────────────
    train = pd.concat([raw_train, pos_train, hn_train]).sample(frac=1, random_state=seed).reset_index(drop=True)
    val   = pd.concat([raw_val,   pos_val,   hn_val  ]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test  = raw_test.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Validar contra valores esperados ─────────────────────────────────────
    _validate(train, cfg["expected"]["train"], "train")
    _validate(val,   cfg["expected"]["val"],   "val")
    _validate(test,  cfg["expected"]["test"],  "test")

    # ── Guardar ───────────────────────────────────────────────────────────────
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(  output_dir / "val.csv",   index=False)
    test.to_csv( output_dir / "test.csv",  index=False)

    print(f"\n✅ Splits guardados en {output_dir}")
    print(f"   train : {len(train):>6,} filas  ({train[label_col].mean():.2%} DC)")
    print(f"   val   : {len(val):>6,} filas  ({val[label_col].mean():.2%} DC)")
    print(f"   test  : {len(test):>6,} filas  ({test[label_col].mean():.2%} DC)")
    print("\n⚠️  test.csv no debe usarse hasta Sprint 10.")


def _validate(df: pd.DataFrame, expected: dict, split_name: str) -> None:
    label_col = "RESULT"
    actual_total = len(df)
    actual_dc    = df[label_col].mean()
    actual_pos   = df[label_col].sum()

    tol_total = 0.01   # 1% tolerancia en cantidad
    tol_dc    = 0.005  # 0.5pp tolerancia en tasa DC

    ok = True
    if abs(actual_total - expected["total"]) / expected["total"] > tol_total:
        print(f"⚠️  [{split_name}] total: esperado {expected['total']}, obtenido {actual_total}")
        ok = False
    if abs(actual_dc - expected["dc_rate"]) > tol_dc:
        print(f"⚠️  [{split_name}] dc_rate: esperado {expected['dc_rate']:.4f}, obtenido {actual_dc:.4f}")
        ok = False
    if ok:
        print(f"✓  [{split_name}] validación OK — {actual_total:,} filas, {actual_dc:.2%} DC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye train/val/test desde los pools.")
    parser.add_argument("--config",     required=True, help="Path a configs/dataset.yaml")
    parser.add_argument("--pools_dir",  required=True, help="Directorio con los 3 CSVs de pools")
    parser.add_argument("--output_dir", required=True, help="Directorio donde guardar train/val/test")
    args = parser.parse_args()

    build_splits(args.config, args.pools_dir, args.output_dir)
