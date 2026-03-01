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
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.csv_reader import read_pool_csv


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_splits(config_path: str, pools_dir: str, output_dir: str) -> None:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    label_col = "RESULT"

    pools_dir = Path(pools_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar pools ──────────────────────────────────────────────────────────
    # Usamos csv_reader porque los CSVs tienen comillas sin escapar en campos JSON
    print("Cargando pools...")
    def load_pool(path):
        df = read_pool_csv(path)
        df = df.dropna(subset=[label_col])
        df = df[df[label_col].isin([0, 1])]
        return df.reset_index(drop=True)

    raw      = load_pool(pools_dir / cfg["pools"]["raw"]["file"])
    pos      = load_pool(pools_dir / cfg["pools"]["positive"]["file"])
    hard_neg = load_pool(pools_dir / cfg["pools"]["hard_negative"]["file"])

    print(f"  raw      : {len(raw):>7,} filas  ({raw[label_col].mean():.2%} DC)")
    print(f"  positivos: {len(pos):>7,} filas  ({pos[label_col].mean():.2%} DC)")
    print(f"  hard_neg : {len(hard_neg):>7,} filas  ({hard_neg[label_col].mean():.2%} DC)")

    # ── Dividir raw pool (estratificado para preservar 1.5% DC) ──────────────
    raw_cfg = cfg["pools"]["raw"]

    # Paso 1: separar test del resto (trainval), estratificado por RESULT
    raw_trainval, raw_test = train_test_split(
        raw,
        test_size=raw_cfg["test"],
        stratify=raw[label_col],
        random_state=seed,
    )

    # Paso 2: separar train y val del trainval, estratificado por RESULT
    raw_train, raw_val = train_test_split(
        raw_trainval,
        test_size=raw_cfg["val"],
        stratify=raw_trainval[label_col],
        random_state=seed,
    )

    # ── Dividir pool de positivos ─────────────────────────────────────────────
    # Todos son DC=1 → no hace falta estratificar
    pos_cfg = cfg["pools"]["positive"]
    pos_train = pos.sample(n=min(pos_cfg["train"], len(pos)), random_state=seed)
    pos_remaining = pos.drop(pos_train.index)
    pos_val = pos_remaining.sample(n=min(pos_cfg["val"], len(pos_remaining)), random_state=seed)

    # ── Dividir hard negatives ────────────────────────────────────────────────
    # Todos son DC=0 → no hace falta estratificar
    hn_cfg = cfg["pools"]["hard_negative"]
    hn_train = hard_neg.sample(n=min(hn_cfg["train"], len(hard_neg)), random_state=seed)
    hn_remaining = hard_neg.drop(hn_train.index)
    hn_val = hn_remaining.sample(n=min(hn_cfg["val"], len(hn_remaining)), random_state=seed)

    # ── Concatenar y shufflear ────────────────────────────────────────────────
    train = (pd.concat([raw_train, pos_train, hn_train])
               .sample(frac=1, random_state=seed)
               .reset_index(drop=True))
    val   = (pd.concat([raw_val, pos_val, hn_val])
               .sample(frac=1, random_state=seed)
               .reset_index(drop=True))
    test  = raw_test.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Verificar que no hay solapamiento de item_id ──────────────────────────
    _check_no_overlap(train, val, test)

    # ── Validar contra valores esperados ─────────────────────────────────────
    _validate(train, cfg["expected"]["train"], "train", label_col)
    _validate(val,   cfg["expected"]["val"],   "val",   label_col)
    _validate(test,  cfg["expected"]["test"],  "test",  label_col)

    # ── Guardar ───────────────────────────────────────────────────────────────
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(  output_dir / "val.csv",   index=False)
    test.to_csv( output_dir / "test.csv",  index=False)

    print(f"\n✅ Splits guardados en {output_dir}")
    print(f"   train : {len(train):>6,} filas  ({train[label_col].mean():.2%} DC)")
    print(f"   val   : {len(val):>6,} filas  ({val[label_col].mean():.2%} DC)")
    print(f"   test  : {len(test):>6,} filas  ({test[label_col].mean():.2%} DC)")
    print("\n⚠️  test.csv NO debe usarse hasta Sprint 10.")


def _check_no_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    id_col = "item_id"
    if id_col not in train.columns:
        return  # Si no hay item_id, salteamos la verificación
    t, v, te = set(train[id_col]), set(val[id_col]), set(test[id_col])
    overlaps = {
        "train∩val":  len(t & v),
        "train∩test": len(t & te),
        "val∩test":   len(v & te),
    }
    for k, n in overlaps.items():
        if n > 0:
            raise ValueError(f"Solapamiento detectado: {k} = {n} items")
    print("✓  Sin solapamiento entre splits")


def _validate(df: pd.DataFrame, expected: dict, split_name: str, label_col: str) -> None:
    actual_total = len(df)
    actual_dc    = df[label_col].mean()

    tol_total = 0.01   # 1% tolerancia en cantidad
    tol_dc    = 0.005  # 0.5pp tolerancia en tasa DC

    ok = True
    if abs(actual_total - expected["total"]) / expected["total"] > tol_total:
        print(f"⚠️  [{split_name}] total: esperado {expected['total']:,}, obtenido {actual_total:,}")
        ok = False
    if abs(actual_dc - expected["dc_rate"]) > tol_dc:
        print(f"⚠️  [{split_name}] dc_rate: esperado {expected['dc_rate']:.4f}, obtenido {actual_dc:.4f}")
        ok = False
    if ok:
        print(f"✓  [{split_name}] OK — {actual_total:,} filas, {actual_dc:.2%} DC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construye train/val/test desde los pools.")
    parser.add_argument("--config",     required=True, help="Path a configs/dataset.yaml")
    parser.add_argument("--pools_dir",  required=True, help="Directorio con los 3 CSVs de pools")
    parser.add_argument("--output_dir", required=True, help="Directorio donde guardar train/val/test")
    args = parser.parse_args()

    build_splits(args.config, args.pools_dir, args.output_dir)
