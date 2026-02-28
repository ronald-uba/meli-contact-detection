"""
test_build_splits.py
--------------------
Tests para verificar que build_splits genera splits correctos.
Usa datos sintéticos para no depender de los CSVs reales.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.build_splits import build_splits, load_config


@pytest.fixture
def synthetic_pools(tmp_path):
    """Genera pools sintéticos con la distribución correcta."""
    rng_seed = 42
    label_col = "RESULT"

    # raw: 100k casos, 1.5% DC
    n_raw = 100_000
    raw = pd.DataFrame({
        "item_id": range(n_raw),
        label_col: ([1] * int(n_raw * 0.015)) + ([0] * (n_raw - int(n_raw * 0.015))),
        "ITE_ITEM_TITLE": [f"item_{i}" for i in range(n_raw)],
    })
    raw = raw.sample(frac=1, random_state=rng_seed).reset_index(drop=True)

    # positivos: 14k, todos DC=1
    pos = pd.DataFrame({
        "item_id": range(100_000, 114_000),
        label_col: [1] * 14_000,
        "ITE_ITEM_TITLE": [f"pos_{i}" for i in range(14_000)],
    })

    # hard negatives: 7k, todos DC=0
    hn = pd.DataFrame({
        "item_id": range(114_000, 121_000),
        label_col: [0] * 7_000,
        "ITE_ITEM_TITLE": [f"hn_{i}" for i in range(7_000)],
    })

    pools_dir = tmp_path / "pools"
    pools_dir.mkdir()
    raw.to_csv(pools_dir / "raw_100k.csv",          index=False)
    pos.to_csv(pools_dir / "positives_14k.csv",     index=False)
    hn.to_csv( pools_dir / "hard_negatives_7k.csv", index=False)

    return pools_dir


@pytest.fixture
def config_path(tmp_path):
    """Copia el dataset.yaml real al directorio temporal."""
    real_config = Path(__file__).parent.parent / "configs" / "dataset.yaml"
    dest = tmp_path / "dataset.yaml"
    dest.write_text(real_config.read_text())
    return str(dest)


def test_build_splits_sizes(synthetic_pools, config_path, tmp_path):
    output_dir = tmp_path / "splits"
    build_splits(config_path, str(synthetic_pools), str(output_dir))

    train = pd.read_csv(output_dir / "train.csv")
    val   = pd.read_csv(output_dir / "val.csv")
    test  = pd.read_csv(output_dir / "test.csv")

    assert len(train) == 66_900, f"train size: {len(train)}"
    assert len(val)   == 24_100, f"val size: {len(val)}"
    assert len(test)  == 30_000, f"test size: {len(test)}"


def test_test_split_no_enrichment(synthetic_pools, config_path, tmp_path):
    """Test split debe usar SOLO el raw pool — sin positivos ni hard negatives."""
    output_dir = tmp_path / "splits"
    build_splits(config_path, str(synthetic_pools), str(output_dir))

    test = pd.read_csv(output_dir / "test.csv")

    # Todos los item_id deben ser del raw pool (0..99999)
    assert test["item_id"].max() < 100_000, "test contiene items fuera del raw pool"


def test_dc_rates(synthetic_pools, config_path, tmp_path):
    output_dir = tmp_path / "splits"
    build_splits(config_path, str(synthetic_pools), str(output_dir))

    label_col = "RESULT"
    train = pd.read_csv(output_dir / "train.csv")
    val   = pd.read_csv(output_dir / "val.csv")
    test  = pd.read_csv(output_dir / "test.csv")

    assert abs(train[label_col].mean() - 0.2053) < 0.01, f"train dc_rate: {train[label_col].mean():.4f}"
    assert abs(val[label_col].mean()   - 0.0546) < 0.01, f"val dc_rate: {val[label_col].mean():.4f}"
    assert abs(test[label_col].mean()  - 0.015)  < 0.005, f"test dc_rate: {test[label_col].mean():.4f}"


def test_no_overlap(synthetic_pools, config_path, tmp_path):
    """Ningún item_id debe aparecer en más de un split."""
    output_dir = tmp_path / "splits"
    build_splits(config_path, str(synthetic_pools), str(output_dir))

    train = set(pd.read_csv(output_dir / "train.csv")["item_id"])
    val   = set(pd.read_csv(output_dir / "val.csv")["item_id"])
    test  = set(pd.read_csv(output_dir / "test.csv")["item_id"])

    assert len(train & val)  == 0, "overlap train-val"
    assert len(train & test) == 0, "overlap train-test"
    assert len(val & test)   == 0, "overlap val-test"
