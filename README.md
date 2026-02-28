# meli-contact-detection

Detección de datos de contacto en listings de Mercado Libre mediante fine-tuning multimodal de LLMs.

**Proyecto de Maestría en Inteligencia Artificial — FIUBA**
Alumno: Ronald Uthurralt | Director: Mg. Franco Arito

---

## Objetivo

Detectar si una publicación de MeLi contiene datos de contacto (teléfonos, WhatsApp, URLs) que permitan evadir la plataforma, usando un modelo multimodal (texto + imágenes) fine-tuneado con LoRA.

## Labels

| Label | Descripción |
|-------|-------------|
| `DC-adrede` | Datos de contacto intencionales |
| `DC-involuntario` | Datos de contacto no intencionales |
| `DC-negativo` | Sin datos de contacto |

## Estructura

```
meli-contact-detection/
├── configs/              # Parámetros reproducibles (YAML)
│   ├── dataset.yaml      # Mixing pools train/val/test
│   └── qwen25_3b.yaml    # Hiperparámetros de entrenamiento
├── notebooks/            # Experimentación por sprint
│   ├── 00_eda.ipynb
│   ├── 01_build_splits.ipynb
│   └── 02_baseline_10imgs.ipynb
├── scripts/
│   └── colab_setup.py    # Setup automático en Colab
├── src/
│   ├── data/
│   │   ├── build_splits.py   # Genera train/val/test desde los pools
│   │   ├── collator.py       # Collator multimodal con label masking
│   │   └── dataset.py        # Dataset builder (CSV → HF Dataset)
│   ├── training/
│   │   └── train.py          # Script de entrenamiento
│   ├── inference/
│   │   └── predict.py        # Predicción individual y batch
│   └── engine/
│       ├── decision.py       # Motor de decisión con umbrales
│       └── thresholds.py     # Calibración de umbrales
└── tests/
    └── test_build_splits.py
```

## Setup en Colab

```python
# En la primera celda del notebook:
exec(open("/content/drive/MyDrive/contact-detection/scripts/colab_setup.py").read())
```

## Datos (Google Drive — no versionados en git)

```
contact-detection/
├── data/
│   ├── pools/            # Inmutables — nunca modificar
│   │   ├── raw_100k.csv
│   │   ├── positives_14k.csv
│   │   └── hard_negatives_7k.csv
│   └── splits/           # Output de build_splits.py
│       ├── train.csv
│       ├── val.csv
│       └── test.csv      # NO usar hasta Sprint 10
├── outputs/              # Checkpoints y modelos fine-tuneados
└── logs/                 # TensorBoard / WandB
```

## Sprints

| Sprint | Foco |
|--------|------|
| 1–2 | EDA, build_splits, baseline texto |
| 3–4 | Fine-tuning multimodal (10 imgs) |
| 5–6 | Optimización LoRA, curriculum |
| 7–8 | Decision engine, calibración umbrales |
| 9 | Evaluación final (val) |
| 10 | Evaluación test — solo una vez |
| 11 | Documentación, POC, entrega |

## Métrica principal

**F2-score** (recall pesa el doble que precision — penaliza falsos negativos).
