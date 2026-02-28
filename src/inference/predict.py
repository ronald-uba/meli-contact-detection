"""
predict.py
----------
Funciones de inferencia: predict_one (un ejemplo) y predict_batch (DataFrame).
"""

import json
import re
from typing import Optional

import torch
from PIL import Image


def predict_one(
    example: dict,
    model,
    processor,
    max_new_tokens: int = 400,
) -> tuple[str, Optional[dict]]:
    """
    Corre inferencia sobre un único ejemplo del Dataset.

    Returns:
        (raw_text, parsed_json) — parsed_json es None si no se pudo parsear.
    """
    paths = example["image_path"]
    if isinstance(paths, str):
        paths = [paths]

    images = [Image.open(p).convert("RGB") for p in paths]

    user_content = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": example["prompt_text"]})

    prompt = processor.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=[images],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decodificar solo los tokens nuevos
    input_len = inputs["input_ids"].shape[1]
    raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

    parsed = _parse_json(raw)
    return raw, parsed


def _parse_json(text: str) -> Optional[dict]:
    """Intenta extraer el primer JSON válido del texto generado."""
    # Buscar bloque ```json ... ``` o { ... }
    for pattern in [r"```json\s*(.*?)\s*```", r"(\{.*?\})"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def predict_batch(df, model, processor, max_new_tokens: int = 400) -> list[dict]:
    """
    Corre predicción sobre un DataFrame.
    Retorna lista de dicts con item_id, raw, resultado, explicacion.
    """
    results = []
    for _, row in df.iterrows():
        ex = row.to_dict()
        raw, parsed = predict_one(ex, model, processor, max_new_tokens)
        results.append({
            "item_id":    ex.get("item_id", ""),
            "raw":        raw,
            "resultado":  parsed.get("resultado") if parsed else None,
            "explicacion": parsed.get("explicacion") if parsed else None,
            "label":      ex.get("label"),
        })
    return results
