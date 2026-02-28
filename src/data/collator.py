"""
collator.py
-----------
Collator multimodal con label masking correcto para fine-tuning instruccional.
Solo los tokens de la respuesta del asistente contribuyen al loss.
"""

import torch
from PIL import Image
from transformers import AutoProcessor


def build_multimodal_collator(processor: AutoProcessor):
    """
    Retorna una función collator lista para pasar a SFTTrainer/Trainer.

    Args:
        processor: AutoProcessor del modelo (Qwen2.5-VL).

    Returns:
        Callable que recibe una lista de ejemplos y retorna un batch tensoriado.
    """

    def multimodal_collator(batch):
        images_batch = []
        full_texts   = []
        prompt_texts = []

        for ex in batch:
            paths = ex["image_path"]
            if isinstance(paths, str):
                paths = [paths]

            imgs = []
            for p in paths:
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB"))
            images_batch.append(imgs)

            answer = ex["answer"]

            user_content = [{"type": "image"} for _ in imgs]
            user_content.append({"type": "text", "text": ex["prompt_text"]})

            # Prompt solo — para calcular la longitud en tokens
            prompt_texts.append(processor.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            ))

            # Conversación completa — incluye <|im_end|> al final de la respuesta
            full_texts.append(processor.apply_chat_template(
                [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": answer},
                ],
                tokenize=False,
                add_generation_prompt=False,
            ))

        # Encode conversaciones completas
        enc_full = processor(
            text=full_texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # Encode solo prompts para medir su longitud en tokens
        enc_prompts = processor(
            text=prompt_texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prompt_lens = enc_prompts["attention_mask"].sum(dim=1).tolist()

        # Labels: -100 en el prompt, loss solo en tokens de respuesta + <|im_end|>
        labels = torch.full_like(enc_full["input_ids"], -100)
        for i, plen in enumerate(prompt_lens):
            plen    = int(plen)
            seq_len = int(enc_full["attention_mask"][i].sum().item())
            labels[i, plen:seq_len] = enc_full["input_ids"][i, plen:seq_len]

        enc_full["labels"] = labels
        return enc_full

    return multimodal_collator


def smoke_test(collator, dataset, n: int = 2) -> None:
    """Verifica que el collator produce labels parcialmente enmascarados."""
    batch = [dataset[i] for i in range(min(n, len(dataset)))]
    out = collator(batch)
    label_frac = (out["labels"] != -100).float().mean().item()
    print("✅ collator ok")
    print(f"   input_ids    : {out['input_ids'].shape}")
    print(f"   pixel_values : {out.get('pixel_values', 'N/A')}")
    print(f"   label_frac   : {label_frac:.3f}  ← debe ser << 1")
    assert label_frac < 0.5, "label_frac demasiado alto — revisar masking"
