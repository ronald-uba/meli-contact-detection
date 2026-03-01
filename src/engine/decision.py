"""
decision.py
-----------
Motor de decisión post-modelo.
Convierte la salida del LLM en una decisión binaria (DC / no-DC)
aplicando umbrales de confianza configurables.
"""

from dataclasses import dataclass
from typing import Optional


VALID_LABELS = {"DC-adrede", "DC-involuntario", "DC-negativo"}
POSITIVE_LABELS = {"DC-adrede", "DC-involuntario"}


@dataclass
class Decision:
    item_id: str
    raw_label: Optional[str]   # Lo que dijo el modelo
    is_dc: bool                # Decisión binaria final
    confidence: str            # "high" | "low" | "parse_error"
    explanation: Optional[str]


def decide(
    item_id: str,
    parsed: Optional[dict],
    require_explanation: bool = True,
) -> Decision:
    """
    Aplica reglas de negocio sobre la salida parseada del LLM.

    Args:
        item_id: Identificador del listing.
        parsed: Dict con keys 'resultado' y 'explicacion' (puede ser None).
        require_explanation: Si True, sin explicación baja a confianza 'low'.
    """
    if parsed is None:
        return Decision(item_id=item_id, raw_label=None, is_dc=False,
                        confidence="parse_error", explanation=None)

    # ── Schema nuevo: has_contact_data / source_field / reason_short ──────────
    if "has_contact_data" in parsed:
        hcd = parsed.get("has_contact_data")
        try:
            is_dc = bool(int(hcd))
        except (TypeError, ValueError):
            return Decision(item_id=item_id, raw_label=str(hcd), is_dc=False,
                            confidence="parse_error", explanation=None)
        raw_label  = "DC" if is_dc else "no-DC"
        expl       = (parsed.get("reason_short") or "").strip()
        confidence = "high" if (not require_explanation or expl) else "low"
        return Decision(item_id=item_id, raw_label=raw_label, is_dc=is_dc,
                        confidence=confidence, explanation=expl or None)

    # ── Schema legacy: resultado / explicacion ────────────────────────────────
    label = parsed.get("resultado", "").strip()
    expl  = parsed.get("explicacion", "").strip()

    if label not in VALID_LABELS:
        return Decision(item_id=item_id, raw_label=label, is_dc=False,
                        confidence="parse_error", explanation=expl or None)

    is_dc = label in POSITIVE_LABELS

    # Confianza: high solo si hay explicación (o no se requiere)
    confidence = "high"
    if require_explanation and not expl:
        confidence = "low"

    return Decision(item_id=item_id, raw_label=label, is_dc=is_dc,
                    confidence=confidence, explanation=expl or None)
