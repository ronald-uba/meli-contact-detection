"""
csv_reader.py
-------------
Lector custom para los CSVs de pools de MeLi donde los campos JSON
(pictures, attributes) contienen comillas internas sin escapar.

El formato correcto (RFC 4180) requeriría doblar las comillas: "" dentro de "..."
Pero estos CSVs las dejan sin escapar: {"key":"val"} en lugar de {""key"":""val""}

Estrategia de parseo:
  - Campos JSON ({...} o [{...}]) → bracket counting ignorando comillas internas.
  - Campos "[" seguido de letra → NO se trata como JSON (es texto libre como EXPLANATION).
  - Campos texto → la comilla cierra solo si va seguida de `,` o fin de línea.
  - Últimas 3 columnas (EXPLANATION, SOURCE_FIELD, pictures_dp):
    SOURCE_FIELD y pictures_dp se parsean desde la DERECHA del renglón;
    EXPLANATION es el tramo que queda entre la posición izquierda y el inicio
    de SOURCE_FIELD.  Esto maneja de forma robusta el caso en que EXPLANATION
    contiene comillas no escapadas y comillas-coma internas.

Función principal:
    read_pool_csv(path, usecols=None) → pd.DataFrame
"""

from pathlib import Path
from typing import Optional

import pandas as pd


COLUMNS = [
    "site_id",
    "item_id",
    "ITE_ITEM_DOM_DOMAIN_ID",
    "ITE_ITEM_TITLE",
    "pictures",
    "ITE_ITEM_DESCRIPTION",
    "attributes",
    "RESULT_4_1_MINI",
    "RESULT_4_1",
    "RESULT",
    "tiene_41",
    "EXPLANATION",
    "SOURCE_FIELD",
    "pictures_dp",
]


# ── Right-side field extractor ────────────────────────────────────────────────

def _parse_field_from_right(line: str, end: int) -> tuple:
    """
    Parsea un campo CSV desde la derecha del renglón.

    `end` es el índice del último carácter del campo (inclusive).

    Retorna (field_value, comma_pos) donde:
      - field_value : valor del campo (None para null/vacío).
      - comma_pos   : índice de la `,` separadora ANTES de este campo,
                      o -1 si el campo está al principio del renglón.

    Casos soportados:
      1. Campo no entrecomillado: `null`, número, vacío.
      2. Campo entrecomillado simple: `"texto"` (puede contener comas).
      3. Campo JSON array entrecomillado: `"["url1","url2"]"`.
    """
    i = end
    n = len(line)

    if i < 0 or i >= n:
        return None, -1

    if line[i] == '"':
        # ── Campo entrecomillado ──────────────────────────────────────────────
        content_last = i - 1  # último char del contenido (antes del cierre `"`)

        if content_last >= 0 and line[content_last] == ']':
            # JSON array envuelto en comillas: "["url1","url2"]"
            # Bracket counting desde la derecha para hallar el `[` de apertura.
            depth = 1
            j = content_last - 1
            while j >= 0:
                c = line[j]
                if c == ']':
                    depth += 1
                elif c == '[':
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            # j apunta a `[`; la comilla de apertura debe estar en j-1
            if j >= 1 and line[j - 1] == '"':
                comma_pos = j - 2
                raw = line[j : content_last + 1]   # ["url1","url2"]
                return raw, comma_pos
            # Estructura inesperada → fallback como string simple
            # (continúa hacia el bloque "string simple" de abajo)

        # ── String simple (puede contener comas): buscar `"` precedida de `,` ──
        # Escaneamos hacia atrás desde content_last hasta encontrar `"`
        # tal que el carácter anterior sea `,` (o estemos al inicio).
        j = content_last
        while j >= 0:
            if line[j] == '"' and (j == 0 or line[j - 1] == ','):
                break
            j -= 1
        if j < 0:
            return None, -1
        content = line[j + 1 : content_last + 1]   # texto entre comillas
        comma_pos = j - 1
        return (None if content in ("null", "") else content), comma_pos

    else:
        # ── Campo sin comillas: escanear hacia atrás hasta `,` ───────────────
        j = i
        while j >= 0 and line[j] != ",":
            j -= 1
        raw = line[j + 1 : i + 1].strip()
        return (None if raw in ("null", "") else raw), j - 1


# ── Row parser ────────────────────────────────────────────────────────────────

def _parse_row(line: str, n_cols: int = None) -> list:
    """
    Parsea una línea del CSV con campos JSON de comillas sin escapar.

    Parámetros
    ----------
    line   : La línea cruda (sin \\n/\\r final).
    n_cols : Número esperado de columnas.  Cuando se indica, las últimas 3
             columnas (EXPLANATION, SOURCE_FIELD, pictures_dp) se extraen
             con la estrategia derecha-izquierda, evitando los problemas de
             comillas no escapadas en EXPLANATION.

    Estrategia izquierda → derecha (columnas 0 … n_cols-4):
      - `"` cierra el campo solo si va seguida de `,` o fin de línea.
      - `""` se trata como comilla escapada (CSV estándar).
      - Cualquier otra `"` interna se trata como contenido.
      - Para campos JSON `"{...}"` o `"[{...}]"` / `"["..."]"`:
        bracket counting de { } o [ ].
        NOTA: `"[texto` (letra tras `[`) NO se trata como JSON.

    Estrategia derecha → izquierda (últimas 2 columnas: SOURCE_FIELD, pictures_dp):
      Ver _parse_field_from_right.
    """
    fields = []
    pos = 0
    n = len(line)

    # Umbral: cuántos campos parsear desde la izquierda antes de cambiar a derecha
    n_left = (n_cols - 3) if (n_cols is not None and n_cols >= 3) else None

    while pos <= n:

        # ── Cambio a estrategia de derecha ────────────────────────────────────
        if n_left is not None and len(fields) == n_left:
            # Parsear pictures_dp (última columna) desde la derecha
            pics_dp_val, comma_before_pics = _parse_field_from_right(line, n - 1)

            # Parsear SOURCE_FIELD (penúltima columna) desde la derecha
            source_val, comma_before_source = _parse_field_from_right(
                line, comma_before_pics - 1
            )

            # EXPLANATION = tramo de la línea entre pos (izquierda) y comma_before_source
            expl_raw = line[pos : comma_before_source]
            if expl_raw and expl_raw[0] == '"' and expl_raw[-1] == '"':
                expl_val = expl_raw[1:-1].replace('""', '"')
            elif expl_raw.strip().lower() in ("null", ""):
                expl_val = None
            else:
                expl_val = expl_raw.strip() or None

            fields.extend([expl_val, source_val, pics_dp_val])
            break

        # ── Fin de línea ──────────────────────────────────────────────────────
        if pos == n:
            fields.append(None)
            break

        ch = line[pos]

        if ch == '"':
            next_ch = line[pos + 1] if pos + 1 < n else ""

            # Solo usamos bracket counting para campos que son JSON real:
            #   "{"key":...}  →  objeto JSON (pictures)   [pos+2 debe ser '"']
            #   "[{..."       →  array de objetos (attributes)
            #   "["..."       →  array de strings (pictures_dp desde la izquierda)
            # NO usamos bracket counting para:
            #   "{ texto"   →  título que empieza con '{'  [pos+2 es espacio/letra]
            #   "[letra"    →  texto libre como EXPLANATION
            use_bracket = next_ch in "{[" and not (
                next_ch == "{" and (pos + 2 >= n or line[pos + 2] != '"')
            ) and not (
                next_ch == "[" and (pos + 2 >= n or line[pos + 2] not in ("{", '"'))
            )

            if use_bracket:
                # ── Campo JSON: bracket counting ──────────────────────────────
                opening = next_ch
                closing = "}" if opening == "{" else "]"
                depth = 0
                i = pos + 1
                while i < n:
                    c = line[i]
                    if c == opening:
                        depth += 1
                    elif c == closing:
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                field = line[pos + 1 : i + 1]
                pos = i + 1
                if pos < n and line[pos] == '"':
                    pos += 1  # saltar comilla de cierre
                fields.append(field)

            else:
                # ── Campo string: `"` cierra solo si va seguida de `,` o fin ──
                i = pos + 1
                while i < n:
                    if line[i] == '"':
                        # Comilla doble → escapada, saltar las dos
                        if i + 1 < n and line[i + 1] == '"':
                            i += 2
                            continue
                        # Comilla seguida de `,` o fin → cierra el campo
                        if i + 1 >= n or line[i + 1] == ',':
                            break
                        # Comilla interna sin escapar → es contenido
                    i += 1
                field = line[pos + 1 : i].replace('""', '"')
                fields.append(None if field == "" else field)
                pos = i + 1  # saltar comilla de cierre

        else:
            # ── Campo sin comillas: número, null o vacío ──────────────────────
            i = pos
            while i < n and line[i] != ",":
                i += 1
            raw = line[pos:i].strip()
            fields.append(None if raw in ("null", "") else raw)
            pos = i

        if pos < n and line[pos] == ",":
            pos += 1
        elif pos >= n:
            break

    return fields


# ── Public reader ─────────────────────────────────────────────────────────────

def read_pool_csv(
    path: str,
    usecols: Optional[list] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lee un CSV de pool MeLi y retorna un DataFrame.

    Args:
        path    : Path al CSV.
        usecols : Lista de nombres de columnas a retener. None = todas.
        verbose : Imprimir estadísticas de parseo.
    """
    path = Path(path)
    rows = []
    errors = 0

    with open(path, encoding="utf-8") as f:
        # Saltar header
        header = f.readline()
        col_names = [c.strip() for c in header.rstrip("\n").split(",")]
        n_cols = len(col_names)

        for lineno, line in enumerate(f, start=2):
            line = line.rstrip("\n\r")
            if not line:
                continue
            try:
                fields = _parse_row(line, n_cols=n_cols)
                if len(fields) != n_cols:
                    errors += 1
                    continue
                rows.append(fields)
            except Exception:
                errors += 1
                continue

    if verbose:
        print(f"  {path.name}: {len(rows):,} filas OK, {errors:,} errores de parseo")

    df = pd.DataFrame(rows, columns=col_names)

    # Castear columnas numéricas conocidas
    for col in ["RESULT_4_1_MINI", "RESULT_4_1", "RESULT", "tiene_41"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if usecols:
        df = df[[c for c in usecols if c in df.columns]]

    return df
