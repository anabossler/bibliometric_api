#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ce_vocab.py — Léxico CE-talk / CE-practice y mapa de mega-áreas.

FUENTE ÚNICA DE VERDAD del estudio. Importado por:
    ce_review_metrics, robustez_16_clusters, sensibilidad_vocabulario,
    oa_bias_check, validate_tmo_convergence.

ATENCIÓN: cualquier cambio en los patrones altera TODOS los resultados
posteriores. Al modificar el léxico, sube `__version__` y vuelve a ejecutar
la cadena completa; `__version__` se registra en los metadatos de cada script.

Decisiones de medida (documentadas para el anexo metodológico)
--------------------------------------------------------------
1. Se cuentan OCURRENCIAS, no presencia. Usa `unique=True` para contar
   términos distintos en lugar de menciones.
2. Las alternativas se ordenan de patrón más largo a más corto, de modo que en
   una misma posición gana la coincidencia más específica ("mechanical
   recycling" antes que "recycl*") y ningún texto se cuenta dos veces.
3. Los espacios se expresan como `\\s+` (tolera saltos de línea y espacios
   dobles típicos de abstracts extraídos de PDF) y se admite guion donde el uso
   compuesto es habitual ("circular-economy", "closed-loop").
4. Los acrónimos (CE, LCA) se buscan con sensibilidad a mayúsculas mediante el
   flag local `(?-i:...)`, para evitar falsos positivos con las secuencias
   "ce" y "lca" en minúscula.
5. Un texto como "life-cycle assessment (LCA)" produce 2 ocurrencias: es una
   propiedad deliberada del conteo por menciones, no un error.

Los patrones de la versión 1.0 se conservan en `LEGACY_TERMS_V1_0` para poder
reproducir exactamente los conteos publicados con esa versión.

Uso como módulo
---------------
    from ce_vocab import CE_TALK_RE, CE_PRACTICE_RE, count_talk, count_practice

Uso como script
---------------
    python ce_vocab.py --dump > lexicon_appendix.csv   # tabla para el anexo
    python ce_vocab.py --selftest                      # comprobaciones básicas

Licencia: MIT.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Mapping

__version__ = "1.1.0"

__all__ = [
    "MEGA_AREAS", "CLUSTER_TO_MEGA", "mega_area",
    "CE_TALK_TERMS", "CE_PRACTICE_TERMS",
    "CE_TALK_RE", "CE_PRACTICE_RE",
    "build_regex", "count_talk", "count_practice", "count_all",
    "matched_terms", "practice_talk_ratio",
    "LEGACY_TERMS_V1_0", "legacy_regexes_v1_0",
]

# ---------------------------------------------------------------------------
# 1. Mega-áreas
#    Cubren 11 de los clusters de la solución de 16: los no listados (y el
#    cluster de ruido) devuelven None por diseño.
# ---------------------------------------------------------------------------

MEGA_AREAS: dict[str, tuple[str, ...]] = {
    "Materials & Technical Recycling":    ("C10", "C11"),
    "Industrial Sectors & Applied Cases": ("C7", "C8", "C9"),
    "Energy & Resource Systems":          ("C2", "C14"),
    "Business, Policy & Governance":      ("C4", "C6"),
    "Sustainability Framing & Society":   ("C3", "C5"),
}
CLUSTER_TO_MEGA: dict[str, str] = {c: m for m, cs in MEGA_AREAS.items() for c in cs}


def mega_area(cluster) -> str | None:
    """Devuelve la mega-área de un cluster. Acepta 10, '10', '10.0', 'C10', 'c10'."""
    m = re.search(r"-?\d+", str(cluster))
    return CLUSTER_TO_MEGA.get(f"C{int(m.group(0))}") if m else None


# ---------------------------------------------------------------------------
# 2. Léxico  {etiqueta legible: patrón}
#    La etiqueta es la que aparece en la tabla del anexo y en `matched_terms`.
# ---------------------------------------------------------------------------

CE_TALK_TERMS: dict[str, str] = {
    "circular economy":        r"\bcircular[\s-]+econom\w*\b",
    "circularity":             r"\bcircularit(?:y|ies)\b",
    "closed loop":             r"\bclosed[\s-]?loops?\b",
    "sustainability paradigm": r"\bsustainab\w+\s+paradigms?\b",
    "sustainability transition": r"\bsustainability\s+transitions?\b",
    "CE framework":            r"(?-i:\bCE\b)[\s-]+frameworks?\b",
    "CE strategy":             r"(?-i:\bCE\b)[\s-]+strateg(?:y|ies)\b",
    "circular business model": r"\bcircular\s+business[\s-]+models?\b",
}

CE_PRACTICE_TERMS: dict[str, str] = {
    "recycling (generic)":     r"\brecycl\w+\b",
    "mechanical recycling":    r"\bmechanical\s+recycl\w+\b",
    "chemical recycling":      r"\bchemical\s+recycl\w+\b",
    "remanufacturing":         r"\bremanufactur\w+\b",
    "refurbishment":           r"\brefurbish\w*\b",
    "repair":                  r"\brepair\w*\b",
    "reuse":                   r"\breus\w+\b",
    "waste reduction (verb)":  r"\breduc\w+\s+waste\b",
    "waste reduction (noun)":  r"\bwaste\s+reduction\b",
    "upcycling":               r"\bupcycl\w+\b",
    "downcycling":             r"\bdowncycl\w+\b",
    "life cycle assessment":   r"\blife[\s-]?cycle\s+assessment\b",
    "LCA (acronym)":           r"(?-i:\bLCA\b)",
    "end of life":             r"\bend[\s-]?of[\s-]?life\b",
    "material recovery":       r"\bmaterial\s+recovery\b",
    "energy recovery":         r"\benergy\s+recovery\b",
    "reverse logistics":       r"\breverse\s+logistics\b",
    "biodegradation":          r"\bbiodegrad\w+\b",
    "composting":              r"\bcompost\w*\b",
}


# ---------------------------------------------------------------------------
# 3. Compilación
# ---------------------------------------------------------------------------

def build_regex(terms: Mapping[str, str], flags: int = re.IGNORECASE) -> re.Pattern:
    """Compila {etiqueta: patrón} en una única expresión sin grupos de captura.

    Sin grupos de captura, `findall` sigue devolviendo las cadenas coincidentes
    (compatible con el uso previo `len(RE.findall(text))`).
    """
    ordered = _order(terms)
    return re.compile("|".join(f"(?:{pat})" for _, pat in ordered), flags)


def _order(terms: Mapping[str, str]) -> list[tuple[str, str]]:
    """Más largo primero; desempate alfabético para que sea determinista."""
    return sorted(terms.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def _build_labeled(terms: Mapping[str, str]) -> tuple[re.Pattern, list[str]]:
    """Variante con grupos nombrados, para atribuir cada match a su término."""
    ordered = _order(terms)
    pattern = "|".join(f"(?P<t{i}>{pat})" for i, (_, pat) in enumerate(ordered))
    return re.compile(pattern, re.IGNORECASE), [label for label, _ in ordered]


CE_TALK_RE = build_regex(CE_TALK_TERMS)
CE_PRACTICE_RE = build_regex(CE_PRACTICE_TERMS)

_LABELED = {
    "talk": _build_labeled(CE_TALK_TERMS),
    "practice": _build_labeled(CE_PRACTICE_TERMS),
}


# ---------------------------------------------------------------------------
# 4. Conteo
# ---------------------------------------------------------------------------

def _as_text(value) -> str:
    """None y NaN cuentan como texto vacío, no como la cadena 'nan'."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return value if isinstance(value, str) else str(value)


def matched_terms(text, kind: str = "practice") -> Counter:
    """Counter {etiqueta: ocurrencias} — útil para auditar el léxico."""
    if kind not in _LABELED:
        raise ValueError(f"kind debe ser 'talk' o 'practice', no {kind!r}")
    rx, labels = _LABELED[kind]
    hits: Counter = Counter()
    for m in rx.finditer(_as_text(text)):
        hits[labels[int(m.lastgroup[1:])]] += 1
    return hits


def count_talk(text, unique: bool = False) -> int:
    """Ocurrencias de CE-talk (o términos distintos si unique=True)."""
    if unique:
        return len(matched_terms(text, "talk"))
    return len(CE_TALK_RE.findall(_as_text(text)))


def count_practice(text, unique: bool = False) -> int:
    """Ocurrencias de CE-practice (o términos distintos si unique=True)."""
    if unique:
        return len(matched_terms(text, "practice"))
    return len(CE_PRACTICE_RE.findall(_as_text(text)))


def count_all(text, unique: bool = False) -> dict[str, int]:
    return {"talk": count_talk(text, unique), "practice": count_practice(text, unique)}


def practice_talk_ratio(text, unique: bool = False) -> float:
    """practice/talk a nivel de documento; NaN si no hay CE-talk."""
    c = count_all(text, unique)
    return c["practice"] / c["talk"] if c["talk"] else float("nan")


# ---------------------------------------------------------------------------
# 5. Versión 1.0 (reproducibilidad de resultados ya publicados)
# ---------------------------------------------------------------------------

LEGACY_TERMS_V1_0: dict[str, tuple[str, ...]] = {
    "talk": (
        r"\bcircular econom\w*\b", r"\bcircularity\b", r"\bclosed-?loop\b",
        r"\bsustainab\w+ paradigm\b", r"\bsustainability transition\b",
        r"\bCE framework\b", r"\bCE strateg\w+\b", r"\bcircular business model\w*\b",
    ),
    "practice": (
        r"\brecycl\w+\b", r"\bremanufactur\w+\b", r"\brefurbish\w+\b", r"\brepair\w*\b",
        r"\breus\w+\b", r"\breduc\w+ waste\b", r"\bwaste reduction\b", r"\bupcycl\w+\b",
        r"\bdowncycl\w+\b", r"\blife[- ]cycle assessment\b", r"\bLCA\b",
        r"\bend[- ]of[- ]life\b", r"\bmaterial recovery\b", r"\benergy recovery\b",
        r"\breverse logistics\b", r"\bmechanical recycling\b", r"\bchemical recycling\b",
        r"\bbiodegrad\w+\b", r"\bcompost\w+\b",
    ),
}


def legacy_regexes_v1_0() -> dict[str, re.Pattern]:
    """Expresiones exactas de la v1.0 (orden original, IGNORECASE global)."""
    return {k: re.compile("|".join(v), re.IGNORECASE)
            for k, v in LEGACY_TERMS_V1_0.items()}


# ---------------------------------------------------------------------------
# 6. Validación al importar
# ---------------------------------------------------------------------------

def _validate() -> None:
    for kind, terms in (("talk", CE_TALK_TERMS), ("practice", CE_PRACTICE_TERMS)):
        for label, pat in terms.items():
            try:
                re.compile(pat)
            except re.error as exc:
                raise ValueError(f"patrón inválido en {kind}[{label!r}]: {exc}") from exc
    shared = set(CE_TALK_TERMS.values()) & set(CE_PRACTICE_TERMS.values())
    if shared:
        raise ValueError(f"patrones duplicados entre talk y practice: {sorted(shared)}")


_validate()


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

_SELFTEST_CASES = [
    # (texto, talk esperado, practice esperado)
    ("Circular economy and circularity in a closed loop system.", 3, 0),
    ("circular  economy\nacross lines", 1, 0),          # espacios y saltos de línea
    ("A circular-economy strategy", 1, 0),              # forma compuesta
    ("We refurbish and compost the residues.", 0, 2),   # stems desnudos
    ("Mechanical recycling of PET", 0, 1),              # sin doble conteo
    ("life-cycle assessment (LCA)", 0, 2),              # menciones separadas
    ("the lca of once", 0, 0),                          # acrónimo en minúscula
    ("CE frameworks and CE strategies", 2, 0),
    (None, 0, 0),
]


def _selftest() -> int:
    failures = 0
    for text, exp_talk, exp_practice in _SELFTEST_CASES:
        got = (count_talk(text), count_practice(text))
        if got != (exp_talk, exp_practice):
            failures += 1
            print(f"FALLO {text!r}: esperado {(exp_talk, exp_practice)}, obtenido {got}")
    print(f"ce_vocab {__version__}: {len(_SELFTEST_CASES) - failures}/"
          f"{len(_SELFTEST_CASES)} comprobaciones OK")
    return 1 if failures else 0


def _dump() -> None:
    import csv
    import sys
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["lexicon_version", "class", "label", "pattern"])
    for kind, terms in (("talk", CE_TALK_TERMS), ("practice", CE_PRACTICE_TERMS)):
        for label, pat in terms.items():
            w.writerow([__version__, kind, label, pat])


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=f"ce_vocab {__version__}")
    ap.add_argument("--dump", action="store_true",
                    help="imprime el léxico en CSV (tabla del anexo)")
    ap.add_argument("--selftest", action="store_true",
                    help="ejecuta las comprobaciones de los patrones")
    ns = ap.parse_args()

    if ns.dump:
        try:
            _dump()
        except BrokenPipeError:      # p. ej. `python ce_vocab.py --dump | head`
            sys.stderr.close()
    elif ns.selftest:
        sys.exit(_selftest())
    else:
        print(f"ce_vocab {__version__} | CE-talk: {len(CE_TALK_TERMS)} términos | "
              f"CE-practice: {len(CE_PRACTICE_TERMS)} términos")
        print("Usa --dump o --selftest. Documentación: docstring del módulo.")
