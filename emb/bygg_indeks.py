"""Bygger embedding-indeksen fra regler.json.

Dette er "indekseringstiden" i RAG-oppsettet og kjøres hver gang reglene
endres (typisk i CI eller som pre-commit):

  1. Les kunnskapsgrafen (noder + kanter).
  2. Lag én gjenfinningstekst per node (tittel + sammendrag + kontekst-tagger).
     Nodene her er korte; lengre regeltekster bør deles i biter på
     ~200–400 tokens med litt overlapp, med flere biter som peker på samme node.
  3. Embed alle tekstene som DOKUMENTER (input_type="document" hos Voyage).
  4. Lagre vektormatrisen + metadata (inkl. embedder-id) i emb/indeks/.

Kjør:  python3 bygg_indeks.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embedder import lag_embedder

HER = Path(__file__).parent
INDEKS_KATALOG = HER / "indeks"


def gjenfinningstekst(node: dict) -> str:
    """Teksten som embeddes for en node. Kontekst-taggene skrives om til ord
    ("rib.laster.nytte-kategori" -> "laster nytte kategori") slik at de bidrar
    til treff i vektorrommet."""
    tagger = " ".join(
        t.replace("rib.", "").replace(".", " ").replace("-", " ")
        for t in node.get("kontekst", [])
    )
    return f"{node['tittel']}\n{node['tekst']}\nStikkord: {tagger}"


def bygg() -> None:
    graf = json.loads((HER / "regler.json").read_text(encoding="utf-8"))
    noder = graf["noder"]

    embedder = lag_embedder()
    tekster = [gjenfinningstekst(n) for n in noder]
    print(f"Embedder {len(tekster)} regelnoder med {embedder.id} …")
    vektorer = embedder.embed_dokumenter(tekster)

    INDEKS_KATALOG.mkdir(exist_ok=True)
    np.save(INDEKS_KATALOG / "vektorer.npy", vektorer)
    meta = {
        "embedder_id": embedder.id,
        "dimensjon": int(vektorer.shape[1]),
        "node_ider": [n["id"] for n in noder],
    }
    (INDEKS_KATALOG / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"OK: {vektorer.shape[0]} vektorer ({vektorer.shape[1]} dim) "
        f"lagret i {INDEKS_KATALOG}/"
    )


if __name__ == "__main__":
    bygg()
