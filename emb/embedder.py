"""Embedding-laget: gjør tekst om til vektorer.

To implementasjoner bak samme grensesnitt:

  * VoyageEmbedder  – produksjonskvalitet. Anthropic har ikke eget
    embeddings-API og anbefaler Voyage AI (https://www.voyageai.com).
    Krever VOYAGE_API_KEY i miljøet.
  * LokalEmbedder   – deterministisk fallback (hashede tegn-n-gram) som gjør
    at hele pipelinen kan kjøres og testes uten API-nøkkel. Kvaliteten er
    langt under en ekte embeddingmodell – bruk den kun til utvikling/demo.

Viktigste regel i praksis: indeks og spørring MÅ bruke samme embedder
(samme modell, samme versjon). Indeksen lagrer derfor en `embedder_id`,
og GrafRAG nekter å svare hvis id-en ikke stemmer med aktiv embedder.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np

VOYAGE_MODELL = "voyage-3.5"
LOKAL_DIM = 512


class VoyageEmbedder:
    """Ekte embeddings via Voyage AI. Asymmetrisk: dokumenter og spørringer
    embeddes med ulik input_type, som gir bedre gjenfinning."""

    id = f"voyage:{VOYAGE_MODELL}"

    def __init__(self) -> None:
        import voyageai  # importeres først her, så fallback fungerer uten pakken

        self._klient = voyageai.Client()  # leser VOYAGE_API_KEY fra miljøet

    def embed_dokumenter(self, tekster: list[str]) -> np.ndarray:
        vektorer: list[list[float]] = []
        for i in range(0, len(tekster), 128):  # API-et tar begrensede batcher
            svar = self._klient.embed(
                tekster[i : i + 128], model=VOYAGE_MODELL, input_type="document"
            )
            vektorer.extend(svar.embeddings)
        return _normaliser(np.asarray(vektorer, dtype=np.float32))

    def embed_sporring(self, tekst: str) -> np.ndarray:
        svar = self._klient.embed([tekst], model=VOYAGE_MODELL, input_type="query")
        return _normaliser(np.asarray(svar.embeddings, dtype=np.float32))[0]


class LokalEmbedder:
    """Hashede tegn-n-gram (3–5) i en fast-dimensjonal vektor.

    Fanger ordoverlapp og bøyninger på norsk godt nok til en demo, men har
    ingen semantisk forståelse ("snø" og "vinterlast" ligner ikke).
    """

    id = f"lokal:ngram-v1:{LOKAL_DIM}"

    def embed_dokumenter(self, tekster: list[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in tekster])

    def embed_sporring(self, tekst: str) -> np.ndarray:
        return self._embed(tekst)

    def _embed(self, tekst: str) -> np.ndarray:
        v = np.zeros(LOKAL_DIM, dtype=np.float32)
        ren = re.sub(r"[^a-z0-9æøå ]", " ", tekst.lower())
        ren = " " + re.sub(r"\s+", " ", ren).strip() + " "
        for n in (3, 4, 5):
            for i in range(len(ren) - n + 1):
                h = hashlib.md5(ren[i : i + n].encode()).digest()
                idx = int.from_bytes(h[:4], "little") % LOKAL_DIM
                fortegn = 1.0 if h[4] % 2 else -1.0
                v[idx] += fortegn
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v


def _normaliser(m: np.ndarray) -> np.ndarray:
    normer = np.linalg.norm(m, axis=1, keepdims=True)
    normer[normer == 0] = 1.0
    return m / normer


def lag_embedder():
    """Velg embedder ut fra miljøet: Voyage hvis nøkkel finnes, ellers lokal."""
    if os.environ.get("VOYAGE_API_KEY"):
        return VoyageEmbedder()
    print(
        "ADVARSEL: VOYAGE_API_KEY er ikke satt – bruker LokalEmbedder "
        "(kun for utvikling/demo, vesentlig dårligere gjenfinning)."
    )
    return LokalEmbedder()
