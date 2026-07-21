"""Graf-RAG: henter de riktige reglene til et spørsmål.

Hentingen skjer i tre steg ("spørretiden" i RAG-oppsettet):

  1. EMBED SPØRRINGEN med samme embedder som indeksen ble bygget med
     (input_type="query" hos Voyage – asymmetrisk mot dokumentene).
  2. VEKTORSØK: cosinus-likhet mot alle regelvektorer. Siden alle vektorer
     er L2-normalisert er cosinus = prikkprodukt, én matrise-multiplikasjon.
     De beste treffene blir "frønoder".
  3. GRAF-EKSPANSJON: følg kantene ett hopp ut fra frønodene. En praksisregel
     drar med seg standarden den er hjemlet i, og omvendt. Naboer arver
     frønodens score ganget med en dempingsfaktor, så direkte treff alltid
     rangeres foran naboer.

Resultatet er en rangert liste noder + relasjonene mellom dem, som
`bygg_kontekst()` formaterer til en kontekstblokk for LLM-en.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from embedder import lag_embedder

HER = Path(__file__).parent
NABO_DEMPING = 0.55   # naboens score = frøscore * demping
MIN_SCORE = 0.08      # under dette regnes treffet som støy og droppes


@dataclass
class Treff:
    node: dict
    score: float
    via: str | None = None          # node-id treffet ble ekspandert fra
    relasjoner: list[str] = field(default_factory=list)


class GrafRAG:
    def __init__(self) -> None:
        graf = json.loads((HER / "regler.json").read_text(encoding="utf-8"))
        self.noder = {n["id"]: n for n in graf["noder"]}
        self.kanter = graf["kanter"]

        # naboliste (grafen behandles som urettet ved ekspansjon)
        self.naboer: dict[str, list[dict]] = defaultdict(list)
        for k in self.kanter:
            self.naboer[k["fra"]].append({**k, "nabo": k["til"]})
            self.naboer[k["til"]].append({**k, "nabo": k["fra"]})

        meta = json.loads((HER / "indeks" / "meta.json").read_text(encoding="utf-8"))
        self.vektorer = np.load(HER / "indeks" / "vektorer.npy")
        self.node_ider = meta["node_ider"]

        self.embedder = lag_embedder()
        if self.embedder.id != meta["embedder_id"]:
            raise RuntimeError(
                f"Indeksen er bygget med {meta['embedder_id']!r}, men aktiv "
                f"embedder er {self.embedder.id!r}. Kjør bygg_indeks.py på nytt "
                "– indeks og spørring må alltid bruke samme embeddingmodell."
            )

    def hent(self, sporsmal: str, antall_fro: int = 4, maks_noder: int = 8) -> list[Treff]:
        # 1) embed spørringen
        q = self.embedder.embed_sporring(sporsmal)

        # 2) vektorsøk (cosinus = prikkprodukt på normaliserte vektorer)
        scorer = self.vektorer @ q
        fro_idx = np.argsort(scorer)[::-1][:antall_fro]

        treff: dict[str, Treff] = {}
        for i in fro_idx:
            score = float(scorer[i])
            if score < MIN_SCORE:
                continue
            nid = self.node_ider[i]
            treff[nid] = Treff(node=self.noder[nid], score=score)

        # 3) graf-ekspansjon, ett hopp fra hver frønode
        for fro_id, fro in list(treff.items()):
            for kant in self.naboer[fro_id]:
                nabo_id = kant["nabo"]
                relasjon = f"{self.noder[kant['fra']]['tittel']} —{kant['relasjon']}→ " \
                           f"{self.noder[kant['til']]['tittel']}"
                nabo_score = fro.score * NABO_DEMPING
                if nabo_id in treff:
                    eksisterende = treff[nabo_id]
                    eksisterende.score = max(eksisterende.score, nabo_score)
                    eksisterende.relasjoner.append(relasjon)
                else:
                    treff[nabo_id] = Treff(
                        node=self.noder[nabo_id], score=nabo_score,
                        via=fro_id, relasjoner=[relasjon],
                    )

        rangert = sorted(treff.values(), key=lambda t: t.score, reverse=True)
        return rangert[:maks_noder]

    def bygg_kontekst(self, treff: list[Treff]) -> str:
        """Formater hentede regler som kontekstblokk til LLM-en."""
        deler = []
        for t in treff:
            n = t.node
            del_ = (
                f"<regel id=\"{n['id']}\" type=\"{n['type']}\">\n"
                f"{n['tittel']}\n{n['tekst']}"
            )
            if t.relasjoner:
                del_ += "\nRelasjoner: " + "; ".join(sorted(set(t.relasjoner)))
            deler.append(del_ + "\n</regel>")
        return "\n\n".join(deler)


if __name__ == "__main__":
    import sys

    rag = GrafRAG()
    sporsmal = " ".join(sys.argv[1:]) or "Hvilke krav gjelder for snølast på flate tak med solceller?"
    print(f"Spørsmål: {sporsmal}\n")
    for t in rag.hent(sporsmal):
        via = f"  (via {t.via})" if t.via else ""
        print(f"{t.score:.3f}  {t.node['id']:32s} {t.node['tittel']}{via}")
