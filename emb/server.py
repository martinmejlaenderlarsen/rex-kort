"""Chat-server: LLM-chat foran, graf-RAG med regler bak.

Flyt per melding:

  nettleser ──POST /api/chat──▶ server
      1. GrafRAG.hent(spørsmål)          (embedding + vektorsøk + graf)
      2. Kildene sendes til klienten først (SSE-hendelse "kilder")
      3. Claude (claude-opus-4-8) svarer strømmet, med reglene som kontekst
         og instruks om å sitere regel-id-er

Krever ANTHROPIC_API_KEY for chat. /api/sok fungerer uten (kun henting).
Kjør:  uvicorn server:app --reload --port 8100   (fra emb/-katalogen)
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from graf_rag import GrafRAG

HER = Path(__file__).parent
MODELL = "claude-opus-4-8"

# Stabil systemprompt – caches hos Anthropic (cache_control på blokken).
# De hentede reglene legges i brukerturen, ETTER cache-grensen, siden de
# varierer per spørsmål.
SYSTEMPROMPT = """Du er en fagassistent for rådgivende ingeniører bygg (RIB) i Norge.

Du får relevante regler vedlagt i <regel>-blokker i hver brukermelding, hentet
fra en kunnskapsgraf med standarder, forskrifter og praksisregler.

Regler for svarene dine:
- Svar på norsk, presist og praktisk rettet.
- Bygg svaret på de vedlagte reglene. Henvis til dem med id i hakeparentes,
  f.eks. [ns-en-1991-1-3] eller [regel-snodrift], der du bruker dem.
- Regeltekstene er sammendrag, ikke autoritativ standardtekst. Ikke oppgi
  konkrete tallverdier (faktorer, laster, grenseverdier) som om de sto i
  vedlegget – henvis i stedet til hvilken standard verdien må slås opp i.
- Dekker ikke de vedlagte reglene spørsmålet, si det tydelig i stedet for å
  gjette. Foreslå gjerne hvilket fagområde/standard brukeren bør sjekke.
- Relasjonslinjene i reglene viser hvordan regler henger sammen – bruk dem til
  å peke på tilgrensende krav brukeren bør være klar over."""

app = FastAPI(title="EMB regelchat")
rag = GrafRAG()
klient = anthropic.Anthropic()  # leser ANTHROPIC_API_KEY fra miljøet


class Melding(BaseModel):
    rolle: str   # "bruker" | "assistent"
    tekst: str


class ChatForesporsel(BaseModel):
    sporsmal: str
    historikk: list[Melding] = []


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def forside():
    return FileResponse(HER / "chat.html")


@app.get("/api/sok")
def sok(q: str):
    """Ren henting uten LLM – nyttig for feilsøking og tuning av gjenfinning."""
    treff = rag.hent(q)
    return {
        "sporsmal": q,
        "treff": [
            {
                "id": t.node["id"],
                "tittel": t.node["tittel"],
                "type": t.node["type"],
                "score": round(t.score, 4),
                "via": t.via,
                "relasjoner": t.relasjoner,
            }
            for t in treff
        ],
    }


@app.post("/api/chat")
def chat(fs: ChatForesporsel):
    treff = rag.hent(fs.sporsmal)
    kontekst = rag.bygg_kontekst(treff)

    meldinger = [
        {"role": "user" if m.rolle == "bruker" else "assistant", "content": m.tekst}
        for m in fs.historikk
    ]
    meldinger.append(
        {
            "role": "user",
            "content": (
                f"Relevante regler hentet fra kunnskapsgrafen:\n\n{kontekst}\n\n"
                f"Spørsmål: {fs.sporsmal}"
            ),
        }
    )

    def strom():
        yield _sse(
            {
                "type": "kilder",
                "kilder": [
                    {
                        "id": t.node["id"],
                        "tittel": t.node["tittel"],
                        "type": t.node["type"],
                        "score": round(t.score, 3),
                        "via": t.via,
                    }
                    for t in treff
                ],
            }
        )
        try:
            with klient.messages.stream(
                model=MODELL,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=[
                    {
                        "type": "text",
                        "text": SYSTEMPROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=meldinger,
            ) as strom_:
                for tekst in strom_.text_stream:
                    yield _sse({"type": "delta", "tekst": tekst})
            yield _sse({"type": "slutt"})
        except anthropic.AuthenticationError:
            yield _sse(
                {"type": "feil", "melding": "ANTHROPIC_API_KEY mangler eller er ugyldig."}
            )
        except anthropic.RateLimitError:
            yield _sse(
                {"type": "feil", "melding": "Ratebegrensning hos API-et – prøv igjen om litt."}
            )
        except anthropic.APIStatusError as e:
            yield _sse({"type": "feil", "melding": f"API-feil ({e.status_code}): {e.message}"})
        except anthropic.APIConnectionError:
            yield _sse({"type": "feil", "melding": "Fikk ikke kontakt med API-et."})
        except Exception as e:  # f.eks. TypeError når ingen API-nøkkel er satt
            yield _sse({"type": "feil", "melding": f"{type(e).__name__}: {e}"})

    return StreamingResponse(strom(), media_type="text/event-stream")
