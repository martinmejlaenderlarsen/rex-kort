# EMB – LLM-chat med graf-RAG over regler

Et komplett, kjørbart oppsett der en **LLM-chat** står foran og en **graf-RAG**
med regler står bak. Chatten henter de riktige reglene via embeddings
(vektorsøk) og en kunnskapsgraf, og svarer med kildehenvisninger.

```
                       ┌──────────────────────── emb/ ────────────────────────┐
                       │                                                      │
 nettleser             │   server.py                     graf_rag.py          │
 chat.html ──spørsmål──▶  POST /api/chat ──────────────▶ 1) embed spørring    │
     ▲                 │        │                        2) vektorsøk (cos)   │
     │                 │        │                        3) graf-ekspansjon   │
     │   SSE-strøm     │        ▼                             │               │
     └── kilder+svar ──┤  Claude (claude-opus-4-8)  ◀── regler som kontekst   │
                       │                                                      │
                       │  bygg_indeks.py (kjøres når regler endres)           │
                       │  regler.json ──▶ embeddinger ──▶ indeks/vektorer.npy │
                       └──────────────────────────────────────────────────────┘
```

## Filene

| Fil | Rolle |
|---|---|
| `regler.json` | Kunnskapsgrafen: 34 noder (standarder, forskrifter, praksisregler for RIB) og kanter mellom dem (`hjemlet_i`, `henviser_til`, `relatert_til`, …). Tekstene er sammendrag for gjenfinning – **ikke** autoritativ standardtekst. |
| `embedder.py` | Embedding-laget. Voyage AI (`voyage-3.5`) i produksjon, lokal deterministisk fallback for utvikling uten nøkkel. |
| `bygg_indeks.py` | Indekseringstid: embedder alle regelnoder og lagrer vektormatrisen i `indeks/`. |
| `graf_rag.py` | Spørretid: query-embedding → cosinus-søk → graf-ekspansjon → kontekstblokk. |
| `server.py` | FastAPI-server: `POST /api/chat` (Claude + strømming), `GET /api/sok` (kun henting, for feilsøking). |
| `chat.html` | Chat-UI i samme stil som kort-appen. Viser hvilke regler som ble hentet under hvert svar. |

## Kom i gang

```bash
cd emb
pip install -r requirements.txt

export VOYAGE_API_KEY=...      # embeddings (uten: lokal demo-fallback)
export ANTHROPIC_API_KEY=...   # chat (kreves for /api/chat)

python3 bygg_indeks.py                       # bygg vektorindeksen
python3 graf_rag.py "snølast på flate tak"   # test henting i terminalen
uvicorn server:app --reload --port 8100      # start chatten → http://localhost:8100
```

`GET /api/sok?q=...` viser hva som hentes for et spørsmål uten å bruke
LLM-tokens – bruk den til å tune gjenfinningen.

## Hvordan implementeres emb (embeddings) i praksis?

En embedding er en vektor (her: ~1000 flyttall) som plasserer en tekst i et
rom der **tekster med lik betydning ligger nær hverandre** – også når de ikke
deler ord ("snødrift ved parapet" ligger nær "fonndannelse på tak").
Det er dette som lar chatten finne riktig regel uansett hvordan brukeren
formulerer seg. Slik er det implementert her, punkt for punkt:

### 1. To tidspunkter: indeksering og spørring

Embeddings brukes på to helt adskilte tidspunkter:

- **Indekseringstid** (`bygg_indeks.py`): hver regelnode embeddes én gang og
  lagres. Kjøres bare når reglene endres – f.eks. som CI-steg når
  `regler.json` committes.
- **Spørretid** (`graf_rag.py`): brukerens spørsmål embeddes per melding
  (én rask API-kall/beregning), og sammenlignes mot de lagrede vektorene.

LLM-en er aldri involvert i selve hentingen – embeddings er en egen, billig
modell. LLM-en ser først reglene etter at de er hentet.

### 2. Samme modell på begge sider – alltid

Vektorer fra ulike modeller (eller ulike versjoner av samme modell) lever i
ulike rom og kan ikke sammenlignes. Derfor lagrer indeksen en `embedder_id`
(`indeks/meta.json`), og `GrafRAG` **nekter å starte** hvis aktiv embedder
ikke matcher indeksen. Bytter du embeddingmodell, må hele indeksen bygges på
nytt – det er derfor re-indeksering skal være et automatisert steg, ikke noe
man husker på.

### 3. Asymmetriske embeddings: dokument vs. spørring

Moderne embeddingmodeller skiller mellom å embedde et *dokument* og en
*spørring* (`input_type="document"` / `"query"` hos Voyage). Et kort spørsmål
og en lang regeltekst skal treffe hverandre selv om de er ulike i form – den
asymmetriske treningen håndterer akkurat det. Se `VoyageEmbedder` i
`embedder.py`.

*(Anthropic har ikke eget embeddings-API og anbefaler Voyage AI. Selve
chatten går mot Claude; kun vektoriseringen går mot Voyage.)*

### 4. Hva som embeddes: én gjenfinningstekst per regelnode

`gjenfinningstekst()` i `bygg_indeks.py` bygger teksten som embeddes:
`tittel + sammendrag + kontekst-tagger` (taggene skrives om til vanlige ord
så de bidrar i vektorrommet). Nodene her er korte, så én vektor per node
holder. For lengre regeltekster (kapitler, veiledere) gjelder:

- del i biter på ~200–400 tokens med litt overlapp,
- embed hver bit for seg, men la alle bitene peke tilbake på samme node i
  grafen – da forblir grafen "sannheten" og vektorene bare inngangsdører.

### 5. Søket: cosinus-likhet som ett prikkprodukt

Alle vektorer L2-normaliseres ved lagring. Da er cosinus-likhet det samme som
prikkprodukt, og hele søket er én linje: `scorer = vektorer @ q`
(`graf_rag.py`). Med 34 regler er det øyeblikkelig; en numpy-matrise holder
fint opp til titusenvis av biter. Først når korpuset blir større enn det, er
det verdt å bytte til en vektordatabase (pgvector, Qdrant, Chroma) – API-et i
`GrafRAG.hent()` forblir det samme.

### 6. Grafen: derfor holder ikke vektorsøk alene

Vektorsøket finner *inngangspunkter*, men regler henger sammen: en
praksisregel er hjemlet i en standard, en standard har utførelseskrav i en
annen. Spør du om toleranser mellom prefab og plasstøpt, treffer vektoren
`regel-toleransekjeder` – og grafen drar automatisk med `ns-en-13670` som
regelen er hjemlet i, pluss relaterte grensesnittsregler. Implementasjonen:

- topp-k vektortreff blir *frønoder*,
- kantene følges ett hopp ut; naboer arver frøets score × `NABO_DEMPING`
  (0.55), så direkte treff alltid rangeres foran,
- relasjonene sendes med i konteksten, så LLM-en kan si *hvorfor* en regel
  er relevant ("hjemlet i …").

To enkle terskler holder støy ute: `MIN_SCORE` kutter svake frø, og
`maks_noder` begrenser hvor mye kontekst LLM-en får.

### 7. Konteksten til LLM-en – og cache-vennlig prompt

De hentede reglene formateres som `<regel id="...">`-blokker og legges i
**brukermeldingen**, mens systemprompten holdes byte-identisk med
`cache_control: ephemeral` (`server.py`). Da gjenbrukes systemprompten fra
Anthropics prompt-cache på hver melding, og bare de varierende reglene +
spørsmålet prosesseres på nytt. Systemprompten instruerer Claude om å sitere
regel-id-er (`[ns-en-1991-1-3]`), å ikke oppgi tallverdier som ikke står i
konteksten, og å si ifra når reglene ikke dekker spørsmålet.

### 8. Drift: hva du må huske

- **Re-indekser ved endring**: `regler.json` endret ⇒ kjør `bygg_indeks.py`.
  Legg det som CI-steg.
- **Versjonér embedderen**: bytt aldri modell uten full re-indeksering
  (håndheves av `embedder_id`-sjekken).
- **Kostnad**: embeddings er billige (brøkdel av LLM-tokens); den dyre delen
  er LLM-svaret. Graf-ekspansjonen koster ingenting ekstra – den gjenbruker
  vektortreffene.
- **Kvalitet måles i hentingen først**: hvis svarene er dårlige, sjekk
  `GET /api/sok` før du justerer prompten. Feil regel inn ⇒ feil svar ut.

## Fallback uten nøkler (kun utvikling)

Uten `VOYAGE_API_KEY` brukes `LokalEmbedder` (hashede tegn-n-gram). Den gjør
pipelinen kjørbar og testbar offline, men har ingen semantisk forståelse –
den matcher bokstavoverlapp, ikke betydning. Ikke bruk den i produksjon.

## Videre arbeid

- Bytt ut demo-reglene i `regler.json` med virksomhetens faktiske regelverk
  (samme node/kant-format).
- Chunking av fulle regeltekster (punkt 4 over) når kildene blir lengre.
- Vektordatabase når korpuset passerer ~10⁴–10⁵ biter.
- Logg spørsmål → hentede regler → svar for å evaluere gjenfinningen over tid.
