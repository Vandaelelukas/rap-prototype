# RAG Prototype — Project Context

## Project overview
- **Klant:** Van Houcke NV (Jabbeke)
- **Gebouwd door:** Lukas Vandaele (LeadLine)
- **Doel:** RAG-kennissysteem waarmee medewerkers vragen stellen over BC-processen en Innomotics-offertes
- **Deployment:** Streamlit Cloud → `rap-prototype-6p2pcsrzjcnnaztaqu2tja.streamlit.app`
- **Repo:** `github.com/Vandaelelukas/rap-prototype` (branch: `main`)

---

## Tech stack

| Laag | Keuze |
|---|---|
| Frontend | Streamlit Cloud |
| Vector DB | Supabase pgvector |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM (antwoorden) | `gpt-4.1` |
| LLM (herschrijven) | `gpt-4.1-mini` |
| LLM (classificeren) | `gpt-4.1-nano` |
| Retrieval | LlamaIndex hybrid (pgvector + BM25) |
| PDF extractie | Docling (primair) + pdfplumber (fallback) |
| Telemetry | Arize Phoenix (`van_houcke_rag` project) |
| Authenticatie | Microsoft MSAL (multi-tenant, `/organizations`) |

---

## Bestandsstructuur

```
rap-prototype/
├── app.py                        # Streamlit frontend + volledige RAG pipeline
├── ingest.py                     # Documenten inladen + embedden naar Supabase
├── cleanup.py                    # Fathom transcriptie → BC procesdocument
├── cleanup_pdf.py                # PDF → tekst via pdfplumber (oud)
├── cleanup_pdf_docling.py        # PDF → tekst via Docling (actief)
├── scrape_docs.py                # Microsoft BC docs scrapen
├── eval_run.py                   # Evaluatie script (lokal uitvoeren)
├── evaluatie.jsonl               # 20 testvragen met verwachte antwoorden
├── nodes_cache.pkl               # BM25 node cache (gegenereerd door ingest.py)
├── requirements.txt              # Streamlit Cloud dependencies
└── requirements-local.txt        # Lokale dependencies (docling, pdfplumber, etc.)
```

---

## Requirements

### `requirements.txt` (Streamlit Cloud)
```
openai
llama-index
llama-index-vector-stores-supabase
llama-index-retrievers-bm25
llama-index-llms-openai
rank-bm25
python-dotenv
streamlit
vecs
arize-phoenix-otel
openinference-instrumentation-llama-index
msal
```

### `requirements-local.txt`
```
docling
pdfplumber
beautifulsoup4
requests
```

---

## Documenten in het systeem

### BC-processen (`documenttype: bc_proces`)
- `BC_inkooporder.txt` — volledig inkoopproces t.e.m. geboekte factuur
- `document.txt` — assemblageorder aanmaken
- `document2.txt` — assemblageorder vrijgeven
- `ATO - ATS.txt` — Assemble to Order vs Assemble to Stock (gescraped)

### Innomotics offertes (`documenttype: offerte`)
- `5084129_AA_IPW56_cleaned.txt` — Gamma rings NBR (feb 2024)
- `5097042_AA_Hoekcontactlager_cleaned.txt` — SKF hoekcontactlagers (jul 2024)
- `5132300_AA_55KW-IE4-225_cleaned.txt` — 55kW IE4 motor (mei 2025)

---

## RAG pipeline architectuur

```
Vraag binnenkomt
     ↓
herschrijf_vraag() — gpt-4.1-mini
     ↓
classificeer_vraag() — gpt-4.1-nano → JSON
{
  "documenttype": "offerte" | "bc_proces" | "algemeen",
  "leverancier": "Innomotics" | null,
  "referentienummer": "5132300" | null
}
     ↓
bouw_retriever() — MetadataFilters op documenttype + leverancier + referentienummer
     ↓
QueryFusionRetriever — pgvector (top-k=5) + BM25 (top-k=5) → reciprocal_rerank
     ↓
RetrieverQueryEngine — gpt-4.1 genereert antwoord
     ↓
Antwoord + 🔍 filter label tonen
```

---

## Metadata per document

Gegenereerd door `ingest.py` tijdens ingest:

| Veld | Beschrijving | Voorbeeld |
|---|---|---|
| `documenttype` | `bc_proces` / `offerte` / `algemeen` | `offerte` |
| `bestandsnaam` | originele bestandsnaam | `5132300_AA_cleaned.txt` |
| `referentienummer` | offertenummer (regex) | `5132300` |
| `leverancier` | genormaliseerd (zonder GmbH etc.) | `Innomotics` |

**Normalisatie leverancier:** `Innomotics GmbH` → `Innomotics`, `Innomotics s.r.o.` → `Innomotics`

---

## Chunking parameters

```python
SentenceSplitter(chunk_size=512, chunk_overlap=50)
```

---

## Prompt (anti-hallucinatie)

```
Je bent een assistent die vragen beantwoordt op basis van de beschikbare documentatie.
Gebruik ALLEEN de informatie uit de context hieronder om de vraag te beantwoorden.
Als het antwoord niet in de context staat, zeg dan expliciet:
'Dit staat niet in de beschikbare documentatie.'
Verzin nooit informatie die niet in de context staat.
Geef altijd een VOLLEDIG antwoord. Als er meerdere methoden of stappen zijn,
som ze allemaal op. Stop niet na de eerste optie.
```

---

## Authenticatie (MSAL)

- **Library:** `msal` (geen Streamlit native `st.login()` — te buggy met Microsoft)
- **Authority:** `https://login.microsoftonline.com/organizations` (multi-tenant)
- **Scopes:** `["User.Read"]`
- **Flow:** Authorization code flow → token inwisselen → user in `st.session_state`

### Azure app registratie
- **App naam:** RAG Van Houcke
- **Client ID:** `59e2527d-3e59-4a33-ae0e-115caef81621`
- **Supported account types:** Multitenant (organizations)
- **Redirect URI:** `https://rap-prototype-6p2pcsrzjcnnaztaqu2tja.streamlit.app`

### Streamlit Secrets vereist
```toml
AZURE_CLIENT_ID = "..."
AZURE_CLIENT_SECRET = "..."
AZURE_TENANT_ID = "..."
AZURE_REDIRECT_URI = "https://rap-prototype-6p2pcsrzjcnnaztaqu2tja.streamlit.app"
OPENAI_API_KEY = "..."
SUPABASE_DB_URL = "postgresql://..."
PHOENIX_API_KEY = "..."
PHOENIX_COLLECTOR_ENDPOINT = "https://app.phoenix.arize.com"
```

---

## Telemetry (Arize Phoenix)

- **Project:** `van_houcke_rag` (productie), `van_houcke_rag_eval` (evaluatie)
- **Account:** `lukasvandaele0 Space` op `app.phoenix.arize.com`
- **Integratie:** `openinference-instrumentation-llama-index` + `arize-phoenix-otel`
- **Initialisatie:** na authenticatiecheck om herinitialisatie te vermijden

```python
if "phoenix_initialized" not in st.session_state:
    tracer_provider = register(project_name="van_houcke_rag")
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    st.session_state.phoenix_initialized = True
```

---

## Evaluatie

### Evaluatieset
- **Bestand:** `evaluatie.jsonl` (20 vragen)
- **Categorieën:** `offerte` (10), `bc_proces` (8), `anti-hallucinatie` (2)
- **Kolommen:** `id`, `categorie`, `input`, `expected_output`, `bron`

### Uitvoeren
```bash
python eval_run.py
```
Traces verschijnen in Phoenix onder `van_houcke_rag_eval`.

### Resultaten sessie 1
- **Score:** 19/20 correct (95%)
- **Falend:** E09 (datum offerte 5097042) — keyword scoring artefact, antwoord was correct
- **Conclusie:** systeem werkt goed op huidige documentset

---

## Workflow: nieuwe documenten toevoegen

### BC-procesvideo (Fathom transcriptie)
1. Kopieer transcriptie in `cleanup.py`
2. `python cleanup.py` → genereert `document_X.txt`
3. Verplaats naar `documenten/`

### Innomotics offerte (PDF)
1. Zet PDF in `pdfs/` map
2. `pip install -r requirements-local.txt`
3. `python cleanup_pdf_docling.py` → genereert `*_cleaned.txt` in `documenten/`

### Na toevoegen van documenten
1. Verwijder oude data uit Supabase: `DELETE FROM vecs.documenten;`
2. `python ingest.py` → nieuwe `nodes_cache.pkl` + Supabase index
3. Commit en push `nodes_cache.pkl` naar repo

---

## Bekende issues & beslissingen

| Issue | Beslissing |
|---|---|
| `st.login()` crasht met Microsoft OAuth (500 op `/oauth2callback`) | Vervangen door MSAL library |
| Streamlit native auth `[auth]` sectie met `common` tenant → `InvalidClaimError` | Geen `common`, gebruik `organizations` authority |
| `langfuse.llama_index` bestaat niet in Langfuse 4.x | Overgestapt naar Arize Phoenix |
| `nodes_cache.pkl` leeg → BM25 crash | Dubbele fallback: documenttype-only → alle nodes |
| `chromadb` in requirements maar nergens gebruikt | Verwijderd |
| `docling` te zwaar voor Streamlit Cloud | Enkel in `requirements-local.txt` |

---

## Volgende stappen (geprioriteerd)

1. **Authenticatie stabiel maken** — Microsoft login loop definitief oplossen
2. **Meer BC-processen** transcriberen en inladen (meer Fathom opnames)
3. **Metadata routing verfijnen** — `proces_type` toevoegen (`inkoop`, `assemblage`, `verkoop`)
4. **Evaluatieset uitbreiden** naarmate meer documenten worden toegevoegd
5. **Fase 3:** Teams integratie
6. **Fase 4:** Uitrol naar andere klanten (LeadLine product)
