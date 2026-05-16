import os
import json
import pickle
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25.base import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from openai import OpenAI as OpenAIClient

# ── Langfuse telemetry ──────────────────────────────────────────────────────
# Pikt LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY en LANGFUSE_HOST automatisch
# op uit environment variables (lokaal via .env, Streamlit Cloud via Secrets)
from langfuse.llama_index import LlamaIndexInstrumentor
instrumentor = LlamaIndexInstrumentor()
instrumentor.start()
# ───────────────────────────────────────────────────────────────────────────

load_dotenv()

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = LlamaOpenAI(model="gpt-4.1", temperature=0)

st.set_page_config(page_title="RAG Assistent", page_icon="🤖", layout="centered")
st.title("📚 RAG Kennissysteem")
st.caption("Stel vragen over de beschikbare documentatie")

HISTORY_WINDOW = 6


def bouw_geschiedenis_tekst(berichten: list, max_beurten: int = HISTORY_WINDOW) -> str:
    recente = berichten[-(max_beurten * 2):]
    if not recente:
        return ""
    regels = []
    for b in recente:
        prefix = "Gebruiker" if b["rol"] == "user" else "Assistent"
        regels.append(f"{prefix}: {b['inhoud']}")
    return "\n".join(regels)


def herschrijf_vraag(vraag: str, geschiedenis: str) -> str:
    if not geschiedenis:
        return vraag

    client = OpenAIClient()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Herschrijf de laatste vraag als een zelfstandige, volledige zoekopdracht "
                    "die geen eerdere context nodig heeft. Voeg relevante entiteiten of onderwerpen "
                    "toe vanuit de gespreksgeschiedenis als de vraag daar naar verwijst. "
                    "Geef ALLEEN de herschreven vraag terug, niets anders."
                )
            },
            {
                "role": "user",
                "content": f"Gespreksgeschiedenis:\n{geschiedenis}\n\nLaatste vraag: {vraag}"
            }
        ]
    )
    return response.choices[0].message.content.strip()


def classificeer_vraag(vraag: str) -> dict:
    client = OpenAIClient()
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Analyseer de vraag en geef een JSON-object terug met deze velden:\n\n"
                    "- documenttype (verplicht): één van deze drie waarden:\n"
                    "  * 'offerte': vragen over prijzen, artikelnummers, leveranciers, "
                    "levertijden, minimumbestellingen, incoterms, offertegeldigheid\n"
                    "  * 'bc_proces': vragen over hoe iets werkt of uitgevoerd wordt in "
                    "Business Central, stappen, navigatie, knoppen, statussen\n"
                    "  * 'algemeen': alles wat niet duidelijk in één van bovenstaande past\n\n"
                    "- leverancier (optioneel): de leveranciersnaam als die expliciet vermeld "
                    "wordt in de vraag, anders null\n\n"
                    "- referentienummer (optioneel): het offertereferentienummer als dat "
                    "expliciet vermeld wordt in de vraag, anders null\n\n"
                    "Geef ALLEEN een geldig JSON-object terug, geen uitleg of extra tekst.\n"
                    'Voorbeeld: {"documenttype": "offerte", "leverancier": "Innomotics", "referentienummer": null}'
                )
            },
            {
                "role": "user",
                "content": vraag
            }
        ]
    )

    try:
        resultaat = json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        resultaat = {}

    # Veiligheidscheck
    if resultaat.get("documenttype") not in ["offerte", "bc_proces", "algemeen"]:
        resultaat["documenttype"] = "algemeen"
    if "leverancier" not in resultaat:
        resultaat["leverancier"] = None
    if "referentienummer" not in resultaat:
        resultaat["referentienummer"] = None

    return resultaat


def bouw_qa_prompt(geschiedenis: str) -> PromptTemplate:
    geschiedenis_blok = (
        f"Gespreksgeschiedenis (ter context):\n{geschiedenis}\n\n"
        if geschiedenis else ""
    )
    return PromptTemplate(
        "Je bent een assistent die vragen beantwoordt op basis van de beschikbare documentatie.\n"
        "Gebruik ALLEEN de informatie uit de context hieronder om de vraag te beantwoorden.\n"
        "Als het antwoord niet in de context staat, zeg dan expliciet: "
        "'Dit staat niet in de beschikbare documentatie.'\n"
        "Verzin nooit informatie die niet in de context staat.\n"
        "Geef altijd een VOLLEDIG antwoord. Als er meerdere methoden of stappen zijn, "
        "som ze allemaal op. Stop niet na de eerste optie.\n"
        "Je mag verwijzen naar wat je eerder in het gesprek hebt gezegd als dat relevant is.\n\n"
        + geschiedenis_blok +
        "Context uit documentatie:\n{context_str}\n\n"
        "Vraag: {query_str}\n\n"
        "Antwoord:"
    )


@st.cache_resource
def laad_systeem():
    db_connection = os.getenv("SUPABASE_DB_URL")

    vector_store = SupabaseVectorStore(
        postgres_connection_string=db_connection,
        collection_name="documenten"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )

    with open("nodes_cache.pkl", "rb") as f:
        nodes = pickle.load(f)

    return index, nodes


def bouw_retriever(index, nodes, classificatie: dict) -> QueryFusionRetriever:
    documenttype = classificatie["documenttype"]
    leverancier = classificatie.get("leverancier")
    referentienummer = classificatie.get("referentienummer")

    if documenttype in ["offerte", "bc_proces"]:
        filter_lijst = [ExactMatchFilter(key="documenttype", value=documenttype)]

        if leverancier:
            filter_lijst.append(ExactMatchFilter(key="leverancier", value=leverancier))

        if referentienummer:
            filter_lijst.append(
                ExactMatchFilter(key="referentienummer", value=str(referentienummer))
            )

        filters = MetadataFilters(filters=filter_lijst)
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,
            filters=filters
        )

        gefilterde_nodes = [
            n for n in nodes
            if n.metadata.get("documenttype") == documenttype
            and (not leverancier or leverancier.lower() in
                 str(n.metadata.get("leverancier", "")).lower())
            and (not referentienummer or
                 str(referentienummer) in str(n.metadata.get("referentienummer", "")))
        ]

        if not gefilterde_nodes:
            gefilterde_nodes = [
                n for n in nodes
                if n.metadata.get("documenttype") == documenttype
            ]

    else:
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
        gefilterde_nodes = nodes

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=gefilterde_nodes,
        similarity_top_k=5
    )

    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank"
    )


def bouw_filter_label(classificatie: dict) -> str:
    documenttype = classificatie["documenttype"]
    leverancier = classificatie.get("leverancier")
    referentienummer = classificatie.get("referentienummer")

    label = f"**{documenttype}**"
    if leverancier:
        label += f" · leverancier: **{leverancier}**"
    if referentienummer:
        label += f" · ref: **{referentienummer}**"
    return label


with st.spinner("Systeem laden..."):
    index, nodes = laad_systeem()

if "berichten" not in st.session_state:
    st.session_state.berichten = []

# Unieke session ID per Streamlit sessie voor Langfuse session tracking
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

for bericht in st.session_state.berichten:
    with st.chat_message(bericht["rol"]):
        st.write(bericht["inhoud"])
    if bericht["rol"] == "assistant" and "classificatie" in bericht:
        label = bouw_filter_label(bericht["classificatie"])
        st.caption(f"🔍 Gezocht in: {label}")

vraag = st.chat_input("Stel je vraag...")

if vraag:
    with st.chat_message("user"):
        st.write(vraag)
    st.session_state.berichten.append({"rol": "user", "inhoud": vraag})

    with st.chat_message("assistant"):
        with st.spinner("Antwoord zoeken..."):
            geschiedenis = bouw_geschiedenis_tekst(st.session_state.berichten[:-1])
            standalone_vraag = herschrijf_vraag(vraag, geschiedenis)
            classificatie = classificeer_vraag(standalone_vraag)

            hybrid_retriever = bouw_retriever(index, nodes, classificatie)

            qa_prompt = bouw_qa_prompt(geschiedenis)

            # ── Langfuse session tracking ───────────────────────────────────
            instrumentor.set_trace_params(
                session_id=st.session_state.session_id,
                user_id="van-houcke",
                tags=[classificatie["documenttype"]]
            )
            # ───────────────────────────────────────────────────────────────

            query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever,
                text_qa_template=qa_prompt
            )

            antwoord = query_engine.query(standalone_vraag)
            st.write(str(antwoord))
            label = bouw_filter_label(classificatie)
            st.caption(f"🔍 Gezocht in: {label}")

    st.session_state.berichten.append({
        "rol": "assistant",
        "inhoud": str(antwoord),
        "classificatie": classificatie
    })
