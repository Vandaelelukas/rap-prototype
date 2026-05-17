import os
import json
import uuid
import pickle
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Assistent", page_icon="🤖", layout="centered")


# ── Helper voor secrets ──────────────────────────────────────────────────────
def get_secret(key):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key)


# ── Microsoft authenticatie via MSAL ─────────────────────────────────────────
import msal

CLIENT_ID = get_secret("AZURE_CLIENT_ID")
CLIENT_SECRET = get_secret("AZURE_CLIENT_SECRET")
TENANT_ID = get_secret("AZURE_TENANT_ID")
REDIRECT_URI = get_secret("AZURE_REDIRECT_URI")
# Gebruik 'organizations' voor multi-tenant work/school accounts
# of 'common' voor alle accounts inclusief persoonlijke Microsoft accounts
AUTHORITY = f"https://login.microsoftonline.com/organizations"
SCOPES = ["User.Read"]


@st.cache_resource
def get_msal_app():
    return msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )


def login_flow():
    query_params = st.query_params
    auth_code = query_params.get("code")

    if auth_code and "user" not in st.session_state:
        # Wissel auth code in voor token
        result = get_msal_app().acquire_token_by_authorization_code(
            auth_code,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI,
        )
        if "id_token_claims" in result:
            claims = result["id_token_claims"]
            st.session_state.user = {
                "name": claims.get("name", "Onbekend"),
                "email": claims.get("preferred_username", ""),
            }
            st.query_params.clear()
            st.rerun()
        else:
            st.error(f"Login mislukt: {result.get('error_description', 'Onbekende fout')}")
            st.stop()

    if "user" not in st.session_state:
        st.title("📚 RAG Kennissysteem")
        st.write("Meld je aan met je Microsoft account.")
        auth_url = get_msal_app().get_authorization_request_url(
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI,
        )
        st.link_button("Aanmelden met Microsoft", auth_url)
        st.stop()


login_flow()


# ── Phoenix tracing — pas na authenticatie ───────────────────────────────────
if "phoenix_initialized" not in st.session_state:
    try:
        from phoenix.otel import register
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        tracer_provider = register(project_name="van_houcke_rag")
        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception:
        pass
    st.session_state.phoenix_initialized = True


# ── LlamaIndex imports & setup ───────────────────────────────────────────────
from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.retrievers.bm25.base import BM25Retriever
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from openai import OpenAI as OpenAIClient

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = LlamaOpenAI(model="gpt-4.1", temperature=0)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.write(f"👤 {st.session_state.user['name']}")
    st.write(f"✉️ {st.session_state.user['email']}")
    if st.button("Afmelden"):
        del st.session_state.user
        st.rerun()


st.title("📚 RAG Kennissysteem")
st.caption("Stel vragen over de beschikbare documentatie")

HISTORY_WINDOW = 6


# ── RAG functies ─────────────────────────────────────────────────────────────
def bouw_geschiedenis_tekst(berichten, max_beurten=HISTORY_WINDOW):
    recente = berichten[-(max_beurten * 2):]
    if not recente:
        return ""
    return "\n".join(
        f"{'Gebruiker' if b['rol'] == 'user' else 'Assistent'}: {b['inhoud']}"
        for b in recente
    )


def herschrijf_vraag(vraag, geschiedenis):
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


def classificeer_vraag(vraag):
    client = OpenAIClient()
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Analyseer de vraag en geef een JSON-object terug met deze velden:\n\n"
                    "- documenttype (verplicht): één van:\n"
                    "  * offerte: vragen over prijzen, artikelnummers, leveranciers, levertijden, incoterms\n"
                    "  * bc_proces: vragen over hoe iets werkt in Business Central, stappen, navigatie\n"
                    "  * algemeen: alles wat niet duidelijk in bovenstaande past\n\n"
                    "- proces_type (optioneel, alleen invullen bij bc_proces):\n"
                    "  * inkoop: inkooporders, bestellingen, ontvangst, factureren\n"
                    "  * assemblage: assemblageorders, componentenlijsten, eindartikelen\n"
                    "  * verkoop: verkooporders, klantorders, verkoopfacturen\n"
                    "  * magazijn: picking, picklijsten, locaties, zones\n"
                    "  * voorraad: voorraadaanpassingen, voorraadtellingen, stock\n"
                    "  * financieel: grootboek, betalingen, journaalposten\n"
                    "  * null als het niet duidelijk is welk BC-proces het betreft\n\n"
                    "- leverancier (optioneel): leveranciersnaam als expliciet vermeld, anders null\n\n"
                    "- referentienummer (optioneel): offertenummer als expliciet vermeld, anders null\n\n"
                    "Geef ALLEEN een geldig JSON-object terug, geen uitleg of extra tekst.\n"
                    "Voorbeeld: "
                    '{"documenttype": "bc_proces", "proces_type": "inkoop", "leverancier": null, "referentienummer": null}'
                )
            },
            {"role": "user", "content": vraag}
        ]
    )
    try:
        resultaat = json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        resultaat = {}

    if resultaat.get("documenttype") not in ["offerte", "bc_proces", "algemeen"]:
        resultaat["documenttype"] = "algemeen"

    geldige_proces_types = ["inkoop", "assemblage", "verkoop", "magazijn", "voorraad", "financieel"]
    if resultaat.get("proces_type") not in geldige_proces_types:
        resultaat["proces_type"] = None

    resultaat.setdefault("leverancier", None)
    resultaat.setdefault("referentienummer", None)
    return resultaat


def bouw_qa_prompt(geschiedenis):
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
    db_connection = get_secret("SUPABASE_DB_URL")
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


def bouw_retriever(index, nodes, classificatie):
    documenttype = classificatie["documenttype"]
    proces_type = classificatie.get("proces_type")
    leverancier = classificatie.get("leverancier")
    referentienummer = classificatie.get("referentienummer")

    if documenttype in ["offerte", "bc_proces"]:
        filter_lijst = [ExactMatchFilter(key="documenttype", value=documenttype)]

        if proces_type:
            filter_lijst.append(ExactMatchFilter(key="proces_type", value=proces_type))
        if leverancier:
            filter_lijst.append(ExactMatchFilter(key="leverancier", value=leverancier))
        if referentienummer:
            filter_lijst.append(
                ExactMatchFilter(key="referentienummer", value=str(referentienummer))
            )

        filters = MetadataFilters(filters=filter_lijst)
        vector_retriever = VectorIndexRetriever(
            index=index, similarity_top_k=5, filters=filters
        )

        gefilterde_nodes = [
            n for n in nodes
            if n.metadata.get("documenttype") == documenttype
            and (not proces_type or n.metadata.get("proces_type") == proces_type)
            and (not leverancier or leverancier.lower() in
                 str(n.metadata.get("leverancier", "")).lower())
            and (not referentienummer or
                 str(referentienummer) in str(n.metadata.get("referentienummer", "")))
        ]
        # Fallback 1: zonder optionele filters
        if not gefilterde_nodes:
            gefilterde_nodes = [
                n for n in nodes if n.metadata.get("documenttype") == documenttype
            ]
        # Fallback 2: alle nodes
        if not gefilterde_nodes:
            gefilterde_nodes = nodes
    else:
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
        gefilterde_nodes = nodes

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=gefilterde_nodes, similarity_top_k=5
    )

    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank"
    )


def bouw_filter_label(classificatie):
    label = f"**{classificatie['documenttype']}**"
    if classificatie.get("proces_type"):
        label += f" · proces: **{classificatie['proces_type']}**"
    if classificatie.get("leverancier"):
        label += f" · leverancier: **{classificatie['leverancier']}**"
    if classificatie.get("referentienummer"):
        label += f" · ref: **{classificatie['referentienummer']}**"
    return label




# ── App logica ───────────────────────────────────────────────────────────────
with st.spinner("Systeem laden..."):
    index, nodes = laad_systeem()

if "berichten" not in st.session_state:
    st.session_state.berichten = []

for bericht in st.session_state.berichten:
    with st.chat_message(bericht["rol"]):
        st.write(bericht["inhoud"])
    if bericht["rol"] == "assistant" and "classificatie" in bericht:
        st.caption(f"🔍 Gezocht in: {bouw_filter_label(bericht['classificatie'])}")

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

            retriever = bouw_retriever(index, nodes, classificatie)
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                text_qa_template=bouw_qa_prompt(geschiedenis)
            )

            antwoord = query_engine.query(standalone_vraag)
            st.write(str(antwoord))
            st.caption(f"🔍 Gezocht in: {bouw_filter_label(classificatie)}")

    st.session_state.berichten.append({
        "rol": "assistant",
        "inhoud": str(antwoord),
        "classificatie": classificatie
    })