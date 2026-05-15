import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25.base import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from openai import OpenAI as OpenAIClient

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

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
        model="gpt-4o-mini",
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

    # Laad gecachete nodes voor BM25 (aangemaakt door ingest.py)
    with open("nodes_cache.pkl", "rb") as f:
        nodes = pickle.load(f)

    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank"
    )

    return hybrid_retriever


with st.spinner("Systeem laden..."):
    hybrid_retriever = laad_systeem()

if "berichten" not in st.session_state:
    st.session_state.berichten = []

for bericht in st.session_state.berichten:
    with st.chat_message(bericht["rol"]):
        st.write(bericht["inhoud"])

vraag = st.chat_input("Stel je vraag...")

if vraag:
    with st.chat_message("user"):
        st.write(vraag)
    st.session_state.berichten.append({"rol": "user", "inhoud": vraag})

    with st.chat_message("assistant"):
        with st.spinner("Antwoord zoeken..."):
            geschiedenis = bouw_geschiedenis_tekst(st.session_state.berichten[:-1])
            standalone_vraag = herschrijf_vraag(vraag, geschiedenis)

            qa_prompt = bouw_qa_prompt(geschiedenis)
            query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever,
                text_qa_template=qa_prompt
            )

            antwoord = query_engine.query(standalone_vraag)
            st.write(str(antwoord))

    st.session_state.berichten.append({"rol": "assistant", "inhoud": str(antwoord)})