import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.supabase import SupabaseVectorStore

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Assistent", page_icon="🤖", layout="centered")
st.title("📚 RAG Kennissysteem")
st.caption("Stel vragen over de beschikbare documentatie")

qa_prompt = PromptTemplate(
    "Je bent een assistent die vragen beantwoordt op basis van de beschikbare documentatie.\n"
    "Gebruik ALLEEN de informatie uit de context hieronder om de vraag te beantwoorden.\n"
    "Als het antwoord niet in de context staat, zeg dan expliciet: "
    "'Dit staat niet in de beschikbare documentatie.'\n"
    "Verzin nooit informatie die niet in de context staat.\n\n"
    "Context:\n{context_str}\n\n"
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

    documents = SimpleDirectoryReader(input_dir="documenten").load_data()
    splitter = SentenceSplitter(chunk_size=512)
    nodes = splitter.get_nodes_from_documents(documents)

    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3
    )

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=3
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=3,
        num_queries=1,
        mode="reciprocal_rerank"
    )

    return RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        text_qa_template=qa_prompt
    )

with st.spinner("Systeem laden..."):
    query_engine = laad_systeem()

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
            antwoord = query_engine.query(vraag)
            st.write(str(antwoord))
    st.session_state.berichten.append({"rol": "assistant", "inhoud": str(antwoord)})