import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import PromptTemplate
from llama_index.vector_stores.supabase import SupabaseVectorStore

# Laad environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Pagina configuratie
st.set_page_config(
    page_title="RAG Assistent",
    page_icon="🤖",
    layout="centered"
)

st.title("📚 RAG Kennissysteem")
st.caption("Stel vragen over de beschikbare documentatie")

# Anti-hallucinatie prompt
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

# Initialiseer systeem eenmalig via Streamlit cache
@st.cache_resource
def laad_systeem():
    db_connection = os.getenv("SUPABASE_DB_URL")
    vector_store = SupabaseVectorStore(
        postgres_connection_string=db_connection,
        collection_name="documenten"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Gebruik bestaande index, geen nieuwe documenten inladen
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    return index.as_query_engine(text_qa_template=qa_prompt)

# Laad systeem
with st.spinner("Systeem laden..."):
    query_engine = laad_systeem()

# Chatgeschiedenis bijhouden
if "berichten" not in st.session_state:
    st.session_state.berichten = []

# Toon chatgeschiedenis
for bericht in st.session_state.berichten:
    with st.chat_message(bericht["rol"]):
        st.write(bericht["inhoud"])

# Invoerveld
vraag = st.chat_input("Stel je vraag...")

if vraag:
    # Toon gebruikersvraag
    with st.chat_message("user"):
        st.write(vraag)
    st.session_state.berichten.append({"rol": "user", "inhoud": vraag})

    # Genereer antwoord
    with st.chat_message("assistant"):
        with st.spinner("Antwoord zoeken..."):
            antwoord = query_engine.query(vraag)
            st.write(str(antwoord))
    st.session_state.berichten.append({"rol": "assistant", "inhoud": str(antwoord)})