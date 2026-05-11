import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import PromptTemplate
from llama_index.vector_stores.supabase import SupabaseVectorStore

# Laad environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Controleer connectie
db_connection = os.getenv("SUPABASE_DB_URL")
print(f"Connectie gevonden: {db_connection[:50]}...")

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

# Supabase vector store
print("Verbinding maken met Supabase...")
vector_store = SupabaseVectorStore(
    postgres_connection_string=db_connection,
    collection_name="documenten"
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Laad documenten en maak index
print("Documenten laden...")
documents = SimpleDirectoryReader(input_dir="documenten").load_data()
print(f"{len(documents)} document(en) geladen")

print("Index opbouwen in Supabase...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
print("Index opgeslagen in Supabase")

# Query engine
query_engine = index.as_query_engine(text_qa_template=qa_prompt)

# Interactieve chatloop
print("\nRAG systeem klaar. Stel je vragen (typ 'stop' om te beëindigen):\n")

while True:
    vraag = input("Jouw vraag: ")
    if vraag.lower() == "stop":
        break
    antwoord = query_engine.query(vraag)
    print(f"\nAntwoord: {antwoord}\n")