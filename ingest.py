import os
import re
import pickle
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.supabase import SupabaseVectorStore

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

db_connection = os.getenv("SUPABASE_DB_URL")

vector_store = SupabaseVectorStore(
    postgres_connection_string=db_connection,
    collection_name="documenten"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


def extraheer_metadata(bestandsnaam, tekst):
    metadata = {}

    if "## Stappen" in tekst and "Navigeer naar" in tekst:
        metadata["documenttype"] = "bc_proces"
    elif "Referentienummer" in tekst and "Innomotics" in tekst:
        metadata["documenttype"] = "offerte"
    else:
        metadata["documenttype"] = "algemeen"

    metadata["bestandsnaam"] = bestandsnaam

    ref_match = re.search(r"Referentienummer[:\s]+([A-Z0-9\-]+)", tekst)
    if ref_match:
        metadata["referentienummer"] = ref_match.group(1)

    lev_match = re.search(r"Leverancier[:\s]+(.+)", tekst)
    if lev_match:
        metadata["leverancier"] = lev_match.group(1).strip()

    return metadata


print("Documenten laden...")
documents = SimpleDirectoryReader(input_dir="documenten").load_data()
print(f"{len(documents)} document(en) gevonden")

for doc in documents:
    bestandsnaam = doc.metadata.get("file_name", "")
    metadata = extraheer_metadata(bestandsnaam, doc.text)
    doc.metadata.update(metadata)
    print(
        f"✓ {bestandsnaam} → type: {metadata.get('documenttype')} | "
        f"ref: {metadata.get('referentienummer', 'n/a')} | "
        f"leverancier: {metadata.get('leverancier', 'n/a')}"
    )

print("\nChunken...")
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
nodes = splitter.get_nodes_from_documents(documents)
print(f"{len(nodes)} chunks aangemaakt")

print("\nNodes opslaan voor BM25 (runtime gebruik)...")
with open("nodes_cache.pkl", "wb") as f:
    pickle.dump(nodes, f)
print("✓ nodes_cache.pkl opgeslagen")

print("\nIngesteren naar Supabase...")
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
)

print("✓ Klaar! Alle documenten met metadata opgeslagen in Supabase")