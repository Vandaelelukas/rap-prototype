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


# ── Proces type keywords ──────────────────────────────────────────────────────
# Per proces_type: lijst van keywords die in de tekst moeten voorkomen
PROCES_TYPE_KEYWORDS = {
    "inkoop": [
        "inkooporder", "inkoop", "leverancier", "bestelling",
        "pgr inkoop", "inkoopfactuur", "purchase order"
    ],
    "assemblage": [
        "assemblageorder", "assemblagebeheer", "assembly",
        "componentenlijst", "eindartikel", "assemble"
    ],
    "verkoop": [
        "verkooporder", "verkoop", "klantorder", "sales order",
        "verkoopfactuur", "verkooplijn", "offerte aan klant"
    ],
    "magazijn": [
        "magazijn", "picking", "picklijst", "warehouse",
        "locatie", "voorraadlocatie", "bin", "zone"
    ],
    "voorraad": [
        "voorraad", "inventory", "stock", "voorraadaanpassing",
        "voorraadtelling", "negatieve voorraad"
    ],
    "financieel": [
        "grootboek", "boekingsgroep", "betaling", "bankrekening",
        "reconciliatie", "journaalpost", "general ledger"
    ],
}


def normaliseer_leverancier(waarde: str) -> str:
    waarde = waarde.strip()
    suffixen = [
        "GmbH", "s.r.o.", "NV", "BV", "B.V.", "N.V.", "Ltd", "LLC",
        "Inc", "Corp", "S.A.", "AG", "OHG", "KG"
    ]
    for suffix in suffixen:
        waarde = re.sub(rf"\b{re.escape(suffix)}\b", "", waarde, flags=re.IGNORECASE)
    return re.sub(r"[,\s]+$", "", waarde).strip()


def detecteer_proces_type(tekst: str) -> str:
    tekst_lower = tekst.lower()
    scores = {}
    for proces_type, keywords in PROCES_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in tekst_lower)
        if score > 0:
            scores[proces_type] = score

    if not scores:
        return "algemeen"

    return max(scores, key=scores.get)


def extraheer_metadata(bestandsnaam: str, tekst: str) -> dict:
    metadata = {}

    # Documenttype
    if "## Stappen" in tekst and ("Navigeer naar" in tekst or "Business Central" in tekst):
        metadata["documenttype"] = "bc_proces"
    elif "Referentienummer" in tekst and ("Innomotics" in tekst or "offerte" in tekst.lower()):
        metadata["documenttype"] = "offerte"
    else:
        metadata["documenttype"] = "algemeen"

    metadata["bestandsnaam"] = bestandsnaam

    # Proces type — enkel voor BC-processen
    if metadata["documenttype"] == "bc_proces":
        metadata["proces_type"] = detecteer_proces_type(tekst)
    else:
        metadata["proces_type"] = None

    # Referentienummer
    ref_match = re.search(r"Referentienummer[:\s*\-]*([A-Z0-9]{4,})", tekst, re.IGNORECASE)
    if ref_match:
        metadata["referentienummer"] = ref_match.group(1).strip()

    # Leverancier
    lev_match = re.search(r"Leverancier[:\s*\-]+(.+)", tekst, re.IGNORECASE)
    if lev_match:
        raw = lev_match.group(1).split("\n")[0]
        metadata["leverancier"] = normaliseer_leverancier(raw)

    return metadata


print("Documenten laden...")
documents = SimpleDirectoryReader(input_dir="documenten").load_data()
print(f"{len(documents)} document(en) gevonden")

for doc in documents:
    bestandsnaam = doc.metadata.get("file_name", "")
    metadata = extraheer_metadata(bestandsnaam, doc.text)
    doc.metadata.update(metadata)
    print(
        f"✓ {bestandsnaam}\n"
        f"  type: {metadata.get('documenttype')} | "
        f"proces: {metadata.get('proces_type', 'n/a')} | "
        f"ref: {metadata.get('referentienummer', 'n/a')} | "
        f"leverancier: {metadata.get('leverancier', 'n/a')}"
    )

print("\nChunken...")
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(documents)
print(f"{len(nodes)} chunks aangemaakt")

print("\nNodes opslaan voor BM25...")
with open("nodes_cache.pkl", "wb") as f:
    pickle.dump(nodes, f)
print("✓ nodes_cache.pkl opgeslagen")

print("\nIngesteren naar Supabase...")
index = VectorStoreIndex(nodes, storage_context=storage_context)
print("✓ Klaar! Alle documenten met metadata opgeslagen in Supabase")