import os
import json
import pickle
import time
from dotenv import load_dotenv

load_dotenv()

# ── Arize Phoenix tracing ─────────────────────────────────────────────────────
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

tracer_provider = register(
    project_name="van_houcke_rag_eval",
    endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com") + "/v1/traces",
    headers={"api_key": os.getenv("PHOENIX_API_KEY", "")},
)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
# ─────────────────────────────────────────────────────────────────────────────

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

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = LlamaOpenAI(model="gpt-4.1", temperature=0)


def laad_evaluatie():
    """Laadt evaluatie data — ondersteunt .jsonl, .csv en .json"""
    import pathlib

    if pathlib.Path("evaluatie.jsonl").exists():
        print("📂 evaluatie.jsonl gevonden")
        items = []
        with open("evaluatie.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Normaliseer kolomnamen
                    item["vraag"] = item.get("input", item.get("vraag", ""))
                    item["verwacht_antwoord"] = item.get("expected_output", item.get("verwacht_antwoord", ""))
                    item["categorie"] = item.get("categorie", "onbekend")
                    item["id"] = item.get("id", "?")
                    items.append(item)
        return items

    elif pathlib.Path("evaluatie.csv").exists():
        import csv
        print("📂 evaluatie.csv gevonden")
        items = []
        with open("evaluatie.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["vraag"] = row.get("input", row.get("vraag", ""))
                row["verwacht_antwoord"] = row.get("expected_output", row.get("verwacht_antwoord", ""))
                items.append(row)
        return items

    elif pathlib.Path("evaluatie.json").exists():
        print("📂 evaluatie.json gevonden")
        with open("evaluatie.json", "r", encoding="utf-8") as f:
            return json.load(f)

    else:
        raise FileNotFoundError(
            "Geen evaluatiebestand gevonden. "
            "Zet evaluatie.jsonl, evaluatie.csv of evaluatie.json in de project folder."
        )


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
            {"role": "user", "content": vraag}
        ]
    )
    try:
        resultaat = json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        resultaat = {}

    if resultaat.get("documenttype") not in ["offerte", "bc_proces", "algemeen"]:
        resultaat["documenttype"] = "algemeen"
    if "leverancier" not in resultaat:
        resultaat["leverancier"] = None
    if "referentienummer" not in resultaat:
        resultaat["referentienummer"] = None
    return resultaat


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
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5, filters=filters)

        gefilterde_nodes = [
            n for n in nodes
            if n.metadata.get("documenttype") == documenttype
            and (not leverancier or leverancier.lower() in
                 str(n.metadata.get("leverancier", "")).lower())
            and (not referentienummer or
                 str(referentienummer) in str(n.metadata.get("referentienummer", "")))
        ]
        if not gefilterde_nodes:
            gefilterde_nodes = [n for n in nodes if n.metadata.get("documenttype") == documenttype]
        if not gefilterde_nodes:
            gefilterde_nodes = nodes
    else:
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
        gefilterde_nodes = nodes

    bm25_retriever = BM25Retriever.from_defaults(nodes=gefilterde_nodes, similarity_top_k=5)
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank"
    )


qa_prompt = PromptTemplate(
    "Je bent een assistent die vragen beantwoordt op basis van de beschikbare documentatie.\n"
    "Gebruik ALLEEN de informatie uit de context hieronder om de vraag te beantwoorden.\n"
    "Als het antwoord niet in de context staat, zeg dan expliciet: "
    "'Dit staat niet in de beschikbare documentatie.'\n"
    "Verzin nooit informatie die niet in de context staat.\n"
    "Geef altijd een VOLLEDIG antwoord. Als er meerdere methoden of stappen zijn, "
    "som ze allemaal op. Stop niet na de eerste optie.\n\n"
    "Context uit documentatie:\n{context_str}\n\n"
    "Vraag: {query_str}\n\n"
    "Antwoord:"
)


def run_eval():
    print("Systeem laden...")
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
    print("✓ Systeem geladen\n")

    evaluatie = laad_evaluatie()
    print(f"✓ {len(evaluatie)} testvragen geladen\n")

    resultaten = []
    correct = 0
    fout = 0

    print(f"{'─'*80}")
    print(f"{'ID':<6} {'CAT':<20} {'VRAAG':<45} {'OK?'}")
    print(f"{'─'*80}")

    for item in evaluatie:
        id_ = item.get("id", "?")
        categorie = item.get("categorie", "onbekend")
        vraag = item.get("vraag", "")
        verwacht = item.get("verwacht_antwoord", "").lower()

        try:
            classificatie = classificeer_vraag(vraag)
            retriever = bouw_retriever(index, nodes, classificatie)
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                text_qa_template=qa_prompt
            )
            antwoord = str(query_engine.query(vraag)).strip()
        except Exception as e:
            antwoord = f"FOUT: {e}"

        kernwoorden = [w for w in verwacht.split() if len(w) > 4]
        matches = sum(1 for w in kernwoorden if w in antwoord.lower())
        score = matches / len(kernwoorden) if kernwoorden else 0
        ok = "✅" if score >= 0.4 else "❌"

        if score >= 0.4:
            correct += 1
        else:
            fout += 1

        vraag_kort = vraag[:42] + "..." if len(vraag) > 42 else vraag
        print(f"{id_:<6} {categorie:<20} {vraag_kort:<45} {ok}")

        resultaten.append({
            "id": id_,
            "categorie": categorie,
            "vraag": vraag,
            "verwacht": item.get("verwacht_antwoord", ""),
            "antwoord": antwoord,
            "classificatie": classificatie if 'classificatie' in dir() else {},
            "score": round(score, 2),
            "ok": score >= 0.4
        })

        time.sleep(1)

    print(f"{'─'*80}")
    print(f"\n📊 Resultaat: {correct}/{len(evaluatie)} correct ({round(correct/len(evaluatie)*100)}%)")
    print(f"   ✅ Correct: {correct}")
    print(f"   ❌ Fout:    {fout}")

    with open("eval_resultaten.json", "w", encoding="utf-8") as f:
        json.dump(resultaten, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Volledige resultaten opgeslagen in eval_resultaten.json")
    print("✓ Traces zichtbaar in Phoenix onder project: van_houcke_rag_eval")

    falend = [r for r in resultaten if not r["ok"]]
    if falend:
        print(f"\n{'─'*80}")
        print("❌ Falende vragen:\n")
        for r in falend:
            print(f"  [{r['id']}] {r['vraag']}")
            print(f"  Verwacht: {r['verwacht']}")
            print(f"  Gekregen: {r['antwoord'][:200]}...")
            print()


if __name__ == "__main__":
    run_eval()