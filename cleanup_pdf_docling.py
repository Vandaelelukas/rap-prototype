import os
from dotenv import load_dotenv
from openai import OpenAI
from docling.document_converter import DocumentConverter

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

converter = DocumentConverter()

def extraheer_tekst(pdf_pad):
    resultaat = converter.convert(pdf_pad)
    return resultaat.document.export_to_markdown()

def cleanup_offerte(tekst, bestandsnaam):
    prompt = f"""
Je krijgt de tekst van een offerte PDF van Innomotics aan Van Houcke NV.

Extraheer ALLEEN de commercieel relevante informatie en structureer die zo:

## Offerte details
- Leverancier: 
- Klant:
- Referentienummer:
- Datum:
- Geldigheid:

## Producten / Onderdelen
[Maak voor elk product een regel: Omschrijving | Artikelnummer/MLFB | Prijs per stuk EUR | Levertijd]

## Commerciële voorwaarden
- Minimumbestelling:
- Incoterms:
- Overige relevante voorwaarden:

Gooi ALLES weg wat juridische boilerplate is: export compliance, sancties, overdrachtsrechten, softwarelicenties.

Bestandsnaam: {bestandsnaam}

Tekst:
{tekst[:6000]}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Verwerk alle PDFs
pdf_map = "pdfs"
output_map = "documenten"
os.makedirs(output_map, exist_ok=True)

pdf_bestanden = [f for f in os.listdir(pdf_map) if f.endswith(".pdf")]

for bestand in pdf_bestanden:
    pad = os.path.join(pdf_map, bestand)
    print(f"Verwerken: {bestand}")
    
    tekst = extraheer_tekst(pad)
    resultaat = cleanup_offerte(tekst, bestand)
    
    uitvoer_naam = bestand.replace(".pdf", "_cleaned.txt")
    uitvoer_pad = os.path.join(output_map, uitvoer_naam)
    
    with open(uitvoer_pad, "w", encoding="utf-8") as f:
        f.write(resultaat)
    
    print(f"✓ Opgeslagen als {uitvoer_naam}")

print("\nKlaar! Alle offertes verwerkt met Docling.")