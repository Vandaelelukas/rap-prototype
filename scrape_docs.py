import requests
from bs4 import BeautifulSoup
import os

# Microsoft BC docs pagina's om op te halen
urls = [
    "https://learn.microsoft.com/en-us/dynamics365/business-central/assembly-how-to-assemble-items",
    "https://learn.microsoft.com/en-us/dynamics365/business-central/assembly-how-work-assembly-bom",
    "https://learn.microsoft.com/en-us/dynamics365/business-central/assembly-assemble-to-order-or-assemble-to-stock",
    "https://learn.microsoft.com/en-us/dynamics365/business-central/assembly-how-to-post-assembly-orders",
    "https://learn.microsoft.com/en-us/dynamics365/business-central/assembly-how-to-undo-assembly-posting"
]

os.makedirs("documenten", exist_ok=True)

for url in urls:
    print(f"Ophalen: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Haal alleen de hoofdinhoud op
    artikel = soup.find("main") or soup.find("article") or soup.find("body")
    
    if artikel:
        # Verwijder navigatie en overbodige elementen
        for tag in artikel.find_all(["nav", "footer", "script", "style", "aside"]):
            tag.decompose()
        
        tekst = artikel.get_text(separator="\n", strip=True)
        
        # Bestandsnaam uit URL
        naam = url.split("/")[-1] + ".txt"
        pad = os.path.join("documenten", naam)
        
        with open(pad, "w", encoding="utf-8") as f:
            f.write(f"Bron: {url}\n\n")
            f.write(tekst)
        
        print(f"✓ Opgeslagen als {naam}")
    else:
        print(f"✗ Geen inhoud gevonden voor {url}")

print("\nKlaar! Alle documenten opgeslagen in /documenten")