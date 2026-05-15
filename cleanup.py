import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ruwe transcriptie
transcriptie = """
Impromptu Google Meet Meeting - May 14
VIEW RECORDING - 6 mins (No highlights): https://fathom.video/share/xM8FiuLkdS3C6GB5tLvrA7NRGi5QWMCe

---

0:11 - Lukas Vandaele (LeadLine)
  Goed, gaan we bekijken hoe we het proces van een inkoop binnen Business Central kunnen uitvoeren. De eerste stap is om bovenaan in het zoekvenster te zoeken naar inkooporder.  Volgens komen we op de lijst van de bestaande inkooporders. Dus indien we een nieuw inkooporder willen maken, is het als tweede stap belangrijk om op nieuw te drukken.  Volgens mij komen we op een leeg inkooporder en als derde stap is het belangrijk om gegevens in te vullen van de leverancier.  En als dat gebeurd is, wordt alle informatie van de leverancier overgeschreven op het inkooporder en kunnen we beginnen met het toevoegen van artikelen die moeten bestaard worden bij deze leverancier.  Dat is de vierde stap. Als vijfde stap gaan we natuurlijk aantallen definiëren, hoeveel willen we er bestellen. En in die nodig zal er als zesde stap ook een prijs moeten meegeven worden, een directe kostprijs exclusief btw.  Belangrijk voor het aanpassen van een prijs en de inkoopprijslijst. Kunnen we dit doen vanop het inkopen. Er bestaat namelijk een vinkje, Add New Price.  Dit is op de regels die aangevinkt zal staan, eenmaal we een prijs hebben ingevuld bij de directe kostprijs exclusief btw.  Op dit moment is de prijs nog niet toegevoegd aan de prijslijst. Dit kunnen we doen door op de knop Adjust Purchase Prices te drukken.  Een andere methode zou zijn om het inkooporder vrij te geven. Het is wel belangrijk mocht je dit doen dat de vink, Add New Price, duidelijk aangevinkt staat.  Vervolgens als volgende stap kunnen we het order vrijgeven. Als zevende stap kunnen we de leverancier op de hoogte brengen via afdrukken en verzenden en vervolgens drukken op verzenden e-mail.  Het e-mailvenster zal dan openen. Je ziet daarbij de e-mail en de pdf van het inkooporder als bijlage. Eenmaal de e-mail is verzonden, wordt de orderlijnstatus op besteld gezet.  Als volgende stap krijgen we hoogstwaarschijnlijk een bevestiging van de leverancier. Die vullen we dan in met de toegezegde ontvangstdatum.  Indien de toegezegde ontvangstdatum wordt ingevuld, verandert de orderlijnstatus naar bevestigd. We kunnen gaan kijken om het order te... Ontvangen via de PGR Inkoop Globale Ontvangst.  Om daar te geraken is het dus belangrijk om te navigeren naar PGR Inkoop Globale Ontvangst. Volgens klik je de inkooporderregels aan die je wilt ontvangen.  Hier kunnen we kiezen voor een volledige of een deelontvangst. Dit hangt af van het te ontvangen aantal. Ik ga nu een volledige ontvangst doen van alle stukken.  Ik heb de regels aangeduid en dan klik ik op Boeken. Dan wordt er een pdf gedownload. Als we terug navigeren naar het inkooporder van daarnet, zullen we zien dat de orderlijnstatus op ontvangen staat en dat er enkel nog een te factureren aantal staat.  Dit wil zeggen dat de leverancier nog zijn factuur moet sturen en om dat te doen kunnen we het factuurnummer van de leverancier gaan invullen.  Vervolgens drukken we op boeken en dan op factureren en dan maken we een geboekte inkoopfactuur."""

cleanup_prompt = f"""
Je bent een expert in Business Central procesdocumentatie.
Zet deze ruwe transcriptie om naar een gestructureerd procesdocument.

Gebruik ALTIJD deze exacte structuur:

## Proces
[Naam van het proces, één zin]

## Doel
[Wat bereik je met dit proces, één zin]

## Wanneer gebruik je dit
[In welke situatie voer je dit proces uit]

## Vereisten
[Wat moet klaar zijn voordat je begint: data, instellingen, rechten]

## Stappen
[Genummerde stappen, elke stap bevat:]
1. Navigeer naar: [exact menupad in BC]
   Actie: [wat je doet]
   Resultaat: [wat er gebeurt]

## Maatwerk en aandachtspunten
[Specifieke aanpassingen of instellingen die afwijken van standaard BC]

## Veelgemaakte fouten
[Wat gaat er vaak mis en hoe vermijd je dat]

## Resultaat
[Wat is de eindstatus na het volledig doorlopen van het proces]

Regels:
- Vervang ALLE visuele verwijzingen door expliciete menupad beschrijvingen
- Gebruik exacte BC terminologie zoals die in het systeem staat
- Geen gesproken taalruis zoals "dus", "voilà", "zeg maar"
- Elke stap moet zelfstandig leesbaar zijn zonder de video

Transcriptie:
{transcriptie}
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": cleanup_prompt}
    ]
)

resultaat = response.choices[0].message.content
print(resultaat)

# Sla op als nieuw document
with open("document_offerte.txt", "w", encoding="utf-8") as f:
    f.write(resultaat)

print("\n✓ Document opgeslagen als document_offerte.txt")