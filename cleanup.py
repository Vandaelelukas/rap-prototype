import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ruwe transcriptie
transcriptie = """
Kennismakingsgesprek: Broes Royaux & Lukas Vandaele - May 11
VIEW RECORDING - 31 mins (No highlights): https://fathom.video/share/yfwBazWBkyUDSaS_EfYdRR6agNaASur5

---

0:00 - Lukas Vandaele (LeadLine)
  Ik ga een korte voorstelling doen. Mijn naam is Lukas van LeadLine en ik heb inderdaad uw gegevens een keer gekregen van dealen.  Ik weet niet wat de relatie daar precies is.

0:11 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ik heb nog een gegeven van mij. Ah, oké.

0:16 - Lukas Vandaele (LeadLine)
  Vandaar, misschien kan het ook een keer interessant zijn, een keer te spreken met elkaar. Ik ga misschien nu al direct zeggen, Broes ben je zeker niet om iets te verkopen of wat dan ook.  Het is puur een keer touch base en kijken of we al dan niet voor elkaar iets kunnen betekenen. We zien wel van daaruit hoe dat er voortvloeit.  Oké, dus zoals ik al zei, ik ben van de firma LeadLine. Wij zijn een start-up. Ik zeg wij, ik heb dat samen met mijn broer opgericht, een dik jaar geleden ondertussen.  Maar daarvoor heb ik zelf al heel wat ervaring in IT, bij ERP-implementaties. Dus bij dealen, het is ook daar dat wij...  We hebben ook Business Central nu onlangs geïmplementeerd bij toolsen en bij Van Hoeken. Vandaaruit zullen we dan gaan kijken met LeadLine wat er eventueel de mogelijkheden waren.  En concreet onze diensten, je kunt het onder de noemer AI zetten, maar natuurlijk dat is redelijk breed. Dus concreet wat we doen zijn AI automations, dat zijn automatisaties van alle vormen.  Dat kan gaan van simpele automatisaties van, ik zeg nu maar iets, het bijhouden van een mailbox of het organiseren van een mailbox, tot ook complexere zaken zoals lead-generatiesysteem of outbound-systeem.  Om een keer een concreet voorbeeld te geven bij Van Hoeken bijvoorbeeld. Daar hebben we, of is Dylan eigenlijk met het idee gekomen, om eigenlijk een lead-generatiesysteem te maken die eigenlijk de concurrentie...  We hadden een keer een analyse gedaan, wat zijn de mogelijkheden, en uiteindelijk hebben we het dan wel iets mooi in productie kunnen gaan brengen, dus die gaat eigenlijk aan de hand van een Excel-lijst, zeg maar, of een klantenlijst, gaat eigenlijk met die data verschillende zoekopdrachten gaan doen, en gaat eigenlijk het output geven van, kijk, dit zijn de meest waarschijnlijke concurrenten van een klant, en op die manier heeft hij dus eigenlijk ook, ja, nieuwe potentiële klanten doen, zeg maar.  Dat is één voorbeeld ervan, een ander voorbeeldje kan bijvoorbeeld zijn, ze sturen heel veel, ja, naar een fabriek voor offertes op te vragen, en wat dan ook, momenteel, ja, er kruipt daar wat tijd tussen, en natuurlijk, dat is tijd dat er een klant aan het wachten is, en misschien naar andere partijen aan het gaan is.  Wat hebben we daar voorzien, is eigenlijk van, oké, als wij een offerte toegestuurd krijgen van, dat bepaalt mijn adres van die fabriek.  Dan gaan we eigenlijk al die pdf's die worden toegestuurd gaan opslaan, we gaan wat logica erin gaan steken van oké is het voor een echte motor of is het voor wat onderdelen bijvoorbeeld.  Afhankelijk daarvan gaan we dan de juiste naam gaan geven en gaan we dat ook gaan opslaan in de OneDrive die eigenlijk dient als een soort van data bijstel.  Natuurlijk, dat moet wel gevoed worden op termijn, maar de bedoeling is wel dat de werknemers eigenlijk zo worden getraind van oké, nieuwe offerten, laten we eerst een keer gaan kijken in die OneDrive als we dat al niet al gevraagd hebben.  Indien ja, wat is de efficiëntie daar is dat je gewoon sneller kunt gaan antwoorden op je klant, een concretere prijs ook direct kan geven waardoor je misschien ook sneller de deal rond krijgt.  Dus denk dat je zelf, of ook toch, dat je zelf ook al een beetje ziet, want het kan heel breed gaan.  We gaan altijd in samenspraak ook met de klant, want natuurlijk, jullie kennen jullie business het best natuurlijk. Het is altijd een beetje een samenwerking natuurlijk.  We hebben een AI chatbot.

3:59 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ja.

4:00 - Lukas Vandaele (LeadLine)
  ik heb dat gezien.

4:01 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Het is op onze site. Ik denk dat het daar wel nog wel wat ruimte is om dat te verbeteren.  Maar wij werken eigenlijk heel particulier hier. Dus klanten zitten daar eigenlijk redelijk actief in vragen te stellen en dergelijke.  En die vragen die ze stellen zijn meestal wel redelijk wederkerend. Het zijn meestal wel dezelfde vragen. En met een renovatie, een comma 12 centimeter opbouw.  Zo van die dingen. Is het interessant om isolatie te plaatsen en vloeiverwarming. Of doe ik dat beter op een andere manier en zo.  Nu reageert de chatbot al redelijk onderbouwd. Die leer blijft ook van zichzelf bijleren en die antwoorden ze wel altijd juist.  Grondendeels juist. Maar de effectieve, bijvoorbeeld het inplannen van een werfbezoek. En dergelijke dagen. Dat gebeurt daar maar altijd door iemand bij ons op kantoor.  Dus ik denk dat het daar... En daarin wel nog optimalisatie mogelijk is. Het opmaken van een offerte gebeurt ook maar nog altijd manueel bij ons.  En op zich is dat iets die vrij eenvoudig in elkaar te steken is, denk ik, om de rector die in jouw chatbot voor de eerste keer al een...  Die chatbot is van Watermelon.

5:20 - Lukas Vandaele (LeadLine)
  Ja, ik heb het gezien inderdaad. Nu Watermelon, ja, ik ben een kort naar die site ook gegaan. Maar het is dan de bedoeling om daarop verder te bouwen of zeggen van bijvoorbeeld zo'n offertesysteem, want dat is ook iets dat we al regelmaat gehoord hebben.  We hebben dus ook een automatisatie degelijk, maar dat staat wel manueel, dat staat niet per se met zo'n chatbot geïntegreerd.  Is bijvoorbeeld dat we inderdaad een template klaar hebben staan, een template als offerte zeg maar, in de huisstijl van Watermelon, zeg maar.  En dat we eigenlijk aan de hand van bepaalde vragen die die templates gaan gaan invullen. Ja. Dat is ook een database achter met de prijzen en weet ik ook veel wat.  Dat is dan allemaal te bekijken natuurlijk, maar dat is ook al iets dat we al veel hoort en dat we ook al verschillende keren hebben geïmplementeerd.  Maar is dat dan bijvoorbeeld iets dat je zegt van dat moet echt, ja, dat is op basis van die chatbot dat moet ingevuld worden, of is dat eerder mensen in het kantoor dan die dat...

6:21 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Wat ik nu al zo, Mike, dat de klanten soms een beetje op afhaken, is dat ze toch nog welke contactformulier moeten invallen om een offerte te verkrijgen.  Dat je een van stroom in wil krijgen van ja, kijk, dat die chatbot zegt van ja, wil je een offerte?  Geef mij die gegevens. En op basis daarvan kan ik jouw offerte al onmiddellijk uitwerken, dat die wel nog een keer gecontroleerd wordt door iemand van ons.  Dat die offerte binnen de 24 uur bewezen, dat spreekt nog welke dagen in de mailbox. Dat moet allemaal heel kort op de bal gaan en dergelijke, dus we moeten daar wel mee zijn.  Maar die chatbot... Dat is voor ons wel heel belangrijk, want die heeft al die info, die heeft al die data, waaruit die al de juiste antwoorden genereert, dus die zouden wel moeten kunnen even behouden.  Er zit ook al een heel serieus bedrag van investeringen in, maar jammer zijn ook dat zoveel je zoiets kunt opstellen die die info uit die chatbot houdt en...

7:23 - Lukas Vandaele (LeadLine)
  Nou, dat ga ik een moeten bekijken natuurlijk. Het probleem is bij ons in business, is allemaal maatwerk. Dus ik kan nu moeilijk zeggen, het gaat zo of het gaat zo zijn, kun je nu de site van Waldemelen een keer voor mij.  Het is een keer te bekijken, wat zijn daar de mogelijkheden. Als zij beschikken over een API, ik weet niet of u dat bekend is, waarbij wij eigenlijk kunnen gaan de chatbot laten communiceren met onze automatisatie, dan zie ik daar zeker wat mogelijkheden in.  Beschikken ze daar niet over, dan is dat weer een heel moeilijk verhaal. Dat zijn zaken, het is daarom dat we nu een keer gewoon deze kaal doen.
  ACTION ITEM: Test Watermelon chatbot as private user; note improvements - WATCH: https://fathom.video/share/yfwBazWBkyUDSaS_EfYdRR6agNaASur5?timestamp=480.9999  Als zij van mocht dan een keer... Dan doen wij dat. Dat is volledig vrijblijvend. En dan kijken we van daaruit van goed, het is in de mogelijkheden of juist niet.  En van daaruit, je moet dan wel, als we er al dan niet mee verder gaan.

8:11 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Maar kun je al een die chatbot invullen of dan? Ik heb er nog niet mee gechat, dat niet, nee.  Je moet al een keer doen, gewoon als particulier zijnde.

8:22 - Lukas Vandaele (LeadLine)
  Je zit ook in de sector.

8:25 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Je moet je wel van eerst merken van oké, dit moet beter of dit kan anders. Op zich, de verantwoorden zijn onderbouwd en zijn redelijk goed.  Het enige dat volgens mij een beetje stoef oploopt, is het geloop van de offertes. Dus dat er echt effectief nog moet het formulier ingevuld worden op onze site.  Dat is weer wegklikken vanuit die chatbot. Ik denk dat dat beter kan. Wat wel nog niet mogelijk is, is dat die chatbot toegang heeft tot ons agenda bijvoorbeeld.  Dat moet ook wel een hardeet. Ja. Dat is wel wat ween die dat doen. Bevor ik je gezegd, ja, ik kan kijken van...  En stel dat je die datums woord, nu hebben we hem dat al geleerd, dat hij dat niet mag doen.  Mensen vragen dat dan. We gaan er dan vanuit dat dat mogelijk is, maar dat is niet altijd even mogelijk in ons naam.

9:13 - Lukas Vandaele (LeadLine)
  Tuurlijk, tuurlijk, tuurlijk. Maar ik denk wel, voor de chatbot-ontwikkeling zelf, is dat niet native ingebouwd? Zijn dat geen mogelijkheden van Watermelon zelf dan?

9:27 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Dat weet ik niet.

9:29 - Lukas Vandaele (LeadLine)
  Ik zou persoonlijk denken, als dat een core business is, want chatbot zelf is nu een beetje minder on... Wij gaan echt wel over de automatisaties, zeg maar.  Chatbots zelf, als dat een core business is, dan zou ik wel verwachten dat er daar een mogelijkheid is om dat te gaan linken met de agenda, bijvoorbeeld.  Dat lijkt mij maar de meest logische stap dat je een chatbot aanbiedt. Want het is juist wel dat veel bedrijven dat willen.  En dat vroeger ook nog gedaan, chatbots. Dus daarmee, we zijn een beetje van afgestapt en dan er inderdaad...

9:58 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Allee, ik zeg het, we zijn in de stap.

10:00 - Lukas Vandaele (LeadLine)
  Het budget is er niet altijd om op te boksen tegen een watermelon bijvoorbeeld, wat dan ook, dus vandaar zijn wij ons echt wel gaan focussen op het maatwerk, zeg maar.

10:11 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  We hebben heel onze site die de offertes maakt en dergelijke, dus we hebben een eigen website, volledig web-based die de offertes maakt, de leverbond maakt, de bestelbond en de planning, dus die niet door duo in elkaar is gestoken.  Dat loopt eigenlijk allemaal vrij goed, vanaf dat die offerte bevestigd is, wordt er van eerste keer een leverbond, daar staat wel een plannings-item aangemaakt, met de juiste gegevens van de klant, dus dat is ook gekoppeld aan Xact enzo, dus op zich loopt dat allemaal vrij goed.  Het eerste dat volgens mij beter kan geautomatiseerd worden, is inderdaad offerte aanvragen of offertes opmaken, maar op zich zijn wij met een product die heel gemakkelijk is om een offerte op te maken.  Als die klant een formulier invult, dan weet je ook vooral... We kunnen opmaken bijvoorbeeld, dat we ook hebben op onze site, die zegt van kijk, heb 80 vierkante meter isolatie nodig op een dikte van 5 centimeter, die selecteert welk type isolatie dat die wil, en dan nog voor hetzelfde oppervlak 60 centimeter chape, dus dat is iets die volgens mij echt instant kan aangeleverd worden.  Maar, omdat we wel het wat van een filter op zetten, ja, dat wel effectief leads zijn, dat is niet zomaar gewoon formulieren vullen, want de gegevens nodig van die mensen, als er geen telefoonnummer of geen e-mailadres bij je staat, dan moet dat niet opmaken.  Ja, inderdaad, inderdaad.

11:42 - Lukas Vandaele (LeadLine)
  Maar ik denk dat de grootste struikelblok, als ik het nu zo hoor, is dan het inval van die gegevens, maar heb ze eigenlijk sowieso nodig.

11:51 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ja, en dan merken we soms dat ze daar op strijd komen in die chatbot. Dat ze niet altijd alles invullen, en dat ze weer weg zijn, is dat een lead type bij mij.  Ja, inderdaad, ja, natuurlijk.

12:04 - Lukas Vandaele (LeadLine)
  Wat ik had hier nu gevraagd, kan ik een offerte aanvraag nemen eigenlijk direct naar de pagina gestuurd? Ja, helemaal.

12:11 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Dus dat denk ik. Dat dat beter kan, dat is echt, kijk, nu moet je nog altijd op die link duwen, een offerte aanvraag invullen.  Al die heves invullen, volgens mij kunnen we daarop verder werken dat dat van eerste keer automatiseren wordt en dat die daar, als ze dat volledig ingevuld hebben, dat ze meteen een offerte in hun mailbox krijgen.

12:37 - Lukas Vandaele (LeadLine)
  Ja, inderdaad, inderdaad. Als ik het zo goed op eerste zeg, zou het eigenlijk een samenwerking van Watermelon en ons moeten zijn.  Want, opnieuw, wij gaan eigenlijk vooral over de automatisatie, het opvragen van de data. Goed, we hebben die data nodig, Maar het opvragen van de data gebeurt via die chatbot.  Allee, wij als LeadLine zijn... ... Ik ik niet in de beschikbaarheid om te gaan ontwikkelen in jullie huidige chatbot en dat sowieso iets zou moeten zijn dat je dan met de mensen van Weathermill moet overleggen.  Hetzelfde voor die agenda, bijna wij, maar eigenlijk vanaf het moment dat het wordt verstuurd of dat al de gegevens verzameld zijn en je wil dat er automatisch in offerte één wordt opgemaakt en ook twee direct wordt verstuurd naar het teamadres dat we het opgeven, dat zijn dan concreet zaken waar wij in beeld komen.  Dat je gewoon een goed beeld krijgt van wat wij wel en niet doen. Dat je geen bals gehoopt krijgt.  Kan jij nog een keer hel, maar ik was er juist aan het team. Wel, als ik het nu zo hoor, dus inderdaad het opvragen van de gegevens in de chatbot zelf, dat zijn zaken dat wij wellicht niet gaan kunnen doen.  Dat is Wallermel, dat is hun development, hun manier van die chatbot. Nu, van het moment dat al de data verzameld is en dat er eigenlijk automatisch een offerte moet gemaakt worden.  Maar goed, ik zeg het opnieuw, als wij die gegevens kunnen krijgen, en we kunnen een automatisatie erachter steken voor te maken van die offerte, die dan ook wordt verstuurd naar dat mailadres, daar komen wij een beetje in beeld te zijn.  Het is dat dat ik wil duidelijk maken naar jou, dat je niet verwacht dat zij het hier allemaal fixen, dat het gewoon realistisch is ook.  Dus wat dat concreet gevraagd wordt, lijkt mij een samenwerking van de twee partijen, dat dat mogelijk is natuurlijk.

14:33 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Zie je jezelf nog voor iets anders, los van al hetgeen dat we al gezegd hebben, dat je zegt, voor jullie lijkt mij dat wel iets wat we...

14:42 - Lukas Vandaele (LeadLine)
  Ja, we hebben eigenlijk twee typen klanten, ik zal het misschien zo zeggen. En ik kan het niet zelf zeggen, zoals jij nu, van hé, we hebben dit proces en wij willen dat graag geautomatiseerd zien.  En we hebben dan ook mensen die inderdaad zeggen van ja, wat raden jullie? Ja. Ja. Nu, binnen jullie specifieke niches hebben wij nog niet concreet automatisaties gemaakt, maar wat wij wel ook doen, zijn eigenlijk analysesessessies.  Dat is dan eigenlijk dat we een keer een call specifiek daar aan toewijzen, zeg maar, om te gaan bekijken wat zijn de processen.  Dat kan via Cloud, maar dat kan ook ons site bijvoorbeeld. En van daaruit komen er wel heel wat zaken, en dat zijn kleine zaken die niet altijd zelf moeten ontwikkeld worden ofzo.  Dat zijn ook gewoon beter gebruik maken van de tools die er bijvoorbeeld al zijn. Ik zeg maar, de mailbox enzo, er zitten ook heel wat automatisatiemogelijkheden.  Dat mensen niet van op de hoogte zijn. Dus dat zijn eigenlijk zo'n beetje de twee zaken. Maar concreet, voor jullie business, kan ik rustig een keer in een mailtje doorsturen van wat wij al hebben gerealiseerd, met wat meer uitleg.

15:53 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ja, we zijn al jaren en dag klank bij DUO, dat is nu Dynamite. Ja. Ja. Ja. Ja. Die zijn hier vorige week ook geweest omdat zij nu, ja, dat zijn allemaal firma's die nu samen fusioneerd zijn.  Om ook die processen en die workflow een keer te bekijken en hoe loopt dat hier en wat gebeurt er allemaal.  Dus we moeten gewoon een beetje opletten dat we dan niet in een KSV water zitten, want zij zijn er ook al mee bezig en zij beheerden onze offertesites en dergelijke, dus zij weten al heel veel informatie.
  ACTION ITEM: Email Broes: Watermelon API/calendar; Drupal quote-form; Drupal quote-tool; examples/use cases; data needs - WATCH: https://fathom.video/share/yfwBazWBkyUDSaS_EfYdRR6agNaASur5?timestamp=980.9999  Tuurlijk, tuurlijk.

16:31 - Lukas Vandaele (LeadLine)
  Ja, wel, wat ik misschien voorstaal, Broes, is dat ik een keer op mijn mail gewoon nog een keer nastuur van, ik denk voor u, of voor Royaux, dat dit mogelijk iets kan zijn.  En we zien dan van daaruit wel, wat dat betreft het watermelon en het offerteverhaal, ja, ik denk dat je misschien best eerst gewoon een keer bij watermelon zelf luistert van, toch zeker van die agenda, ik denk, en mij lijkt het dat dat zeker mogelijk moet zijn, daar komen wij zelf niet aan te passen.  En dat we ook van daaruit dan bekijken. Wat ik misschien wel nog zou zeggen is, we zijn nu ook, maar dat is wel nog volledig in ontwikkeling, bij Vanhoeken zijn we nu ook bezig met een soort van AI-brain te maken.  Een beetje een rare naam als je dat zo zegt. Maar wat houdt dat eigenlijk concreet in, heel veel bedrijven, net zoals jullie waarschijnlijk, beschikken over heel wat data die verspreid zit over documenten.  De een zit in je hoofd, de ander zit in iemand anders in hoofd, de dienen werknemer weet dat bepaald proces heel goed, zo enzo.  En natuurlijk de moeilijkheid daarvan is om alles gecentraliseerd te krijgen. Dat we momenteel bezig zijn bij Vanhoeken, is eigenlijk om een systeem op te zetten, waarbij dat er eigenlijk, net zoals de chatbot die jullie nu eigenlijk beschikken, maar dan voor intern gebruik.  Die eigenlijk gaan kunnen navragen over processen, gebruik van systemen bijvoorbeeld, zijn er nu Business Central geïmplementeerd, er zijn heel wat vragen.  Deelen zelf ook, moet veel antwoorden en nog wat bijsturen en doen. Maar ook over contracten. Wat is er in 2000 zoveel afgesproken geweest met leverancier X of met klant Y enzovoort.  Dat proberen we eigenlijk allemaal te centraliseren en dat doorzoekbaar te maken. Nu, persoonlijk denk ik dat elk bedrijf daar gebruik van kan maken.  Maar dat is wel nog iets, ik wil er nu nog bij zijn, dit is nu nog in ontwikkeling bij Vanhoeken.  Maar dat is wel sowieso ook iets dat we in de toekomst product 1000 willen gaan aanbieden aan meerdere klanten.

18:35 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ja, dat is wel iets waar we ook altijd mee willen zijn. We zitten in de markt tegenwoordig dat alles heel snel moet gebeuren.  We moeten heel kort op de bal spelen. Als je het niet doet, heb je de werf niet. Dus tussen offerten en plaatsingen zitten soms maar een week.  Je moet echt schakelen. We kunnen dat ook, want die flexibiliteit in ons. Maar dat betekent wel dat je vroeger ging ter plaatse, je maakte je werkverslag de dagen nadien, je zette het op mail, dat dat allemaal veel sneller en sneller en sneller moet gebeuren om misverstand te veranderen.  Ja, natuurlijk. Dus dat zijn wel dingen, dat ik 100% van overtuigd ben, dat dat met een AI wel veel gemakkelijker kan verlopen, dat ik uit de werf kom, dat je automatisch al hebt het verslag doorgemaild en dat je denoteert dat dat is afgesproken.  Mijn broer doet dat al vaak op andere werkvergaderingen, voor projectplanning, maar hij heeft zo'n recorder die alles opneemt, die dat dan samenvat en een hapklare mail klaar zet voor te versturen.  Ik denk dat dat iets is dat we zeer kunnen implementeren en dat we in de toekomst naartoe moeten gaan.  Ja, sowieso, sowieso.

19:58 - Lukas Vandaele (LeadLine)
  Ik zeg het, allez, het is ook... Ik spreek een beetje tegen mijn eigen winkel, maar je zult al verbaasd zijn van de tools die er gewoon al bestaan.

20:05 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ja, ja.

20:06 - Lukas Vandaele (LeadLine)
  En het is daarom, we zijn ook zelf, we zijn nu nog maar een jaar bezig, maar we zitten in een branche die heel snel ontwikkelt.  En het is ook zo dat wij soms dingen zeggen van goed, op een gegeven moment gaan we ook gewoon meer naar een soort van advies moeten gaan, omdat mensen gewoon niet weten welke tools er bestaan.

20:24 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ja, ja.

20:25 - Lukas Vandaele (LeadLine)
  Allee, de zaken die jij nu zegt, kan ik nu in mijn eigen voordeel spreken en zeggen, ah ja, we kunnen dat vledig maken.  Goed, dat is allemaal custom, die kosten lopen ook op natuurlijk. Maar er zijn ongetwijfeld al tools die daar al meer research en development in gestoken hebben, en die misschien ook meer naar jullie behoeften...

20:45 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Nou ja, dat is ook door enorm dat Dynamite hier is geweest vorige week. Ja. Omdat zij toch al redelijk deel van onze data beschikbaar hebben.  Ja. Zit daar nog data die bij de marketingkantoor... Dat ze onze website, die particuliere website, beheren. De site waar we al onze offertes op maken is volledig web-based.  Dat is Zetanby Dynamite. Volgens mij had ik daar tussenin veel data verloren. Dat is hetgeen dat we een beetje gaan optimaliseren samen met onze workflow.  Dat het gewoon gemakkelijker is dan weleens geweest zijn dat die offerte bam, in bewezen van spreken, tien minuten later in de brievenbus.

21:30 - Lukas Vandaele (LeadLine)
  En concreet, kun je nog een concreet dat proces uit? Stel nu, ik ben particulier en ik wil een offerte aanbraken.

21:40 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  De meesten komen via de site of bellen. Via de site kun alles even, via de contactformulier. Op basis daarvan worden er veel offertes al vrijblijvend opgemaakt.  Maar we gaan altijd gaan pushen naar een wervenbezoek ter plaatse. Als je één keer ter plaatse ben geweest, heb je je voet in de deur en dan is het veel makkelijker om te verkopen.  We zijn ook zelfs zaakvoerder, dus naar vertrouwen toe is dat een heel stuk beter naar klanten toe en daar ook bewust voor gekozen om dat beheerbaar te houden voor ons twee.  Dus op zich, veel goed potentieel hebben we daardoor niet, omdat we het nog allemaal zelf moeten kunnen beheren en het ruipt heel veel tijd in particuliere afspraken, maar ook veel meer over nadenken, dus daardoor kunnen we niemand nog extra erbij zijn.  Maar dat proces kunnen we verbeteren, waardoor we veel meer afspraken zouden kunnen doen en we meer verkoop kunnen genereren.  Maar het proces dat een particulier doorgaans doorloopt, is ofwel bellen ofwel via de site ofwel mailen, als ze aanvinken van ja, ik wil een bezoek ter plaatse, wordt ook de dag zelf of dag nadien een afspraak ingeplant, wanneer dat dat schikt voor die klant, dan ga ik ter plaatse, meet ik het op en bezoek ik op basis daarvan de offerte.

22:53 - Lukas Vandaele (LeadLine)
  Ja, en die offerte wordt volledig manueel nog gemaakt. Maar niemand...

23:00 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Dat is via een site waar al onze producten zitten daarin. Ik heb een offerte in twee minuten gemaakt, dus we mogen manueel de naam, het adres, e-mail, telefoonnummer ingeven, welk product, welke hoeveelheden en dat nog een controleren en dan wordt er automatisch een pdf opgemaakt en verstuurd via mail.

23:20 - Lukas Vandaele (LeadLine)
  Oké, maar eigenlijk worden er geen offertes gemaakt zonder werkbezoek, klopt dat?

23:25 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Maar wel, klanten een meet staat hebben bijvoorbeeld, of plannen, of ze weten meer een renovatie van exact, ik heb zestig vierkante meter vandaan nodig, en ze sturen een mail, dan wordt er ook direct een offerte opgemaakt.  Doorgaans proberen we wel echt binnen 24 uur een offerte buiten te navragen, als er geen plaatsbezoekte plaatsen gewenst is.  Ja, inderdaad, anders moet er eerst een afspraak gemaakt worden.

23:50 - Lukas Vandaele (LeadLine)
  Ja, wordt er inderdaad uit te kijken, Maar, act aan dat zo hoor, dus inderdaad, bij je werkbezoek, heb je wel eigenlijk een systeem, ise.  verddorieде? regelmatig of bijvoorbeeld, ik bonus Je moet dat wel nog de gegevens invullen, maar dat is ook erg logisch, want dat plaatsbezoek zorgt er ook voor dat je nieuwe data hebt.  Je kunt niet van tevoren gaan automatiseren. Maar wat misschien wel mogelijk is, is inderdaad voor die kleinere aanvragen, als ik het zo mag doen, die eigenlijk direct een offert kunnen opgemaakt worden.  Momenteel is dat wel nog steeds de manuele handeling.

24:22 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ja, de mensen vullen dat contactformulier in, metgene dat ze willen, dan wordt dat bezorgd aan de sectoresse, dat zit dan in haar mailbox, die heeft dat contactformulier en op basis van dat contactformulier maakt zij de offert op.  Ja. Volgens mij, metgene dat er van data al ingegeven wordt in die site, met dat contactformulier moet er in principe onmiddellijk een offerte kunnen gegenereerd worden.  En is dat dan nog niet 100% juist, maakt dat niet uit, want we moeten hem toch aanpassen na plaatsbezoek.  En wat we echt willen pushen, is dat de mensen dat plaatsbezoek accepteëren. En als je dat plaatsbezoek hebt... Ja.  Maar je moet er quasi vanuit gaan dat dat tot de bestelling komt. Want de prijzen in onze markt zijn allemaal relatief gelijk, behalve dat je over de grote meter spreekt, dan is er echt al de prijzenoorlog aan de hand, maar bij particuliere werven zitten onze prijzen heel goed en kunnen grotere spelers daar moeilijker aan die prijzen geraten, maar dat dat bij grote werven nog een keer zit.  Dat wij nog met een eindschreven personeel zijn. Ik denk dat die workflow veel verbeterd kan worden. En dat dat ook voor de bureau een aanzienlijke werkklasvermindering zou geven.  Dat dat ook wel van eerst keer, ja...

25:38 - Lukas Vandaele (LeadLine)
  Ja, over, ja, sorry, dat kan ik onderbreek, over wat grote spreekt, of wat die offerten is dan een idee van?

25:47 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Op jaarbasis gemaakt worden?

25:49 - Lukas Vandaele (LeadLine)
  Ja, jaar of maand misschien. Misschien dat dat...

25:53 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ik vind het wel het makkelijkste test van te zijn. Kijk, doorgaans zit dat op het 2000... ... ... Ja, 2050, praktisch de 2000 en 2540 is dat er op jarenbald is gemaakt.

26:06 - Lukas Vandaele (LeadLine)
  Ja, dat is aanzienlijk inderdaad.

26:08 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Een verkooppercentage daarvan zou theoretisch rond de 30% moeten blijven. Ja.

26:16 - Lukas Vandaele (LeadLine)
  Maar als ik het dan zo goed hoor, Broes, denk ik eerlijk gezegd, inderdaad. Ik zeg ook altijd bij de AI-automatisaties, moet niet direct de Rolls Royce van de automatisaties gaan toepassen of gaan doen.  Dat is gewoon klein, al dan niet klein, maar toch met de meeste return on investment, zeg maar, gaan werken.  En mijn persoonlijk acteur lijkt dat inderdaad formulier op de site. Goed, het is misschien nog een trigger, dat is dan iets dat je zegt, om dat formulier in te vullen.  Dat kan al dan niet met een chatbot. Maar het ding is wel, als het wordt ingevuld, is het inderdaad nog werkdruk die bij de secretarissen ligt.  En ja, ga ik dat ook enigszins wat tijd doen. En Persoonlijk zie ik daar de snelste manier van werken, of de snelste return on investment, als je daar al een simpele automatisatie afsteekt.  Dus het formulier wordt ingevuld, wordt direct een offerte van opgemaakt en ter goedkeuring verstuurd naar de secretaris of die dat dan ook moest zijn.  Ik denk dat je daar al de snelste winst gaat kunnen pakken. En goed, je wilt dat dan op termijn misschien in die chatbot.  Het is niet zo dat, als we die automatisatie zouden maken, dat dat volledig in de vuilbak is, moest dat vanuit de chatbot komen.  Ik denk dat dat op zich ook wel goed gaat meevallen. Dus het is ook in die optiek niet echt verloren, weggesmeten geld om het zo te zeggen.  Nee, ja. Dus, ik weet niet, als jij wilt, wil ik rustig een dan nog een keer verder bekijken en een keer een vrijblijvende offerte verstuurd, wat dat zou zijn.

27:59 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ja. Ja. Ja. Ja.

28:00 - Lukas Vandaele (LeadLine)
  En dan zien we van daaruit, dan gaan we met elkaar verder, ja of nee. Ja wel, dat maakt zeker, dat is moeilijk, dat kun je dan moeilijk weggen ook, hè.

28:08 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Maar ja, dat is dan niet goed. Dat weet ik maar nooit, ik kan het er maar mee zien. Nee, dat mag je zeker doen, dus dan mag je mij uitsturen met degene die je van mij nodig hebt, om dat mogelijk te maken.

28:19 - Lukas Vandaele (LeadLine)
  Ja, het is dat inderdaad, want ik met Siwana inderdaad nog even vragen, eigenlijk, momenteel, dat wordt ingevuld, er is echt gewoon een mailtje dat dan bij de secretarisse komt met dat formulier, ja, oké.  En op basis van wat is dat formulier gemaakt, is dat gewoon onderdeel van jullie website, of is dat een specifiek...  Ja, dat is op basis van die contactaanvraag die op de site staat, offerteaanvraag, contact, dus daar moet je voornaam, naam, straat, postcode, e-mail, Ja, ik heb het hier voor mij, maar het is niet bijvoorbeeld dat we, alleen, het gaat er ook, ik ga nu maar iets in een plugin, of weet ik, kijk, veel wat, die het daar gebruikt voor wordt, nee, omdat wij achterliggend wel een beetje de trigger moeten hebben van, oké...  Stuur het naar ons door in plaats van de secretaris?

29:04 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Nee, dat offerteaanvraagformulier maakt gewoon een mailtje waar dat allemaal in staat en dat wordt doorgegeven aan de info.

29:12 - Lukas Vandaele (LeadLine)
  Ja, oké.

29:13 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Maar er wordt nog niet automatisch een klantenfiche aangemaakt in ons offerteprogramma. En welk programma is dat?

29:21 - Lukas Vandaele (LeadLine)
  Dat is Drupal.

29:24 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Jullie site zal dan ook met Drupal gemaakt zijn? Ja. Ja, Drupal beest. Ja, oké.

29:31 - Lukas Vandaele (LeadLine)
  Misschien kunnen we dan de automatisatie ook zo maken, opnieuw te bekijken. Maar dat er daar ook automatisch al inderdaad een offerte contact wordt gemaakt, zeg maar.

29:42 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ja, dat lijkt me. Volgens mij gaat er daar wel nog wat leads in verloor doen.

29:46 - Lukas Vandaele (LeadLine)
  Ja, ik kan me dat goed voorstellen. Zeker aan vast. Oké, maar dan spremen we dat zo af, Broes. Dus dan ga ik verder een kleine analyse doen daarvan.  Had ik nog iets nodig van data ga ik het sowieso vragen naar jou via mail. Ja. Ja. Ja. Ja.  Ik ga kijken dus inderdaad voor die offertes te automatiseren als het wordt aangevraagd, ook wat we met Drupal al dan niet kunnen doen, mag dat dan nog verwachten op mail, en ik zal ondertussen ook een keer kijken wat ik zelf nog denk wat dat eventueel handig kan zijn voor jullie als bedrijf.  Perfect.

30:17 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Goed? Ja.

30:19 - Lukas Vandaele (LeadLine)
  Oké, super, kijk, dikke merci als het is om de tijd te nemen en we horen elkaar nog iets aan.

30:24 - Broes Royaux (Royaux Chape en Isolatie | Brugge)
  Ja, oké. Oké, goeie. Goeie. Goeie. Goeie."""

cleanup_prompt = f"""
Je bent een expert in het omzetten van gesproken transcripties naar gestructureerde kennisdocumenten.

Hieronder staat een ruwe transcriptie van een zakelijk gesprek.
Zet dit om naar een gestructureerd kennisdocument dat:
- Zelfstandig leesbaar is zonder de audio of video
- Alle visuele verwijzingen ("zoals je ziet", "dit scherm") vervangt door expliciete beschrijvingen
- Gesproken taalruis verwijdert ("dus eh", "allee", "zeg maar")
- De kerninhoud behoudt maar structureert in duidelijke secties
- Gebruik maakt van deze vaste structuur:

## Onderwerp
[Wat gaat dit document over]

## Context
[Wanneer is dit relevant, voor wie]

## Kerninhoud
[De belangrijkste informatie gestructureerd]

## Pijnpunten / Uitdagingen
[Wat loopt er nu niet goed]

## Voorgestelde oplossingen
[Wat zijn de mogelijke oplossingen]

## Vervolgacties
[Concrete volgende stappen]

Transcriptie:
{transcriptie}

Geef alleen het gestructureerde document terug, geen uitleg erover.
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