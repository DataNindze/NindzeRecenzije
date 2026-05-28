# Project: Croatian Sentiment Reviews Corpus

# 1.ČLANOVI TIMA
- Lorena Čizmek
- Dora Posilović
- Sara Henč
- Nika Haraminčić

# 2.OPIS PROJEKTA 
Cilj ovog projekta bio je izrada cjelovitog korpusa prikupljanjem rečenica na hrvatskom jeziku sa web stranice najdoktor.com. Nad navedenim skupom podataka napravljena je analiza sentimenta. Svi prikupljeni podatci bili su javno dostupni. Projekt je pokrenut u sklopu kolegija Obrada prirodnog jezika na Filozofskom fakultetu Sveučilišta u Zagrebu.

# 3.PRIKUPLJANJE PODATAKA
U svrhu bržeg prikupljanja potrebne količine podataka razvijen je program za automatsko prikupljanje potrebne količine javno dostupnih komentara s portala najdoktor.com. 

# 4.PILOT ANOTIRANJE
Nakon izrade cjelokupnog korpusa napravljeno je pilot-anotiranje sentimenta.Ova faza projekta napravljena je s ciljem usklađivanja i provjere konzistentnosti svih anotatora naše grupe prije početka obrade cijelog skupa podataka. 

Klasifikacija sentimenta temeljila se na ljestvici od 5 stupnjeva:
*negativan (negative): izražavanje nezadovoljstva ili kritike
*neutralan (neutral): informativni komentari bez izricanja negativnog ili pozitivnog
*pozitivan (positive): izražavanje zadovoljstva ili pohvale
*mješovit (mixed): komentari koji sadrže pozitivne i negativne elemente
*sarkastičan (sarcasm): ironični ili sarkastični komentari

Za potrebe ovog pilot-istraživanja iz cjelovitog korpusa izdvojen je nasumičan uzorak od 150 rečenica. Svim članicama tima dodijeljen je identičan skup podataka koji je uz izvorni tekst sadržavao prazan stupac za unos vlastite oznake. Proces anotacije odvijao se potpuno neovisno - svaka je članica samostalno procijenila i označila svih 150 rečenica. 
Nakon završetka pojedinačnog ocjenjivanja, uslijedila je zajednička analiza i evaluacija dobivenih rezultata. Koristeći Python biblioteke Pandas i statsmodels, izračunata je Fleiss Kappa vrijednost koja je iznosila 0.78. 


# 5.FINALNO ANOTIRANJE
Nakon pilot-anotiranja provedeno je anotiranje cjelokupnog korpusa na isti način koristeći jednaku ljestvicu od 5 stupnjeva. Anotaciju su provele 3 članice grupe, dok je četvrta imala ulogu data curator-a, čija je uloga bila odlučivanje o finalnoj anotaciji u slučajevima kada bi se anotacije ostalih članica razilazile. 

# 6.STROJNO UČENJE
Nakon oblikovanja i anotiranja korpusa, provedena je klasifikacija primjenom 2 klasična algoritma strojnog učenja: Logistic Regression i Multinomial Naive Bayes. Cilj je bio usporediti njihovu uspješnost predviđanja sentimenta rečenica iz prikupljenog korpusa. 

Za izdvajanje značajki i numeričku reprezentaciju teksta korišten je TF-IDF (Term Frequency-inverse Document Frequency) iz biblioteke scikit-learn. Vektorizator je konfiguriran uz obuhvaćanje unigrama i bigrama te sa sljedećim parametrima: 

| Parametar | Vrijednost | Obrazloženje |
| :--- | :--- | :--- |
| `lowercase` | `True` | Sav tekst se prije analize pretvara u mala slova radi ujednačavanja tokena. |
| `ngram_range` | `(1, 2)` | Koriste se unigrami i bigrami, čime se omogućuje prepoznavanje dvočlanih veza (npr. negacija). |
| `min_df` | `2` | Ignoriraju se tokeni (riječi i fraze) koji se u cijelom korpusu pojavljuju samo jednom. |
| `max_features` | `50000` | Vokabular je ograničen na najinformativnijih 50 000 tokena kako bi se smanjila dimenzionalnost. |

Model logističke regresije (`LogisticRegression`) trenira se uz parametar `class_weight="balanced"` kako bi se kompenzirala prirodna nejednaka distribucija klasa unutar korpusa, gdje su određene kategorije zastupljenije u odnosu na specifične komentare poput mješovitih (`mixed`) ili sarkastičnih (`sarcasm`). Također, definiran je maksimalni broj iteracija `max_iter=1000` radi osiguravanja konvergencije algoritma, uz postavljen `random_state=42` koji jamči potpunu ponovljivost rezultata. Drugi klasifikator, multinomijalni Naive Bayes (`MultinomialNB`), koristi standardnu vrijednost parametra zaglađivanja `alpha=1.0`.

Cijeli je proces automatiziran i strukturiran unutar Python modula radi preglednosti, modularnosti i ponovne upotrebe koda:

* **`machine_learning_results.py`**: Glavna skripta koja objedinjuje cjelokupni pipeline strojnog učenja. Ona učitava i čisti podatke (funkcije `read_dataset` i `clean_dataset`), inicijalizira TF-IDF vektorizator, provodi treniranje i evaluaciju četiriju različitih eksperimentalnih postavki te generira izlazne izvještaje.
* **Struktura eksperimenata**: Skripta evaluira ponašanje modela kroz četiri zasebne konfiguracije (kombiniranjem algoritama i dvaju različitih skupova za treniranje: vlastitog skupa `Train-3` te proširenog, kombiniranog skupa `TRAIN`). Svaki od istreniranih modela testira se na četiri neovisna testna skupa (`Test 1: group 1`, `Test 2: group 2`, `Test 3: group 3 (OURS)` i `Test 4: group 4`).

Za svaku pojedinu kombinaciju (testni skup, model, trening-skup) program računa četiri standardne evaluacijske metrike:
* `accuracy`: udio točno klasificiranih rečenica u odnosu na ukupan broj primjera.
* `precision`: preciznost ponderirana prema broju primjeraka po klasi (`weighted`).
* `recall`: odziv ponderiran prema broju primjeraka po klasi (`weighted`).
* `F1`: harmonijska sredina preciznosti i odziva, također ponderirana prema distribuciji klasa (`weighted`).

Istrenirani klasifikatori i pripadajući transformatori trajno se pohranjuju u mapu `models/` kao `.joblib` datoteke. Dinamički nazivi datoteka prate strukturu eksperimenta (npr. `logistic_regression_train_3_model.joblib` i pripadajući `logistic_regression_train_3_tfidf_vectorizer.joblib`), što omogućuje kasniju izravnu inferenciju i primjenu nad novim tekstualnim unosima bez potrebe za ponovnim fittanjem i treniranjem sustava.

Rezultati svih provedenih eksperimenata ispisuju se u terminalu, a zatim automatski strukturiraju i trajno pohranjuju u datoteku `ml_results.csv` (u CSV formatu s točka-zarez separatorom) te u obliku preglednih tablica unutar dokumenta `results.md`.

## Potrebne biblioteke za pokretanje (machine learning)

```bash
pip install scikit-learn pandas joblib tabulate
```
# 7.DUBOKO UČENJE 


Nakon provedbe klasičnih metoda, klasifikacija osjećaja proširena je primjenom dvaju modela plitkog dubokog učenja (Shallow Deep Learning): konvolucijske neuronske mreže (`CNN`) i ponavljajuće neuronske mreže s prigušenim povratnim jedinicama (`GRU`). Cilj je bio ispitati i usporediti potencijal neuronskih mreža u modeliranju nelinearnih odnosa i sekvencijalnih značajki unutar medicinskih recenzija.
 
Za numeričku reprezentaciju teksta i izlučivanje semantičkih značajki korišteni su pretrenirani ugradbeni vektori riječi (*word embeddings*) za hrvatski jezik iz projekta **FastText** (`cc.hr.300.vec`), dok su riječi izvan tog vokabulara inicijalizirane nasumično. Vektorizacija i priprema podataka konfigurirane su uz sljedeće parametre:
 
### Rezultati dubokog učenja:
 
| # | method | algorithm | train | Test 1: group 1 | Test 2: group 2 | Test 3: group 3 (OURS) | Test 4: group 4 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **2.a** | Shallow Deep Learning | CNN | TRAIN | P: 0.712, R: 0.509, F1: 0.588, Acc: 0.509 | P: 0.594, R: 0.388, F1: 0.441, Acc: 0.388 | P: 0.709, R: 0.584, F1: 0.632, Acc: 0.584 | P: 0.752, R: 0.535, F1: 0.609, Acc: 0.535 |
| **2.b** | Shallow Deep Learning | GRU | TRAIN | P: 0.714, R: 0.568, F1: 0.621, Acc: 0.568 | P: 0.624, R: 0.550, F1: 0.579, Acc: 0.550 | P: 0.738, R: 0.646, F1: **0.682**, Acc: 0.646 | P: 0.746, R: 0.556, F1: 0.624, Acc: 0.556 |
 
Oba modela trenirana su na spojenom skupu podataka `TRAIN`. Kako bi se neutralizirala izrazita neujednačenost klasa u korpusu, tijekom izračuna funkcije gubitka unutar `PyTorch`-a primijenjene su težine klasa (`class_weights=True`) izračunane na temelju njihove distribucije u trening-skupu.
 
Mreže su implementirane sa specifičnim arhitektonskim značajkama:
 
* **`CNN`**: Koristi višestruke veličine konvolucijskih filtara (3, 4 i 5) kako bi istovremeno obuhvatio n-grame različitih duljina, uz postavljenih 100 filtara po svakoj veličini i primjenu `Adam` optimizatora.
 
* **`GRU`**: Konfiguriran je kao dvosmjerni (*bidirectional*) GRU sa skrivenim slojem veličine 128 (`hidden_size=128`), čime se omogućuje kontekstualna analiza rečenice s lijeve na desnu stranu i obrnuto, također uz primjenu `Adam` optimizatora.
 
Cijeli je eksperiment automatiziran i strukturiran unutar Python modula radi preglednosti i lakše reprodukcije eksperimenata:
 
* **`deep_learning_experiment.py`**: Glavna skripta koja upravlja cjelokupnim procesom dubokog učenja. Ona obuhvaća pretprocesiranje teksta, izgradnju rječnika, inicijalizaciju PyTorch skupa podataka (`Dataset`) i punjača podataka (`DataLoader`), izvršavanje petlje za treniranje kroz maksimalno 30 epoha (`EPOCHS=30`) s paketima veličine 32 (`BATCH_SIZE=32`) te evaluaciju modela.
 
* **Validacija i rano zaustavljanje (*Early Stopping*)**: Za validaciju je kreiran jedinstveni skup spajanjem četiriju validacijskih datoteka. Gubitak na validacijskom skupu računao se nakon svake epohe, a primijenjen je mehanizam ranog zaustavljanja s tolerancijom od 3 epohe (`patience=3`) kako bi se treniranje prekinulo u optimalnom trenutku.
 
Za provjeru robusnosti, oba modela testirana su na četiri neovisna testna skupa (`Test 1: group 1`, `Test 2: group 2`, `Test 3: group 3 (OURS)` i `Test 4: group 4`) kroz četiri standardne metrike ponderirane prema broju primjera po klasi: `accuracy`, `weighted precision`, `weighted recall` i `weighted F1-score`.
 
Istrenirane neuronske mreže pohranjuju se u mapu `dl_models/` kao PyTorch `.pt` datoteke (`cnn_TRAIN.pt` i `gru_TRAIN.pt`), dok se povijest kretanja gubitka kroz epohe bilježi u pripadajuće `.csv` datoteke.
 
Konačni rezultati i usporedbe automatski se strukturiraju u obliku tablica i pohranjuju u dokumente `deep_learning_results.csv` i `results_deep_learning.md`, dok se pripadajuće matrice zabune izvoze u mapu `confusion_matrices/`.
 
## Potrebne biblioteke za pokretanje (Deep Learning)
 
```bash
pip install torch pandas numpy scikit-learn openpyxl tabulate
```





