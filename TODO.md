# TODO — Integrazione multimodale PlanetScope, Sentinel-2 e AUX

Questo documento accompagna la Fase 2. Una voce va marcata con `[x]` solo
dopo aver implementato la modifica e aver eseguito il relativo controllo o test.

Nota di stato: le voci gia' `[x]` indicano codice implementato e controlli
statici eseguiti. Le verifiche su PyTorch/GDAL e dati reali sono raccolte
nella sezione 15 e restano obbligatorie prima dei criteri finali.

## 1. Contratto dei campioni e controlli iniziali

- [X] Definire forme, dtype e significato di ogni tensore.
- [X] Planet pre/post: richiedere esattamente forma `(3, H, W)`.
- [X] Sentinel pre/post: richiedere forma `(N_TEMPORAL, 10, H, W)`.
- [X] Maschera target: richiedere forma `(1, H, W)`.
- [X] Maschere temporali Sentinel: richiedere forma `(N_TEMPORAL,)` e dtype booleano.
- [X] AUX: definire forma, canali, unità di misura e normalizzazione esplicita
  (`asinh(DEM/1000)`, slope/90, seno e coseno dell'aspect).
- [X] Rifiutare tensori con NaN, Inf, numero di canali o dimensioni errati.
- [X] Eliminare il fallback Sentinel fisso a `(10, 128, 128)`.
- [X] Controllare che il DataLoader mantenga tutte le modalità previste.
- [X] Aggiungere log iniziali con nomi e forme degli input ricevuti dal modello.

## 2. Augmentation geometrica sincronizzata — critico

- [X] Centralizzare nel dataset multimodale la generazione dei parametri casuali.
- [X] Estrarre una sola volta flip, rotazione ed eventuale crop per campione.
- [X] Applicare gli stessi parametri a Planet pre/post.
- [X] Applicare gli stessi parametri a tutti i frame Sentinel pre/post.
- [X] Applicare gli stessi parametri alla mask.
- [X] Applicare gli stessi parametri agli AUX spaziali.
- [X] Separare e documentare le trasformazioni radiometriche specifiche dei sensori.
- [X] Aggiungere un test sintetico con marker spaziali riconoscibili.
- [X] Verificare nel test che tutte le modalità restino allineate dopo l'augmentation.

## 3. Allineamento Planet pre/post — critico

- [x] Confrontare CRS, affine transform, bounds, risoluzione e dimensioni prima del patching.
- [x] Interrompere la generazione in caso di mismatch non corretto.
- [x] Riproiettare Planet post sulla griglia Planet pre quando necessario.
- [x] Riproiettare mask e raster ausiliari sulla griglia corretta quando necessario.
- [x] Non assegnare a dati non riproiettati la trasformazione della patch Planet pre.
- [x] Aggiungere test con raster aventi CRS o transform differenti.

## 4. Preprocessing Sentinel-2

- [x] Scegliere esplicitamente il livello Sentinel da utilizzare, preferibilmente L2A.
- [x] Correggere la ricerca dei nomi delle bande L2A.
- [x] Verificare ordine e presenza delle dieci bande.
- [x] Impedire il mescolamento o la sovrascrittura silenziosa di L1C e L2A.
- [x] Conservare livello, tile, data e prodotto sorgente nei metadati.
- [x] Rendere configurabili i percorsi attualmente hardcoded.
- [x] Caricare le credenziali solo durante il download, non all'import del modulo.

## 5. Gestione di più tile Sentinel — critico

- [x] Raccogliere tutti i tile che coprono la stessa patch e data.
- [x] Mosaicare i tile prima o durante la riproiezione.
- [x] Impedire che un tile sovrascriva il precedente.
- [x] Rendere deterministico l'ordinamento di tile, livelli e date.
- [x] Calcolare la percentuale di copertura valida di ogni patch.
- [x] Scartare o segnalare le patch senza copertura sufficiente.

## 6. NoData e nuvole Sentinel

- [x] Gestire esplicitamente `src_nodata`, `dst_nodata` e inizializzazione della destinazione.
- [x] Non confondere zero radiometrico, padding temporale e assenza di dati.
- [x] Riproiettare e utilizzare Sen2Cor/SCL.
- [x] Produrre una maschera spaziale dei pixel Sentinel validi.
- [x] Definire una soglia massima di nuvole o pixel invalidi.
- [x] Evitare che il resampling bilineare mescoli NoData e pixel validi.

## 7. Selezione temporale

- [x] Richiedere almeno un frame reale sia pre sia post, salvo diversa decisione documentata.
- [x] Correggere l'intervallo di download Lombok per includere acquisizioni post-evento.
- [x] Validare automaticamente intervalli di download e cutoff di tutti gli eventi.
- [x] Salvare o registrare nei log le date selezionate per ogni campione.
- [x] Definire una distanza temporale massima dall'evento.
- [x] Rendere esplicita la politica per campioni temporalmente incompleti.

## 8. Diagnostica geospaziale riproducibile

- [x] Creare uno script che controlli CRS, transform, bounds, risoluzione e dimensioni.
- [x] Verificare ordine degli assi e coordinate dei pixel.
- [x] Verificare copertura e percentuale di pixel validi.
- [x] Generare un report CSV o JSON per campione.
- [x] Salvare viste affiancate Planet/Sentinel.
- [x] Salvare overlay con feature o bordi confrontabili.
- [x] Permettere una selezione casuale riproducibile dei campioni.
- [x] Fallire quando lo scostamento supera la tolleranza definita.

## 9. Integrazione AUX nel modello — critico

- [x] Usare AUX raster allineati alla stessa area Planet per la fusione spaziale.
- [x] Normalizzare DEM e slope.
- [x] Rappresentare aspect in forma circolare, ad esempio seno e coseno.
- [x] Restituire AUX dal dataset multimodale.
- [x] Mantenere AUX nel DataLoader.
- [x] Aggiungere un encoder AUX al modello.
- [x] Fondere le feature AUX con Planet e Sentinel.
- [x] Passare AUX in training, validation e inference.
- [x] Verificare che i parametri AUX ricevano gradienti non nulli.
- [x] Aggiungere un test nel quale una variazione di AUX modifica l'output.

## 10. Dataset, cache e campioni scartati

- [x] Includere root, eventi, patch size e configurazione nella chiave delle cache.
- [x] Invalidare cache contenenti percorsi mancanti o configurazioni obsolete.
- [x] Rilevare duplicati di `(event, patch_id)`.
- [x] Segnalare il numero di campioni esclusi e il motivo.
- [x] Rendere esplicito l'esito dell'intersezione Planet/Sentinel.
- [x] Decidere se riattivare o rimuovere il sampler bilanciato.

## 11. Decoder e forme interne

- [x] Sostituire l'upsampling tramite `repeat_interleave` con una vera operazione spaziale 2D.
- [x] Controllare le forme delle feature a tutti i livelli.
- [x] Aggiungere un test con marker per verificare che l'upsampling non sposti strutture localizzate.
- [x] Testare tutte le patch size dichiarate come supportate.

## 12. Training, metriche e risultati

- [x] Correggere la chiamata a `plot_pr_curve`.
- [x] Correggere l'associazione fra miglior F1 e threshold.
- [x] Gestire il punto della curva PR privo di soglia.
- [x] Aggiornare score, soglia ed early stopping prima di salvare il checkpoint.
- [x] Salvare epoca, soglia, metriche e configurazione nel best checkpoint.
- [x] Scrivere `config.json`, `history.csv` e `results.json`.
- [x] Alimentare e chiudere correttamente TensorBoard.
- [x] Registrare il motivo della terminazione.
- [x] Salvare il contenuto di `--description`.
- [x] Aggiornare automaticamente `experiments.csv`.
- [x] Valutare `pos_weight` o sampling per lo sbilanciamento delle classi.

## 13. Inference e test end-to-end — implementazione

- [x] Creare uno script di inference dedicato.
- [x] Caricare modello, configurazione e threshold dal checkpoint.
- [x] Riutilizzare esattamente il preprocessing della validation.
- [x] Salvare probabilità e maschera binaria georeferenziate.
- [x] Creare fixture sintetiche separate dai dati originali.
- [x] Aggiungere uno smoke test sintetico di training, validation, checkpoint e inference.
- [x] Controllare i gradienti dei rami Planet, Sentinel e AUX.
- [x] Conservare CRS, transform e dtype negli output del test di inference.

## 14. Documentazione e riproducibilità

- [x] Aggiornare `requirements.txt`, inclusa la gestione GDAL.
- [x] Documentare struttura delle directory e comandi di esecuzione.
- [x] Documentare livello Sentinel, bande, normalizzazione e NoData.
- [x] Documentare la politica sui campioni incompleti.
- [x] Aggiornare `DOCUMENTAZIONE_PROGETTO.md` con lo stato corrente e i limiti delle verifiche.
- [x] Separare chiaramente risultati dichiarati e risultati riproducibili.

## 15. Validazione runtime prima della chiusura

- [ ] Installare le dipendenze dichiarate in un ambiente pulito.
- [ ] Eseguire tutti i test PyTorch/GDAL, inclusi AUX, decoder, mosaico e allineamento.
- [ ] Generare il report `diagnose_geospatial.py` su un campione casuale riproducibile.
- [ ] Esaminare manualmente i PNG Planet/Sentinel/mask prodotti dalla diagnostica.
- [ ] Rigenerare le patch Sentinel con L2A, `s2_valid.tif` e mosaico multi-tile.
- [ ] Eseguire uno smoke test training/validation di una o due epoche.
- [ ] Eseguire lo smoke test sintetico training/validation/checkpoint/inference.
- [ ] Eseguire uno smoke test su pochi campioni reali.
- [ ] Eseguire inference sul checkpoint prodotto e controllare probabilita' e mask.
- [ ] Se una verifica fallisce, correggere il codice e ripetere questa sezione.

## Criteri finali di completamento

- [ ] Tutte le modalità dello stesso campione sono geograficamente congruenti.
- [ ] Le augmentation geometriche sono dimostrate sincronizzate da un test.
- [ ] Planet, Sentinel e AUX modificano realmente l'output e ricevono gradienti.
- [ ] Training e validation completano almeno uno smoke test senza errori.
- [ ] L'inference produce una maschera georeferenziata verificabile.
- [ ] I comandi di riproduzione funzionano in un ambiente pulito.
- [ ] Risultati, configurazione e checkpoint sono salvati insieme.
