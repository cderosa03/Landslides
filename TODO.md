# TODO — Integrazione multimodale PlanetScope, Sentinel-2 e AUX

Questo documento accompagna la Fase 2. Una voce va marcata con `[x]` solo
dopo aver implementato la modifica e aver eseguito il relativo controllo o test.

## 1. Contratto dei campioni e controlli iniziali

- [X] Definire forme, dtype e significato di ogni tensore.
- [X] Planet pre/post: richiedere esattamente forma `(3, H, W)`.
- [X] Sentinel pre/post: richiedere forma `(N_TEMPORAL, 10, H, W)`.
- [X] Maschera target: richiedere forma `(1, H, W)`.
- [X] Maschere temporali Sentinel: richiedere forma `(N_TEMPORAL,)` e dtype booleano.
- [X] AUX: definire forma, canali, unità di misura e normalizzazione esplicita
  (attualmente unità sorgente, senza normalizzazione).
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

- [ ] Raccogliere tutti i tile che coprono la stessa patch e data.
- [ ] Mosaicare i tile prima o durante la riproiezione.
- [ ] Impedire che un tile sovrascriva il precedente.
- [ ] Rendere deterministico l'ordinamento di tile, livelli e date.
- [ ] Calcolare la percentuale di copertura valida di ogni patch.
- [ ] Scartare o segnalare le patch senza copertura sufficiente.

## 6. NoData e nuvole Sentinel

- [ ] Gestire esplicitamente `src_nodata`, `dst_nodata` e inizializzazione della destinazione.
- [ ] Non confondere zero radiometrico, padding temporale e assenza di dati.
- [ ] Riproiettare e utilizzare Sen2Cor/SCL.
- [ ] Produrre una maschera spaziale dei pixel Sentinel validi.
- [ ] Definire una soglia massima di nuvole o pixel invalidi.
- [ ] Evitare che il resampling bilineare mescoli NoData e pixel validi.

## 7. Selezione temporale

- [ ] Richiedere almeno un frame reale sia pre sia post, salvo diversa decisione documentata.
- [ ] Correggere l'intervallo di download Lombok per includere acquisizioni post-evento.
- [ ] Validare automaticamente intervalli di download e cutoff di tutti gli eventi.
- [ ] Salvare o registrare nei log le date selezionate per ogni campione.
- [ ] Definire una distanza temporale massima dall'evento.
- [ ] Rendere esplicita la politica per campioni temporalmente incompleti.

## 8. Diagnostica geospaziale riproducibile

- [ ] Creare uno script che controlli CRS, transform, bounds, risoluzione e dimensioni.
- [ ] Verificare ordine degli assi e coordinate dei pixel.
- [ ] Verificare copertura e percentuale di pixel validi.
- [ ] Generare un report CSV o JSON per campione.
- [ ] Salvare viste affiancate Planet/Sentinel.
- [ ] Salvare overlay con feature o bordi confrontabili.
- [ ] Permettere una selezione casuale riproducibile dei campioni.
- [ ] Fallire quando lo scostamento supera la tolleranza definita.

## 9. Integrazione AUX nel modello — critico

- [ ] Decidere se gli AUX devono coprire la stessa area Planet o un contesto più ampio.
- [ ] Normalizzare DEM e slope.
- [ ] Rappresentare aspect in forma circolare, ad esempio seno e coseno.
- [ ] Restituire AUX dal dataset multimodale.
- [ ] Mantenere AUX nel DataLoader.
- [ ] Aggiungere un encoder AUX al modello.
- [ ] Fondere le feature AUX con Planet e Sentinel.
- [ ] Passare AUX in training, validation e inference.
- [ ] Verificare che i parametri AUX ricevano gradienti non nulli.
- [ ] Aggiungere un test nel quale una variazione di AUX modifica l'output.

## 10. Dataset, cache e campioni scartati

- [ ] Includere root, eventi, patch size e configurazione nella chiave delle cache.
- [ ] Invalidare cache contenenti percorsi mancanti o configurazioni obsolete.
- [ ] Rilevare duplicati di `(event, patch_id)`.
- [ ] Segnalare il numero di campioni esclusi e il motivo.
- [ ] Rendere esplicito l'esito dell'intersezione Planet/Sentinel.
- [ ] Decidere se riattivare o rimuovere il sampler bilanciato.

## 11. Decoder e forme interne

- [ ] Sostituire l'upsampling tramite `repeat_interleave` con una vera operazione spaziale 2D.
- [ ] Controllare le forme delle feature a tutti i livelli.
- [ ] Verificare che l'upsampling non sposti strutture localizzate.
- [ ] Testare tutte le patch size dichiarate come supportate.

## 12. Training, metriche e risultati

- [ ] Correggere la chiamata a `plot_pr_curve`.
- [ ] Correggere l'associazione fra miglior F1 e threshold.
- [ ] Gestire il punto della curva PR privo di soglia.
- [ ] Aggiornare score, soglia ed early stopping prima di salvare il checkpoint.
- [ ] Salvare epoca, soglia, metriche e configurazione nel best checkpoint.
- [ ] Scrivere `config.json`, `history.csv` e `results.json`.
- [ ] Alimentare e chiudere correttamente TensorBoard.
- [ ] Registrare il motivo della terminazione.
- [ ] Salvare il contenuto di `--description`.
- [ ] Aggiornare automaticamente `experiments.csv`.
- [ ] Valutare `pos_weight` o sampling per lo sbilanciamento delle classi.

## 13. Inference e test end-to-end

- [ ] Creare uno script di inference dedicato.
- [ ] Caricare modello, configurazione e threshold dal checkpoint.
- [ ] Riutilizzare esattamente il preprocessing della validation.
- [ ] Salvare probabilità e maschera binaria georeferenziate.
- [ ] Creare fixture sintetiche separate dai dati originali.
- [ ] Eseguire un passo minimo di training, validation e inference.
- [ ] Controllare i gradienti dei rami Planet, Sentinel e AUX.
- [ ] Eseguire uno smoke test su pochi campioni reali quando saranno disponibili.

## 14. Documentazione e riproducibilità

- [ ] Aggiornare `requirements.txt`, inclusa la gestione GDAL.
- [ ] Documentare struttura delle directory e comandi di esecuzione.
- [ ] Documentare livello Sentinel, bande, normalizzazione e NoData.
- [ ] Documentare la politica sui campioni incompleti.
- [ ] Aggiornare `DOCUMENTAZIONE_PROGETTO.md` con le correzioni realmente verificate.
- [ ] Separare chiaramente risultati dichiarati e risultati riproducibili.

## Criteri finali di completamento

- [ ] Tutte le modalità dello stesso campione sono geograficamente congruenti.
- [ ] Le augmentation geometriche sono dimostrate sincronizzate da un test.
- [ ] Planet, Sentinel e AUX modificano realmente l'output e ricevono gradienti.
- [ ] Training e validation completano almeno uno smoke test senza errori.
- [ ] L'inference produce una maschera georeferenziata verificabile.
- [ ] I comandi di riproduzione funzionano in un ambiente pulito.
- [ ] Risultati, configurazione e checkpoint sono salvati insieme.
