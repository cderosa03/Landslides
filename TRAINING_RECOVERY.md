# Training dopo timeout e riavvio del container

I log disponibili mostrano due episodi diversi: timeout del DataLoader per
`swinunet_128_25` alle 16:48 del 15 settembre 2026 (circa 20,7 GiB di RAM),
e RAM prossima a 64 GiB alle 21:52 seguita dal riavvio del container durante
il periodo dell'esperimento `_26`. L'OOM di quest'ultimo resta un'ipotesi:
non sono disponibili gli eventi del vecchio container.

## Modifiche

- DataLoader con `spawn`: i worker non ereditano CUDA, GDAL e thread del padre.
- Cache GDAL esplicita di 256 MiB **per processo**, VSI cache disabilitata,
  un thread GDAL per processo. Questo non limita l'intera RAM del processo,
  la page cache del kernel o gli altri programmi nel container.
- Memoria pinned disabilitata di default; `--pin-memory` permette di riattivarla.
- Stack Sentinel preallocati e normalizzazione in-place, mantenendo date,
  bande, padding e valori normalizzati. L'indice Planet usa una configurazione
  serializzabile senza il mismatch tuple/list che invalidava sempre la cache.
- `run_logs/<timestamp_pid>/training.log` contiene avanzamento ed eccezioni;
  `memory.jsonl` registra ogni 10 secondi memoria totale del cgroup, anonima,
  cache, shmem, eventi OOM e RSS del padre. `stacks.log` mostra la posizione del
  padre dopo 300 secondi senza avanzamento; `worker_*_stacks.log` contiene
  snapshot periodici dei worker, anche quando sono semplicemente in attesa.
- Checkpoint scritti prima in un file temporaneo, poi sostituiti. Il salvataggio
  resta a fine epoca, dopo la validation: non e una ripresa a meta epoca.
- Punti della curva precision/recall non definiti sono trattati come zero
  prima della scelta della soglia, evitando che NaN propaghi a F1 e IoU.

La scelta di spawn segue [Rasterio](https://rasterio.readthedocs.io/en/stable/topics/concurrency.html).
I limiti della cache seguono [GDAL](https://gdal.org/en/stable/user/configoptions.html#performance-and-caching).

## Sul server, nella cartella del progetto

Il launcher attiva esplicitamente `.venv/bin/activate` e usa GPU 0 salvo un valore
`CUDA_VISIBLE_DEVICES` gia impostato. Mantiene modello small, batch 8,
training EmiliaRomagna2023 e validation Michoacan2022.

```bash
bash run_stable_training.sh smoke
```

La prova cerca soltanto le patch Sentinel valide necessarie a 20 batch di
training e fino a 4 di validation, poi indicizza Planet solo per quelle patch.
Le directory candidate sono visitate in ordine casuale riproducibile; le patch
senza coppia temporale o senza tutti i file Planet non consumano il limite.
Il contatore mostra le singole patch e il log registra l'avanzamento ogni
15 secondi durante la scansione. Il training completo continua a usare tutti
i campioni validi. La prova si interrompe dopo 45 minuti se non termina prima.
Il successo e indicato da `SMOKE TEST PASSED`. Gli esperimenti `_smoke`
sono diagnostici: metriche escluse da `experiments.csv`, checkpoint non
riutilizzabili con `--resume` per il training completo.

La prova controlla caricamento, forward/backward, validation e salvataggio;
non garantisce la stabilita per ore, ne la qualita scientifica del modello.
Verificare i tempi di caricamento e memoria prima di impegnare piu giorni.

Dopo una prova riuscita, il training completo si avvia con:

```bash
nohup bash run_stable_training.sh train > "launch_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

Il launcher mantiene 110 epoche massime (10 warm-up + 100 principali) e
l'early stopping esistente sulla AUPRC di validation, con patience 20.
Usa `--match-train-to-val --positive-fraction 0.5`: ogni epoca estrae senza
ripetizioni `min(numero training, numero validation)` campioni training,
puntando al 50% di patch con frana e completando con patch senza frana.
La selezione cambia con seed `42 + indice epoca`, quindi la stessa epoca
ripresa da un checkpoint usa gli stessi indici. La validation resta completa
e fissa. La scansione delle maschere legge direttamente i GeoTIFF `mask.tif`,
senza caricare le serie Sentinel.
Con 14.725 patch training e 6.679 validation, sono 6.679 campioni e 835 batch
per fase con batch size 8. Tutte le patch training restano nel pool disponibile.

Questa e una riduzione del numero di aggiornamenti per epoca, non un
bilanciamento tra pixel/patch positivi e negativi. Per la tesi riportare
campioni per epoca, criterio di selezione e aggiornamenti effettivi: 110 epoche
limitate non equivalgono a 110 passaggi sull'intero training set. I conteggi
sono salvati in config.json. La durata stimata dalla prova breve e circa
1,8 ore/epoca, circa 8 giorni per 110 epoche, soggetta alle prestazioni NFS
e da verificare sulla prima epoca completa. L'early stopping non garantisce
un arresto anticipato.

La scansione iniziale considera comunque l'intero pool training. Per usare
epoche che coprono tutti i campioni, invocare train.py senza --match-train-to-val.

Non ripetere il comando mentre un training e attivo. I risultati e i
checkpoint restano nella nuova directory `exp/swinunet_128_*` indicata dal log.
Per riprendere un esperimento completo usare `train.py --resume <directory>`
con gli stessi eventi, modello, scheduler e iperparametri originali.
Per riprendere questo protocollo includere anche --match-train-to-val.

In caso di errore inviare la directory diagnostica del nuovo tentativo.
Se il test fallisce ancora con spawn, una successiva prova con
`--num-workers 0` puo aiutare a localizzare il caricamento bloccato tramite
`stacks.log`. Non disabilitare il timeout e lasciare lavorare alla cieca.
