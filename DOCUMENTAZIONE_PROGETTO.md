# Documentazione del progetto Landslides

## Obiettivo

Il progetto segmenta le frane a livello di pixel combinando immagini
PlanetScope pre/post-evento, serie temporali Sentinel-2 e dati topografici. La
rete restituisce un logit per pixel; sigmoid e soglia producono rispettivamente
la probabilità e la maschera binaria.

Gli eventi configurati sono Lombok 2018, Filippine 2019, Michoacán 2022 ed
Emilia-Romagna 2023. Per default i primi tre eventi diversi da Lombok sono
usati per il training e Lombok per la validazione.

## Componenti

- `build_ps_dataset.py`: download e patching PlanetScope, mask e AUX;
- `s2_builder/S2_download.py`: download Sentinel-2;
- `s2_builder/combine_bands.py`: composizione delle bande L2A;
- `s2_builder/make_patches.py`: mosaico e riproiezione Sentinel sulla griglia
  Planet;
- `dataset/`: indicizzazione, caricamento, controlli e augmentation;
- `models/swinunet.py`: modello multimodale;
- `train.py`: training, validazione e artifact degli esperimenti;
- `inference.py`: inferenza di una patch e output georiferiti;
- `diagnose_geospatial.py`: report e immagini per verificare l'allineamento;
- `tests/`: test statici, sintetici e geospaziali.

## Contratto di un campione

| Campo | Forma | Significato |
|---|---:|---|
| `planet_pre` | `(3,H,W)` | PlanetScope prima dell'evento, valori divisi per 255 |
| `planet_post` | `(3,H,W)` | PlanetScope dopo l'evento, valori divisi per 255 |
| `s2_pre` | `(6,10,H,W)` | fino a sei frame Sentinel pre-evento |
| `s2_post` | `(6,10,H,W)` | fino a sei frame Sentinel post-evento |
| `s2_valid_pre/post` | `(6,)` bool | distingue frame reali e padding |
| `aux` | `(4,H,W)` | DEM, slope e aspect normalizzati |
| `mask` | `(1,H,W)` | target binario |

I controlli rifiutano forme, dtype, canali, NaN e Inf incompatibili.

## PlanetScope e AUX

Planet post, mask, DEM, slope e aspect vengono letti o riproiettati sulla
griglia di Planet pre. Ogni patch contiene soltanto gli AUX allineati
`dem.tif`, `slope.tif` e `aspect.tif`. I precedenti raster `*_wide.tif` non
sono usati dal modello e non vengono più generati.

Il tensore AUX contiene:

1. `asinh(DEM_metri / 1000)`;
2. `clamp(slope_gradi, 0, 90) / 90`;
3. `sin(aspect_gradi)`;
4. `cos(aspect_gradi)`.

La rappresentazione circolare evita la discontinuità fra 0 e 360 gradi.

## Sentinel-2

La pipeline usa esplicitamente prodotti `MSIL2A`. L'ordine delle bande è B02,
B03, B04, B08, B05, B06, B07, B8A, B11 e B12; i valori vengono divisi per
10000.

Per ogni data vengono raccolti tutti i tile sovrapposti alla patch. I tile sono
ordinati deterministicamente, riproiettati e mosaicati prima del salvataggio.
La Scene Classification Layer esclude classi invalide configurabili, mentre
`s2_valid.tif` conserva la validità spaziale. Per default una data viene
scartata se la copertura valida è inferiore al 95%.

Vengono selezionati al massimo sei frame per lato entro 90 giorni dalla data
dell'evento. È obbligatorio almeno un frame reale pre e uno post; gli slot
rimanenti sono padding zero e sono esclusi dalla media temporale mediante le
maschere booleane. L'intervallo di Lombok include acquisizioni post-evento.

Le patch Sentinel generate con versioni precedenti della pipeline devono essere
rigenerate per ottenere L2A, mosaico multi-tile e `s2_valid.tif` coerenti.

## Augmentation

L'augmentation geometrica esiste soltanto in
`MultiModalLandslideDataset`. Flip e rotazione vengono estratti una volta per
campione e applicati nello stesso modo a Planet pre/post, tutti i frame
Sentinel, AUX e mask. I dataset dei singoli sensori rifiutano l'augmentation
autonoma per evitare disallineamenti accidentali.

## Modello

PlanetScope e Sentinel usano encoder Swin separati. Un adattatore 1×1 converte
le dieci bande Sentinel in tre canali; ogni frame è codificato dallo stesso
encoder e le feature sono mediate usando soltanto i frame validi.

Per ogni scala vengono calcolate le differenze post-pre Planet e Sentinel. Un
encoder convoluzionale produce la piramide AUX. Le tre feature sono concatenate
e fuse prima del decoder. Il decoder usa interpolazione bilineare spaziale per
le skip connection e per il ritorno alla risoluzione della patch.

## Training e risultati

Il training usa BCE con logit, AdamW, warm-up opzionale, scheduler coseno ed
early stopping sull'AUPRC. `--pos-weight` e `--balanced-sampling` sono opzioni
esplicite per lo sbilanciamento.

La soglia è scelta massimizzando F1 sui punti della curva precision-recall che
possiedono una soglia. Ogni esperimento produce:

```text
config.json
history.csv
results.json
checkpoint_last.pth
best_model.pth
val_pr_curve.csv
val_pr_curve.png
logs/
```

Il best checkpoint contiene pesi, epoca, soglia, metriche e configurazione. La
ripresa è stretta: checkpoint precedenti all'integrazione AUX sono
incompatibili e richiedono un nuovo training.

`inference.py` legge modello e soglia dal best checkpoint e salva
`probability.tif` e `mask.tif` con la georeferenziazione di Planet pre.

## Stato delle verifiche

La struttura sopra descrive il codice corrente. Parsing e controlli statici
sono stati eseguiti durante lo sviluppo. I test che richiedono PyTorch, timm,
GDAL e dati reali devono ancora essere eseguiti nell'ambiente finale; lo stato
puntuale è mantenuto in `TODO.md`.

Non vengono considerati riproducibili risultati presenti soltanto in un CSV.
Un risultato può essere dichiarato verificato solo insieme a configurazione,
storico, checkpoint e log della relativa esecuzione. Per questo motivo i valori
di vecchi CSV privi di tali artifact non sono riportati come risultato del
modello corrente.

I comandi operativi e l'installazione sono documentati in `README.md`.
