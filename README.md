# Landslides

Pipeline multimodale per segmentare frane combinando PlanetScope pre/post,
serie temporali Sentinel-2 L2A e informazioni topografiche.

## Ambiente

È consigliato Python 3.11. Su Ubuntu/Debian, GDAL deve essere installato prima
delle dipendenze Python e il binding deve avere la stessa versione della
libreria di sistema:

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv libgdal-dev gdal-bin
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-binary gdal "GDAL==$(gdal-config --version)"
python -m pip install -r requirements.txt
```

Per GPU, la build di PyTorch deve essere scelta in base alla versione CUDA
della macchina. Le credenziali Planet e Copernicus non sono incluse nel
repository.

## Struttura dei dati

La radice passata a `--dataset-root` contiene una cartella per evento e una per
patch:

```text
patches/
  Lombok2018/
    1/
      pre.tif
      post.tif
      mask.tif
      dem.tif
      slope.tif
      aspect.tif
      s2/
        YYYYMMDD/
          s2_10m.tif
          s2_20m.tif
          s2_valid.tif
```

Tutti i raster della patch sono sulla griglia di `pre.tif`. I vecchi file
`*_wide.tif` non sono usati né generati: la fusione AUX richiede corrispondenza
pixel-a-pixel con PlanetScope.

Sentinel usa esclusivamente prodotti L2A. Le dieci bande, in ordine, sono B02,
B03, B04, B08, B05, B06, B07, B8A, B11 e B12. I DN sono divisi per 10000.
La SCL esclude NoData, saturi/difettosi, ombre, nuvole e cirri; `s2_valid.tif`
conserva la validità spaziale dopo mosaico e riproiezione.

Per ogni lato temporale vengono usati al massimo sei frame entro 90 giorni
dall'evento. Un campione privo di almeno un frame reale pre oppure post viene
escluso. Le posizioni rimanenti sono padding zero accompagnato da una maschera
temporale booleana, quindi non partecipano alla media delle feature.

I quattro canali AUX sono:

```text
asinh(DEM_metri / 1000)
clamp(slope_gradi, 0, 90) / 90
sin(aspect_gradi)
cos(aspect_gradi)
```

## Preparazione

I percorsi principali sono configurabili tramite CLI o variabili d'ambiente.
Eseguire nell'ordine:

```bash
python build_ps_dataset.py --base-dir /dati/PlanetScope
S2_IMAGES_PATH=/dati/Sentinel/images python s2_builder/S2_download.py
S2_IMAGES_PATH=/dati/Sentinel/images python s2_builder/combine_bands.py
PS_PATCHES_PATH=/dati/PlanetScope/patches \
S2_IMAGES_PATH=/dati/Sentinel/images \
python s2_builder/make_patches.py
```

Dopo modifiche al preprocessing Sentinel, rigenerare le directory `s2/` per
ottenere mosaici L2A e `s2_valid.tif` aggiornati.

## Controllo geospaziale

```bash
python diagnose_geospatial.py \
  --patches-root /dati/PlanetScope/patches \
  --max-samples 20 --seed 42 --plots \
  --output-dir diagnostics
```

Il comando produce report CSV/JSON, confronti affiancati e overlay. Deve
terminare senza mismatch; i PNG devono anche essere controllati visivamente.

## Training

```bash
python train.py \
  --description "smoke multimodale" \
  --dataset-root /dati/PlanetScope/patches \
  --model-size tiny --patch-size 128 \
  --warmup-epochs 0 --epochs 1 --batch-size 2 --num-workers 12
```

Per un training completo aumentare `--epochs` e, se necessario, usare
`--pos-weight` oppure `--balanced-sampling`. Per riprendere un esperimento:

```bash
python train.py --description "ripresa" \
  --dataset-root /dati/PlanetScope/patches \
  --resume exp/swinunet_128
```

La ripresa richiede un checkpoint prodotto dall'architettura corrente con AUX;
i checkpoint precedenti non vengono caricati parzialmente.

Ogni esperimento salva insieme:

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

`exp/experiments.csv` è un indice riassuntivo; i risultati sono riproducibili
solo se sono disponibili anche configurazione, storico e checkpoint associati.

## Inferenza

```bash
python inference.py \
  --checkpoint exp/swinunet_128/best_model.pth \
  --dataset-root /dati/PlanetScope/patches \
  --event Lombok2018 --patch-id 1 \
  --output-dir inference/Lombok2018_1
```

La soglia viene letta dal checkpoint; `--threshold` permette di sovrascriverla.
Gli output `probability.tif` e `mask.tif` conservano CRS, transform e dimensioni
di `pre.tif`.

## Test

```bash
python -m unittest discover -s tests -v
```

I test includono contratti, augmentation sincronizzata, riproiezione,
mosaico, decoder, AUX e un ciclo sintetico training-validation-inference. Lo
smoke test sintetico non sostituisce la validazione su patch reali descritta in
`TODO.md`.
