# Procedura operativa sul server (stato attuale)

Questa procedura usa solo gli eventi per i quali sono presenti immagini
Sentinel-2 sia pre sia post evento. Non avvia automaticamente il training.

## Situazione verificata

- `Michoacan2022`: pre e post presenti.
- `EmiliaRomagna2023`: pre e post presenti.
- `Lombok2018`: solo pre disponibile.
- `Philippines2019`: solo pre disponibile.

Lombok e Filippine non vengono quindi usati dal modello temporale attuale, che
richiede almeno una data reale in entrambe le fasi.

## 1. Attendere il processo attuale

Non avviare un secondo `make_patches.py` mentre quello attuale è in esecuzione:

```bash
pgrep -af 's2_builder/make_patches.py'
```

Quando il comando non restituisce più righe, l'elaborazione è terminata.

## 2. Impostare i percorsi

In un nuovo terminale Jupyter:

```bash
cd /home/jovyan/nfs/tesista3/landslide-detection
export DATASET_ROOT=/home/jovyan/nfs/tesista3/landslide-detection/Landslides/PlanetScope/patches
export S2_IMAGES_PATH=/home/jovyan/nfs/tesista3/landslide-detection/Landslides/Sentinel/images
```

## 3. Verificare le patch generate

Controllare che Michoacán ed Emilia-Romagna contengano file Sentinel completi:

```bash
for ev in Michoacan2022 EmiliaRomagna2023; do
  echo -n "$ev: "
  find "$DATASET_ROOT/$ev" -name 's2_valid.tif' | wc -l
done
```

Una data è completa solo se contiene tutti questi file:

```text
s2_10m.tif
s2_20m.tif
s2_valid.tif
```

## 4. Diagnostica geospaziale (facoltativa ma consigliata)

Eseguire il controllo su un evento con pre e post:

```bash
python diagnose_geospatial.py \
  --patches-root "$DATASET_ROOT" \
  --events Michoacan2022 \
  --max-samples 1 \
  --seed 42 \
  --plots \
  --output-dir diagnostics_michoacan
```

## 5. Avviare il training manualmente

Per mantenere separati training e validation:

```bash
python train.py \
  --description "Training Michoacan, validation Emilia-Romagna" \
  --train-events Michoacan2022 \
  --val-events EmiliaRomagna2023 \
  --num-workers 12
```

Non usare la configurazione predefinita, perché imposta `Lombok2018` come
validation e quell'evento non ha Sentinel post.

## 6. Lombok e Filippine

Non eseguire altri comandi per questi eventi nella configurazione attuale:

- non rilanciare `make_patches.py` per loro;
- non usare le loro patch nel training temporale;
- non considerarli un test valido del ramo Sentinel pre/post.

Potranno essere recuperati solo se si ottengono nuovi prodotti Sentinel-2
post-evento. In quel caso sarà necessario rieseguire download, combinazione e
un'estrazione mirata delle sole patch interessate.
