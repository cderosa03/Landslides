#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

mode=${1:-smoke}
case "$mode" in
    smoke|train) ;;
    *) printf 'Uso: bash run_stable_training.sh smoke|train\n' >&2; exit 2 ;;
esac

python_bin=.venv/bin/python
if [[ ! -x "$python_bin" ]]; then
    python_bin=python
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

args=(
    --description "Emilia-Romagna train, Michoacan validation; bounded GDAL cache and spawn"
    --train-events EmiliaRomagna2023
    --val-events Michoacan2022
    --batch-size 8 --val-batch-size 8
    --num-workers 2 --loader-timeout 300
    --gdal-cache-mb 256 --cpu-threads 4 --no-pin-memory
    --profile-batches 20
)
if [[ "$mode" == smoke ]]; then
    # Exercise both phases and checkpoint writing without waiting for an epoch
    # on the full dataset. A timeout returns a failure, never a successful test.
    exec timeout --signal=INT --kill-after=30s 45m \
        "$python_bin" -u train.py "${args[@]}" --smoke-batches 20
else
    exec "$python_bin" -u train.py "${args[@]}" \
        --epochs 100 --warmup-epochs 10 --match-train-to-val --positive-fraction 0.5
fi
