## Environment Setup

### 1. Install Python 3.11 and create a virtual environment:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

```bash
sudo apt update --option Acquire::http::Proxy=http://proxy.uninsubria.it:3128/
sudo apt install -y libgdal-dev gdal-bin --option Acquire::http::Proxy=http://proxy.uninsubria.it:3128/

pip install -U setuptools wheel
export GDAL_CONFIG=/usr/bin/gdal-config
pip install --no-binary gdal "GDAL==$(gdal-config --version)"
```

CUDA_VISIBLE_DEVICES=2 python train.py --description "prova risorse"
CUDA_VISIBLE_DEVICES=3 python train.py --description "prova Emilia" --train-events EmiliaRomagna2023

CUDA_VISIBLE_DEVICES=2 python train.py --description "ripresa" --resume exp/swinunet_128_9