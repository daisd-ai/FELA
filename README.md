# FELA

This repository accompanies the paper **“Introducing FELA - Flexible Entity Linking Approach”**, presented at HybridAIMS 2025 in conjunction with the 37th International Conference on Advanced Information Systems Engineering.
The paper is available [here](https://ceur-ws.org/Vol-3996/paper-2.pdf).

## Requirements

- Linux with an NVIDIA GPU (tested on NVIDIA H100)
- CUDA-compatible drivers
- Conda (miniconda or Anaconda)
- Python 3.11.11

> **Important:** A GPU is required to run this project.

## Installation

1. **Create and activate the Conda environment with FAISS (GPU):**
   ```bash
   conda create -n fela python=3.11.11
   conda activate fela
   conda install -c pytorch -c nvidia faiss-gpu=1.9.0
   ```
   If you encounter issues, consult the official [FAISS installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Download the required assets into the `data/` directory.

- **FAISS index (≈5 GB):**
  ```bash
  wget "https://box.pionier.net.pl/f/0f9c117b7bd24a0ab409/?dl=1" -O data/faiss_ubinary_wikidata_v2.index
  ```

- **Wikidata mapping (≈6 GB):**
  ```bash
  wget "https://box.pionier.net.pl/f/62aefc56e82843858760/?dl=1" -O data/wikidata_id_to_profile.json
  ```

Update paths in the configuration if your directory structure differs:
```python
DATA_DIR = "data/datasets"
WIKIDATA_FILE = "data/wikidata_id_to_profile.json"
FAISS_INDEX = "data/faiss_ubinary_wikidata_v2.index"
```

## Configuration

Select the evaluation dataset by setting `DATASET_NAME`:
```python
DATASET_NAME = "Tweeki_gold"
# DATASET_NAME = "RSS_500"
# DATASET_NAME = "reuters-128_wikidata"
```

Adjust GPU memory usage as needed:
```python
GPU_MEM_UTIL = 0.2
```

## First Run

On the first execution the script downloads models. Allow additional time for this initial setup. Subsequent runs should be faster.

```bash
python main.py
```
