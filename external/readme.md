# Metabolic Network Gap-filling Pipeline

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive pipeline for metabolic network reconstruction and gap-filling using machine learning and hypergraph neural network approaches (MuSHIN).

## Overview

The pipeline includes:

1. **Reaction Pool Creation**: Merges genome-scale metabolic models (GEMs) with reference reaction pools.
2. **Reaction Score Prediction**: Predicts reaction scores using the MuSHIN neural network.
3. **Similarity Calculation**: Computes similarity metrics between reactions.
4. **Metabolic Gap-filling**: Selects reactions intelligently to enhance model accuracy.

## Quick Start

```bash
git clone https://github.com/YourRepo/metabolic-gapfill.git
cd metabolic-gapfill
pip install -r requirements.txt
python main.py
```

## Directory Structure

```
├── data/
│   ├── metadata/
│   ├── Models_EGC_removed/
│   ├── pools/
│   └── gems/
├── results/
│   ├── fba_result/
│   ├── similarity_scores/
│   └── universe/
└── src/
    ├── pipeline.py
    ├── gapfill.py
    ├── predict.py
    ├── similarity.py
    ├── utils.py
    └── main.py
```

## Installation

### Dependencies

```bash
pip install -r requirements.txt
```

### External Files

MuSHIN neural network requires PubChem SMILES data:
- Place it in: `MuSHIN-main/external/PubChem10M_SMILES_BPE_450k`

## Usage

### Run the Complete Pipeline

```bash
python main.py
```

### Run Components Separately

- **Create Reaction Pool**:
```bash
python -c "from utils import create_pool; create_pool()"
```

- **Predict Reaction Scores**:
```bash
python -c "from predict import predict; predict()"
```

- **Calculate Similarity Scores**:
```bash
python -c "from similarity import get_similarity_score; get_similarity_score(top_N=2000, n_jobs=8)"
```

- **Gap-filling**:
```bash
python -c "from gapfill import main_improved; main_improved()"
```

### Advanced Pipeline Configuration

```python
from pipeline import run_metabolic_gapfill_pipeline

results = run_metabolic_gapfill_pipeline(
    directory="Zimmermann2021GenBiol",
    pipeline="carveme",
    method="GPR_POOL",
    n_jobs=8,
    nselect_values=[100, 200],
    strategy="advanced"
)
```


## License

This project is licensed under the MIT License.

