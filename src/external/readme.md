# External Validation

```bash
# Enter the external validation directory
cd src/external
```

This module performs external validation of the MuSHIN model on draft genome-scale metabolic models (GEMs), such as those derived from anaerobic bacteria (e.g., Zimmermann et al.). It evaluates MuSHIN’s ability to enhance phenotype prediction accuracy by gap-filling using top-ranked reactions from a universal pool.

## Installation
We recommend installing dependencies from the provided requirements file:
```bash
pip install -r src/external/requirements.txt
```

## Overview

The pipeline includes:

1. **Reaction Pool Creation**: Merges genome-scale metabolic models (GEMs) with reference reaction pools.
2. **Reaction Score Prediction**: Predicts reaction scores using the MuSHIN neural network.
3. **Similarity Calculation**: Computes similarity metrics between reactions.
4. **Metabolic Gap-filling**: Selects reactions intelligently to enhance model accuracy.

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
