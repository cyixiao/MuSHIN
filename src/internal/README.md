# Internal Validation

```bash
# Enter the internal validation directory
cd src/internal
```

This module performs **internal validation** of the MuSHIN model on curated genome-scale metabolic models (GEMs) such as **BiGG** and **AGORA**. It evaluates MuSHIN’s ability to recover artificially removed reactions from a given metabolic network.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Data Preparation

1. **Download** the GEM XML you want to test  
   * BiGG <http://bigg.ucsd.edu/models>  
   * AGORA <https://www.vmh.life/>  

2. **Place** the file (e.g. `iAF1260b.xml`) in `data/internal/<model_name>/`.

3. **Pre-process** the model:

```bash
python src/internal/datasets/process_data.py --dataset iAF1260b
```

---

## Running Internal Validation — example

```bash
python main.py \
  --train iAF1260b \
  --raw_path ../results/internal_results.jsonl \
  --seed 42 \
  --iteration 10 \
  --epoch 100 \
  --algorithm similarity \
  --create_negative True \
  --atom_ratio 0.5 \
  --negative_ratio 1 \
  --emb_dim 64 \
  --conv_dim 128 \
  --head 6 \
  --L 2 \
  --p 0.1 \
  --g_lambda 1 \
  --lr 1e-2 \
  --weight_decay 1e-3 \
  --batch_size 256 \
  --s2m_batch_size 32 \
  --enable_hygnn \
  --enable_reaction_fp
```

---

## Argument Definitions

| Argument | Description |
|----------|-------------|
| `--train` | Name of the GEM dataset folder under `data/` (e.g. `iAF1260b`). |
| `--raw_path` | Path to append raw JSONL results (e.g. `results/internal_results.jsonl`). |
| `--seed` | Random seed for reproducibility. |
| `--iteration` | Number of independent runs to average. |
| `--epoch` | Maximum number of training epochs. |
| `--algorithm` | Feature pipeline: `similarity` (fingerprint + GIP) or `smiles2vec` (ChemBERTa). |
| `--create_negative` | Whether to generate synthetic negative reactions (`True`/`False`). |
| `--atom_ratio` | Fraction of atoms replaced when creating negatives (0–1). |
| `--negative_ratio` | Number of negatives per positive sample. |
| `--emb_dim` | Node-embedding dimension. |
| `--conv_dim` | Output dimension of each HypergraphConv layer. |
| `--head` | Number of attention heads. |
| `--L` | Number of HypergraphConv layers. |
| `--p` | Dropout probability. |
| `--g_lambda` | Gaussian kernel bandwidth for similarity features. |
| `--lr` | Learning rate. |
| `--weight_decay` | Weight-decay (L2 regularization) coefficient. |
| `--batch_size` | Training batch size (reactions). |
| `--s2m_batch_size` | Batch size for SMILES-to-vector conversion. |
| `--enable_hygnn` / `--disable_hygnn` | Toggle the hypergraph GNN component. |
| `--enable_reaction_fp` / `--disable_reaction_fp` | Toggle reaction-level fingerprint embeddings. |

# Result Post-Processing

This utility aggregates **raw model outputs** (saved as `.jsonl` files) and computes common evaluation metrics (F1, precision, recall, accuracy, AUROC, AUPRC).  
It scans a directory of raw results, applies a user-selected thresholding strategy, and writes a new `.jsonl` file containing the processed scores.

---

## Quick Usage

```bash
python utils/metrics.py \
  --raw_results_path results/raw_runs \
  --output_file results/metrics/processed_results.jsonl \
  --threshold_mode value \
  --threshold_value 0.5
```

---

## Arguments

| Argument | Description |
|----------|-------------|
| `--raw_path` | Directory containing raw `.jsonl` files produced by `main.py`. |
| `--output_file` | Destination path for the processed results `.jsonl` (parent folders are created automatically). |
| `--threshold_mode` | Thresholding strategy used to binarise predictions. Options: `value`, `mean`, `median`. |
| `--threshold_value` | Numeric threshold applied when `--threshold_mode value` is selected. Ignored for `mean` and `median`. |

---

## Output Format

Each line of the output file is a JSON object:

```json
{
  "model": "iAF1260b",
  "algorithm": "similarity",
  "results": [
    {
      "f1": 0.87,
      "precision": 0.81,
      "recall": 0.94,
      "accuracy": 0.88,
      "auroc": 0.92,
      "aupr": 0.89
    },
    …
  ]
}
```

---
