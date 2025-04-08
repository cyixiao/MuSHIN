# Metabolic Network Gap-filling Pipeline

This repository provides a streamlined pipeline for **genome-scale metabolic model reconstruction and gap-filling**. It integrates machine learning and network analysis methods to identify and add missing reactions, enhancing model accuracy and biological relevance.

## Overview

## External Validation

Validated using fermentation product analysis from the **Zimmermann 2021 GenBiol** dataset.

## System Requirements

## Installation

```bash
git clone https://github.com/your-username/metabolic-gapfill.git
cd metabolic-gapfill
pip install -r requirements.txt
```

## Dataset

Datasets available via OSF:

🔗 [Zimmermann 2021 Dataset](https://doi.org/10.17605/OSF.IO/98KMJ)

**Required Files:**
- `media.tsv`
- `organisms2.csv`
- `ferm_prod_exp.csv`
- Genome-scale models (SBML format)
- Reaction libraries (BiGG, ModelSEED)

## Usage

Run default pipeline:

```bash
python main.py
```

Customized execution:

```bash
python main.py \
  --directory Zimmermann2021GenBiol \
  --pipeline carveme \
  --method GPR_POOL \
  --nselect 100 200 \
  --strategy advanced \
  --n_jobs -1
```

### Main Parameters

- `--directory`: Dataset directory (default: `Zimmermann2021GenBiol`)
- `--pipeline`: `carveme` or `modelseed`
- `--method`: Reaction selection method (`GPR_POOL`)
- `--nselect`: Number of reactions to select
- `--strategy`: `advanced` or `balanced`
- `--n_jobs`: Number of CPUs (`-1` for all cores)



## License

MIT License © Your Name or Organization

