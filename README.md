# ReadMe

Repo for https://www.biorxiv.org/content/10.1101/2024.12.23.630166v1.abstract

## Scripts

### 1) `generate_data.py`
Generate synthetic datasets used by the VAE-C experiments.
```bash
python generate_data.py 
```

### 2) `train_vae.py`
Train models for a given **situation** (dataset setup).
```bash
python train_vae.py 1
```

### 3) `generate_results_table.py`
Aggregate metrics into a results table for a given situation.
```bash
python generate_results_table.py 1 
```

### 4) `create_all_archs_vary_beta_figs_with_stats.py`
Reproduce the main figures for a given situation, including stats overlays.
```bash
python create_all_archs_vary_beta_figs_with_stats.py 1
```

### Repeat process for situations 2,3, and 4

## Environment

```bash
pip install -r requirements.txt
```
