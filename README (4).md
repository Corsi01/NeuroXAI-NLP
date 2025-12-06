# Mapping Transformer Layer Hierarchy to Brain Functional Organization Through Integrated Gradients

Master's Project - Natural Language Processing  
Università degli Studi di Milano-Bicocca  
Author: Andrea Corsico

## Overview

This project investigates how GPT-2 layer representations align with the functional organization of the human language network using fMRI encoding models and Integrated Gradients attribution analysis.

## Key Findings

- Contextualized representations (GPT-2) substantially outperform static embeddings (FastText)
- Intermediate layers (7-8) achieve peak encoding performance
- Mean and max pooling capture complementary information with distinct regional preferences
- Feature importance patterns follow anatomical (frontal vs. temporal) rather than functional (syntactic vs. semantic) organization

## Repository Structure

```
notebooks/
  1_data_preparation.ipynb    # EDA, ROI extraction, feature generation, embedding visualization
  2_encoding_baseline.ipynb   # FastText baseline models (5 ROIs)
  3_encoding_models.ipynb     # GPT-2 layer-wise encoding models (60 combinations)
  4_analysis.ipynb            # Performance evaluation & interpretability analysis

src/
  IntegratedGradients.py      # IG computation
  optimization.py             # Optuna hyperparameter optimization
  train_utils.py              # Training utilities
  utils.py                    # General utilities

data/
  allParcels-language-SN220.nii   # Fedorenko language network masks

report/
  main.tex                    # LaTeX report
  references.bib              # Bibliography
```

## Data

This project uses the Algonauts 2025 Challenge dataset derived from CNeuroMod (Friends fMRI).

## Methods

1. Feature Extraction: Layer-wise embeddings from GPT-2 small (768-dim x 12 layers)
2. Pooling: Concatenated mean + max pooling (1536-dim)
3. Encoding Models: MLP with Optuna-optimized hyperparameters
4. Attribution: Integrated Gradients for feature importance analysis

## Requirements

```
torch
transformers
nilearn
optuna
scikit-learn
scipy
pandas
numpy
matplotlib
seaborn
spacy
```

## References

- Radford et al. (2019). Language models are unsupervised multitask learners.
- Schrimpf et al. (2021). The neural architecture of language: Integrative modeling converges on predictive processing.
- Sundararajan et al. (2017). Axiomatic attribution for deep networks.
- Fedorenko et al. (2010). New method for fMRI investigations of language.

## License

MIT License