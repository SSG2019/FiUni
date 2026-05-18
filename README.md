# FiUni

This repository contains the partial implementation and experimental artifacts for our paper:

**Unifying Detection and Adaptation in Task-Free Continual Learning**

FiUni is a Fisher geometry-based framework for task-free continual parameter-efficient fine-tuning. It uses Fisher principal subspaces to perform batch-level latent task detection and adaptive LoRA subspace construction.

## Repository Structure

```text
.
├── bash/                  # Training and evaluation scripts
├── data/                  # Benchmark data used in our experiments
├── FiUni/                 # Core training code for FiUni
├── outputs/               # Experimental outputs reported in the paper
├── calc_train_scores.py   # Script for calculating final training/evaluation scores
├── pyproject.toml         # Project configuration
└── README.md
```

## Current Release

This repository currently includes the main training scripts, FiUni training files, benchmark data, and experimental outputs used in the paper.

We release a partial version of the code at this stage. The complete and cleaned implementation will be publicly released after the paper is accepted.

## Usage

Training scripts are provided under the `bash/` directory. Users can refer to these scripts to reproduce the main experiments on the supported benchmarks.

The `FiUni/` directory contains the core training files of FiUni. The `data/` directory contains the benchmark data used in our experiments.

The `outputs/` directory contains the experimental results reported in the paper. The script `calc_train_scores.py` can be used to aggregate and calculate final scores from saved outputs.
