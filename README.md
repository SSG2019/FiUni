# FiUni

Official implementation of **Unifying Detection and Adaptation in Task-Free Continual Learning**.

FiUni is a Fisher geometry-based framework for task-free continual parameter-efficient fine-tuning. It uses Fisher/K-FAC principal subspaces for batch-level latent task detection and adaptive LoRA subspace construction.

---

## 📁 Repository Structure

```text
.
├── bash/                  # Scripts for training, similarity analysis, and boundary detection
├── data/                  # Benchmark data used in the paper
├── FiUni/                 # Main implementation of FiUni
├── fiunilib/              # Supporting library and utility modules
├── outputs/               # All reported results and training logs in the paper
├── calc_train_scores.py   # Script for calculating final scores
├── pyproject.toml         # Project configuration
└── README.md
```

---

## ⚙️ Environment Setup

Depending on your CUDA version and local environment, you may need to install a compatible version of PyTorch manually before installing the remaining dependencies.

```bash
cd FiUni

conda create -n fiuni python=3.9
conda activate fiuni

pip install -e .
```

---

## 📊 Data

The `data/` directory contains the benchmark data used in the paper:

```text
data/
├── CL/                # Continual learning benchmark data for SC and LS
└── TRACE-Benchmark/   # TRACE benchmark data
```

---

## 🤖 Model Path

The training scripts use local model paths by default. Please modify `MODEL_NAME` in the corresponding script before running.

For example, in `bash/train/train_ls_llama.sh`:

```bash
MODEL_NAME="${HOME}/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B"
```

Change this path to your own local checkpoint path if needed.

---

## 🚀 Usage

All scripts are provided under `bash/`.

### 🔍 Similarity Analysis

The similarity script computes Fisher/K-FAC principal subspace similarities between tasks or data windows. It can be used to analyze whether Fisher subspace similarity reflects task-level geometric relations.

```bash
bash bash/run_similarity.sh
```

For running similarity analysis over multiple settings, use:

```bash
bash bash/run_similarity_all.sh
```

### 🧭 Boundary Detection

The boundary detection script performs batch-level latent task detection in a task-free data stream. It estimates the Fisher principal subspace of incoming batches and compares it with historical subspaces to identify potential latent task changes.

```bash
bash bash/detect_boundary.sh
```

### 🏋️ Training

The training scripts reproduce FiUni continual fine-tuning on the benchmarks used in the paper. The following command is an example for running FiUni on the LS benchmark with LLaMA:

```bash
bash bash/train/train_ls_llama.sh
```

Before running training, please check the model path, data path, task order, and random seed settings in the corresponding script.

---

## 📂 Results and Logs

The `outputs/` directory contains all experimental results and training logs reported in the paper. These files can be used to inspect the original training records and verify the reported scores.

---

## 🧮 Score Calculation

After training, calculate the final scores with:

```bash
python calc_train_scores.py
```

Please modify the paths in `calc_train_scores.py` if your outputs are saved in a different directory.