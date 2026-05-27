
---

# SecureMamba

**Robust Biosafety Screening via Functional Semantic Reconstruction**

A Dual-Path Mamba Block-based framework that detects malicious biosequences **even after heavy obfuscation, fragmentation, and rearrangement**.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Performance Highlights](#performance-highlights)
4. [Architecture](#architecture)
5. [Applications](#applications)
6. [Visualization Analysis](#visualization-analysis)
7. [Requirements](#requirements)
8. [Data Availability](#data-availability)
9. [Model Weights](#model-weights)
10. [Installation](#installation)
11. [Configuration](#configuration)
12. [Running](#running)
13. [Prediction](#prediction)
14. [Output Files](#output-files)
15. [Project Structure](#project-structure)
16. [FAQ](#faq)
17. [License](#license)

---

## Overview

Traditional screening fails against AI-modified pathogens.  
SecureMamba upgrades detection from **syntactic matching** to **functional semantic understanding** for synthetic biology & DNA storage safety.

![image](image/model.png)

---

## Key Features

- 🧬 **Anti-obfuscation**: Resists mutations, fragmentation, reverse-complementation
- 🧠 **Dual-Path Mamba**: Semantic perception + adversarial perception
- 🌍 **Universal**: Cross-species toxin detection with shared toxicity grammar
- 📦 **DNA-Storage Ready**: Low false-positive, high-throughput, robust in high-entropy environments
- 🔬 **Interpretable**: Anchors directly to conserved functional domains

---

## Performance Highlights

- **97.7%** accuracy on virulence sequences
- **98.4%** on obfuscated variants
- **82.1%** cross-species toxin detection
- **0.14%** false-positive rate in DNA storage

---

## Architecture

```
NT-V3 → Dual-Path Mamba Block → Contrastive Learning → Classifier
```

---

## Applications

- DNA synthesis safety screening
- High-throughput synthetic biology monitoring
- Secure DNA data storage
- Cross-species toxin & pathogen detection

---

## Visualization Analysis

We can analyze the important regions identified by the model through gradient attribution analysis methods to gain deeper biological significance and model rationality.

Simply run the `attribution_analysis.py` script to obtain visualization results. The specific steps are shown in the video below.

https://github.com/user-attachments/assets/f2f97147-c2a0-448b-abbb-2d4e51a8c7a0

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.11.6 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |

### Core Dependencies

| Package | Version |
|---------|---------|
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| torchaudio | 2.5.1+cu121 |
| mamba-ssm | 2.2.6.post3 |
| causal-conv1d | 1.5.3.post1 |
| transformers | 4.57.1 |
| biopython | 1.85 |
| numpy | 2.2.6 |
| pandas | 2.3.2 |
| scikit-learn | 1.7.1 |
| matplotlib | 3.9.1 |
| seaborn | 0.13.2 |
| tqdm | 4.67.1 |

---

## Data Availability

The training and testing datasets used in SecureMamba are publicly available on Zenodo:

- **Dataset DOI (version v1)**: [10.5281/zenodo.20374288](https://doi.org/10.5281/zenodo.20374288)
- **Dataset DOI (all versions)**: [10.5281/zenodo.20374287](https://doi.org/10.5281/zenodo.20374287)

### Dataset Files

| File | Size | Description |
|------|------|-------------|
| `train_data_pos.fasta` | 52.8 MB | Positive training samples (toxic/virulent sequences) |
| `train_data_neg.fasta` | 192.1 MB | Negative training samples (non-toxic/benign sequences) |
| `btx_test.fasta` | 2.2 MB | BTXPred test set |
| `vfdb_test.fasta` | 16.9 MB | Virulence Factor Database (VFDB) test set |
| `toxin_test.fasta` | 9.2 MB | ToxinPred test set |

### Download Datasets

```bash
# Download using wget
wget https://zenodo.org/record/20374288/files/train_data_pos.fasta
wget https://zenodo.org/record/20374288/files/train_data_neg.fasta
wget https://zenodo.org/record/20374288/files/btx_test.fasta
wget https://zenodo.org/record/20374288/files/vfdb_test.fasta

# Create data directory and move files
mkdir -p train_data
mv *.fasta train_data/
```

---

## Model Weights

The pre-trained SecureMamba model weights are available on Zenodo:

- **Model DOI (version v1)**: [10.5281/zenodo.20365513](https://doi.org/10.5281/zenodo.20365513)

### Download Model

```bash
# Download model weights from Zenodo
wget https://zenodo.org/record/20365513/files/best_nucleotide_v3_mamba_model.pth
```

Place the downloaded model file in the project root directory or specify the path in prediction commands.

---

## Installation

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Mamba-SSM
pip install mamba-ssm

# Install other dependencies
pip install transformers biopython numpy pandas scikit-learn matplotlib seaborn tqdm
```

> **Note**: The environment shows torch 2.5.1+cu121. For CUDA 12.1 compatibility, use:
> ```bash
> pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Configuration

Edit `config.py` to adjust settings:

```python
# Data
positive_fasta = "train_data_pos.fasta"
negative_fasta = "train_data_neg.fasta"
max_seq_length = 2048
downsample_ratio = 3.0

# Model
transformer_model_repo = "InstaDeepAI/NTv3_8M_pre"
d_model = 256
n_layer = 2
block_type = "dual_path"

# Training
batch_size = 16
learning_rate = 2e-6
num_epochs = 20
use_amp = True
run_mode = "all"  # preprocess, generate, train, all, train-only
```

---

## Running

### Prepare Data

Place downloaded FASTA files in `train_data/`:
```
train_data/
├── train_data_pos.fasta          # Positive samples (52.8 MB)
├── train_data_neg.fasta          # Negative samples (192.1 MB)
├── btx_test.fasta                # Botulinum toxin test set (2.2 MB)
└── vfdb_test.fasta               # Virulence factor test set (16.9 MB)
```

### Run Training

```bash
python main.py
```

### Run Modes

Set `run_mode` in `config.py`:

| Mode | Description |
|------|-------------|
| `preprocess` | FASTA → segments |
| `generate` | Generate semantic variants |
| `train` | Train model (requires variants) |
| `all` | Full pipeline |
| `train-only` | Train only |

---

## Prediction

```bash
python predict.py --fasta input.fasta --model best_model.pth --threshold 0.5
```

### Parameters

| Argument | Required | Description |
|----------|----------|-------------|
| `--fasta` | ✅ | Input FASTA file |
| `--model` | ✅ | Trained model (.pth) |
| `--threshold` | ❌ | Classification threshold (default: 0.5) |
| `--batch-size` | ❌ | Batch size (default: 8) |
| `--output-dir` | ❌ | Output directory |

### Example with Downloaded Model

```bash
python predict.py --fasta train_data/btx_test.fasta --model best_nucleotide_v3_mamba_model.pth --threshold 0.6 --output-dir ./results
```

---

## Output Files

### Training (`output_dir`)

| File | Description |
|------|-------------|
| `variants_data.pkl` | Generated variants |
| `best_nucleotide_v3_mamba_model.pth` | Best checkpoint |
| `nucleotide_v3_training_log.json` | Training log |

### Prediction (`output_dir`)

| File | Description |
|------|-------------|
| `predictions_threshold_{value}.csv` | Results table |
| `prediction_visualization_threshold_{value}.png` | Charts |
| `detailed_threshold_analysis.png` | Threshold analysis |

---

## Project Structure

```
├── config.py                 # Configuration
├── main.py                   # Entry point
├── model.py                  # NTv3 + Mamba model
├── data_preprocessing.py     # Data preprocessing
├── synonymous_variants.py    # Variant generation
├── realtime_trainer.py       # Trainer
├── predict.py                # Prediction script
├── attribution_analysis.py   # Attribution analysis script
├── best_nucleotide_v3_mamba_model.pth  # Model weights (download from Zenodo)
└── train_data/               # Training data (download from Zenodo)
    ├── train_data_pos.fasta
    ├── train_data_neg.fasta
    ├── btx_test.fasta
    └── vfdb_test.fasta
```

---

## FAQ

**CUDA out of memory?** Reduce `batch_size` and `max_seq_length`.

**Slow variant generation?** Increase `min_pos_base_diff` or reduce `max_semantic_attempts`.

**HF model download slow?** Add mirror:
```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

**Where to download datasets?** Zenodo DOI: [10.5281/zenodo.20374288](https://doi.org/10.5281/zenodo.20374288)

**Where to download model weights?** Zenodo DOI: [10.5281/zenodo.20365513](https://doi.org/10.5281/zenodo.20365513)

---

## License

MIT License

Copyright (c) 2026 kosha316

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Citation

If you use SecureMamba in your research, please cite:

```bibtex
@software{SecureMamba2026,
  author = {kosha316},
  title = {SecureMamba: Robust Biosafety Screening via Functional Semantic Reconstruction},
  year = {2026},
  doi = {10.5281/zenodo.20365513},
  url = {https://doi.org/10.5281/zenodo.20365513}
}
```

**Datasets citation**:
```bibtex
@data{SecureMamba_Datasets_2026,
  author = {kosha316},
  title = {SecureMamba Training and Testing Datasets},
  year = {2026},
  doi = {10.5281/zenodo.20374288},
  url = {https://doi.org/10.5281/zenodo.20374288}
}
```

---
