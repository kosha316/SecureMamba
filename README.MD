
---

# SecureMamba

**Robust Biosafety Screening via Functional Semantic Reconstruction**

A Dual-Path Mamba Block-based framework that detects malicious biosequences **even after heavy obfuscation, fragmentation, and rearrangement**.

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

### Actual Environment Packages

| Package | Version | Build | Channel |
|---------|---------|-------|---------|
| python | 3.11.6 | hab00c5b_0_cpython | conda-forge |
| torch | 2.5.1+cu121 | pypi_0 | pypi |
| torchvision | 0.20.1+cu121 | pypi_0 | pypi |
| torchaudio | 2.5.1+cu121 | pypi_0 | pypi |
| mamba-ssm | 2.2.6.post3 | pypi_0 | pypi |
| causal-conv1d | 1.5.3.post1 | pypi_0 | pypi |
| transformers | 4.57.1 | pypi_0 | pypi |
| biopython | 1.85 | py311h49ec1c0_2 | conda-forge |
| numpy | 2.2.6 | py311h5d046bc_0 | conda-forge |
| pandas | 2.3.2 | py311hed34c8f_0 | conda-forge |
| scikit-learn | 1.7.1 | py311hc3e1efb_0 | conda-forge |
| matplotlib | 3.9.1 | py311h38be061_1 | conda-forge |
| seaborn | 0.13.2 | hd8ed1ab_3 | conda-forge |
| tqdm | 4.67.1 | pyhd8ed1ab_1 | conda-forge |
| scipy | 1.16.1 | py311h1e13796_1 | conda-forge |
| requests | 2.32.5 | pyhd8ed1ab_0 | conda-forge |
| pyyaml | 6.0.2 | py311h2dc5d0c_2 | conda-forge |
| huggingface-hub | 0.36.0 | pypi_0 | pypi |
| datasets | 4.4.1 | pypi_0 | pypi |
| tokenizers | 0.22.1 | pypi_0 | pypi |
| safetensors | 0.6.2 | pypi_0 | pypi |
| accelerate | 1.11.0 | pypi_0 | pypi |
| einops | 0.8.1 | pypi_0 | pypi |

### CUDA Packages (conda-forge/nvidia)

| Package | Version | Build | Channel |
|---------|---------|-------|---------|
| cuda-cudart | 11.8.89 | 0 | nvidia |
| cuda-cupti | 11.8.87 | 0 | nvidia |
| cuda-nvrtc | 11.8.89 | 0 | nvidia |
| cuda-nvtx | 11.8.86 | 0 | nvidia |
| libcublas | 11.11.3.6 | 0 | nvidia |
| libcufft | 10.9.0.58 | 0 | nvidia |
| libcurand | 10.3.10.19 | 0 | nvidia |
| libcusolver | 11.4.1.48 | 0 | nvidia |
| libcusparse | 11.7.5.86 | 0 | nvidia |

### CUDA Packages (PyPI - cu12)

| Package | Version |
|---------|---------|
| nvidia-cublas-cu12 | 12.1.3.1 |
| nvidia-cuda-runtime-cu12 | 12.1.105 |
| nvidia-cudnn-cu12 | 9.1.0.70 |
| nvidia-cuda-nvrtc-cu12 | 12.1.105 |
| nvidia-cusparse-cu12 | 12.1.0.106 |
| nvidia-nvtx-cu12 | 12.1.105 |

---

## Installation

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Mamba-SSM
pip install mamba-ssm

# SecureMamba Model Acquisition
# https://doi.org/10.5281/zenodo.20365513

# Install other dependencies
pip install transformers biopython numpy pandas scikit-learn matplotlib seaborn tqdm
```

> **Note**: The actual environment shows torch 2.5.1+cu121 installed. The `cu118` index above installs CUDA 11.8-compatible wheels. For CUDA 12.1 compatibility as shown in the environment, use:
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

Place FASTA files in `train_data/`:
```
train_data/
├── train_data_pos.fasta          # Positive samples
└── train_data_neg.fasta          # Negative samples
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

### Example

```bash
python predict.py --fasta test.fasta --model results/best_model.pth --threshold 0.6
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
└── train_data/               # Training data
```

---

## FAQ

**CUDA out of memory?** Reduce `batch_size` and `max_seq_length`.

**Slow variant generation?** Increase `min_pos_base_diff` or reduce `max_semantic_attempts`.

**HF model download slow?** Add mirror:
```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

**CUDA version mismatch?** The environment has both CUDA 11.8 (conda-forge/nvidia) and CUDA 12.1 (PyPI packages). PyTorch uses CUDA 12.1 runtime (`torch 2.5.1+cu121`). Ensure `nvidia-smi` shows driver supporting CUDA 12.1.

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

**Model DOI**: https://doi.org/10.5281/zenodo.20365513

---

