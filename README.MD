# SecureMamba
**Robust Biosafety Screening via Functional Semantic Reconstruction**

A dual-path Mamba-based framework that detects malicious biosequences **even after heavy obfuscation, fragmentation, and rearrangement**.

---

## Overview
Traditional screening fails against AI-modified pathogens.  
SecureMamba upgrades detection from **syntactic matching** to **functional semantic understanding** for synthetic biology & DNA storage safety.

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

## 📋 Requirements

| Component | Version |
|-----------|---------|
| Python | 3.9+ |
| CUDA | 11.8+ (recommended) |
| GPU RAM | 16GB+ |

### Core Dependencies

```bash
torch>=2.0.0
mamba-ssm>=2.0.0
transformers>=4.35.0
biopython>=1.81
numpy>=1.24.0
scikit-learn>=1.3.0
```

---

## 🚀 Installation


# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Mamba
pip install mamba-ssm

# Install other dependencies
pip install transformers biopython numpy pandas scikit-learn matplotlib seaborn tqdm
```

---

## ⚙️ Configuration

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

## 🏃 Running

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
| `train-only` | Train from existing variants |
| `all` | Full pipeline |


---

## 🔮 Prediction

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

## 📁 Output Files

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

## 📊 Project Structure

```
├── config.py                 # Configuration
├── main.py                   # Entry point
├── model.py                  # NTv3 + Mamba model
├── data_preprocessing.py     # Data preprocessing
├── synonymous_variants.py    # Variant generation
├── realtime_trainer.py       # Trainer
├── predict.py                # Prediction script
└── train_data/               # Training data
```

---

## ❓ FAQ

**CUDA out of memory?** Reduce `batch_size` and `max_seq_length`.

**Slow variant generation?** Increase `min_pos_base_diff` or reduce `max_semantic_attempts`.

**HF model download slow?** Add mirror:
```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```


## License
MIT
