
# SecureMamba - Complete Installation Guide

This guide provides a **batch-by-batch** installation method that installs **every package** from the SecureMamba environment, avoiding the common "Solving environment" hang.

---

## Table of Contents

1. [Batch Installation (All Packages)](#batch-installation-all-packages)
   - [Batch 1: Environment Setup](#batch-1-environment-setup)
   - [Batch 2: PyTorch & CUDA Runtime](#batch-2-pytorch--cuda-runtime)
   - [Batch 3: Core Scientific Packages](#batch-3-core-scientific-packages)
   - [Batch 4: Visualization & Plotting](#batch-4-visualization--plotting)
   - [Batch 5: Machine Learning & Statistics](#batch-5-machine-learning--statistics)
   - [Batch 6: Bioinformatics](#batch-6-bioinformatics)
   - [Batch 7: Numerical Computing](#batch-7-numerical-computing)
   - [Batch 8: Compression Packages](#batch-8-compression-packages)
   - [Batch 9: Network Packages](#batch-9-network-packages)
   - [Batch 10: Image & GUI](#batch-10-image--gui)
   - [Batch 11: System Utilities](#batch-11-system-utilities)
   - [Batch 12: Hugging Face Ecosystem](#batch-12-hugging-face-ecosystem)
   - [Batch 13: Transformers](#batch-13-transformers)
   - [Batch 14: Dimensionality Reduction](#batch-14-dimensionality-reduction)
   - [Batch 15: Mamba-SSM (Critical - Pre-built Wheels)](#batch-15-mamba-ssm-critical---pre-built-wheels)
   - [Batch 16: Remaining Pip Packages](#batch-16-remaining-pip-packages)
2. [Complete Automation Script](#complete-automation-script)
3. [Verification](#verification)
4. [Troubleshooting](#troubleshooting)
5. [Download Datasets & Model](#download-datasets--model)

---

## Batch Installation (All Packages)

This method installs **every package** in small batches to avoid dependency conflicts.

### Batch 1: Environment Setup

```bash
# Create base environment
conda create -n securemamba python=3.11 -y
conda activate securemamba

# Upgrade pip
pip install pip==25.2 --upgrade
pip install --upgrade setuptools wheel
```

### Batch 2: PyTorch & CUDA Runtime

```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install CUDA runtime libraries
conda install -c nvidia -y cuda-cudart=11.8 cuda-cupti=11.8 cuda-nvrtc=11.8
```

### Batch 3: Core Scientific Packages

```bash
conda install -c conda-forge -y \
    numpy=2.2.6 \
    scipy=1.16.1 \
    pandas=2.3.2
```

### Batch 4: Visualization & Plotting

```bash
conda install -c conda-forge -y \
    matplotlib=3.9.1 \
    seaborn=0.13.2 \
    contourpy=1.3.3 \
    fonttools=4.59.2 \
    kiwisolver=1.4.9 \
    cycler=0.12.1 \
    pyparsing=3.2.3
```

### Batch 5: Machine Learning & Statistics

```bash
conda install -c conda-forge -y \
    scikit-learn=1.7.1 \
    statsmodels=0.14.5 \
    threadpoolctl=3.6.0
```

### Batch 6: Bioinformatics

```bash
conda install -c conda-forge -y \
    biopython=1.85
```

### Batch 7: Numerical Computing

```bash
conda install -c conda-forge -y \
    numba=0.61.2 \
    sympy=1.13.1 \
    networkx=3.5 \
    gmpy2=2.2.1
```

### Batch 8: Compression Packages

```bash
conda install -c conda-forge -y \
    zstd=1.5.6 \
    lz4-c=1.9.4 \
    brotli=1.1.0 \
    zstandard=0.23.0
```

### Batch 9: Network Packages

```bash
conda install -c conda-forge -y \
    requests=2.32.5 \
    urllib3=2.5.0 \
    certifi=2025.11.12 \
    idna=3.10 \
    charset-normalizer=3.4.3 \
    pysocks=1.7.1
```

### Batch 10: Image & GUI

```bash
conda install -c conda-forge -y \
    pillow=9.4.0 \
    jpeg=9e \
    pyqt=5.15.9
```

### Batch 11: System Utilities

```bash
conda install -c conda-forge -y \
    tqdm=4.67.1 \
    joblib=1.5.2 \
    pyyaml=6.0.2 \
    toml=0.10.2 \
    tomli=2.2.1 \
    colorama=0.4.6 \
    unicodedata2=16.0.0 \
    six=1.17.0 \
    python-dateutil=2.9.0.post0 \
    pytz=2025.2 \
    packaging=23.2 \
    platformdirs=4.5.0 \
    psutil=7.1.3 \
    filelock=3.19.1 \
    tornado=6.5.2 \
    markupsafe=3.0.2 \
    jinja2=3.1.6
```

### Batch 12: Hugging Face Ecosystem

```bash
pip install \
    huggingface-hub==0.36.0 \
    datasets==4.4.1 \
    hf-xet==1.1.9 \
    safetensors==0.6.2 \
    h5py==3.15.1
```

### Batch 13: Transformers

```bash
# transformers auto-installs tokenizers and regex
pip install transformers==4.57.1
```

### Batch 14: Dimensionality Reduction

```bash
# umap-learn auto-installs pynndescent
pip install umap-learn==0.5.9.post2
```

### Batch 15: Mamba-SSM (Critical - Pre-built Wheels)

**Use these pre-compiled wheels to avoid compilation issues:**

```bash
# Download causal-conv1d pre-built wheel
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Install causal-conv1d
pip install causal_conv1d-1.5.3.post1+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Download mamba-ssm pre-built wheel
wget https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Install mamba-ssm
pip install mamba_ssm-2.2.6.post3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

**Direct download links:**
- [causal_conv1d-1.5.3.post1+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl)
- [mamba_ssm-2.2.6.post3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl](https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl)

### Batch 16: Remaining Pip Packages

```bash
pip install \
    accelerate==1.11.0 \
    einops==0.8.1 \
    lion-pytorch==0.2.3 \
    aiohttp==3.13.2 \
    aiosignal==1.4.0 \
    aiohappyeyeballs==2.6.1 \
    frozenlist==1.8.0 \
    multidict==6.7.0 \
    yarl==1.22.0 \
    propcache==0.4.1 \
    anyio==4.11.0 \
    httpcore==1.0.9 \
    httpx==0.28.1 \
    sniffio==1.3.1 \
    numexpr==2.14.1 \
    tables==3.10.2 \
    blosc2==3.11.1 \
    msgpack==1.1.2 \
    msgpack-numpy==0.4.8 \
    ndindex==1.10.1 \
    pyarrow==22.0.0 \
    dill==0.4.0 \
    multiprocess==0.70.18 \
    fsspec==2025.9.0 \
    xxhash==3.6.0 \
    lmdb==1.7.5 \
    patsy==1.0.1 \
    py-cpuinfo==9.0.0 \
    ply==3.11
```

> **Note:** The following packages are auto-installed as dependencies and do not need manual installation:
> - `tokenizers` (by transformers)
> - `regex` (by transformers)
> - `ninja` (by mamba-ssm)
> - `typing_extensions` (by torch)
> - `pynndescent` (by umap-learn)
> - `h11` (by httpcore)

---

## Complete Automation Script

Save as `install_all.sh`:

```bash
#!/bin/bash
# SecureMamba Complete Installation Script

set -e

echo "=========================================="
echo "SecureMamba Complete Installation"
echo "=========================================="

# Initialize conda
eval "$(conda shell.bash hook)"

# Batch 1
echo "[1/16] Creating environment..."
conda create -n securemamba python=3.11 -y
conda activate securemamba
pip install pip==25.2 --upgrade
pip install --upgrade setuptools wheel

# Batch 2
echo "[2/16] Installing PyTorch and CUDA..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118
conda install -c nvidia -y cuda-cudart=11.8 cuda-cupti=11.8 cuda-nvrtc=11.8

# Batch 3
echo "[3/16] Installing core scientific packages..."
conda install -c conda-forge -y numpy=2.2.6 scipy=1.16.1 pandas=2.3.2

# Batch 4
echo "[4/16] Installing visualization packages..."
conda install -c conda-forge -y matplotlib=3.9.1 seaborn=0.13.2 \
    contourpy=1.3.3 fonttools=4.59.2 kiwisolver=1.4.9 cycler=0.12.1 pyparsing=3.2.3

# Batch 5
echo "[5/16] Installing ML packages..."
conda install -c conda-forge -y scikit-learn=1.7.1 statsmodels=0.14.5 threadpoolctl=3.6.0

# Batch 6
echo "[6/16] Installing bioinformatics..."
conda install -c conda-forge -y biopython=1.85

# Batch 7
echo "[7/16] Installing numerical computing..."
conda install -c conda-forge -y numba=0.61.2 sympy=1.13.1 networkx=3.5 gmpy2=2.2.1

# Batch 8
echo "[8/16] Installing compression packages..."
conda install -c conda-forge -y zstd=1.5.6 lz4-c=1.9.4 brotli=1.1.0 zstandard=0.23.0

# Batch 9
echo "[9/16] Installing network packages..."
conda install -c conda-forge -y requests=2.32.5 urllib3=2.5.0 certifi=2025.11.12 \
    idna=3.10 charset-normalizer=3.4.3 pysocks=1.7.1

# Batch 10
echo "[10/16] Installing image/GUI packages..."
conda install -c conda-forge -y pillow=9.4.0 jpeg=9e pyqt=5.15.9

# Batch 11
echo "[11/16] Installing utilities..."
conda install -c conda-forge -y tqdm=4.67.1 joblib=1.5.2 pyyaml=6.0.2 toml=0.10.2 \
    tomli=2.2.1 colorama=0.4.6 unicodedata2=16.0.0 six=1.17.0 python-dateutil=2.9.0.post0 \
    pytz=2025.2 packaging=23.2 platformdirs=4.5.0 psutil=7.1.3 filelock=3.19.1 \
    tornado=6.5.2 markupsafe=3.0.2 jinja2=3.1.6

# Batch 12
echo "[12/16] Installing Hugging Face packages..."
pip install huggingface-hub==0.36.0 datasets==4.4.1 hf-xet==1.1.9 safetensors==0.6.2 h5py==3.15.1

# Batch 13
echo "[13/16] Installing Transformers..."
pip install transformers==4.57.1

# Batch 14
echo "[14/16] Installing UMAP..."
pip install umap-learn==0.5.9.post2

# Batch 15
echo "[15/16] Installing Mamba-SSM from pre-built wheels..."
wget -q https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
wget -q https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install causal_conv1d-1.5.3.post1+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install mamba_ssm-2.2.6.post3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
rm -f causal_conv1d*.whl mamba_ssm*.whl

# Batch 16
echo "[16/16] Installing remaining pip packages..."
pip install \
    accelerate==1.11.0 \
    einops==0.8.1 \
    lion-pytorch==0.2.3 \
    aiohttp==3.13.2 \
    aiosignal==1.4.0 \
    aiohappyeyeballs==2.6.1 \
    frozenlist==1.8.0 \
    multidict==6.7.0 \
    yarl==1.22.0 \
    propcache==0.4.1 \
    anyio==4.11.0 \
    httpcore==1.0.9 \
    httpx==0.28.1 \
    sniffio==1.3.1 \
    numexpr==2.14.1 \
    tables==3.10.2 \
    blosc2==3.11.1 \
    msgpack==1.1.2 \
    msgpack-numpy==0.4.8 \
    ndindex==1.10.1 \
    pyarrow==22.0.0 \
    dill==0.4.0 \
    multiprocess==0.70.18 \
    fsspec==2025.9.0 \
    xxhash==3.6.0 \
    lmdb==1.7.5 \
    patsy==1.0.1 \
    py-cpuinfo==9.0.0 \
    ply==3.11

echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo "Activate: conda activate securemamba"
echo "Verify:   python verify.py"
```

Run the script:

```bash
chmod +x install_all.sh
./install_all.sh
```

---

## Verification

Save as `verify.py`:

```python
#!/usr/bin/env python3
"""Complete environment verification for SecureMamba"""

import sys

def main():
    print("=" * 50)
    print("SecureMamba Verification")
    print("=" * 50)
    
    packages = {
        "torch": "torch",
        "torchvision": "torchvision",
        "torchaudio": "torchaudio",
        "transformers": "transformers",
        "datasets": "datasets",
        "huggingface_hub": "huggingface_hub",
        "accelerate": "accelerate",
        "einops": "einops",
        "mamba_ssm": "mamba_ssm",
        "causal_conv1d": "causal_conv1d",
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "sklearn": "sklearn",
        "statsmodels": "statsmodels",
        "Bio": "Bio",
        "numba": "numba",
        "tqdm": "tqdm",
        "yaml": "yaml",
        "requests": "requests",
        "aiohttp": "aiohttp",
        "umap": "umap",
        "tables": "tables",
        "pyarrow": "pyarrow",
    }
    
    failed = []
    for name, import_name in packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: NOT FOUND")
            failed.append(name)
    
    print("\n--- CUDA ---")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
    except:
        print("CUDA check failed")
    
    print("\n--- Summary ---")
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")
    else:
        print("✅ All packages verified!")
    
    print("=" * 50)
    return len(failed) == 0

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
```

Run verification:

```bash
conda activate securemamba
python verify.py
```

---

## Troubleshooting

### Issue: `Solving environment` hangs

**Fix:** Use batch installation instead of single YAML file.

### Issue: Mamba-SSM installation fails

**Fix:** Use the pre-built wheels provided in Batch 15. Direct download links:

- [causal_conv1d-1.5.3.post1+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl)
- [mamba_ssm-2.2.6.post3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl](https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl)

### Issue: `libcudart.so` not found

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Issue: Conda channels slow (China users)

```bash
cat > ~/.condarc << 'EOF'
channels:
  - conda-forge
  - pytorch
  - nvidia
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
```

### Issue: Hugging Face download slow (China users)

```bash
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

---

## Download Datasets & Model

### Model Weights

```bash
mkdir -p Model
wget -P Model/ https://zenodo.org/records/20365513/files/best_full_model_model.pth
```

### Datasets

```bash
mkdir -p train_data
cd train_data

wget https://zenodo.org/record/20374288/files/train_data_pos.fasta
wget https://zenodo.org/record/20374288/files/train_data_neg.fasta
wget https://zenodo.org/record/20374288/files/btx_test.fasta
wget https://zenodo.org/record/20374288/files/vfdb_test.fasta
wget https://zenodo.org/record/20374288/files/toxin_test.fasta

cd ..
```

### Expected Directory Structure

```
SecureMamba/
├── Model/
│   └── best_nucleotide_v3_mamba_model.pth
├── train_data/
│   ├── train_data_pos.fasta
│   ├── train_data_neg.fasta
│   ├── btx_test.fasta
│   ├── vfdb_test.fasta
│   └── toxin_test.fasta
├── config.py
├── main.py
├── predict.py
├── install_all.sh
├── verify.py
└── [other source files]
```

---

## Running SecureMamba

### Training

```bash
conda activate securemamba
python main.py
```

### Prediction

```bash
conda activate securemamba
python predict.py --fasta train_data/btx_test.fasta --model Model/best_nucleotide_v3_mamba_model.pth
```

---

**Last Updated:** 2026  
**Maintainer:** kosha316

