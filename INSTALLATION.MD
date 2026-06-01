
# SecureMamba - Complete Installation Guide

This guide provides a **batch-by-batch** installation method that installs **every package** from the SecureMamba environment, avoiding the common "Solving environment" hang.

---

## Table of Contents

1. [Quick Installation (Mamba)](#quick-installation-mamba)
2. [Batch Installation (All Packages)](#batch-installation-all-packages)
   - [Batch 1: Python & Pip](#batch-1-python--pip)
   - [Batch 2: PyTorch Ecosystem](#batch-2-pytorch-ecosystem)
   - [Batch 3: Core Scientific Packages](#batch-3-core-scientific-packages)
   - [Batch 4: Visualization & Plotting](#batch-4-visualization--plotting)
   - [Batch 5: Machine Learning & Statistics](#batch-5-machine-learning--statistics)
   - [Batch 6: Bioinformatics](#batch-6-bioinformatics)
   - [Batch 7: Numerical Computing](#batch-7-numerical-computing)
   - [Batch 8: Data Processing & Compression](#batch-8-data-processing--compression)
   - [Batch 9: Network & Async](#batch-9-network--async)
   - [Batch 10: Image & GUI](#batch-10-image--gui)
   - [Batch 11: System & Utilities](#batch-11-system--utilities)
   - [Batch 12: Hugging Face Ecosystem](#batch-12-hugging-face-ecosystem)
   - [Batch 13: Transformers & NLP](#batch-13-transformers--nlp)
   - [Batch 14: Dimensionality Reduction](#batch-14-dimensionality-reduction)
   - [Batch 15: Mamba-SSM (Critical)](#batch-15-mamba-ssm-critical)
   - [Batch 16: Remaining Pip Packages](#batch-16-remaining-pip-packages)
   - [Batch 17: Conda-Only Packages](#batch-17-conda-only-packages)
3. [Verification](#verification)
4. [Troubleshooting](#troubleshooting)
5. [Download Datasets & Model](#download-datasets--model)

---

### Verify Your System

```bash
# Check GPU and CUDA driver
nvidia-smi

# Expected: CUDA Version 11.8 or higher
```

---

## Quick Installation (Mamba)

**Fastest method** - uses mamba to install all packages at once:

```bash
# 1. Clone repository
git clone https://github.com/kosha316/SecureMamba.git
cd SecureMamba

# 2. Install mamba
conda activate base
conda install mamba -c conda-forge -y

# 3. Create environment with all packages
mamba env create -f environment.yml

# 4. Activate
conda activate securemamba

# 5. Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**If mamba also hangs**, use the **Batch Installation** below.

---

## Batch Installation (All Packages)

This method installs **every package** from your environment in small batches to avoid dependency conflicts.

### Setup

```bash
# Create base environment
conda create -n securemamba python=3.11 -y
conda activate securemamba

# Upgrade pip
pip install --upgrade pip
```

---

### Batch 1: Python & Pip

```bash
# Already installed with environment
pip install --upgrade setuptools wheel
```

---

### Batch 2: PyTorch Ecosystem

```bash
# PyTorch with CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

---

### Batch 3: Core Scientific Packages

```bash
conda install -c conda-forge -y \
    numpy=2.2.6 \
    scipy=1.16.1 \
    pandas=2.3.2
```

---

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

---

### Batch 5: Machine Learning & Statistics

```bash
conda install -c conda-forge -y \
    scikit-learn=1.7.1 \
    statsmodels=0.14.5 \
    threadpoolctl=3.6.0
```

---

### Batch 6: Bioinformatics

```bash
conda install -c conda-forge -y \
    biopython=1.85
```

---

### Batch 7: Numerical Computing

```bash
conda install -c conda-forge -y \
    numba=0.61.2 \
    sympy=1.13.1 \
    networkx=3.5
```

---

### Batch 8: Data Processing & Compression

```bash
conda install -c conda-forge -y \
    zstd=1.5.6 \
    lz4-c=1.9.4 \
    brotli=1.1.0 \
    zstandard=0.23.0
```

---

### Batch 9: Network & Async

```bash
conda install -c conda-forge -y \
    requests=2.32.5 \
    urllib3=2.5.0 \
    certifi=2025.11.12 \
    idna=3.10 \
    charset-normalizer=3.4.3 \
    pysocks=1.7.1
```

---

### Batch 10: Image & GUI

```bash
conda install -c conda-forge -y \
    pillow=9.4.0 \
    jpeg=9e \
    pyqt=5.15.9
```

---

### Batch 11: System & Utilities

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

---

### Batch 12: Hugging Face Ecosystem

```bash
pip install \
    huggingface-hub==0.36.0 \
    datasets==4.4.1 \
    hf-xet==1.1.9 \
    safetensors==0.6.2 \
    tokenizers==0.22.1
```

---

### Batch 13: Transformers & NLP

```bash
pip install \
    transformers==4.57.1 \
    regex==2025.9.1 \
    tokenizers==0.22.1
```

---

### Batch 14: Dimensionality Reduction

```bash
pip install \
    umap-learn==0.5.9.post2 \
    pynndescent==0.5.13
```

---

### Batch 15: Mamba-SSM (Critical)

**This is the most challenging package. Follow carefully:**

```bash
# Set compilation flags (adjust based on your GPU)
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
export MAX_JOBS=4

# Install causal-conv1d
pip install causal-conv1d==1.5.3.post1

# Install mamba-ssm
pip install mamba-ssm==2.2.6.post3
```

**If the above fails,** try pre-compiled wheels:

```bash
# Download from: https://github.com/Dao-AILab/causal-conv1d/releases
# Look for: causal_conv1d-1.5.3+cu118torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install causal_conv1d-1.5.3+cu118torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install mamba_ssm-2.2.6+cu118torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

---

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
    h11==0.16.0 \
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
    ninja==1.11.1 \
    py-cpuinfo==9.0.0 \
    ply==3.11 \
    typing_extensions==4.15.0
```

---

### Batch 17: Conda-Only Packages (System Libraries)

These are automatically installed with conda packages but listed here for completeness:

```bash
# These should already be installed as dependencies
# Verify they are present:
conda list | grep -E "alsa-lib|brotli|bzip2|cairo|dbus|expat|ffmpeg|fftw|fontconfig|freetype|gettext|glib|gmp|gnutls|graphite2|gstreamer|harfbuzz|icu|krb5|lame|lcms2|libclang|libcups|libdb|libdrm|libedit|libevent|libffi|libflac|libgcc|libgcrypt|libglib|libgpg-error|libhwloc|libiconv|libidn2|libllvm|libogg|libopus|libpng|libpq|libsndfile|libsqlite|libtasn1|libtiff|libtool|libudev|libunistring|libuuid|libva|libvorbis|libvpx|libwebp|libxcb|libxcrypt|libxkbcommon|libxml2|llvm-openmp|mpc|mpfr|mpg123|mysql-common|mysql-libs|ncurses|nettle|nspr|nss|openh264|openjpeg|openssl|p11-kit|pcre2|pixman|pulseaudio|qhull|qt-main|readline|svt-av1|tbb|tk|x264|x265|xcb-util|xkeyboard-config|xorg-libx11|xorg-libxau|xorg-libxdmcp|xorg-libxext|xorg-libxfixes|xorg-libxrender|xz|yaml|zlib
```

---

## Complete Batch Installation Script

Save as `install_all_batches.sh`:

```bash
#!/bin/bash
set -e  # Stop on error

echo "=========================================="
echo "SecureMamba Complete Installation"
echo "=========================================="

# Create environment
echo "[1/17] Creating environment..."
conda create -n securemamba python=3.11 -y
conda activate securemamba
pip install --upgrade pip

# Batch 2
echo "[2/17] Installing PyTorch..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Batch 3
echo "[3/17] Installing core scientific packages..."
conda install -c conda-forge -y numpy=2.2.6 scipy=1.16.1 pandas=2.3.2

# Batch 4
echo "[4/17] Installing visualization packages..."
conda install -c conda-forge -y matplotlib=3.9.1 seaborn=0.13.2 \
    contourpy=1.3.3 fonttools=4.59.2 kiwisolver=1.4.9 cycler=0.12.1 pyparsing=3.2.3

# Batch 5
echo "[5/17] Installing ML packages..."
conda install -c conda-forge -y scikit-learn=1.7.1 statsmodels=0.14.5 threadpoolctl=3.6.0

# Batch 6
echo "[6/17] Installing bioinformatics..."
conda install -c conda-forge -y biopython=1.85

# Batch 7
echo "[7/17] Installing numerical computing..."
conda install -c conda-forge -y numba=0.61.2 sympy=1.13.1 networkx=3.5

# Batch 8
echo "[8/17] Installing compression packages..."
conda install -c conda-forge -y zstd=1.5.6 lz4-c=1.9.4 brotli=1.1.0 zstandard=0.23.0

# Batch 9
echo "[9/17] Installing network packages..."
conda install -c conda-forge -y requests=2.32.5 urllib3=2.5.0 certifi=2025.11.12 \
    idna=3.10 charset-normalizer=3.4.3 pysocks=1.7.1

# Batch 10
echo "[10/17] Installing image/GUI packages..."
conda install -c conda-forge -y pillow=9.4.0 jpeg=9e pyqt=5.15.9

# Batch 11
echo "[11/17] Installing utilities..."
conda install -c conda-forge -y tqdm=4.67.1 joblib=1.5.2 pyyaml=6.0.2 toml=0.10.2 \
    tomli=2.2.1 colorama=0.4.6 unicodedata2=16.0.0 six=1.17.0 python-dateutil=2.9.0.post0 \
    pytz=2025.2 packaging=23.2 platformdirs=4.5.0 psutil=7.1.3 filelock=3.19.1 \
    tornado=6.5.2 markupsafe=3.0.2 jinja2=3.1.6

# Batch 12
echo "[12/17] Installing Hugging Face packages..."
pip install huggingface-hub==0.36.0 datasets==4.4.1 hf-xet==1.1.9 safetensors==0.6.2

# Batch 13
echo "[13/17] Installing Transformers..."
pip install transformers==4.57.1 regex==2025.9.1

# Batch 14
echo "[14/17] Installing UMAP..."
pip install umap-learn==0.5.9.post2 pynndescent==0.5.13

# Batch 15
echo "[15/17] Installing Mamba-SSM (may take a few minutes)..."
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
export MAX_JOBS=4
pip install causal-conv1d==1.5.3.post1
pip install mamba-ssm==2.2.6.post3

# Batch 16
echo "[16/17] Installing remaining pip packages..."
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
    h11==0.16.0 \
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
    ninja==1.11.1 \
    py-cpuinfo==9.0.0 \
    ply==3.11 \
    typing_extensions==4.15.0

echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo "Activate environment: conda activate securemamba"
echo "Run verification: python verify.py"
```

Run the script:

```bash
chmod +x install_all_batches.sh
./install_all_batches.sh
```

---

## Verification

### Quick Check

```bash
conda activate securemamba
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Complete Verification Script

Save as `verify_all.py`:

```python
#!/usr/bin/env python3
"""Verify all SecureMamba packages are installed"""

import sys

def verify():
    print("=" * 60)
    print("SecureMamba Package Verification")
    print("=" * 60)
    
    packages = {
        # Core
        "torch": "torch",
        "torchvision": "torchvision", 
        "torchaudio": "torchaudio",
        
        # ML/NLP
        "transformers": "transformers",
        "datasets": "datasets",
        "huggingface_hub": "huggingface_hub",
        "accelerate": "accelerate",
        "einops": "einops",
        
        # Mamba
        "mamba_ssm": "mamba_ssm",
        "causal_conv1d": "causal_conv1d",
        
        # Scientific
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "scikit_learn": "sklearn",
        "statsmodels": "statsmodels",
        "biopython": "Bio",
        "numba": "numba",
        "sympy": "sympy",
        
        # Utils
        "tqdm": "tqdm",
        "pyyaml": "yaml",
        "requests": "requests",
        "pillow": "PIL",
        "aiohttp": "aiohttp",
        "umap_learn": "umap",
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
    
    # CUDA check
    print("\n--- CUDA ---")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("CUDA check failed")
    
    print("\n--- Summary ---")
    if failed:
        print(f"❌ Failed packages: {', '.join(failed)}")
    else:
        print("✅ All packages verified!")
    
    print("=" * 60)
    return len(failed) == 0

if __name__ == "__main__":
    sys.exit(0 if verify() else 1)
```

Run:

```bash
conda activate securemamba
python verify_all.py
```

---

## Troubleshooting

### Issue: `Solving environment` hangs

**Fix:** Use batch installation instead of single YAML file.

### Issue: Mamba-SSM fails to compile

**Fix 1:** Install pre-compiled wheels
```bash
# Download from: https://github.com/Dao-AILab/causal-conv1d/releases
pip install causal_conv1d-1.5.3+cu118torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install mamba_ssm-2.2.6+cu118torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

**Fix 2:** Use conda-forge
```bash
conda install -c conda-forge mamba-ssm causal-conv1d
```

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

### Issue: CUDA out of memory during training

```python
# In config.py
batch_size = 8  # Reduce from 16
max_seq_length = 1024  # Reduce from 2048
```

---

## Download Datasets & Model

### Model Weights

```bash
mkdir -p Model
wget -P Model/ https://zenodo.org/record/20365513/files/best_nucleotide_v3_mamba_model.pth
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
```

---
