
# SecureMamba - Complete Installation Guide

A step-by-step guide to install SecureMamba and all its dependencies.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation Steps](#detailed-installation-steps)
   - [Step 1: Install Conda/Mamba](#step-1-install-condamamba)
   - [Step 2: Configure Mirrors (China Region)](#step-2-configure-mirrors-china-region)
   - [Step 3: Create Environment from YAML](#step-3-create-environment-from-yaml)
   - [Step 4: Verify Installation](#step-4-verify-installation)
   - [Step 5: Download Model Weights](#step-5-download-model-weights)
   - [Step 6: Download Datasets](#step-6-download-datasets)
4. [Alternative Installation Methods](#alternative-installation-methods)
   - [Method A: Manual Conda + Pip](#method-a-manual-conda--pip)
   - [Method B: Pip Only (No Conda)](#method-b-pip-only-no-conda)
   - [Method C: Docker Installation](#method-c-docker-installation)
5. [Handling Difficult Packages](#handling-difficult-packages)
   - [Mamba-SSM and Causal-Conv1d](#mamba-ssm-and-causal-conv1d)
6. [Troubleshooting](#troubleshooting)
7. [Verification Script](#verification-script)
8. [Environment Export](#environment-export)

---

## Prerequisites

Before starting, ensure your system meets the following requirements:

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux (Ubuntu 20.04+ recommended) |
| **Python** | 3.11.6 |
| **CUDA** | 11.8 or 12.1 |
| **GPU Memory** | 16GB+ recommended |
| **Disk Space** | 50GB+ |
| **RAM** | 32GB+ recommended |

### Check System Requirements

```bash
# Check CUDA version
nvidia-smi

# Check Python version
python3 --version

# Check disk space
df -h

# Check GPU memory
nvidia-smi --query-gpu=memory.total --format=csv
```

---

## Quick Installation

For users who want to get started quickly:

```bash
# Clone repository
git clone https://github.com/kosha316/SecureMamba.git
cd SecureMamba

# Create and activate environment
conda env create -f environment.yml
conda activate securemamba

# Download model weights
wget https://zenodo.org/record/20365513/files/best_nucleotide_v3_mamba_model.pth

# Download test data
wget https://zenodo.org/record/20374288/files/btx_test.fasta

# Run prediction
python predict.py --fasta btx_test.fasta --model best_nucleotide_v3_mamba_model.pth
```

---

## Detailed Installation Steps

### Step 1: Install Conda/Mamba

If you don't have Conda installed:

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts, then restart terminal or source bashrc
source ~/.bashrc

# (Recommended) Install mamba for faster dependency resolution
conda install mamba -n base -c conda-forge -y
```

### Step 2: Configure Mirrors (China Region)

**For users in China**, configure mirrors to speed up downloads:

```bash
# Configure Conda mirror (Tsinghua)
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
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
channel_priority: flexible
EOF

# Configure Pip mirror
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# Configure Hugging Face mirror
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

**For users outside China**, you can skip this step or use default sources.

### Step 3: Create Environment from YAML

Create `environment.yml` in your project root with the complete package list:

```yaml
name: securemamba
channels:
  - conda-forge
  - pytorch
  - nvidia
  - defaults
dependencies:
  # Python core
  - python=3.11.6
  - pip
  - setuptools
  - wheel
  
  # Scientific computing
  - numpy=2.2.6
  - scipy=1.16.1
  - pandas=2.3.2
  - matplotlib=3.9.1
  - seaborn=0.13.2
  - scikit-learn=1.7.1
  - statsmodels=0.14.5
  
  # Bioinformatics
  - biopython=1.85
  
  # Visualization
  - contourpy=1.3.3
  - fonttools=4.59.2
  - kiwisolver=1.4.9
  - cycler=0.12.1
  - pyparsing=3.2.3
  
  # Numerical computing
  - numba=0.61.2
  - sympy=1.13.1
  - networkx=3.5
  
  # Deep learning base (conda)
  - pytorch-mutex=1.0=cuda
  
  # Data processing
  - tqdm=4.67.1
  - joblib=1.5.2
  - threadpoolctl=3.6.0
  - pyyaml=6.0.2
  - toml=0.10.2
  - tomli=2.2.1
  
  # Network
  - requests=2.32.5
  - urllib3=2.5.0
  - certifi=2025.11.12
  - idna=3.10
  - charset-normalizer=3.4.3
  - pysocks=1.7.1
  
  # Image processing
  - pillow=9.4.0
  - jpeg=9e
  
  # Compression
  - zstd=1.5.6
  - lz4-c=1.9.4
  - brotli=1.1.0
  - zstandard=0.23.0
  
  # GUI (optional)
  - pyqt=5.15.9
  
  # Other dependencies
  - tornado=6.5.2
  - markupsafe=3.0.2
  - jinja2=3.1.6
  - colorama=0.4.6
  - unicodedata2=16.0.0
  - six=1.17.0
  - python-dateutil=2.9.0.post0
  - pytz=2025.2
  - packaging=23.2
  - platformdirs=4.5.0
  - psutil=7.1.3
  - filelock=3.19.1
  
  # Pip-installed packages (conda unavailable)
  - pip:
    # PyTorch ecosystem (CUDA 12.1)
    - torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    - torchvision==0.20.1+cu121
    - torchaudio==2.5.1+cu121
    
    # Transformers ecosystem
    - transformers==4.57.1
    - tokenizers==0.22.1
    - safetensors==0.6.2
    - huggingface-hub==0.36.0
    - datasets==4.4.1
    - hf-xet==1.1.9
    
    # Mamba series (requires CUDA compilation)
    - mamba-ssm==2.2.6.post3
    - causal-conv1d==1.5.3.post1
    
    # Deep learning tools
    - einops==0.8.1
    - lion-pytorch==0.2.3
    - accelerate==1.11.0
    
    # Async networking
    - aiohttp==3.13.2
    - aiosignal==1.4.0
    - aiohappyeyeballs==2.6.1
    - frozenlist==1.8.0
    - multidict==6.7.0
    - yarl==1.22.0
    - propcache==0.4.1
    - anyio==4.11.0
    - httpcore==1.0.9
    - httpx==0.28.1
    - h11==0.16.0
    - sniffio==1.3.1
    
    # Scientific computing extensions
    - numexpr==2.14.1
    - tables==3.10.2
    - blosc2==3.11.1
    - msgpack==1.1.2
    - msgpack-numpy==0.4.8
    - ndindex==1.10.1
    
    # Data processing
    - pyarrow==22.0.0
    - dill==0.4.0
    - multiprocess==0.70.18
    - fsspec==2025.9.0
    - xxhash==3.6.0
    - lmdb==1.7.5
    
    # Dimensionality reduction
    - umap-learn==0.5.9.post2
    - pynndescent==0.5.13
    
    # Statistics
    - patsy==1.0.1
    
    # Utilities
    - ninja==1.11.1
    - py-cpuinfo==9.0.0
    - ply==3.11
    - regex==2025.9.1
    - typing_extensions==4.15.0
```

Now create the environment:

```bash
# Using conda (may take 10-30 minutes)
conda env create -f environment.yml

# Or using mamba (much faster, recommended)
mamba env create -f environment.yml

# Activate environment
conda activate securemamba
```

### Step 4: Verify Installation

Run basic verification:

```bash
python -c "
import torch
import mamba_ssm
import transformers
import Bio

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ CUDA version:', torch.version.cuda)
    print('✓ GPU:', torch.cuda.get_device_name(0))
print('✓ Mamba-SSM:', mamba_ssm.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ Biopython:', Bio.__version__)
print('\n✅ All core packages installed successfully!')
"
```

### Step 5: Download Model Weights

```bash
# Create Model directory
mkdir -p Model

# Download pre-trained model from Zenodo
wget -P Model/ https://zenodo.org/record/20365513/files/best_nucleotide_v3_mamba_model.pth

# Verify download
ls -lh Model/
```

### Step 6: Download Datasets

```bash
# Create data directory
mkdir -p train_data

# Download training and test datasets
cd train_data
wget https://zenodo.org/record/20374288/files/train_data_pos.fasta
wget https://zenodo.org/record/20374288/files/train_data_neg.fasta
wget https://zenodo.org/record/20374288/files/btx_test.fasta
wget https://zenodo.org/record/20374288/files/vfdb_test.fasta
wget https://zenodo.org/record/20374288/files/toxin_test.fasta
cd ..

# Verify downloads
ls -lh train_data/
```

---

## Alternative Installation Methods

### Method A: Manual Conda + Pip

Use this if the YAML installation fails.

```bash
# Step 1: Create base environment
conda create -n securemamba python=3.11 -y
conda activate securemamba

# Step 2: Install PyTorch with CUDA 11.8 (pip recommended)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Step 3: Install conda dependencies (one by one or in groups)
conda install -c conda-forge -y \
    numpy=2.2.6 \
    scipy=1.16.1 \
    pandas=2.3.2 \
    matplotlib=3.9.1 \
    seaborn=0.13.2 \
    biopython=1.85 \
    scikit-learn=1.7.1 \
    statsmodels=0.14.5 \
    numba=0.61.2 \
    sympy=1.13.1 \
    networkx=3.5 \
    tqdm \
    joblib \
    pyyaml \
    requests \
    pillow \
    tornado

# Step 4: Install remaining pip dependencies
pip install \
    transformers==4.57.1 \
    datasets==4.4.1 \
    huggingface-hub==0.36.0 \
    accelerate==1.11.0 \
    einops==0.8.1 \
    aiohttp==3.13.2 \
    umap-learn==0.5.9.post2 \
    tables==3.10.2 \
    pyarrow==22.0.0

# Step 5: Install Mamba-SSM (see section below)
pip install causal-conv1d==1.5.3.post1
pip install mamba-ssm==2.2.6.post3
```

### Method B: Pip Only (No Conda)

For users who prefer pip and venv:

```bash
# Install system dependencies first (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential libhdf5-dev

# Create virtual environment
python3.11 -m venv securemamba_env
source securemamba_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install scientific packages
pip install \
    numpy==2.2.6 \
    scipy==1.16.1 \
    pandas==2.3.2 \
    matplotlib==3.9.1 \
    seaborn==0.13.2 \
    biopython==1.85 \
    scikit-learn==1.7.1 \
    statsmodels==0.14.5 \
    numba==0.61.2 \
    sympy==1.13.1

# Install ML packages
pip install \
    transformers==4.57.1 \
    datasets==4.4.1 \
    huggingface-hub==0.36.0 \
    accelerate==1.11.0 \
    einops==0.8.1 \
    causal-conv1d==1.5.3.post1 \
    mamba-ssm==2.2.6.post3

# Install utilities
pip install \
    aiohttp==3.13.2 \
    umap-learn==0.5.9.post2 \
    tables==3.10.2 \
    pyarrow==22.0.0 \
    tqdm \
    pyyaml \
    requests \
    pillow \
    tornado

# Verify
python -c "import torch; print(torch.__version__)"
```

### Method C: Docker Installation

Create a `Dockerfile`:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH

# Clone repository
RUN git clone https://github.com/kosha316/SecureMamba.git /app
WORKDIR /app

# Create conda environment
COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml && conda clean -afy

# Download model weights (optional)
RUN mkdir -p Model && \
    wget -P Model/ https://zenodo.org/record/20365513/files/best_nucleotide_v3_mamba_model.pth

ENTRYPOINT ["conda", "run", "-n", "securemamba"]
CMD ["python", "predict.py", "--help"]
```

Build and run:

```bash
# Build image
docker build -t securemamba .

# Run with GPU support
docker run --gpus all -it securemamba

# Run prediction with mounted data
docker run --gpus all -v $(pwd)/data:/app/data securemamba \
    python predict.py --fasta data/input.fasta --model Model/best_nucleotide_v3_mamba_model.pth
```

---

## Handling Difficult Packages

### Mamba-SSM and Causal-Conv1d

These packages require CUDA compilation and often fail. Here are solutions:

**Solution 1: Pre-compiled wheels (Recommended)**

Visit the releases page to download pre-compiled wheels:
- [causal-conv1d releases](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl)
- [mamba releases](https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl)

Download the wheel matching:
- Your CUDA version (11.8 or 12.1)
- PyTorch version (2.5.1)
- Python version (3.11)
- Linux x86_64

```bash
# Example for CUDA 11.8, PyTorch 2.5, Python 3.11
pip install causal_conv1d-1.5.3+cu118torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install mamba_ssm-2.2.6+cu118torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

**Solution 2: Set compilation flags**

```bash
# Set environment variables before installation
export MAX_JOBS=4  # Reduce parallel compilation
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"  # Match your GPU architecture

pip install causal-conv1d==1.5.3.post1 --no-cache-dir
pip install mamba-ssm==2.2.6.post3 --no-cache-dir
```

**Solution 3: Install from source**

```bash
# Clone and install causal-conv1d
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.5.3
pip install -e .
cd ..

# Clone and install mamba
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v2.2.6
pip install -e .
cd ..
```

**Solution 4: Use conda-forge (if available)**

```bash
# Check if conda-forge has pre-built versions
conda search -c conda-forge mamba-ssm
conda install -c conda-forge mamba-ssm causal-conv1d
```

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| **CondaHTTPError** | Configure mirrors (see Step 2) |
| **UnsatisfiableError** | Use mamba: `mamba env create -f environment.yml` |
| **CUDA out of memory** | Reduce `batch_size` in config.py |
| **Mamba-SSM import error** | Check CUDA version compatibility (needs CUDA 11.8+) |
| **HF download slow** | Set `HF_ENDPOINT=https://hf-mirror.com` |
| **libcuda.so not found** | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` |
| **gcc version error** | Install newer gcc: `conda install -c conda-forge gcc=12` |
| **ninja build error** | `pip install ninja` or `conda install -c conda-forge ninja` |
| **tables/h5py compilation fails** | Install HDF5: `conda install -c conda-forge hdf5` |
| **pyarrow installation fails** | `conda install -c conda-forge pyarrow` |

### CUDA Compatibility

Check your CUDA version and ensure compatibility:

```bash
# Check system CUDA
nvcc --version
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# For CUDA 11.8, use torch with cu118 index
# For CUDA 12.1, use torch with cu121 index
```

### Common Error Messages

**Error: `ModuleNotFoundError: No module named 'torch'`**
```bash
# PyTorch not installed correctly
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

**Error: `OSError: libcudart.so.11.0: cannot open shared object file`**
```bash
# CUDA runtime not found
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

**Error: `CUDA error: no kernel image is available for execution on the device`**
```bash
# GPU architecture not supported
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# Then reinstall mamba-ssm
pip uninstall mamba-ssm causal-conv1d -y
pip install causal-conv1d==1.5.3.post1 mamba-ssm==2.2.6.post3
```

**Error: `ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`**
```bash
# Update libstdc++
conda install -c conda-forge libstdcxx-ng
```

---

## Verification Script

Save this as `verify_env.py`:

```python
#!/usr/bin/env python3
"""Complete environment verification script for SecureMamba"""

import sys
import subprocess

def run_check(name, import_name, version_attr='__version__'):
    try:
        module = __import__(import_name)
        version = getattr(module, version_attr, 'unknown')
        print(f"✓ {name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {name}: Not installed - {e}")
        return False

def main():
    print("=" * 60)
    print("SecureMamba Environment Verification")
    print("=" * 60)
    
    # System checks
    print("\n--- System ---")
    print(f"Python: {sys.version}")
    
    # CUDA check
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("CUDA: Failed to check")
    
    # Package checks
    print("\n--- Core Packages ---")
    packages = [
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("TorchAudio", "torchaudio"),
        ("Transformers", "transformers"),
        ("Datasets", "datasets"),
        ("HuggingFace Hub", "huggingface_hub"),
        ("Mamba-SSM", "mamba_ssm"),
        ("Causal-Conv1D", "causal_conv1d"),
        ("Accelerate", "accelerate"),
        ("Einops", "einops"),
        ("Biopython", "Bio"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("SciPy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("statsmodels", "statsmodels"),
        ("numba", "numba"),
        ("aiohttp", "aiohttp"),
        ("umap-learn", "umap"),
        ("tables", "tables"),
        ("pyarrow", "pyarrow"),
    ]
    
    results = []
    for name, import_name in packages:
        success = run_check(name, import_name)
        results.append(success)
    
    # Summary
    print("\n--- Summary ---")
    passed = sum(results)
    total = len(results)
    print(f"Packages: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ Environment is ready for SecureMamba!")
    else:
        print(f"\n⚠️ {total - passed} packages failed. Check installation guide.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
```

Run verification:

```bash
conda activate securemamba
python verify_env.py
```

---

## Environment Export

To share or backup your environment:

```bash
# Export full environment (including builds)
conda env export > environment_full.yml

# Export without builds (better compatibility)
conda env export --no-builds > environment.yml

# Export only explicitly installed packages
conda env export --from-history > environment_history.yml

# Export pip packages only
pip freeze > requirements.txt
```

To recreate from exported files:

```bash
# From conda YAML
conda env create -f environment.yml

# From pip requirements
pip install -r requirements.txt
```

---

## Post-Installation

After successful installation, you should have:

```
SecureMamba/
├── environment.yml          # Conda environment spec
├── Model/
│   └── best_nucleotide_v3_mamba_model.pth  # Model weights
├── train_data/
│   ├── train_data_pos.fasta
│   ├── train_data_neg.fasta
│   ├── btx_test.fasta
│   ├── vfdb_test.fasta
│   └── toxin_test.fasta
└── [source code files]
```

Ready to run:

```bash
# Run training
python main.py

# Run prediction
python predict.py --fasta train_data/btx_test.fasta --model Model/best_nucleotide_v3_mamba_model.pth
```

---

## Getting Help

If you encounter issues not covered here:

1. Check [GitHub Issues](https://github.com/kosha316/SecureMamba/issues)
2. Check [Mamba-SSM Issues](https://github.com/state-spaces/mamba/issues)
3. Check [PyTorch Forums](https://discuss.pytorch.org/)

---

**Documentation Version**: 2.0  
**Last Updated**: 2026  
**Maintainer**: kosha316
```

