#!/bin/bash

# ProteinMPNN Installation Script
# Supports both standard and MPS-enabled installations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     ProteinMPNN Installation Script      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Detect OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

print_info "Detected OS: $OS"
print_info "Detected Architecture: $ARCH"
echo ""

# Check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.7 or later."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    print_info "Python version: $PYTHON_VERSION"
    
    # Check if version is >= 3.7
    MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 7 ]); then
        print_error "Python 3.7 or later is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
}

# Parse command line arguments
INSTALL_MODE="standard"
USE_CONDA=false
USE_CURRENT_ENV=false
ENV_NAME="proteinmpnn"
DOWNLOAD_WEIGHTS=true
DOWNLOAD_DATASET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mps)
            INSTALL_MODE="mps"
            shift
            ;;
        --cuda)
            INSTALL_MODE="cuda"
            shift
            ;;
        --cpu)
            INSTALL_MODE="cpu"
            shift
            ;;
        --conda)
            USE_CONDA=true
            shift
            ;;
        --current-env)
            USE_CURRENT_ENV=true
            shift
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --no-weights)
            DOWNLOAD_WEIGHTS=false
            shift
            ;;
        --with-dataset)
            DOWNLOAD_DATASET=true
            shift
            ;;
        --help)
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mps              Install with Apple Silicon MPS support"
            echo "  --cuda             Install with CUDA support"
            echo "  --cpu              Install CPU-only version"
            echo "  --conda            Use conda instead of pip"
            echo "  --current-env      Use current Python environment (no venv/conda)"
            echo "  --env-name NAME    Conda environment name (default: proteinmpnn)"
            echo "  --no-weights       Skip downloading model weights"
            echo "  --with-dataset     Download training dataset (16.5GB)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-detect installation mode if not specified
if [ "$INSTALL_MODE" == "standard" ]; then
    if [ "$OS" == "Darwin" ] && [ "$ARCH" == "arm64" ]; then
        INSTALL_MODE="mps"
        print_info "Detected Apple Silicon - using MPS installation"
    elif command -v nvidia-smi &> /dev/null; then
        INSTALL_MODE="cuda"
        print_info "Detected NVIDIA GPU - using CUDA installation"
    else
        INSTALL_MODE="cpu"
        print_info "No GPU detected - using CPU installation"
    fi
fi

echo ""
print_info "Installation mode: $INSTALL_MODE"
echo ""

# Check Python
check_python

# Handle environment setup
if [ "$USE_CURRENT_ENV" = true ]; then
    print_info "Using current Python environment"
    if [ -n "$VIRTUAL_ENV" ]; then
        print_info "Current virtual environment: $VIRTUAL_ENV"
        PYTHON_CMD="python"
        PIP_CMD="pip"
    elif [ -n "$CONDA_DEFAULT_ENV" ]; then
        print_info "Current conda environment: $CONDA_DEFAULT_ENV"
        PYTHON_CMD="python"
        PIP_CMD="pip"
    else
        print_warning "No virtual/conda environment detected, using system Python"
        PIP_CMD="$PYTHON_CMD -m pip --user"
    fi
elif [ "$OS" == "Darwin" ] && [ "$USE_CONDA" = false ]; then
    # Check if we're already in a virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        print_info "macOS detected - creating virtual environment to avoid system Python issues"
        
        VENV_DIR="./venv_proteinmpnn"
        
        if [ ! -d "$VENV_DIR" ]; then
            print_status "Creating virtual environment in $VENV_DIR"
            $PYTHON_CMD -m venv $VENV_DIR
        fi
        
        print_status "Activating virtual environment"
        source $VENV_DIR/bin/activate
        PYTHON_CMD="python"
        PIP_CMD="pip"
        
        print_info "Virtual environment activated: $VIRTUAL_ENV"
    else
        print_info "Already in virtual environment: $VIRTUAL_ENV"
        PYTHON_CMD="python"
        PIP_CMD="pip"
    fi
elif [ "$USE_CONDA" = false ]; then
    PIP_CMD="$PYTHON_CMD -m pip"
fi

# Create/activate conda environment if requested
if [ "$USE_CONDA" = true ]; then
    print_status "Setting up conda environment: $ENV_NAME"
    
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install Anaconda or Miniconda first."
        exit 1
    fi
    
    # Check if environment exists
    if conda env list | grep -q "^$ENV_NAME "; then
        print_info "Environment $ENV_NAME already exists"
        read -p "Do you want to update it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda activate $ENV_NAME
        else
            print_error "Installation cancelled"
            exit 1
        fi
    else
        conda create -n $ENV_NAME python=3.9 -y
        conda activate $ENV_NAME
    fi
    
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

if [ "$USE_CONDA" = false ] && [ -z "$VIRTUAL_ENV" ]; then
    PIP_CMD="$PYTHON_CMD -m pip"
fi

# Install PyTorch based on mode
echo ""
print_status "Installing PyTorch for $INSTALL_MODE..."

case $INSTALL_MODE in
    mps)
        # MPS requires PyTorch 1.12+ on macOS 12.3+
        $PIP_CMD install --upgrade pip
        $PIP_CMD install torch torchvision torchaudio
        print_status "PyTorch installed with MPS support"
        ;;
    cuda)
        # Get CUDA version if available
        if command -v nvidia-smi &> /dev/null; then
            CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
            print_info "Detected CUDA version: $CUDA_VERSION"
            
            # Install appropriate PyTorch version
            if [ "$CUDA_VERSION" == "11.8" ]; then
                $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            elif [ "$CUDA_VERSION" == "12.1" ]; then
                $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            else
                $PIP_CMD install torch torchvision torchaudio
            fi
        else
            $PIP_CMD install torch torchvision torchaudio
        fi
        print_status "PyTorch installed with CUDA support"
        ;;
    cpu)
        $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        print_status "PyTorch installed (CPU-only)"
        ;;
esac

# Install other dependencies
echo ""
print_status "Installing additional dependencies..."

# Core dependencies
$PIP_CMD install numpy scipy

# Optional but recommended dependencies
$PIP_CMD install rich click tqdm

# For notebook support
read -p "Install Jupyter notebook support? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    $PIP_CMD install jupyter ipywidgets
    print_status "Jupyter support installed"
fi

# Download model weights
if [ "$DOWNLOAD_WEIGHTS" = true ]; then
    echo ""
    print_status "Downloading model weights..."
    
    # Create directories if they don't exist
    mkdir -p vanilla_model_weights
    mkdir -p ca_model_weights
    mkdir -p soluble_model_weights
    
    # Function to download with progress
    download_file() {
        url=$1
        output=$2
        
        if [ -f "$output" ]; then
            print_info "File already exists: $output"
        else
            print_info "Downloading: $output"
            curl -L --progress-bar "$url" -o "$output"
        fi
    }
    
    # Download vanilla model weights
    BASE_URL="https://files.ipd.uw.edu/pub/ProteinMPNN"
    
    download_file "$BASE_URL/vanilla_model_weights/v_48_002.pt" "vanilla_model_weights/v_48_002.pt"
    download_file "$BASE_URL/vanilla_model_weights/v_48_010.pt" "vanilla_model_weights/v_48_010.pt"
    download_file "$BASE_URL/vanilla_model_weights/v_48_020.pt" "vanilla_model_weights/v_48_020.pt"
    download_file "$BASE_URL/vanilla_model_weights/v_48_030.pt" "vanilla_model_weights/v_48_030.pt"
    
    # Download CA-only model weights
    download_file "$BASE_URL/ca_model_weights/v_48_002.pt" "ca_model_weights/v_48_002.pt"
    download_file "$BASE_URL/ca_model_weights/v_48_010.pt" "ca_model_weights/v_48_010.pt"
    download_file "$BASE_URL/ca_model_weights/v_48_020.pt" "ca_model_weights/v_48_020.pt"
    
    # Download soluble model weights
    download_file "$BASE_URL/soluble_model_weights/v_48_010.pt" "soluble_model_weights/v_48_010.pt"
    download_file "$BASE_URL/soluble_model_weights/v_48_020.pt" "soluble_model_weights/v_48_020.pt"
    
    print_status "Model weights downloaded successfully"
fi

# Download training dataset if requested
if [ "$DOWNLOAD_DATASET" = true ]; then
    echo ""
    print_warning "Training dataset is 16.5 GB. This will take some time..."
    read -p "Continue with dataset download? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p training_data
        cd training_data
        
        if [ -f "pdb_2021aug02.tar.gz" ]; then
            print_info "Dataset archive already exists"
        else
            print_info "Downloading training dataset..."
            curl -L --progress-bar "https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz" -o pdb_2021aug02.tar.gz
        fi
        
        if [ ! -d "pdb_2021aug02" ]; then
            print_info "Extracting dataset..."
            tar -xzf pdb_2021aug02.tar.gz
            print_status "Dataset extracted to training_data/pdb_2021aug02/"
        else
            print_info "Dataset already extracted"
        fi
        
        cd ..
    fi
fi

# Test installation
echo ""
print_status "Testing installation..."

$PYTHON_CMD -c "
import torch
import numpy as np

print('PyTorch version:', torch.__version__)
print('NumPy version:', np.__version__)

# Check device availability
if torch.cuda.is_available():
    print('CUDA is available')
    print('CUDA device count:', torch.cuda.device_count())
    print('CUDA device name:', torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    print('MPS is available')
    print('MPS is built:', torch.backends.mps.is_built())
else:
    print('Running on CPU')

# Quick test
x = torch.randn(2, 3)
if torch.cuda.is_available():
    x = x.cuda()
    print('Test tensor on CUDA:', x.device)
elif torch.backends.mps.is_available():
    x = x.to('mps')
    print('Test tensor on MPS:', x.device)
else:
    print('Test tensor on CPU:', x.device)
" || {
    print_error "Installation test failed"
    exit 1
}

# Create example run script
cat > run_example.sh << 'EOF'
#!/bin/bash
# Example ProteinMPNN run script

python protein_mpnn_run.py \
    --pdb_path inputs/1QYS.pdb \
    --pdb_path_chains A \
    --out_folder outputs/ \
    --num_seq_per_target 10 \
    --sampling_temp "0.1" \
    --seed 37
EOF

chmod +x run_example.sh

# Print summary
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Installation Complete!               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""

print_info "Installation summary:"
echo "  • PyTorch mode: $INSTALL_MODE"
if [ "$USE_CONDA" = true ]; then
    echo "  • Conda environment: $ENV_NAME"
fi
if [ "$DOWNLOAD_WEIGHTS" = true ]; then
    echo "  • Model weights: Downloaded"
fi
if [ "$DOWNLOAD_DATASET" = true ] && [ -d "training_data/pdb_2021aug02" ]; then
    echo "  • Training dataset: Downloaded"
fi
echo ""

print_info "Quick start:"
echo "  1. Test basic inference:"
echo "     ./run_example.sh"
echo ""
echo "  2. Test MPS/GPU acceleration:"
echo "     python test_mps_proteinmpnn.py"
echo ""
echo "  3. Quick training test:"
echo "     python test_train.py pdb_2021aug02"
echo ""

if [ "$USE_CONDA" = true ]; then
    print_warning "Remember to activate the environment:"
    echo "     conda activate $ENV_NAME"
fi

echo ""
print_status "Ready to use ProteinMPNN!"