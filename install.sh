#!/bin/bash

# BoxDreamer Installation Script
# ==============================
# This script automates the installation of BoxDreamer and its dependencies.
# Requires: CUDA 12.1, Python 3.11, PyTorch 2.5.1

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Symbols
CHECK="✓"
CROSS="✗"
ARROW="→"
STAR="★"
WARNING="⚠"

# Required versions (from README)
REQUIRED_CUDA_VERSION="12.1"
REQUIRED_PYTHON_VERSION="3.11"
REQUIRED_PYTORCH_VERSION="2.5.1"
CONDA_ENV_NAME="boxdreamer"
# Function to print colored messages
print_header() {
    echo -e "\n${PURPLE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║${NC}  ${STAR} ${CYAN}$1${NC}"
    echo -e "${PURPLE}╚════════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    echo -e "${BLUE}${ARROW}${NC} ${1}..."
}

print_success() {
    echo -e "${GREEN}${CHECK}${NC} ${1}"
}

print_error() {
    echo -e "${RED}${CROSS}${NC} ${1}"
}

print_warning() {
    echo -e "${YELLOW}${WARNING}${NC} ${1}"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} ${1}"
}

print_fatal() {
    echo -e "\n${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║${NC}  ${CROSS} ${RED}FATAL ERROR${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${RED}${1}${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect CUDA version
detect_cuda_version() {
    if command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        echo "$CUDA_VERSION"
    else
        echo "none"
    fi
}

# Function to compare version strings
version_compare() {
    if [[ $1 == $2 ]]; then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # Fill empty positions with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]})); then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then
            return 2
        fi
    done
    return 0
}

# Function to validate CUDA version
validate_cuda_version() {
    local detected_cuda=$1
    local required_cuda=$2

    if [ "$detected_cuda" == "none" ]; then
        print_fatal "CUDA is not installed or nvcc is not in PATH"
        echo -e "${YELLOW}Installation requirements:${NC}"
        echo -e "  • CUDA ${required_cuda} or higher (same major version) is required"
        echo -e "  • Download from: ${BLUE}https://developer.nvidia.com/cuda-toolkit-archive${NC}"
        echo ""
        echo -e "${YELLOW}After installing CUDA, make sure to:${NC}"
        echo -e "  1. Add CUDA to your PATH"
        echo -e "  2. Set LD_LIBRARY_PATH (Linux)"
        echo -e "  3. Verify installation: ${CYAN}nvcc --version${NC}"
        return 1
    fi

    local detected_major=$(echo $detected_cuda | cut -d. -f1)
    local detected_minor=$(echo $detected_cuda | cut -d. -f2)
    local required_major=$(echo $required_cuda | cut -d. -f1)
    local required_minor=$(echo $required_cuda | cut -d. -f2)

    print_info "CUDA compatibility check:"
    echo -e "    Required: ${YELLOW}${required_cuda}${NC} (minimum)"
    echo -e "    Detected: ${YELLOW}${detected_cuda}${NC}"

    if [ "$detected_major" -lt "$required_major" ]; then
        print_fatal "CUDA major version is too old"
        echo -e "${RED}Detected:${NC} CUDA ${detected_cuda}"
        echo -e "${RED}Required:${NC} CUDA ${required_cuda} or higher"
        echo ""
        echo -e "${YELLOW}Why this matters:${NC}"
        echo -e "  • PyTorch ${REQUIRED_PYTORCH_VERSION} is built for CUDA ${required_major}.x"
        echo -e "  • CUDA ${detected_major}.x is not compatible with CUDA ${required_major}.x packages"
        echo -e "  • Major version upgrade needed: ${detected_major}.x → ${required_major}.x"
        echo ""
        echo -e "${YELLOW}Solutions:${NC}"
        echo -e "  1. Install CUDA ${required_cuda} or higher: ${BLUE}https://developer.nvidia.com/cuda-toolkit-archive${NC}"
        echo -e "  2. Or use PyTorch built for CUDA ${detected_major}.x (requires manual installation)"
        return 1
    fi

    if [ "$detected_major" -gt "$required_major" ]; then
        print_warning "CUDA major version is newer than required"
        echo -e "    This may cause compatibility issues!"
        echo -e "    ${YELLOW}Detected:${NC} CUDA ${detected_cuda}"
        echo -e "    ${YELLOW}Required:${NC} CUDA ${required_cuda}"
        echo ""
        echo -e "${YELLOW}Recommendation:${NC}"
        echo -e "  • Install CUDA ${required_major}.x for best compatibility"
        echo -e "  • Or manually install PyTorch for CUDA ${detected_major}.x"
        echo ""
        read -p "$(echo -e ${YELLOW}Continue with CUDA ${detected_cuda}? This may fail. \(y/N\): ${NC})" -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation aborted by user"
            return 1
        fi
        print_warning "Proceeding with CUDA ${detected_cuda} at your own risk"
        return 0
    fi

    if [ "$detected_minor" -lt "$required_minor" ]; then
        print_fatal "CUDA minor version is too old"
        echo -e "${RED}Detected:${NC} CUDA ${detected_cuda}"
        echo -e "${RED}Required:${NC} CUDA ${required_cuda} or higher"
        echo ""
        echo -e "${YELLOW}Why this matters:${NC}"
        echo -e "  • PyTorch ${REQUIRED_PYTORCH_VERSION} requires CUDA ${required_cuda} or higher"
        echo -e "  • CUDA ${detected_cuda} may be missing required features"
        echo -e "  • Some operations may fail at runtime"
        echo ""
        echo -e "${YELLOW}Solutions:${NC}"
        echo -e "  1. ${GREEN}Recommended:${NC} Upgrade to CUDA ${required_cuda} or higher"
        echo -e "     Download: ${BLUE}https://developer.nvidia.com/cuda-${required_cuda}-download-archive${NC}"
        echo -e "  2. ${YELLOW}Alternative:${NC} Use PyTorch built for CUDA ${detected_major}.${detected_minor}"
        echo -e "     (Requires manual installation and may have feature limitations)"
        return 1
    fi

    if [ "$detected_minor" -gt "$required_minor" ]; then
        print_success "CUDA ${detected_cuda} is compatible (newer than required ${required_cuda})"
        echo -e "    ${GREEN}✓${NC} CUDA backward compatibility: packages built for ${required_cuda}"
        echo -e "      will run on ${detected_cuda}"
        return 0
    fi

    if [ "$detected_cuda" == "$required_cuda" ]; then
        print_success "CUDA version matches exactly: ${detected_cuda}"
        return 0
    fi

    return 0
}

select_pytorch_cuda_version() {
    local detected_cuda=$1
    local detected_major=$(echo $detected_cuda | cut -d. -f1)
    local detected_minor=$(echo $detected_cuda | cut -d. -f2)


    case "$detected_major" in
        11)
            if [ "$detected_minor" -ge 8 ]; then
                echo "cu118"
            else
                echo "cu117"
            fi
            ;;
        12)
            if [ "$detected_minor" -ge 4 ]; then
                echo "cu124"
            elif [ "$detected_minor" -ge 1 ]; then
                echo "cu121"
            else
                echo "cu118"
            fi
            ;;
        *)
            echo "cu121"
            ;;
    esac
}

check_pytorch_package_availability() {
    local cuda_suffix=$1
    local pytorch_version=$2

    print_step "Checking PyTorch package availability for ${cuda_suffix}"

    local index_url="https://download.pytorch.org/whl/${cuda_suffix}"

    if curl -s -f -I "$index_url" > /dev/null 2>&1; then
        print_success "PyTorch repository for ${cuda_suffix} is accessible"
        return 0
    else
        print_warning "PyTorch repository for ${cuda_suffix} may not be available"
        return 1
    fi
}

# Parse command line arguments
SKIP_CONDA=false
SKIP_SAM2=false
FORCE_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-conda)
            SKIP_CONDA=true
            shift
            ;;
        --skip-sam2)
            SKIP_SAM2=true
            shift
            ;;
        --force)
            FORCE_INSTALL=true
            print_warning "Force install mode enabled - skipping CUDA version check"
            shift
            ;;
        -h|--help)
            echo "BoxDreamer Installation Script"
            echo ""
            echo "Requirements:"
            echo "  • CUDA ${REQUIRED_CUDA_VERSION}"
            echo "  • Python ${REQUIRED_PYTHON_VERSION}"
            echo "  • PyTorch ${REQUIRED_PYTORCH_VERSION}"
            echo ""
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-conda       Skip conda environment creation"
            echo "  --skip-sam2        Skip SAM2 installation"
            echo "  --force            Force installation (skip CUDA check - NOT RECOMMENDED)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Example:"
            echo "  ./install.sh                  # Full installation with checks"
            echo "  ./install.sh --skip-sam2      # Skip optional SAM2"
            echo "  ./install.sh --force          # Force install (may break)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main installation
print_header "BoxDreamer Installation"

print_info "Required versions:"
echo -e "  • CUDA: ${YELLOW}${REQUIRED_CUDA_VERSION}${NC}"
echo -e "  • Python: ${YELLOW}${REQUIRED_PYTHON_VERSION}${NC}"
echo -e "  • PyTorch: ${YELLOW}${REQUIRED_PYTORCH_VERSION}${NC}"
echo ""

# Step 1: Validate CUDA version
print_header "Validating System Requirements"

print_step "Detecting CUDA version"
DETECTED_CUDA=$(detect_cuda_version)

if [ "$DETECTED_CUDA" != "none" ]; then
    print_success "CUDA ${DETECTED_CUDA} detected"
else
    print_error "CUDA not detected"
fi

if [ "$FORCE_INSTALL" = false ]; then
    print_step "Validating CUDA version"
    if ! validate_cuda_version "$DETECTED_CUDA" "$REQUIRED_CUDA_VERSION"; then
        echo -e "\n${RED}Installation aborted due to CUDA version mismatch.${NC}"
        echo -e "${YELLOW}Use --force to bypass this check (NOT RECOMMENDED)${NC}\n"
        exit 1
    fi
    print_success "CUDA version validated: ${DETECTED_CUDA}"
else
    print_warning "CUDA version check bypassed (--force flag)"
    REQUIRED_CUDA_VERSION="$DETECTED_CUDA"
fi

# Convert CUDA version to PyTorch format
CUDA_SUFFIX=$(echo $REQUIRED_CUDA_VERSION | sed 's/\.//g' | sed 's/^/cu/')
print_info "Using PyTorch CUDA suffix: ${CUDA_SUFFIX}"

# Step 2: Check other prerequisites
print_step "Checking for conda"
if ! command_exists conda; then
    print_fatal "conda is not installed"
    echo -e "${YELLOW}Please install Anaconda or Miniconda:${NC}"
    echo -e "  ${BLUE}https://docs.conda.io/en/latest/miniconda.html${NC}\n"
    exit 1
fi
print_success "conda is installed"
print_success "conda environment '$CONDA_ENV_NAME' is activated"

print_step "Checking for git"
if ! command_exists git; then
    print_fatal "git is not installed"
    echo -e "${YELLOW}Please install git:${NC}"
    echo -e "  Ubuntu/Debian: ${CYAN}sudo apt-get install git${NC}"
    echo -e "  CentOS/RHEL: ${CYAN}sudo yum install git${NC}"
    echo -e "  macOS: ${CYAN}brew install git${NC}\n"
    exit 1
fi
print_success "git is installed"

print_step "Checking for wget"
if ! command_exists wget; then
    print_warning "wget is not installed, will try to install it"
    if command_exists apt-get; then
        sudo apt-get update && sudo apt-get install -y wget
    elif command_exists yum; then
        sudo yum install -y wget
    elif command_exists brew; then
        brew install wget
    else
        print_error "Please install wget manually"
        exit 1
    fi
fi
print_success "wget is available"

# Step 3: Create conda environment
if ! $SKIP_CONDA; then
    print_header "Creating Conda Environment"

    print_step "Creating conda environment '$CONDA_ENV_NAME' with Python ${REQUIRED_PYTHON_VERSION}"

    # Check if environment already exists
    if conda env list | grep -q "^$CONDA_ENV_NAME "; then
        print_warning "Environment '$CONDA_ENV_NAME' already exists"
        read -p "$(echo -e ${YELLOW}Do you want to remove and recreate it? \(y/N\): ${NC})" -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_step "Removing existing environment"
            conda env remove -n $CONDA_ENV_NAME -y
            print_success "Removed existing environment"
        else
            print_info "Using existing environment"
        fi
    fi

    if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
        conda create -n $CONDA_ENV_NAME python=${REQUIRED_PYTHON_VERSION} -y
        print_success "Created conda environment '$CONDA_ENV_NAME'"
    fi

    print_step "Activating conda environment"
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV_NAME
    print_success "Activated conda environment '$CONDA_ENV_NAME'"

    # Verify Python version
    PYTHON_VERSION_INSTALLED=$(python --version 2>&1 | awk '{print $2}')
    print_info "Python version: ${PYTHON_VERSION_INSTALLED}"

    if [[ ! $PYTHON_VERSION_INSTALLED =~ ^${REQUIRED_PYTHON_VERSION} ]]; then
        print_warning "Python version mismatch (expected ${REQUIRED_PYTHON_VERSION}, got ${PYTHON_VERSION_INSTALLED})"
    fi
else
    print_info "Skipping conda environment creation"
    print_warning "Make sure you have Python ${REQUIRED_PYTHON_VERSION} activated"
fi

# Step 4: Install uv
print_header "Installing uv Package Manager"

print_step "Installing uv"
pip install -q uv || pip install uv
print_success "uv installed successfully"

UV_VERSION=$(uv --version 2>/dev/null | awk '{print $2}' || echo "unknown")
print_info "uv version: ${UV_VERSION}"

# Step 5: Install PyTorch
print_header "Installing PyTorch ${REQUIRED_PYTORCH_VERSION}"

print_step "Installing PyTorch with CUDA ${REQUIRED_CUDA_VERSION} support"
print_info "This may take several minutes depending on your network speed..."
print_info "PyTorch index: https://download.pytorch.org/whl/${CUDA_SUFFIX}"

if uv pip install torch==${REQUIRED_PYTORCH_VERSION} \
    torchvision==0.20.1 \
    torchaudio==${REQUIRED_PYTORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_SUFFIX}; then
    print_success "PyTorch ${REQUIRED_PYTORCH_VERSION} installed successfully"
else
    print_fatal "PyTorch installation failed"
    echo -e "${YELLOW}Possible reasons:${NC}"
    echo -e "  • Network connection issues"
    echo -e "  • Invalid CUDA version (${CUDA_SUFFIX})"
    echo -e "  • Incompatible PyTorch version"
    echo ""
    echo -e "${YELLOW}Try:${NC}"
    echo -e "  1. Check your internet connection"
    echo -e "  2. Verify CUDA version: ${CYAN}nvcc --version${NC}"
    echo -e "  3. Visit PyTorch website: ${BLUE}https://pytorch.org${NC}\n"
    exit 1
fi

# Verify PyTorch installation
print_step "Verifying PyTorch installation"
python << 'EOF'
import sys
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: CUDA is not available in PyTorch!")
        sys.exit(1)
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "PyTorch verification passed"
else
    print_fatal "PyTorch verification failed"
    echo -e "${RED}PyTorch was installed but CUDA is not available.${NC}"
    echo -e "${YELLOW}This usually means:${NC}"
    echo -e "  • CUDA driver is not installed"
    echo -e "  • CUDA runtime mismatch"
    echo -e "  • GPU is not detected"
    echo ""
    echo -e "${YELLOW}Check:${NC}"
    echo -e "  1. NVIDIA driver: ${CYAN}nvidia-smi${NC}"
    echo -e "  2. CUDA installation: ${CYAN}nvcc --version${NC}"
    echo -e "  3. LD_LIBRARY_PATH includes CUDA libraries\n"
    exit 1
fi

# Step 6: Install PyTorch3D
print_header "Installing PyTorch3D"

print_step "Installing PyTorch3D from source"
print_info "This may take 5-10 minutes to compile..."

if pip install -q "git+https://github.com/facebookresearch/pytorch3d.git"; then
    print_success "PyTorch3D installed successfully"
else
    print_fatal "PyTorch3D installation failed"
    echo -e "${YELLOW}Common issues:${NC}"
    echo -e "  • Missing build dependencies (gcc, g++, cmake)"
    echo -e "  • CUDA development files not found"
    echo ""
    echo -e "${YELLOW}On Ubuntu/Debian, try:${NC}"
    echo -e "  ${CYAN}sudo apt-get install build-essential cmake${NC}\n"
    exit 1
fi

# Install flash_attn
print_header "Installing Flash Attention"
print_step "Installing Flash Attention"
if pip install flash_attn; then
    print_success "Flash Attention installed successfully"
else
    print_fatal "Flash Attention installation failed"
    echo -e "${YELLOW}Check the error messages above for details.${NC}"
    exit 1
fi

# Install xformers (for torch 2.5.1)
print_header "Installing XFormers"
print_step "Installing XFormers"
if pip install xformers==0.0.28.post3; then
    print_success "XFormers installed successfully"
else
    print_fatal "XFormers installation failed"
    echo -e "${YELLOW}Check the error messages above for details.${NC}"
    exit 1
fi


# Step 7: Install project dependencies
print_header "Installing Project Dependencies"

if [ ! -f "requirements.txt" ]; then
    print_fatal "requirements.txt not found"
    echo -e "${YELLOW}Make sure you're running this script from the project root directory.${NC}\n"
    exit 1
fi

print_step "Installing dependencies from requirements.txt"
print_info "Processing $(wc -l < requirements.txt) requirements..."

if uv pip install -r requirements.txt; then
    print_success "All dependencies installed successfully"
else
    print_fatal "Failed to install some dependencies"
    echo -e "${YELLOW}Check the error messages above for details.${NC}"
    echo -e "${YELLOW}You may need to install some dependencies manually.${NC}\n"
    exit 1
fi

# Step 8: Install BoxDreamer
print_header "Installing BoxDreamer"

print_step "Installing BoxDreamer in editable mode"

if uv pip install -e .; then
    print_success "BoxDreamer installed successfully"
else
    print_fatal "BoxDreamer installation failed"
    echo -e "${YELLOW}Check that setup.py or pyproject.toml exists and is valid.${NC}\n"
    exit 1
fi

# Verify CLI installation
print_step "Verifying CLI installation"
if command_exists boxdreamer-cli; then
    CLI_PATH=$(which boxdreamer-cli)
    print_success "boxdreamer-cli available at: ${CLI_PATH}"
else
    print_warning "boxdreamer-cli not found in PATH"
    echo -e "${YELLOW}You may need to restart your shell or add the script directory to PATH${NC}"
fi

# Step 9: Initialize git submodules
print_header "Initializing Git Submodules"

print_step "Checking git submodules"
if [ ! -f ".gitmodules" ]; then
    print_warning ".gitmodules not found, skipping submodule initialization"
else
    print_step "Updating git submodules (dust3r, mast3r, vggsfm)"
    if git submodule update --init --recursive; then
        print_success "Git submodules initialized successfully"

        # List initialized submodules
        print_info "Initialized submodules:"
        git submodule status | while read line; do
            echo "    • $line"
        done
    else
        print_error "Failed to initialize git submodules"
        echo -e "${YELLOW}This may cause issues if you need these modules.${NC}"
    fi
fi

# Step 10: Download model checkpoints
print_header "Downloading Model Checkpoints"

mkdir -p weights
cd weights

# Download DUSt3R checkpoint
print_step "Downloading DUSt3R checkpoint (~2.2GB)"
DUST3R_URL="https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
DUST3R_FILE="DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

if [ -f "$DUST3R_FILE" ]; then
    print_info "DUSt3R checkpoint already exists, skipping download"
else
    if wget -q --show-progress "$DUST3R_URL" -O "$DUST3R_FILE"; then
        print_success "DUSt3R checkpoint downloaded"
        FILE_SIZE=$(du -h "$DUST3R_FILE" | cut -f1)
        print_info "File size: ${FILE_SIZE}"
    else
        print_error "Failed to download DUSt3R checkpoint"
        echo -e "${YELLOW}You can download it manually from:${NC}"
        echo -e "  ${BLUE}${DUST3R_URL}${NC}"
    fi
fi

# Download Grounding DINO checkpoint
print_step "Downloading Grounding DINO checkpoint (~600MB)"
GDINO_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
GDINO_FILE="groundingdino_swint_ogc.pth"

if [ -f "$GDINO_FILE" ]; then
    print_info "Grounding DINO checkpoint already exists, skipping download"
else
    if wget -q --show-progress "$GDINO_URL" -O "$GDINO_FILE"; then
        print_success "Grounding DINO checkpoint downloaded"
        FILE_SIZE=$(du -h "$GDINO_FILE" | cut -f1)
        print_info "File size: ${FILE_SIZE}"
    else
        print_error "Failed to download Grounding DINO checkpoint"
        echo -e "${YELLOW}You can download it manually from:${NC}"
        echo -e "  ${BLUE}${GDINO_URL}${NC}"
    fi
fi

cd ..

print_step "Listing downloaded checkpoints"
if [ -d "weights" ] && [ "$(ls -A weights)" ]; then
    ls -lh weights/ | grep "\.pth$" | awk '{print "    • " $9 " (" $5 ")"}'
    print_success "Checkpoints ready"
else
    print_warning "No checkpoints found in weights/"
fi

# Step 11: Install SAM2 (optional)
if ! $SKIP_SAM2; then
    print_header "Installing SAM2 (Optional)"

    print_warning "SAM2 installation is optional but recommended for demo usage"
    print_info "SAM2 requires:"
    echo -e "    • CUDA ${REQUIRED_CUDA_VERSION}"
    echo -e "    • Python ${REQUIRED_PYTHON_VERSION}"
    echo -e "    • PyTorch ${REQUIRED_PYTORCH_VERSION}"
    echo ""

    read -p "$(echo -e ${YELLOW}Do you want to install SAM2? \(Y/n\): ${NC})" -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        print_step "Installing SAM2 for CUDA ${REQUIRED_CUDA_VERSION}"

        # Construct SAM2 wheel URL based on detected CUDA version
        SAM2_WHEEL="https://github.com/MiroPsota/torch_packages_builder/releases/download/SAM_2-1.0%2Bc2ec8e1/SAM_2-1.0%2Bc2ec8e1pt${REQUIRED_PYTORCH_VERSION}${CUDA_SUFFIX}-cp311-cp311-linux_x86_64.whl"

        print_info "Downloading from: ${SAM2_WHEEL}"

        if pip install "$SAM2_WHEEL"; then
            print_success "SAM2 installed successfully"

            # Install additional dependencies for SAM2
            print_step "Installing additional dependencies for SAM2"
            if pip install decord pyqt5 gradio transformers; then
                print_success "SAM2 dependencies installed"
            else
                print_warning "Some SAM2 dependencies failed to install"
            fi
        else
            print_warning "SAM2 installation failed"
            echo -e "${YELLOW}This is optional and won't affect core functionality.${NC}"
            echo -e "${YELLOW}You can install it manually later if needed.${NC}"
            echo ""
            echo -e "${CYAN}Manual installation:${NC}"
            echo -e "  ${CYAN}pip install ${SAM2_WHEEL}${NC}"
        fi
    else
        print_info "Skipping SAM2 installation"
    fi
else
    print_info "Skipping SAM2 installation (--skip-sam2 flag)"
fi

# Step 12: Setup VS Code environment (optional)
print_header "Setting up Development Environment"

if [ ! -f ".env" ]; then
    print_step "Creating .env file for VS Code Python environment"
    echo "PYTHONPATH=three/dust3r" > .env
    print_success ".env file created"
else
    print_info ".env file already exists"
    cat .env | while read line; do
        echo "    $line"
    done
fi

# Step 13: Run validation tests
print_header "Running Validation Tests"

print_step "Testing Python imports"
python << 'EOF'
import sys

def test_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        __import__(module_name)
        print(f"  ✓ {display_name}")
        return True
    except ImportError as e:
        print(f"  ✗ {display_name}: {e}")
        return False

all_passed = True
all_passed &= test_import("torch", "PyTorch")
all_passed &= test_import("torchvision")
all_passed &= test_import("pytorch3d", "PyTorch3D")
all_passed &= test_import("cv2", "OpenCV")
all_passed &= test_import("PIL", "Pillow")
all_passed &= test_import("numpy")
all_passed &= test_import("matplotlib")
all_passed &= test_import("loguru")
all_passed &= test_import("omegaconf")
all_passed &= test_import("open3d")

if not all_passed:
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "All critical imports successful"
else
    print_warning "Some imports failed - check error messages above"
fi

# Final summary
print_header "Installation Summary"

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ${CHECK} ${GREEN}BoxDreamer Installation Complete!${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}Installation Details:${NC}"
echo -e "  • Python: ${YELLOW}$(python --version 2>&1 | awk '{print $2}')${NC}"
echo -e "  • CUDA: ${YELLOW}${DETECTED_CUDA}${NC}"
echo -e "  • PyTorch: ${YELLOW}$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'unknown')${NC}"
echo -e "  • Installation Path: ${YELLOW}$(pwd)${NC}"
echo ""

echo -e "${CYAN}Quick Start:${NC}"
echo -e "  ${BLUE}1.${NC} Activate environment:"
echo -e "     ${CYAN}conda activate boxdreamer${NC}"
echo ""
echo -e "  ${BLUE}2.${NC} Verify installation:"
echo -e "     ${CYAN}boxdreamer-cli --help${NC}"
echo ""
echo -e "  ${BLUE}3.${NC} Run a demo:"
echo -e "     ${CYAN}boxdreamer-cli --video test.mp4${NC}"
echo ""
echo -e "  ${BLUE}4.${NC} With Grounding DINO:"
echo -e "     ${CYAN}boxdreamer-cli --video test.mp4 --use_grounding_dino --text_prompt \"a cat\"${NC}"
echo ""

echo -e "${CYAN}Useful Commands:${NC}"
echo -e "  • Check GPU: ${YELLOW}nvidia-smi${NC}"
echo -e "  • Test PyTorch CUDA: ${YELLOW}python -c 'import torch; print(torch.cuda.is_available())'${NC}"
echo -e "  • List installed packages: ${YELLOW}pip list${NC}"
echo ""

echo -e "${CYAN}Documentation:${NC}"
echo -e "  • README: ${BLUE}cat README.md${NC}"
echo -e "  • GitHub: ${BLUE}https://github.com/yourusername/boxdreamer${NC}"
echo ""

if [ -d "weights" ] && [ "$(ls -A weights)" ]; then
    echo -e "${GREEN}${CHECK}${NC} Model checkpoints are ready in ${YELLOW}./weights/${NC}"
else
    echo -e "${YELLOW}${WARNING}${NC} Some model checkpoints may be missing"
fi

echo ""
print_success "Happy researching! 🚀"
echo ""
