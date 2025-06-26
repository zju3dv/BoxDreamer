#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

image_size=224
patch_size=14
use_rgb=True
batch_size=56
CONFIG_NAME="train.yaml" 

Coor="object"

EXP_NAME="boxdreamer_scale_pretrain" 
# ======================================
# 1. Activate the specified Conda environment
# ======================================
# conda env could be set by user
# get the user input

CONDA_ENV_NAME="oneposev3"

# Initialize Conda
# This assumes Conda is installed and the 'conda' command is available
# If Conda is not initialized, you might need to source the appropriate script
. "$(conda info --base)/etc/profile.d/conda.sh"

echo "Activating Conda environment: $CONDA_ENV_NAME"
# conda init bash
# source ~/.bashrc
conda activate "$CONDA_ENV_NAME"

# ======================================
# 2. Prepare to execute the specified Python script
# ======================================

# Get the current working directory
CURRENT_PATH=$(pwd)
echo "Current directory: $CURRENT_PATH"

# Get the number of GPUs available
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Number of GPUs detected: $GPU_COUNT"

# Get the total memory of the first GPU (in MB)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
echo "Memory of each GPU: ${GPU_MEMORY}MB"

# get gpu device name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
echo "GPU type: $GPU_NAME"

# if gpu is A6000, batchsize should  // 2 (debug mode)
# if gpu is 3090 / 4090, batchsize should // 3 (debug mode)
if [ "$GPU_NAME" == "A6000" ]; then
    batch_size=$(($batch_size / 2))
elif [ "$GPU_NAME" == "NVIDIA RTX A6000" ]; then
    batch_size=$(($batch_size / 2))
elif [ "$GPU_NAME" == "NVIDIA RTX 3090" ]; then
    batch_size=$(($batch_size / 3))
elif [ "$GPU_NAME" == "NVIDIA RTX 4090" ]; then
    batch_size=$(($batch_size / 3))
fi

# if gpu is A800, set NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
if [ "$GPU_NAME" == "NVIDIA A800-SXM4-80GB" ]; then
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
fi

CPU_COUNT=$(nproc)
echo "Number of CPUs detected: $CPU_COUNT"

if [ $batch_size -lt $CPU_COUNT ]; then
    WORKERS_NUM=$batch_size
else
    WORKERS_NUM=$CPU_COUNT
fi
echo "Configured workers number: $WORKERS_NUM"

echo "Configured batch size: $batch_size"

# Set the device list based on the number of GPUs
# Generates a list like [0,1,2] for 3 GPUs
DEVICE_LIST=$(seq -s, 0 $(($GPU_COUNT - 1)))
DEVICES="[$DEVICE_LIST]"
echo "Configured devices list: $DEVICES"

# Set the path to the pre-trained checkpoint based on the user's home directory
CKPT_PATH="$HOME/.cache/torch/hub/facebookresearch_dinov2_main"
echo "Pre-trained checkpoint path: $CKPT_PATH"

# ======================================
# 3. Construct and execute the training command
# ======================================

# Define the training script and configuration file
TRAIN_SCRIPT="run.py" 



# first, run env auto check
# python scripts/tools/env-tools/check.py ./ --auto_install --auto_mode --quiet

# Construct the training command using an array to handle spaces and special characters
TRAIN_CMD=(
    python "$TRAIN_SCRIPT"
    --config-name="$CONFIG_NAME"
    exp_name="$EXP_NAME"
    coordinate="$Coor"
    bbox_representation="heatmap"
    use_pretrained=False
    pretrain_name="boxdreamer_scale_pretrain"
    trainer.devices="$DEVICES"
    model.modules.encoder.dino.ckpt_path="$CKPT_PATH"
    # Add additional parameters as needed
    # Example: Set batch size if the training script supports it
    datamodule.batch_size=$batch_size
    datamodule.num_workers=36
    image_size="$image_size"
    patch_size="$patch_size"
    model.modules.use_rgb="$use_rgb"
    model.modules.decoder.num_decoder_layers=12
    datamodule.val_datasets="[LINEMOD]"
    datamodule.train_datasets="[Objaverse, OnePose]"
)

echo "Executing training command: ${TRAIN_CMD[@]}"

# Execute the training command and wait for it to finish
"${TRAIN_CMD[@]}"

echo "Training completed successfully."

# ======================================
# 4. Deactivate the Conda environment
# ======================================

conda deactivate
