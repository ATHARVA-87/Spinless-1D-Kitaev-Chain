#!/bin/bash

# Check if H5 file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_h5_file> [output_directory]"
    exit 1
fi

H5_FILE=$1
OUTPUT_DIR="${2:-output}"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv bdg_venv
source bdg_venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# Debug HDF5 file structure
echo "Inspecting HDF5 file structure..."
python main.py --h5_file "$H5_FILE" --debug

# Run training
echo "Starting training pipeline..."
python main.py \
    --h5_file "$H5_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 256 \
    --num_epochs 100 \
    --early_stopping 10 \
    --lambda_eig 1.0 \
    --num_workers 4 \
    --seed 42

echo "Training complete! Results saved to $OUTPUT_DIR"