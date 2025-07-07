# BdG Machine Learning Pipeline

A PyTorch-based machine learning pipeline for predicting edge localization and Bogoliubov-de Gennes (BdG) eigenvalue spectra from 5D input features.

## Overview

This pipeline is designed to predict:
1. Edge localization (scalar value)
2. BdG eigenvalue spectrum (variable-length array, padded to 180 elements)

From 5D input features:
- L (lattice size)
- mu (chemical potential)
- t (hopping parameter)
- delta (superconducting gap)
- disorder_strength

## Features

- Data loading from HDF5 files with proper masking
- MLP model with GELU activations
- Masked loss function for variable-length eigenvalue arrays
- Out-of-Distribution (OOD) detection
- Mixed precision training for NVIDIA GPUs
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics

## Requirements

- Python 3.10
- PyTorch 2.5.1 with CUDA 12.4
- h5py, numpy, tqdm, torch-optimizer

## Installation

1. Clone this repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

For PyTorch with CUDA support:
```bash
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset Format

The HDF5 file should have the following structure:
- `/train` group with 200,000 samples
- `/test` group with 30,000 samples

Each sample should contain:
- `L`: lattice size (int32)
- `mu`, `t`, `delta`, `disorder_strength`: model parameters (float32)
- `has_mzm`, `topological_invariant`: binary indicators (int32)
- `min_energy`, `edge_localization`: derived measures (float32)
- `spectra/eigenvalues`: variable-length arrays (float64)

## Usage

### Training a Model

```bash
python main.py --h5_file /path/to/data.h5 --output_dir ./output --batch_size 256 --num_epochs 100
```

### Evaluating a Pre-trained Model

```bash
python main.py --h5_file /path/to/data.h5 --no_train --model_path /path/to/best_model.pth
```

### Debugging HDF5 File Structure

```bash
python main.py --h5_file /path/to/data.h5 --debug
```

## Command-Line Arguments

- `--h5_file`: Path to HDF5 file (required)
- `--output_dir`: Directory to save model and logs (default: 'output')
- `--batch_size`: Batch size for training (default: 256)
- `--num_epochs`: Number of epochs to train for (default: 100)
- `--early_stopping`: Patience for early stopping (default: 10)
- `--lambda_eig`: Weight for eigenvalue loss (default: 1.0)
- `--num_workers`: Number of workers for DataLoader (default: 4)
- `--seed`: Random seed for reproducibility (default: 42)
- `--debug`: Print HDF5 file structure
- `--no_train`: Skip training and only evaluate
- `--model_path`: Path to pre-trained model to load

## Code Structure

- `main.py`: Main entry point for the pipeline
- `dataset.py`: Dataset and DataLoader implementation
- `model.py`: Neural network model and loss function
- `trainer.py`: Training and evaluation functions

## Performance

The model is optimized for NVIDIA RTX A5000 GPUs with 24GB memory:
- Mixed precision training for faster computation
- Batch size of 256 for optimal GPU memory utilization
- AdaBound optimizer for fast convergence
- GELU activation for handling negative eigenvalues

## Out-of-Distribution Detection

The pipeline automatically flags test samples that deviate significantly from the training distribution:
- Computes mean and standard deviation of training features
- Flags samples with any feature z-score > 3.0 as OOD
- Reports OOD ratio during testing