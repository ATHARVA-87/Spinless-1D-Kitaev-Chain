# BdG Prediction Pipeline - Usage Guide

This guide explains how to use the `main.py` script for training and evaluating Bogoliubov-de Gennes (BdG) prediction models.

## Prerequisites

Before running the script, ensure you have:

1. Python 3.7+ installed
2. PyTorch and required dependencies installed
3. HDF5 data file containing the BdG simulation data
4. Supporting files: `dataset.py`, `model.py`, and `trainer.py` in the same directory

## Installation

```bash
# Install required dependencies
pip install torch numpy h5py tqdm torch_optimizer matplotlib

# Clone or download the repository
git clone https://github.com/your-username/bdg-prediction.git
cd bdg-prediction
```

## Basic Usage

The simplest way to run the pipeline is:

```bash
python main.py --h5_file /path/to/your/data.h5 --output_dir results
```

This will train a model using default parameters and save results to the `results` directory.

## Command-Line Arguments

The script supports various command-line arguments organized into the following categories:

### Data Options

* `--h5_file`: Path to HDF5 file containing the data (required)
* `--train_test_split`: Fraction of data to use for training (default: 0.8)
* `--num_workers`: Number of workers for DataLoader (default: 4)

### Model Options

* `--input_dim`: Input feature dimension (default: 5)
* `--hidden_dim1`: First hidden layer dimension (default: 256)
* `--hidden_dim2`: Second hidden layer dimension (default: 512)
* `--output_dim`: Output dimension (default: 181)
* `--dropout`: Dropout rate for regularization (default: 0.2)

### Training Options

* `--batch_size`: Batch size for training (default: 256)
* `--num_epochs`: Number of epochs to train for (default: 100)
* `--early_stopping`: Patience for early stopping (default: 10)
* `--save_every`: Save model every N epochs (default: 10, 0 to save only best model)
* `--lambda_eig`: Weight for eigenvalue loss (default: 1.0)
* `--lambda_spectral`: Weight for spectral loss (default: 0.2)
* `--huber_delta`: Threshold for Huber loss (default: 1.0)

### Output Options

* `--output_dir`: Directory to save model and logs (default: "output")
* `--experiment_name`: Experiment name for output directory (default: "")

### Execution Options

* `--seed`: Random seed for reproducibility (default: 42)
* `--debug`: Enable debug logging and print HDF5 file structure
* `--no_train`: Skip training and only evaluate
* `--model_path`: Path to pre-trained model to load
* `--device`: Device to use for training and evaluation (choices: "cpu", "cuda", "auto", default: "auto")

## Examples

### Training a New Model

```bash
python main.py \
    --h5_file data/bdg_simulations.h5 \
    --batch_size 128 \
    --num_epochs 200 \
    --early_stopping 15 \
    --lambda_eig 1.5 \
    --experiment_name high_lambda_test
```

### Fine-tuning an Existing Model

```bash
python main.py \
    --h5_file data/bdg_simulations.h5 \
    --model_path output/previous_run/best_model.pth \
    --num_epochs 50 \
    --lambda_eig 1.0 \
    --lambda_spectral 0.3 \
    --experiment_name fine_tuning
```

### Evaluating a Pre-trained Model

```bash
python main.py \
    --h5_file data/bdg_simulations.h5 \
    --model_path output/my_best_model/best_model.pth \
    --no_train \
    --device cpu
```

### Debugging Data Issues

```bash
python main.py \
    --h5_file data/bdg_simulations.h5 \
    --debug \
    --batch_size 16 \
    --num_epochs 2
```

## Output Directory Structure

The script creates a timestamped directory within the specified output directory:

```
output/
└── run_20250509_123456_experiment_name/
    ├── args.json          # Saved command-line arguments
    ├── run.log            # Detailed log file
    ├── best_model.pth     # Best model weights
    ├── model_epoch_10.pth # Checkpoint saved every save_every epochs
    ├── model_epoch_20.pth
    ├── ...
    ├── training_metrics.json  # Training history and metrics
    └── test_metrics.json      # Test evaluation results
```

## Interpreting Results

- Check `training_metrics.json` for training/validation loss curves
- Examine `test_metrics.json` for final model performance
- The `best_model.pth` file contains the model weights with the lowest validation loss

## Advanced Usage

### Custom Data Processing

If your data requires special preprocessing, you may need to modify the `dataset.py` file to handle your specific HDF5 structure.

### Hyperparameter Tuning

For hyperparameter tuning, you can create a shell script that runs `main.py` with different parameters:

```bash
#!/bin/bash

for lr in 0.001 0.0005 0.0001; do
  for lambda in 0.5 1.0 1.5; do
    python main.py \
      --h5_file data/bdg_simulations.h5 \
      --lambda_eig $lambda \
      --experiment_name "lr${lr}_lambda${lambda}"
  done
done
```

## Troubleshooting

- If you encounter memory issues, try reducing the batch size
- For CUDA out-of-memory errors, switch to CPU with `--device cpu`
- If training is unstable, try using a smaller learning rate by modifying the `Trainer` class
- For data loading issues, use the `--debug` flag to inspect your HDF5 file structure

## Further Information

For more details on the model architecture and training process, refer to the comments in `model.py` and `trainer.py`.