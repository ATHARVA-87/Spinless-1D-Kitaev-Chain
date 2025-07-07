import os
import argparse
import torch
import numpy as np
import json
from datetime import datetime

from dataset import get_dataloaders, inspect_h5_file
from model import BdGPredictor
from trainer import Trainer, evaluate_model


def main():
    """Main function to run the BdG prediction pipeline."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="BdG Prediction Pipeline")
    parser.add_argument('--h5_file', type=str, required=True, help='Path to HDF5 file containing the data')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save model and logs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--early_stopping', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--lambda_eig', type=float, default=1.0, help='Weight for eigenvalue loss')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true', help='Print HDF5 file structure')
    parser.add_argument('--no_train', action='store_true', help='Skip training and only evaluate')
    parser.add_argument('--model_path', type=str, help='Path to pre-trained model to load')
    parser.add_argument('--acc_threshold', type=float, default=0.5, help='Threshold for binary accuracy calculation')
    parser.add_argument('--eig_tolerance', type=float, default=0.1, help='Tolerance for eigenvalue accuracy')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory with timestamp
    if not args.no_train:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save command-line arguments
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Inspect HDF5 file if debug flag is set
    if args.debug:
        inspect_h5_file(args.h5_file)
    
    # Get DataLoaders
    print("Creating DataLoaders...")
    train_loader, test_loader = get_dataloaders(
        args.h5_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = BdGPredictor(input_dim=5, hidden_dim1=256, hidden_dim2=512, output_dim=181)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        output_dir=args.output_dir,
        lambda_eig=args.lambda_eig
    )
    
    # Load pre-trained model if specified
    if args.model_path:
        trainer.load_model(args.model_path)
    
    # Train model if not skipping training
    if not args.no_train:
        print("\nStarting training...")
        metrics = trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,  # Using test_loader as validation for simplicity
            num_epochs=args.num_epochs,
            early_stopping_patience=args.early_stopping
        )
        
        # Save training metrics
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Load best model for evaluation
        trainer.load_model("best_model.pth")
    
    # Evaluate model on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(
        model=trainer.model,
        test_loader=test_loader,
        device=device,
        lambda_eig=args.lambda_eig
    )
    
    # Print accuracy metrics summary
    print("\nAccuracy Metrics Summary:")
    print(f"Edge Accuracy: {test_metrics['edge_accuracy']:.4f}")
    print(f"Eigenvalue Accuracy: {test_metrics['eigenvalue_accuracy']:.4f}")
    print(f"Overall Accuracy: {test_metrics['overall_accuracy']:.4f}")
    
    # Save test metrics
    if not args.no_train:
        with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=4)
    
    print("\nDone!")


if __name__ == "__main__":
    main()