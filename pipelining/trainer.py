import os
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_optimizer import AdaBound

from model import BdGPredictor, masked_mse_loss, compute_accuracy_metrics


class Trainer:
    """Trainer class for BdG prediction model."""
    
    def __init__(self, model: BdGPredictor, device: torch.device, 
                 output_dir: str, lambda_eig: float = 1.0):
        """
        Initialize trainer.
        
        Args:
            model: BdGPredictor model
            device: Device to train on
            output_dir: Directory to save model and logs
            lambda_eig: Weight for eigenvalue loss
        """
        self.model = model.to(device)
        self.device = device
        self.output_dir = output_dir
        self.lambda_eig = lambda_eig
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdaBound(
            self.model.parameters(),
            lr=1e-3,
            final_lr=0.1,
            weight_decay=1e-4
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Initialize best loss for early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_edge_loc_loss = 0.0
        total_eig_loss = 0.0
        total_edge_accuracy = 0.0
        total_eig_accuracy = 0.0
        total_overall_accuracy = 0.0
        
        # Use tqdm for progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                predictions = self.model(features)
                loss, edge_loc_loss, eig_loss = masked_mse_loss(
                    predictions, targets, mask, self.lambda_eig
                )
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Compute accuracy metrics
            with torch.no_grad():
                accuracy_metrics = compute_accuracy_metrics(
                    predictions, targets, mask,
                    threshold=0.5,
                    eigenvalue_tolerance=0.1
                )
            
            # Update metrics
            total_loss += loss.item()
            total_edge_loc_loss += edge_loc_loss.item()
            total_eig_loss += eig_loss.item()
            total_edge_accuracy += accuracy_metrics['edge_binary_accuracy']
            total_eig_accuracy += accuracy_metrics.get('eigenvalue_accuracy', 0.0)
            total_overall_accuracy += accuracy_metrics['overall_accuracy']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'edge_loc': f"{edge_loc_loss.item():.4f}",
                'eig': f"{eig_loss.item():.4f}"
            })
        
        # Calculate average losses and accuracies
        avg_loss = total_loss / len(train_loader)
        avg_edge_loc_loss = total_edge_loc_loss / len(train_loader)
        avg_eig_loss = total_eig_loss / len(train_loader)
        avg_edge_accuracy = total_edge_accuracy / len(train_loader)
        avg_eig_accuracy = total_eig_accuracy / len(train_loader)
        avg_overall_accuracy = total_overall_accuracy / len(train_loader)
        
        return {
            'loss': avg_loss,
            'edge_loc_loss': avg_edge_loc_loss,
            'eig_loss': avg_eig_loss,
            'edge_accuracy': avg_edge_accuracy,
            'eigenvalue_accuracy': avg_eig_accuracy,
            'overall_accuracy': avg_overall_accuracy
        }
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_edge_loc_loss = 0.0
        total_eig_loss = 0.0
        total_edge_accuracy = 0.0
        total_eig_accuracy = 0.0
        total_overall_accuracy = 0.0
        
        # Use tqdm for progress bar
        pbar = tqdm(val_loader, desc="Validating")
        
        with torch.no_grad():
            for batch in pbar:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Forward pass
                predictions = self.model(features)
                loss, edge_loc_loss, eig_loss = masked_mse_loss(
                    predictions, targets, mask, self.lambda_eig
                )
                
                # Compute accuracy metrics
                accuracy_metrics = compute_accuracy_metrics(
                    predictions, targets, mask,
                    threshold=0.5,
                    eigenvalue_tolerance=0.1
                )
                
                # Update metrics
                total_loss += loss.item()
                total_edge_loc_loss += edge_loc_loss.item()
                total_eig_loss += eig_loss.item()
                total_edge_accuracy += accuracy_metrics['edge_binary_accuracy']
                total_eig_accuracy += accuracy_metrics.get('eigenvalue_accuracy', 0.0)
                total_overall_accuracy += accuracy_metrics['overall_accuracy']
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'edge_loc': f"{edge_loc_loss.item():.4f}",
                    'eig': f"{eig_loss.item():.4f}"
                })
        
        # Calculate average losses and accuracies
        avg_loss = total_loss / len(val_loader)
        avg_edge_loc_loss = total_edge_loc_loss / len(val_loader)
        avg_eig_loss = total_eig_loss / len(val_loader)
        avg_edge_accuracy = total_edge_accuracy / len(val_loader)
        avg_eig_accuracy = total_eig_accuracy / len(val_loader)
        avg_overall_accuracy = total_overall_accuracy / len(val_loader)
        
        return {
            'loss': avg_loss,
            'edge_loc_loss': avg_edge_loc_loss,
            'eig_loss': avg_eig_loss,
            'edge_accuracy': avg_edge_accuracy,
            'eigenvalue_accuracy': avg_eig_accuracy,
            'overall_accuracy': avg_overall_accuracy
        }
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
             val_loader: Optional[torch.utils.data.DataLoader] = None,
             num_epochs: int = 100, early_stopping_patience: int = 10) -> Dict[str, list]:
        """
        Train model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Dictionary with training and validation metrics
        """
        # Store metrics
        metrics = {
            'train_loss': [],
            'train_edge_loc_loss': [],
            'train_eig_loss': [],
            'train_edge_accuracy': [],
            'train_eigenvalue_accuracy': [],
            'train_overall_accuracy': [],
            'val_loss': [],
            'val_edge_loc_loss': [],
            'val_eig_loss': [],
            'val_edge_accuracy': [],
            'val_eigenvalue_accuracy': [],
            'val_overall_accuracy': [],
            'epoch_time': []
        }
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train for one epoch
            epoch_start_time = time.time()
            train_metrics = self.train_epoch(train_loader)
            epoch_end_time = time.time()
            
            # Log training metrics
            metrics['train_loss'].append(train_metrics['loss'])
            metrics['train_edge_loc_loss'].append(train_metrics['edge_loc_loss'])
            metrics['train_eig_loss'].append(train_metrics['eig_loss'])
            metrics['train_edge_accuracy'].append(train_metrics['edge_accuracy'])
            metrics['train_eigenvalue_accuracy'].append(train_metrics['eigenvalue_accuracy'])
            metrics['train_overall_accuracy'].append(train_metrics['overall_accuracy'])
            metrics['epoch_time'].append(epoch_end_time - epoch_start_time)
            
            print(f"Training Loss: {train_metrics['loss']:.4f}, "
                  f"Edge Loc Loss: {train_metrics['edge_loc_loss']:.4f}, "
                  f"Eig Loss: {train_metrics['eig_loss']:.4f}, "
                  f"Edge Acc: {train_metrics['edge_accuracy']:.4f}, "
                  f"Eig Acc: {train_metrics['eigenvalue_accuracy']:.4f}, "
                  f"Overall Acc: {train_metrics['overall_accuracy']:.4f}, "
                  f"Time: {epoch_end_time - epoch_start_time:.2f}s")
            
            # Validate if val_loader is provided
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                # Log validation metrics
                metrics['val_loss'].append(val_metrics['loss'])
                metrics['val_edge_loc_loss'].append(val_metrics['edge_loc_loss'])
                metrics['val_eig_loss'].append(val_metrics['eig_loss'])
                metrics['val_edge_accuracy'].append(val_metrics['edge_accuracy'])
                metrics['val_eigenvalue_accuracy'].append(val_metrics['eigenvalue_accuracy'])
                metrics['val_overall_accuracy'].append(val_metrics['overall_accuracy'])
                
                print(f"Validation Loss: {val_metrics['loss']:.4f}, "
                      f"Edge Loc Loss: {val_metrics['edge_loc_loss']:.4f}, "
                      f"Eig Loss: {val_metrics['eig_loss']:.4f}, "
                      f"Edge Acc: {val_metrics['edge_accuracy']:.4f}, "
                      f"Eig Acc: {val_metrics['eigenvalue_accuracy']:.4f}, "
                      f"Overall Acc: {val_metrics['overall_accuracy']:.4f}")
                
                # Update learning rate based on validation loss
                self.scheduler.step(val_metrics['loss'])
                
                # Early stopping
                if val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    self.patience_counter = 0
                    
                    # Save best model
                    self.save_model("best_model.pth")
                    print(f"New best model saved! Loss: {self.best_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"Early stopping counter: {self.patience_counter}/{early_stopping_patience}")
                    
                    if self.patience_counter >= early_stopping_patience:
                        print("Early stopping triggered!")
                        break
            else:
                # No validation set, save model every epoch
                self.save_model(f"model_epoch_{epoch}.pth")
        
        return metrics
    
    def save_model(self, filename: str) -> None:
        """
        Save model to disk.
        
        Args:
            filename: Name of file to save model to
        """
        model_path = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename: str) -> None:
        """
        Load model from disk.
        
        Args:
            filename: Name of file to load model from
        """
        model_path = os.path.join(self.output_dir, filename)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Model loaded from {model_path}")


def evaluate_model(model: BdGPredictor, test_loader: torch.utils.data.DataLoader, 
                  device: torch.device, lambda_eig: float = 1.0) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: BdGPredictor model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        lambda_eig: Weight for eigenvalue loss
        
    Returns:
        Dictionary with test metrics
    """
    model.eval()
    total_loss = 0.0
    total_edge_loc_loss = 0.0
    total_eig_loss = 0.0
    total_edge_accuracy = 0.0
    total_eig_accuracy = 0.0
    total_overall_accuracy = 0.0
    ood_count = 0
    sample_count = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch in pbar:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['mask'].to(device)
            is_ood = batch['is_ood']
            
            # Update OOD count
            ood_count += torch.sum(is_ood).item()
            sample_count += len(is_ood)
            
            # Forward pass
            predictions = model(features)
            loss, edge_loc_loss, eig_loss = masked_mse_loss(
                predictions, targets, mask, lambda_eig
            )
            
            # Compute accuracy metrics
            accuracy_metrics = compute_accuracy_metrics(
                predictions, targets, mask,
                threshold=0.5,
                eigenvalue_tolerance=0.1
            )
            
            # Update metrics
            total_loss += loss.item()
            total_edge_loc_loss += edge_loc_loss.item()
            total_eig_loss += eig_loss.item()
            total_edge_accuracy += accuracy_metrics['edge_binary_accuracy']
            total_eig_accuracy += accuracy_metrics.get('eigenvalue_accuracy', 0.0)
            total_overall_accuracy += accuracy_metrics['overall_accuracy']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'edge_loc': f"{edge_loc_loss.item():.4f}",
                'eig': f"{eig_loss.item():.4f}"
            })
    
    # Calculate average losses and accuracies
    avg_loss = total_loss / len(test_loader)
    avg_edge_loc_loss = total_edge_loc_loss / len(test_loader)
    avg_eig_loss = total_eig_loss / len(test_loader)
    avg_edge_accuracy = total_edge_accuracy / len(test_loader)
    avg_eig_accuracy = total_eig_accuracy / len(test_loader)
    avg_overall_accuracy = total_overall_accuracy / len(test_loader)
    ood_ratio = ood_count / sample_count if sample_count > 0 else 0.0
    
    # Print test results
    print(f"\nTest Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Edge Localization MSE: {avg_edge_loc_loss:.4f}")
    print(f"Eigenvalue MSE: {avg_eig_loss:.4f}")
    print(f"Edge Accuracy: {avg_edge_accuracy:.4f}")
    print(f"Eigenvalue Accuracy: {avg_eig_accuracy:.4f}")
    print(f"Overall Accuracy: {avg_overall_accuracy:.4f}")
    print(f"OOD Ratio: {ood_ratio:.4f} ({ood_count}/{sample_count})")
    
    return {
        'loss': avg_loss,
        'edge_loc_loss': avg_edge_loc_loss,
        'eig_loss': avg_eig_loss,
        'edge_accuracy': avg_edge_accuracy,
        'eigenvalue_accuracy': avg_eig_accuracy,
        'overall_accuracy': avg_overall_accuracy,
        'ood_ratio': ood_ratio
    }