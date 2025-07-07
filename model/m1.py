"""
Bogoliubov-de Gennes (BdG) Eigenvalue and Edge Localization Prediction Pipeline

This script implements a machine learning pipeline to predict the eigenvalues and edge localization
of Bogoliubov-de Gennes (BdG) Hamiltonians for superconducting systems, potentially with disorder.

Key components:
1. Custom Dataset for efficient loading of HDF5 data
2. MLP model architecture for prediction
3. Masked loss function to handle variable-length outputs
4. Out-of-distribution (OOD) detection
5. Training and evaluation pipeline

References:
- BdG formalism: Tinkham, M. (2004). Introduction to Superconductivity. Dover Publications.
- Masked loss functions: Vaswani et al. (2017). "Attention Is All You Need"
- OOD detection: Hendrycks & Gimpel (2016). "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks"
- HDF5 with PyTorch: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import numpy as np
from tqdm import tqdm
import sys
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BdGDataset(Dataset):
    """
    Custom Dataset for loading Bogoliubov-de Gennes (BdG) Hamiltonian data from HDF5 files.
    
    The dataset provides:
    - Features: System size (L), chemical potential (mu), hopping (t), pairing (delta), disorder strength
    - Targets: Edge localization and eigenvalues
    
    References:
    - BdG formalism: Tinkham, M. (2004). Introduction to Superconductivity.
    - HDF5 in PyTorch: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """
    def __init__(self, hdf5_file, group):
        """
        Initialize the BdG dataset.
        
        Args:
            hdf5_file (str): Path to the HDF5 file containing the data
            group (str): Group within the HDF5 file to use (e.g., '/train', '/test')
        """
        self.file = h5py.File(hdf5_file, 'r')
        self.group = self.file[group]
        self.keys = list(self.group.keys())
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.keys)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (features, edge_localization, padded_eigenvalues, mask)
                - features: Tensor of shape (5,) containing L, mu, t, delta, disorder_strength
                - edge_localization: Scalar tensor with the edge localization value
                - padded_eigenvalues: Tensor of shape (180,) with eigenvalues, padded as needed
                - mask: Tensor of shape (180,) with 1s for valid eigenvalues and 0s for padding
        """
        sample = self.group[self.keys[idx]]
        
        # Extract features: L, mu, t, delta, disorder_strength
        features = torch.tensor([
            sample['L'][()], sample['mu'][()], sample['t'][()], 
            sample['delta'][()], sample['disorder_strength'][()]
        ], dtype=torch.float32)
        
        # Extract targets
        edge_loc = torch.tensor(sample['edge_localization'][()], dtype=torch.float32)
        eigenvalues = torch.tensor(sample['eigenvalues'][:], dtype=torch.float32)
        L = int(sample['L'][()])
        
        # Pad eigenvalues to fixed length (180)
        # The actual number of eigenvalues is 2*L, so we pad with zeros for the rest
        padded_eigenvalues = torch.zeros(180, dtype=torch.float32)
        padded_eigenvalues[:2*L] = eigenvalues
        
        # Create mask to indicate valid values (1) and padding (0)
        mask = torch.zeros(180, dtype=torch.float32)
        mask[:2*L] = 1.0
        
        return features, edge_loc, padded_eigenvalues, mask
    
    def close(self):
        """Close the HDF5 file to free resources."""
        self.file.close()


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for predicting BdG eigenvalues and edge localization.
    
    Architecture:
    - Input layer: 5 features (L, mu, t, delta, disorder_strength)
    - Hidden layers: 256 and 512 neurons with GELU activation
    - Output layer: 181 values (1 for edge_loc + 180 for eigenvalues)
    
    References:
    - GELU activation: Hendrycks & Gimpel (2016). "Gaussian Error Linear Units"
    """
    def __init__(self):
        """Initialize the MLP model architecture."""
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 181)  # 1 for edge_loc + 180 for eigenvalues
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input features of shape (batch_size, 5)
            
        Returns:
            Tensor: Predictions of shape (batch_size, 181)
        """
        return self.layers(x)


def compute_loss(predictions, edge_loc_target, eigen_target, mask, lambda_weight=1.0):
    """
    Compute the masked loss function, combining edge localization loss and masked eigenvalue loss.
    
    Args:
        predictions (Tensor): Model predictions, shape (batch_size, 181)
        edge_loc_target (Tensor): Ground truth edge localization, shape (batch_size,)
        eigen_target (Tensor): Ground truth eigenvalues, shape (batch_size, 180)
        mask (Tensor): Mask for valid eigenvalues, shape (batch_size, 180)
        lambda_weight (float): Weight for eigenvalue loss term, default=1.0
    
    Returns:
        Tensor: Combined loss value
        
    References:
    - Masked loss: Vaswani et al. (2017). "Attention Is All You Need"
    """
    # Split predictions into edge localization and eigenvalues
    pred_edge_loc = predictions[:, 0]
    pred_eigen = predictions[:, 1:]
    
    # MSE loss for edge localization
    edge_loss = nn.MSELoss()(pred_edge_loc, edge_loc_target)
    
    # Masked MSE loss for eigenvalues (only consider non-padded values)
    mask = mask.to(predictions.device)
    # Normalize by sum of mask to account for variable number of valid eigenvalues
    eigen_loss = torch.mean(((pred_eigen - eigen_target) ** 2) * mask) / (mask.sum() + 1e-8)
    
    # Combine losses with weighting factor
    return edge_loss + lambda_weight * eigen_loss


def compute_ood_stats(dataset):
    """
    Compute mean and standard deviation statistics for out-of-distribution detection.
    
    Args:
        dataset (BdGDataset): Dataset to compute statistics from
        
    Returns:
        tuple: (mean, std) arrays of shape (5,) for the 5 input features
        
    References:
    - OOD detection: Hendrycks & Gimpel (2016). "A Baseline for Detecting Misclassified and 
      Out-of-Distribution Examples in Neural Networks"
    """
    features = []
    for i in range(len(dataset)):
        feat, _, _, _ = dataset[i]
        features.append(feat.numpy())
    features = np.stack(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return mean, std


def is_ood(features, mean, std, threshold=3.0):
    """
    Check if a sample is out-of-distribution based on z-score thresholding.
    
    Args:
        features (numpy.ndarray): Input features to check
        mean (numpy.ndarray): Mean of training distribution
        std (numpy.ndarray): Standard deviation of training distribution
        threshold (float): Z-score threshold, default=3.0
        
    Returns:
        bool: True if the sample is OOD, False otherwise
    """
    # Calculate normalized distance (z-score) from distribution center
    z_scores = np.abs((features - mean) / (std + 1e-8))
    # Sample is OOD if any feature exceeds the threshold
    return np.any(z_scores > threshold)


def train_model(model, train_loader, val_loader, checkpoint_dir="./", epochs=100, patience=10):
    """
    Train the MLP model with early stopping and learning rate scheduling.
    
    Args:
        model (MLP): The model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        checkpoint_dir (str): Directory to save model checkpoints
        epochs (int): Maximum number of training epochs
        patience (int): Early stopping patience
        
    Returns:
        dict: Training history with losses
        
    References:
    - Early stopping: Prechelt, L. (1998). "Early Stopping - But When?"
    - Learning rate scheduling: Smith, L. N. (2017). "Cyclical Learning Rates for Training Neural Networks"
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # Initialize optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"Training on device: {device}")
    print(f"Model will be saved to: {checkpoint_path}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, edge_loc, eigen, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features, edge_loc, eigen, mask = (
                features.to(device), edge_loc.to(device), eigen.to(device), mask.to(device)
            )
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(features)
            loss = compute_loss(predictions, edge_loc, eigen, mask)
            
            # Backward pass and optimization
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, edge_loc, eigen, mask in val_loader:
                features, edge_loc, eigen, mask = (
                    features.to(device), edge_loc.to(device), eigen.to(device), mask.to(device)
                )
                predictions = model(features)
                val_loss += compute_loss(predictions, edge_loc, eigen, mask).item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Save losses to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping and model checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            print(f"Saving best model to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return history


def evaluate_model(model, test_loader, mean, std, checkpoint_path='best_model.pth'):
    """
    Evaluate the trained model on test data with OOD detection.
    
    Args:
        model (MLP): The model to evaluate
        test_loader (DataLoader): DataLoader for test data
        mean (numpy.ndarray): Mean of training distribution for OOD detection
        std (numpy.ndarray): Standard deviation of training distribution for OOD detection
        checkpoint_path (str): Path to the best model checkpoint
        
    Returns:
        tuple: (test_loss, ood_count, total_samples, metrics_dict)
    """
    # Load best model if available
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Model checkpoint not found at {checkpoint_path}")
    
    model.eval()
    test_loss = 0
    ood_count = 0
    edge_loc_mse = 0
    eigenvalues_mse = 0
    
    with torch.no_grad():
        for features, edge_loc, eigen, mask in tqdm(test_loader, desc="Evaluating"):
            features, edge_loc, eigen, mask = (
                features.to(device), edge_loc.to(device), eigen.to(device), mask.to(device)
            )
            
            # Forward pass
            predictions = model(features)
            loss = compute_loss(predictions, edge_loc, eigen, mask)
            test_loss += loss.item()
            
            # Calculate component losses
            pred_edge_loc = predictions[:, 0]
            pred_eigen = predictions[:, 1:]
            edge_loc_mse += nn.MSELoss()(pred_edge_loc, edge_loc).item()
            masked_eigen_mse = torch.mean(((pred_eigen - eigen) ** 2) * mask) / (mask.sum() + 1e-8)
            eigenvalues_mse += masked_eigen_mse.item()
            
            # OOD detection
            features_np = features.cpu().numpy()
            for feat in features_np:
                if is_ood(feat, mean, std):
                    ood_count += 1
    
    # Calculate average metrics
    total_samples = len(test_loader.dataset)
    test_loss /= len(test_loader)
    edge_loc_mse /= len(test_loader)
    eigenvalues_mse /= len(test_loader)
    
    # Print evaluation results
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Edge Localization MSE: {edge_loc_mse:.6f}")
    print(f"Eigenvalues MSE: {eigenvalues_mse:.6f}")
    print(f"OOD Samples: {ood_count}/{total_samples} ({100*ood_count/total_samples:.2f}%)")
    
    # Return metrics dictionary for further analysis
    metrics = {
        'test_loss': test_loss,
        'edge_loc_mse': edge_loc_mse,
        'eigenvalues_mse': eigenvalues_mse,
        'ood_ratio': ood_count / total_samples
    }
    
    return test_loss, ood_count, total_samples, metrics


def main(hdf5_path, output_dir="./"):
    """
    Main pipeline function that orchestrates the entire workflow.
    
    Args:
        hdf5_path (str): Path to the HDF5 file containing the BdG data
        output_dir (str): Directory to save model checkpoints and results
    """
    print("="*50)
    print("BdG Eigenvalue and Edge Localization Prediction Pipeline")
    print("="*50)
    print(f"Using device: {device}")
    print(f"Data source: {hdf5_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Data Loading
    print("\nInitializing datasets...")
    train_dataset = BdGDataset(hdf5_path, '/train')
    test_dataset = BdGDataset(hdf5_path, '/test')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=128)  # Using test as val for simplicity
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # 2. OOD Statistics Computation
    print("\nComputing OOD statistics from training data...")
    mean, std = compute_ood_stats(train_dataset)
    print(f"Feature means: {mean}")
    print(f"Feature standard deviations: {std}")
    
    # 3. Model Setup
    print("\nInitializing MLP model...")
    model = MLP().to(device)
    print(model)
    
    # 4. Model Training
    print("\nStarting model training...")
    train_history = train_model(model, train_loader, val_loader, checkpoint_dir=output_dir)
    
    # 5. Model Evaluation
    print("\nEvaluating best model...")
    checkpoint_path = os.path.join(output_dir, 'best_model.pth')
    _, _, _, metrics = evaluate_model(model, test_loader, mean, std, checkpoint_path=checkpoint_path)
    
    # 6. Cleanup
    print("\nCleaning up resources...")
    train_dataset.close()
    test_dataset.close()
    
    print("\nPipeline completed successfully!")
    return metrics


if __name__ == "__main__":
    """if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_hdf5_file>")
        sys.exit(1)"""
    
    hdf5_path = "/home/levi/anaconda3/envs/mzm/project_mzm/kitaev_chain_dataset/kitaev_chain_dataset_200000_30000.h5" #sys.argv[1]
    main(hdf5_path)