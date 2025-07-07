import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional


class BdGDataset(Dataset):
    """Dataset for loading BdG eigenvalue prediction data from HDF5 file."""
    
    def __init__(self, h5_file: str, split: str = 'train', max_padding: int = 180):
        """
        Initialize BdG dataset.
        
        Args:
            h5_file: Path to HDF5 file containing the data
            split: 'train' or 'test'
            max_padding: Maximum length to pad eigenvalues to
        """
        self.h5_file = h5_file
        self.split = split
        self.max_padding = max_padding
        
        # Open HDF5 file to get dataset size
        with h5py.File(h5_file, 'r') as f:
            self.length = len(f[split]['L'])
        
        # Compute mean and std for features (for OOD detection)
        if split == 'train':
            self.compute_feature_stats()
    
    def compute_feature_stats(self):
        """Compute mean and std of training features for OOD detection."""
        with h5py.File(self.h5_file, 'r') as f:
            # Get all features
            L = f[self.split]['L'][:]
            mu = f[self.split]['mu'][:]
            t = f[self.split]['t'][:]
            delta = f[self.split]['delta'][:]
            disorder_strength = f[self.split]['disorder_strength'][:]
            
            # Stack features
            features = np.column_stack([L, mu, t, delta, disorder_strength])
            
            # Compute mean and std
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0)
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing the features, target, and mask
        """
        with h5py.File(self.h5_file, 'r') as f:
            # Get input features
            L = f[self.split]['L'][idx]
            mu = f[self.split]['mu'][idx]
            t = f[self.split]['t'][idx]
            delta = f[self.split]['delta'][idx]
            disorder_strength = f[self.split]['disorder_strength'][idx]
            
            # Get targets
            edge_localization = f[self.split]['edge_localization'][idx]
            eigenvalues = f[self.split]['spectra/eigenvalues'][idx]
            
            # Create mask based on L
            mask = np.zeros(self.max_padding, dtype=np.float32)
            # Make sure valid_length is an integer
            valid_length = min(int(2 * L), self.max_padding)
            mask[:valid_length] = 1.0
            
            # Pad eigenvalues to max_padding
            padded_eigenvalues = np.zeros(self.max_padding, dtype=np.float32)
            # Make sure length calculations are integers
            padded_eigenvalues[:min(len(eigenvalues), self.max_padding)] = eigenvalues[:min(len(eigenvalues), self.max_padding)]
            
            # Prepare input features and targets
            features = np.array([L, mu, t, delta, disorder_strength], dtype=np.float32)
            
            # Combine edge_localization with padded eigenvalues
            targets = np.concatenate([[edge_localization], padded_eigenvalues])
            
            # Check if sample is OOD (if in test set and feature_mean/std available)
            is_ood = False
            if hasattr(self, 'feature_mean') and hasattr(self, 'feature_std'):
                # Calculate z-scores for features
                z_scores = np.abs((features - self.feature_mean) / (self.feature_std + 1e-8))
                # Consider OOD if any feature has z-score > 3.0
                is_ood = np.any(z_scores > 3.0)
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'is_ood': torch.tensor(is_ood, dtype=torch.bool)
        }


def get_dataloaders(h5_file: str, batch_size: int = 256, 
                   num_workers: int = 4, pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for train and test sets.
    
    Args:
        h5_file: Path to HDF5 file containing the data
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory in DataLoader
        
    Returns:
        train_loader, test_loader
    """
    # Create train and test datasets
    train_dataset = BdGDataset(h5_file, split='train')
    test_dataset = BdGDataset(h5_file, split='test')
    
    # Copy mean and std from train dataset to test dataset for OOD detection
    test_dataset.feature_mean = train_dataset.feature_mean
    test_dataset.feature_std = train_dataset.feature_std
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def inspect_h5_file(h5_file: str) -> None:
    """
    Inspect the structure of an HDF5 file.
    
    Args:
        h5_file: Path to HDF5 file
    """
    with h5py.File(h5_file, 'r') as f:
        print(f"HDF5 file structure: {h5_file}\n")
        
        def print_attrs(name, obj):
            print(f"{name}: {type(obj).__name__}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
                print(f"  First few values: {obj[:5]}")
                print()
        
        # Recursively visit all groups and datasets
        f.visititems(print_attrs)