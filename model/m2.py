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
    """
    def __init__(self, hdf5_file, group):
        self.file = h5py.File(hdf5_file, 'r')
        self.group = self.file[group]
        self.keys = list(self.group.keys())
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        sample = self.group[self.keys[idx]]
        
        # Extract features: L, mu, t, delta, disorder_strength
        features = torch.tensor([
            float(sample['L'][0]), 
            float(sample['mu'][0]), 
            float(sample['t'][0]),
            float(sample['delta'][0]), 
            float(sample['disorder_strength'][0])
        ], dtype=torch.float32)
        
        # Extract targets
        edge_loc = torch.tensor(float(sample['edge_localization'][0]), dtype=torch.float32)
        eigenvalues = torch.tensor(sample['eigenvalues'][:], dtype=torch.float32)
        L = int(features[0])  # Use L from features tensor
        
        # Pad eigenvalues to fixed length (180)
        padded_eigenvalues = torch.zeros(180, dtype=torch.float32)
        padded_eigenvalues[:min(len(eigenvalues), 180)] = eigenvalues[:180]
        
        # Create mask
        mask = torch.zeros(180, dtype=torch.float32)
        mask[:min(2*L, 180)] = 1.0
        
        return features, edge_loc, padded_eigenvalues, mask
    
    def close(self):
        self.file.close()

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for predicting BdG eigenvalues and edge localization.
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 181)  # 1 for edge_loc + 180 for eigenvalues
        )
    
    def forward(self, x):
        return self.layers(x)

def compute_loss(predictions, edge_loc_target, eigen_target, mask, lambda_weight=1.0):
    pred_edge_loc = predictions[:, 0]
    pred_eigen = predictions[:, 1:]
    edge_loss = nn.MSELoss()(pred_edge_loc, edge_loc_target)
    mask = mask.to(predictions.device)
    eigen_loss = torch.mean(((pred_eigen - eigen_target) ** 2) * mask) / (mask.sum() + 1e-8)
    return edge_loss + lambda_weight * eigen_loss

def compute_ood_stats(dataset):
    features = []
    for i in range(len(dataset)):
        feat, _, _, _ = dataset[i]
        features.append(feat.numpy())
    features = np.stack(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return mean, std

def is_ood(features, mean, std, threshold=3.0):
    z_scores = np.abs((features - mean) / (std + 1e-8))
    return np.any(z_scores > threshold)

def train_model(model, train_loader, val_loader, checkpoint_dir="./", epochs=100, patience=10):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"Training on device: {device}")
    print(f"Model will be saved to: {checkpoint_path}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, edge_loc, eigen, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features, edge_loc, eigen, mask = (
                features.to(device), edge_loc.to(device), eigen.to(device), mask.to(device)
            )
            optimizer.zero_grad()
            predictions = model(features)
            loss = compute_loss(predictions, edge_loc, eigen, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, edge_loc, eigen, mask in val_loader:
                features, edge_loc, eigen, mask = (
                    features.to(device), edge_loc.to(device), eigen.to(device), mask.to(device)
                )
                predictions = model(features)
                val_loss += compute_loss(predictions, edge_loc, eigen, mask).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        scheduler.step(val_loss)
        
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
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
            predictions = model(features)
            loss = compute_loss(predictions, edge_loc, eigen, mask)
            test_loss += loss.item()
            pred_edge_loc = predictions[:, 0]
            pred_eigen = predictions[:, 1:]
            edge_loc_mse += nn.MSELoss()(pred_edge_loc, edge_loc).item()
            masked_eigen_mse = torch.mean(((pred_eigen - eigen) ** 2) * mask) / (mask.sum() + 1e-8)
            eigenvalues_mse += masked_eigen_mse.item()
            features_np = features.cpu().numpy()
            for feat in features_np:
                if is_ood(feat, mean, std):
                    ood_count += 1
    
    total_samples = len(test_loader.dataset)
    test_loss /= len(test_loader)
    edge_loc_mse /= len(test_loader)
    eigenvalues_mse /= len(test_loader)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Edge Localization MSE: {edge_loc_mse:.6f}")
    print(f"Eigenvalues MSE: {eigenvalues_mse:.6f}")
    print(f"OOD Samples: {ood_count}/{total_samples} ({100*ood_count/total_samples:.2f}%)")
    
    metrics = {
        'test_loss': test_loss,
        'edge_loc_mse': edge_loc_mse,
        'eigenvalues_mse': eigenvalues_mse,
        'ood_ratio': ood_count / total_samples
    }
    return test_loss, ood_count, total_samples, metrics

def debug_dataset(hdf5_path, group='/train'):
    print(f"Debugging HDF5 file: {hdf5_path}, group: {group}")
    with h5py.File(hdf5_path, 'r') as f:
        print("Available groups:", list(f.keys()))
        if group in f:
            sample_keys = list(f[group].keys())
            if sample_keys:
                first_key = sample_keys[0]
                print(f"First sample key: {first_key}")
                sample = f[group][first_key]
                print("Sample datasets:", list(sample.keys()))
                print("Sample details:")
                for field in sample.keys():
                    value = sample[field][()]
                    if isinstance(value, np.ndarray) and len(value) > 3:
                        print(f"  {field}: shape={value.shape}, type={type(value)}, first 3 elements={value[:3]}")
                    else:
                        print(f"  {field}: {value}, type={type(value)}")
                attrs = dict(sample.attrs)
                print("Sample attributes:", attrs if attrs else "No attributes found on sample")
        else:
            print(f"Group {group} not found")

def main(hdf5_path, output_dir="./", debug=False):
    print("="*50)
    print("BdG Eigenvalue and Edge Localization Prediction Pipeline")
    print("="*50)
    print(f"Using device: {device}")
    print(f"Data source: {hdf5_path}")
    print(f"Output directory: {output_dir}")
    
    if debug:
        debug_dataset(hdf5_path)
    
    os.makedirs(output_dir, exist_ok=True)
    print("\nInitializing datasets...")
    train_dataset = BdGDataset(hdf5_path, '/train')
    test_dataset = BdGDataset(hdf5_path, '/test')
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    print("\nComputing OOD statistics from training data...")
    mean, std = compute_ood_stats(train_dataset)
    print(f"Feature means: {mean}")
    print(f"Feature standard deviations: {std}")
    
    print("\nInitializing MLP model...")
    model = MLP().to(device)
    print(model)
    
    print("\nStarting model training...")
    train_history = train_model(model, train_loader, val_loader, checkpoint_dir=output_dir)
    
    print("\nEvaluating best model...")
    checkpoint_path = os.path.join(output_dir, 'best_model.pth')
    _, _, _, metrics = evaluate_model(model, test_loader, mean, std, checkpoint_path=checkpoint_path)
    
    print("\nCleaning up resources...")
    train_dataset.close()
    test_dataset.close()
    
    print("\nPipeline completed successfully!")
    return metrics

if __name__ == "__main__":
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        sys.argv.remove("--debug")
    if len(sys.argv) != 2:
        print("Usage: python script.py [--debug] <path_to_hdf5_file>")
        hdf5_path = "/home/levi/anaconda3/envs/mzm/project_mzm/kitaev_chain_dataset/kitaev_chain_dataset_200000_30000.h5"
    else:
        hdf5_path = sys.argv[1]
    main(hdf5_path, debug=debug_mode)